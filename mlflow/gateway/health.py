"""
Target health tracking, ejection and recovery probing for traffic-split routes.

A traffic route distributes requests across several named destinations using
configured traffic percentages. This module adds:

- request-level failover: a request that hits a retryable upstream error
  (connection failure, timeout, 5xx) is transferred to another destination,
  bounded by ``TrafficFailoverConfig.max_failovers``;
- health-based ejection: after ``failure_threshold`` consecutive failures from
  real traffic, a destination is temporarily ejected and the configured weights
  are renormalized over the remaining healthy destinations;
- recovery probing: ejected destinations are probed with a lightweight probe at
  ``probe_interval_seconds`` (after ``cooldown_seconds``) and rejoin on success;
- a process-global registry keyed by destination name, so health state survives
  in-process configuration reloads as long as the destination identity (its
  endpoint name) is unchanged.

Probe failures and request failures are counted separately. Client errors such
as 404/422 never trigger failover, are never counted, and never eject a target.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from mlflow.gateway.base_models import ResponseModel
from mlflow.gateway.config import TrafficFailoverConfig
from mlflow.gateway.exceptions import AIGatewayException

if TYPE_CHECKING:
    from mlflow.gateway.providers.base import BaseProvider

_logger = logging.getLogger(__name__)

FAILURE_SOURCE_REQUEST = "request"
FAILURE_SOURCE_PROBE = "probe"


class TargetState(str, Enum):
    HEALTHY = "healthy"
    EJECTED = "ejected"


class FailureInfoModel(ResponseModel):
    timestamp: float
    source: str
    error_type: str
    status_code: int | None = None
    message: str


class TargetHealthModel(ResponseModel):
    name: str
    state: str
    configured_weight: int
    # The destination's share (0-100) after renormalizing weights over the
    # currently healthy destinations; 0 while ejected.
    effective_weight: float
    consecutive_failures: int
    request_failure_count: int
    request_success_count: int
    probe_failure_count: int
    probe_success_count: int
    last_failure: FailureInfoModel | None = None
    ejected_reason: str | None = None
    ejected_at: float | None = None
    last_probe_at: float | None = None
    next_probe_at: float | None = None


class RouteHealthModel(ResponseModel):
    route: str
    failover_enabled: bool
    destinations: list[TargetHealthModel]


class NoHealthyTargetsError(AIGatewayException):
    """Raised when a traffic route has no destination that can serve traffic."""

    def __init__(self, route_name: str):
        super().__init__(
            status_code=503,
            detail=(
                f"Route '{route_name}' has no available targets: every destination is "
                "ejected or has exhausted its failover attempts."
            ),
        )


def is_retryable_upstream_error(exc: BaseException) -> bool:
    """Whether a failed target call may be retried on another target.

    Only connection failures, timeouts and server-side 5xx errors are
    retryable. Client errors (e.g. 404, 422) and everything that is not clearly
    an upstream-availability problem must surface to the caller unchanged.
    """
    # TimeoutError covers asyncio.TimeoutError on Python 3.11+; the explicit
    # asyncio reference keeps the classification correct on 3.10.
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True

    try:
        import aiohttp

        if isinstance(exc, aiohttp.ClientError):
            return True
    except ImportError:
        pass

    # fastapi.HTTPException carries ``status_code``; AIGatewayException too.
    # aiohttp.ClientResponseError (if it ever leaks) carries ``status``.
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    return isinstance(status, int) and 500 <= status < 600


def _failure_status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status if isinstance(status, int) else None


@dataclass
class TargetHealth:
    """Mutable health record for a single destination identity."""

    name: str
    state: TargetState = TargetState.HEALTHY
    consecutive_failures: int = 0
    request_failure_count: int = 0
    request_success_count: int = 0
    probe_failure_count: int = 0
    probe_success_count: int = 0
    last_failure: FailureInfoModel | None = None
    ejected_reason: str | None = None
    ejected_at: float | None = None
    # Scheduling timestamps use the same injectable wall clock (epoch seconds).
    next_probe_at: float | None = None
    last_probe_at: float | None = None

    def snapshot(
        self,
        configured_weight: int,
        healthy_weight_sum: int | None,
        healthy_count: int = 0,
    ) -> TargetHealthModel:
        if self.state == TargetState.EJECTED:
            effective_weight = 0.0
        elif healthy_weight_sum:
            effective_weight = configured_weight * 100.0 / healthy_weight_sum
        elif healthy_count:
            # Every surviving target carries zero configured weight: selection
            # balances evenly, so mirror that in the snapshot.
            effective_weight = 100.0 / healthy_count
        else:
            effective_weight = 0.0
        return TargetHealthModel(
            name=self.name,
            state=self.state.value,
            configured_weight=configured_weight,
            effective_weight=effective_weight,
            consecutive_failures=self.consecutive_failures,
            request_failure_count=self.request_failure_count,
            request_success_count=self.request_success_count,
            probe_failure_count=self.probe_failure_count,
            probe_success_count=self.probe_success_count,
            last_failure=self.last_failure,
            ejected_reason=self.ejected_reason,
            ejected_at=self.ejected_at,
            last_probe_at=self.last_probe_at,
            next_probe_at=self.next_probe_at,
        )


class TargetHealthRegistry:
    """Process-global store of health records and probe locks keyed by target name.

    The registry deliberately outlives individual app/provider instances: a
    configuration reload rebuilds providers but reuses the record for every
    destination whose endpoint name is unchanged, so ejection/recovery state is
    not reset by a config rewrite.
    """

    def __init__(self) -> None:
        self._states: dict[str, TargetHealth] = {}
        self._probe_locks: dict[str, asyncio.Lock] = {}

    def get(self, name: str) -> TargetHealth:
        if (state := self._states.get(name)) is None:
            state = TargetHealth(name=name)
            self._states[name] = state
        return state

    def reset(self) -> None:
        """Drop all tracked state (used by tests)."""
        self._states.clear()
        self._probe_locks.clear()

    def record_request_failure(
        self, name: str, exc: BaseException, *, threshold: int, cooldown_seconds: float, now: float
    ) -> TargetHealth:
        state = self.get(name)
        state.consecutive_failures += 1
        state.request_failure_count += 1
        state.last_failure = FailureInfoModel(
            timestamp=now,
            source=FAILURE_SOURCE_REQUEST,
            error_type=type(exc).__name__,
            status_code=_failure_status_code(exc),
            message=str(exc) or type(exc).__name__,
        )
        # Eject exactly once at the threshold. Further failures while ejected
        # keep counting but cannot re-trigger a state transition, so concurrent
        # requests cannot flap the state.
        if state.state == TargetState.HEALTHY and state.consecutive_failures >= threshold:
            state.state = TargetState.EJECTED
            state.ejected_reason = f"reached the consecutive failure threshold ({threshold})"
            state.ejected_at = now
            state.next_probe_at = now + cooldown_seconds
            _logger.warning(
                "Gateway target '%s' ejected after %d consecutive failures",
                name,
                state.consecutive_failures,
            )
        return state

    def record_request_success(self, name: str) -> TargetHealth:
        state = self.get(name)
        state.request_success_count += 1
        state.consecutive_failures = 0
        if state.state == TargetState.EJECTED:
            # Defensive: ejected targets are never selected, but if a late
            # in-flight request succeeds, treat it as evidence of recovery.
            state.state = TargetState.HEALTHY
            state.ejected_reason = None
            state.ejected_at = None
            state.next_probe_at = None
        return state

    async def run_due_probe(
        self,
        name: str,
        provider: "BaseProvider",
        config: TrafficFailoverConfig,
        *,
        now: float,
    ) -> None:
        """Run the recovery probe for ``name`` if it is due.

        A per-target lock collapses concurrent callers onto a single probe; the
        losers await and observe the refreshed state.
        """
        state = self.get(name)
        if state.state != TargetState.EJECTED or state.next_probe_at is None:
            return
        if now < state.next_probe_at:
            return

        if (lock := self._probe_locks.get(name)) is None:
            lock = asyncio.Lock()
            self._probe_locks[name] = lock

        async with lock:
            # Re-check under the lock: another request may have probed first.
            state = self.get(name)
            if (
                state.state != TargetState.EJECTED
                or state.next_probe_at is None
                or now < state.next_probe_at
            ):
                return

            state.last_probe_at = now
            try:
                await provider.health_probe()
            except Exception as e:
                # Probe failures are counted separately from request failures
                # and never contribute to the consecutive-request-failure count.
                state.probe_failure_count += 1
                state.last_failure = FailureInfoModel(
                    timestamp=now,
                    source=FAILURE_SOURCE_PROBE,
                    error_type=type(e).__name__,
                    status_code=_failure_status_code(e),
                    message=str(e) or type(e).__name__,
                )
                state.next_probe_at = now + config.probe_interval_seconds
                _logger.debug("Recovery probe for target '%s' failed: %s", name, e)
            else:
                state.probe_success_count += 1
                state.state = TargetState.HEALTHY
                state.consecutive_failures = 0
                state.ejected_reason = None
                state.ejected_at = None
                state.next_probe_at = None
                _logger.info("Gateway target '%s' recovered and rejoined routing", name)


# Process-global registry shared by every traffic route provider in this worker.
health_registry = TargetHealthRegistry()


class RouteHealthManager:
    """Per-route selection/failover bookkeeping bound to the global registry."""

    def __init__(
        self,
        route_name: str,
        names: list[str],
        providers: list["BaseProvider"],
        weights: list[int],
        failover_config: TrafficFailoverConfig,
        registry: TargetHealthRegistry | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not (len(names) == len(providers) == len(weights)):
            raise ValueError("'names', 'providers' and 'weights' must have the same length.")
        self._route_name = route_name
        self._names = names
        self._providers = providers
        self._weights = [int(w) for w in weights]
        self._config = failover_config
        self._registry = registry or health_registry
        self._clock = clock

    @property
    def failover_config(self) -> TrafficFailoverConfig:
        return self._config

    def _healthy_indices(self) -> list[int]:
        return [
            i
            for i, name in enumerate(self._names)
            if self._registry.get(name).state == TargetState.HEALTHY
        ]

    async def _probe_if_due(self) -> None:
        now = self._clock()
        for i, name in enumerate(self._names):
            state = self._registry.get(name)
            if (
                state.state == TargetState.EJECTED
                and state.next_probe_at is not None
                and now >= state.next_probe_at
            ):
                await self._registry.run_due_probe(name, self._providers[i], self._config, now=now)

    async def select(self, exclude: set[str] | None = None) -> tuple[str, "BaseProvider"] | None:
        """Pick a healthy destination with weight renormalized over survivors.

        Triggers due recovery probes before selecting. Returns None when no
        healthy, non-excluded destination remains.
        """
        await self._probe_if_due()
        exclude = exclude or set()
        indices = [i for i in self._healthy_indices() if self._names[i] not in exclude]
        if not indices:
            return None

        weights = np.array([self._weights[i] for i in indices], dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            # All surviving targets carry zero configured weight: balance evenly.
            probabilities = np.full(len(indices), 1.0 / len(indices))
        else:
            probabilities = weights / total
        chosen = int(np.random.choice(np.arange(len(indices)), p=probabilities))
        index = indices[chosen]
        return self._names[index], self._providers[index]

    def record_failure(self, name: str, exc: BaseException) -> None:
        self._registry.record_request_failure(
            name,
            exc,
            threshold=self._config.failure_threshold,
            cooldown_seconds=self._config.cooldown_seconds,
            now=self._clock(),
        )

    def record_success(self, name: str) -> None:
        self._registry.record_request_success(name)

    def snapshot(self) -> RouteHealthModel:
        return snapshot_enabled_route(
            self._route_name, self._names, self._weights, registry=self._registry
        )


def snapshot_enabled_route(
    route_name: str,
    names: list[str],
    weights: list[int],
    *,
    registry: TargetHealthRegistry | None = None,
) -> RouteHealthModel:
    """Build a live health snapshot for a failover-enabled route."""
    registry = registry or health_registry
    states = [registry.get(name).state for name in names]
    healthy_weight_sum = sum(
        weight for weight, state in zip(weights, states) if state == TargetState.HEALTHY
    )
    healthy_count = sum(state == TargetState.HEALTHY for state in states)
    return RouteHealthModel(
        route=route_name,
        failover_enabled=True,
        destinations=[
            registry.get(name).snapshot(weight, healthy_weight_sum, healthy_count)
            for name, weight in zip(names, weights)
        ],
    )


def snapshot_disabled_route(route_name: str, destinations: list[Any]) -> RouteHealthModel:
    """Build a static all-healthy snapshot for routes without failover enabled."""
    return RouteHealthModel(
        route=route_name,
        failover_enabled=False,
        destinations=[
            TargetHealthModel(
                name=destination.name,
                state=TargetState.HEALTHY.value,
                configured_weight=destination.traffic_percentage,
                effective_weight=float(destination.traffic_percentage),
                consecutive_failures=0,
                request_failure_count=0,
                request_success_count=0,
                probe_failure_count=0,
                probe_success_count=0,
            )
            for destination in destinations
        ],
    )
