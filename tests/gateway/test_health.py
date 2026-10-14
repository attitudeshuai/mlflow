import asyncio

import aiohttp
import pytest
from fastapi import HTTPException

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import TrafficFailoverConfig
from mlflow.gateway.health import (
    RouteHealthManager,
    TargetHealthRegistry,
    TargetState,
    is_retryable_upstream_error,
)


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeProbeTarget:
    def __init__(self, *, fail: bool = False, delay: float = 0.0):
        self.fail = fail
        self.delay = delay
        self.probe_calls = 0

    async def health_probe(self) -> None:
        self.probe_calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise aiohttp.ClientConnectionError("boom")


def _manager(
    names=("a", "b"),
    weights=(80, 20),
    config=None,
    registry=None,
    clock=None,
):
    config = config or TrafficFailoverConfig(
        max_failovers=1, failure_threshold=3, cooldown_seconds=30, probe_interval_seconds=10
    )
    clock = clock or FakeClock()
    providers = [FakeProbeTarget() for _ in names]
    registry = registry or TargetHealthRegistry()
    manager = RouteHealthManager(
        route_name="r",
        names=list(names),
        providers=providers,
        weights=list(weights),
        failover_config=config,
        registry=registry,
        clock=clock,
    )
    return manager, providers, registry, clock


@pytest.mark.parametrize(
    "exc",
    [
        HTTPException(status_code=500, detail="x"),
        HTTPException(status_code=503, detail="x"),
        HTTPException(status_code=530, detail="origin error"),
        aiohttp.ClientConnectionError("refused"),
        aiohttp.ClientError("generic client error"),
        asyncio.TimeoutError(),
    ],
)
def test_retryable_errors(exc):
    assert is_retryable_upstream_error(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        HTTPException(status_code=400, detail="x"),
        HTTPException(status_code=404, detail="not found"),
        HTTPException(status_code=422, detail="bad request"),
        ValueError("nonsense"),
        KeyError("model"),
    ],
)
def test_non_retryable_errors(exc):
    assert is_retryable_upstream_error(exc) is False


def test_invalid_failover_config_rejected():
    with pytest.raises(MlflowException, match="failure_threshold"):
        TrafficFailoverConfig(failure_threshold=0)
    with pytest.raises(MlflowException, match="max_failovers"):
        TrafficFailoverConfig(max_failovers=-1)
    with pytest.raises(MlflowException, match="positive"):
        TrafficFailoverConfig(cooldown_seconds=0)
    with pytest.raises(MlflowException, match="positive"):
        TrafficFailoverConfig(probe_interval_seconds=-5)
    # booleans must not be accepted as numbers
    with pytest.raises(MlflowException):
        TrafficFailoverConfig(failure_threshold=True)
    with pytest.raises(MlflowException):
        TrafficFailoverConfig(cooldown_seconds=False)


@pytest.mark.asyncio
async def test_ejection_at_threshold_and_single_transition():
    manager, _, registry, clock = _manager()
    state = registry.get("a")

    manager.record_failure("a", HTTPException(status_code=503, detail="down"))
    manager.record_failure("a", HTTPException(status_code=503, detail="down"))
    assert state.state == TargetState.HEALTHY
    assert state.consecutive_failures == 2

    ejected_at = clock()
    manager.record_failure("a", HTTPException(status_code=530, detail="origin down"))
    assert state.state == TargetState.EJECTED
    assert state.consecutive_failures == 3
    assert state.ejected_reason and "threshold" in state.ejected_reason
    assert state.ejected_at == ejected_at
    assert state.next_probe_at == ejected_at + 30
    assert state.last_failure.source == "request"
    assert state.last_failure.status_code == 530

    # Further failures must not reset or re-trigger the ejection.
    clock.advance(5)
    manager.record_failure("a", HTTPException(status_code=503, detail="down"))
    assert state.state == TargetState.EJECTED
    assert state.ejected_at == ejected_at


@pytest.mark.asyncio
async def test_success_resets_consecutive_failures():
    manager, _, registry, _ = _manager()
    manager.record_failure("a", HTTPException(status_code=503, detail="down"))
    manager.record_failure("a", aiohttp.ClientConnectionError("refused"))
    manager.record_success("a")
    assert registry.get("a").consecutive_failures == 0
    assert registry.get("a").request_success_count == 1


@pytest.mark.asyncio
async def test_weights_renormalize_over_healthy_targets():
    manager, _, registry, clock = _manager(weights=(80, 20))
    for _ in range(3):
        manager.record_failure("a", HTTPException(status_code=503, detail="down"))
    assert registry.get("a").state == TargetState.EJECTED

    # Before cooldown elapses no probe runs; only b is selectable.
    for _ in range(20):
        name, provider = await manager.select()
        assert name == "b"

    snapshot = manager.snapshot()
    by_name = {d.name: d for d in snapshot.destinations}
    assert by_name["a"].state == "ejected"
    assert by_name["a"].effective_weight == 0.0
    assert by_name["b"].effective_weight == 100.0


@pytest.mark.asyncio
async def test_zero_weight_survivor_still_serves():
    manager, _, registry, _ = _manager(weights=(100, 0))
    for _ in range(3):
        manager.record_failure("a", HTTPException(status_code=503, detail="down"))
    name, _ = await manager.select()
    assert name == "b"


@pytest.mark.asyncio
async def test_probe_gated_by_cooldown_then_interval():
    manager, providers, registry, clock = _manager()
    target = providers[0]
    for _ in range(3):
        manager.record_failure("a", HTTPException(status_code=503, detail="down"))

    # Still inside the cooldown: no probe and target remains ejected.
    clock.advance(29)
    await manager.select()
    assert target.probe_calls == 0
    assert registry.get("a").state == TargetState.EJECTED

    # Cooldown elapsed but the probe still fails: stay ejected, wait the
    # configured interval before the next attempt.
    target.fail = True
    clock.advance(1)
    await manager.select()
    assert target.probe_calls == 1
    state = registry.get("a")
    assert state.state == TargetState.EJECTED
    assert state.probe_failure_count == 1
    assert state.request_failure_count == 3  # probe failures are separate
    assert state.consecutive_failures == 3
    assert state.last_failure.source == "probe"
    assert state.next_probe_at == clock() + 10

    # Not due yet.
    clock.advance(9)
    await manager.select()
    assert target.probe_calls == 1

    # Probe succeeds: target rejoins with reset consecutive failures.
    target.fail = False
    clock.advance(1)
    await manager.select()
    assert target.probe_calls == 2
    state = registry.get("a")
    assert state.state == TargetState.HEALTHY
    assert state.probe_success_count == 1
    assert state.consecutive_failures == 0
    assert state.ejected_reason is None
    assert state.next_probe_at is None


@pytest.mark.asyncio
async def test_concurrent_probes_collapse_to_one():
    config = TrafficFailoverConfig(
        failure_threshold=1, cooldown_seconds=30, probe_interval_seconds=10
    )
    manager, providers, registry, clock = _manager(config=config)
    target = providers[0]
    target.delay = 0.02
    manager.record_failure("a", HTTPException(status_code=503, detail="down"))
    clock.advance(31)

    await asyncio.gather(*[manager.select() for _ in range(10)])
    assert target.probe_calls == 1
    assert registry.get("a").state == TargetState.HEALTHY


@pytest.mark.asyncio
async def test_concurrent_failures_eject_once():
    manager, _, registry, _ = _manager()
    state = registry.get("a")

    async def fail():
        manager.record_failure("a", HTTPException(status_code=503, detail="down"))

    await asyncio.gather(*[fail() for _ in range(20)])
    assert state.state == TargetState.EJECTED
    assert state.request_failure_count == 20
    assert state.ejected_reason == "reached the consecutive failure threshold (3)"


@pytest.mark.asyncio
async def test_all_ejected_returns_none():
    manager, _, registry, _ = _manager()
    for name in ("a", "b"):
        for _ in range(3):
            manager.record_failure(name, HTTPException(status_code=503, detail="down"))
    assert await manager.select() is None


@pytest.mark.asyncio
async def test_exclude_avoids_reselecting_attempted_target():
    manager, _, _, _ = _manager(weights=(100, 0))
    name, _ = await manager.select(exclude={"b"})
    assert name == "a"
    assert await manager.select(exclude={"a", "b"}) is None


@pytest.mark.asyncio
async def test_health_state_preserved_when_manager_rebuilt():
    """A config rebuild must reuse health state by target identity."""
    registry = TargetHealthRegistry()
    manager, _, _, _ = _manager(weights=(80, 20), registry=registry)
    for _ in range(3):
        manager.record_failure("a", HTTPException(status_code=503, detail="down"))
    assert registry.get("a").state == TargetState.EJECTED

    # Simulate hot reload: rewritten weights, fresh manager, same registry.
    rebuilt, _, _, _ = _manager(weights=(50, 50), registry=registry)
    snapshot = rebuilt.snapshot()
    by_name = {d.name: d for d in snapshot.destinations}
    assert by_name["a"].state == "ejected"
    assert by_name["a"].consecutive_failures == 3

    name, _ = await rebuilt.select()
    assert name == "b"
