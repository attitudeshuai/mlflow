import aiohttp
import pytest
from fastapi import HTTPException

from mlflow.gateway.config import EndpointConfig, TrafficFailoverConfig
from mlflow.gateway.health import (
    NoHealthyTargetsError,
    RouteHealthManager,
    TargetHealthRegistry,
    TargetState,
)
from mlflow.gateway.providers.base import TrafficRouteProvider

from tests.gateway.providers.test_openai import chat_config


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class FakeTarget:
    def __init__(self, name, failure=None, probe_failure=None):
        self.name = name
        self.failure = failure
        self.probe_failure = probe_failure
        self.chat_calls = 0
        self.probe_calls = 0

    async def chat(self, payload):
        self.chat_calls += 1
        if self.failure is not None:
            raise self.failure
        return {"served_by": self.name}

    async def chat_stream(self, payload):
        self.chat_calls += 1
        if self.failure is not None:
            raise self.failure
        yield "chunk-1"
        yield "chunk-2"

    async def health_probe(self):
        self.probe_calls += 1
        if self.probe_failure is not None:
            raise self.probe_failure


class MidStreamFailTarget(FakeTarget):
    async def chat_stream(self, payload):
        self.chat_calls += 1
        yield "partial"
        raise HTTPException(status_code=503, detail="connection reset")


def _build_provider(
    targets,
    weights=(100, 0),
    *,
    failover=None,
    clock=None,
):
    configs = [EndpointConfig(**chat_config()) for _ in targets]
    names = [t.name for t in targets]
    provider = TrafficRouteProvider(
        configs=configs,
        traffic_splits=list(weights),
        routing_strategy="TRAFFIC_SPLIT",
        target_names=names,
        route_name="r",
        failover_config=failover,
    )
    provider._providers = targets
    if failover is not None and len(targets) > 1:
        # Rebind the manager with an isolated registry and fake targets so
        # tests do not leak health state through the process-global registry.
        provider._health = RouteHealthManager(
            route_name="r",
            names=names,
            providers=targets,
            weights=list(weights),
            failover_config=failover,
            registry=TargetHealthRegistry(),
            clock=clock or (lambda: 1000.0),
        )
    return provider


def _failover(**kwargs):
    defaults = dict(
        max_failovers=1,
        failure_threshold=100,
        cooldown_seconds=30,
        probe_interval_seconds=10,
    )
    defaults.update(kwargs)
    return TrafficFailoverConfig(**defaults)


@pytest.mark.asyncio
async def test_failover_to_healthy_target():
    a = FakeTarget("a", failure=HTTPException(status_code=503, detail="down"))
    b = FakeTarget("b")
    provider = _build_provider([a, b], failover=_failover())

    result = await provider.chat(None)

    assert result == {"served_by": "b"}
    assert a.chat_calls == 1
    assert b.chat_calls == 1


@pytest.mark.asyncio
async def test_connection_error_and_timeout_fail_over():
    a = FakeTarget("a", failure=aiohttp.ClientConnectionError("refused"))
    b = FakeTarget("b")
    provider = _build_provider([a, b], failover=_failover())
    assert (await provider.chat(None))["served_by"] == "b"


@pytest.mark.asyncio
async def test_client_error_is_not_retried_or_counted():
    failure = HTTPException(status_code=404, detail="unknown model")
    a = FakeTarget("a", failure=failure)
    b = FakeTarget("b")
    provider = _build_provider([a, b], failover=_failover(failure_threshold=1))

    with pytest.raises(HTTPException) as exc_info:
        await provider.chat(None)
    assert exc_info.value.status_code == 404
    assert a.chat_calls == 1
    assert b.chat_calls == 0

    snapshot = provider._health.snapshot()
    state_a = next(d for d in snapshot.destinations if d.name == "a")
    assert state_a.consecutive_failures == 0
    assert state_a.state == "healthy"


@pytest.mark.asyncio
async def test_max_failovers_zero_propagates_error():
    a = FakeTarget("a", failure=HTTPException(status_code=503, detail="down"))
    b = FakeTarget("b")
    provider = _build_provider([a, b], failover=_failover(max_failovers=0))

    with pytest.raises(HTTPException) as exc_info:
        await provider.chat(None)
    assert exc_info.value.status_code == 503
    assert a.chat_calls == 1
    assert b.chat_calls == 0


@pytest.mark.asyncio
async def test_all_targets_failed_raises_clear_error():
    a = FakeTarget("a", failure=HTTPException(status_code=503, detail="down"))
    b = FakeTarget("b", failure=HTTPException(status_code=500, detail="down"))
    provider = _build_provider([a, b], failover=_failover())

    with pytest.raises(NoHealthyTargetsError) as exc_info:
        await provider.chat(None)
    assert exc_info.value.status_code == 503
    assert "r" in exc_info.value.detail
    assert "no available targets" in exc_info.value.detail


@pytest.mark.asyncio
async def test_ejection_removes_target_from_later_requests():
    a = FakeTarget("a", failure=HTTPException(status_code=503, detail="down"))
    b = FakeTarget("b")
    provider = _build_provider([a, b], failover=_failover(failure_threshold=2))

    for _ in range(3):
        result = await provider.chat(None)
        assert result["served_by"] == "b"

    # a is tried twice (until threshold) and never again after ejection.
    assert a.chat_calls == 2
    assert b.chat_calls == 3
    state = provider._health.snapshot().destinations[0]
    assert state.state == "ejected"
    assert state.ejected_reason is not None
    assert state.last_failure.status_code == 503


@pytest.mark.asyncio
async def test_all_ejected_immediately_reports_unavailable():
    a = FakeTarget("a", failure=HTTPException(status_code=503, detail="down"))
    b = FakeTarget("b", failure=HTTPException(status_code=530, detail="origin"))
    provider = _build_provider([a, b], failover=_failover(failure_threshold=1, max_failovers=2))

    # First request ejects both targets while trying them.
    with pytest.raises(NoHealthyTargetsError):
        await provider.chat(None)
    # Subsequent requests fail immediately with the explicit error.
    with pytest.raises(NoHealthyTargetsError) as exc_info:
        await provider.chat(None)
    assert exc_info.value.status_code == 503
    assert a.chat_calls == 1
    assert b.chat_calls == 1


@pytest.mark.asyncio
async def test_single_target_enabled_degrades_to_direct_call():
    a = FakeTarget("a", failure=HTTPException(status_code=503, detail="down"))
    provider = _build_provider([a], weights=(100,), failover=_failover())

    assert provider._health is None
    with pytest.raises(HTTPException) as exc_info:
        await provider.chat(None)
    assert exc_info.value.status_code == 503
    assert "no available targets" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_disabled_route_keeps_legacy_behavior():
    a = FakeTarget("a", failure=HTTPException(status_code=503, detail="down"))
    b = FakeTarget("b")
    provider = _build_provider([a, b], failover=None)

    assert provider._health is None
    with pytest.raises(HTTPException) as exc_info:
        await provider.chat(None)
    assert exc_info.value.status_code == 503
    assert b.chat_calls == 0


@pytest.mark.asyncio
async def test_stream_failover_before_first_chunk():
    a = FakeTarget("a", failure=HTTPException(status_code=503, detail="down"))
    b = FakeTarget("b")
    provider = _build_provider([a, b], failover=_failover())

    chunks = [chunk async for chunk in provider.chat_stream(None)]
    assert chunks == ["chunk-1", "chunk-2"]
    assert a.chat_calls == 1
    assert b.chat_calls == 1


@pytest.mark.asyncio
async def test_stream_mid_failure_is_not_retried():
    a = MidStreamFailTarget("a")
    b = FakeTarget("b")
    provider = _build_provider([a, b], failover=_failover())

    with pytest.raises(HTTPException):
        chunks = [chunk async for chunk in provider.chat_stream(None)]
        assert chunks == ["partial"]
    assert a.chat_calls == 1
    assert b.chat_calls == 0


@pytest.mark.asyncio
async def test_probe_success_readmits_target():
    clock = FakeClock()
    a = FakeTarget("a", failure=HTTPException(status_code=503, detail="down"))
    b = FakeTarget("b")
    provider = _build_provider(
        [a, b],
        failover=_failover(failure_threshold=1, cooldown_seconds=30, probe_interval_seconds=10),
        clock=clock,
    )

    # a fails and is ejected; the same request fails over to b.
    assert (await provider.chat(None))["served_by"] == "b"
    assert provider._health.snapshot().destinations[0].state == "ejected"

    # The target recovers; after the cooldown the next request's probe succeeds
    # and a rejoins the weight allocation (it carries 100% of configured weight).
    a.failure = None
    clock.advance(31)
    assert (await provider.chat(None))["served_by"] == "a"
    assert a.probe_calls == 1
    assert provider._health.snapshot().destinations[0].state == TargetState.HEALTHY.value
