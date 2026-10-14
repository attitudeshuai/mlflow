import time
from unittest import mock

import pytest
import yaml
from fastapi.testclient import TestClient

from mlflow.exceptions import MlflowException
from mlflow.gateway.app import create_app_from_config
from mlflow.gateway.config import GatewayConfig, _load_gateway_config
from mlflow.gateway.constants import MLFLOW_GATEWAY_CRUD_ROUTE_V3_BASE, MLFLOW_GATEWAY_ROUTE_BASE
from mlflow.gateway.health import health_registry

from tests.gateway.tools import MockAsyncResponse

CHAT_OK = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "gpt-4o-mini",
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    "choices": [
        {
            "message": {"role": "assistant", "content": "ok", "refusal": None},
            "finish_reason": "stop",
            "index": 0,
        }
    ],
    "headers": {"Content-Type": "application/json"},
}


def _endpoint(name, base):
    return {
        "name": name,
        "endpoint_type": "llm/v1/chat",
        "model": {
            "name": "gpt-4o-mini",
            "provider": "openai",
            "config": {"openai_api_key": "k", "openai_api_base": base},
        },
    }


def _route(name, destinations, failover=None):
    route = {
        "name": name,
        "task_type": "llm/v1/chat",
        "destinations": destinations,
    }
    if failover is not None:
        route["failover"] = failover
    return route


def _config(routes, endpoints=None):
    endpoints = endpoints or [
        _endpoint("ta", "https://a.example.com/v1"),
        _endpoint("tb", "https://b.example.com/v1"),
    ]
    return GatewayConfig(**{"endpoints": endpoints, "routes": routes})


def _client(config):
    return TestClient(create_app_from_config(config))


def _post_side_effect(status_by_base, counters, body=None):
    body = body or CHAT_OK

    def _post(url, *args, **kwargs):
        url = str(url)
        if "a.example.com" in url:
            counters["a"] += 1
            status = status_by_base["a"]
        elif "b.example.com" in url:
            counters["b"] += 1
            status = status_by_base["b"]
        else:
            raise AssertionError(f"unexpected url {url}")
        if status >= 400:
            return MockAsyncResponse({"error": {"message": f"upstream {status}"}}, status=status)
        return MockAsyncResponse(dict(body), status=status)

    return _post


def _head_side_effect(status_by_base, counters):
    def _head(url, *args, **kwargs):
        url = str(url)
        if "a.example.com" in url:
            counters["a"] += 1
            status = status_by_base["a"]
        else:
            counters["b"] += 1
            status = status_by_base["b"]
        return MockAsyncResponse({}, status=status)

    return _head


@pytest.fixture(autouse=True)
def reset_health_registry():
    health_registry.reset()
    yield
    health_registry.reset()


def _health(client, route_name):
    return client.get(f"{MLFLOW_GATEWAY_CRUD_ROUTE_V3_BASE}{route_name}/health")


def test_failover_ejection_recovery_and_health_api():
    config = _config(
        [
            _route(
                "split",
                [
                    # All initial weight on the target that fails deterministically.
                    {"name": "ta", "traffic_percentage": 100},
                    {"name": "tb", "traffic_percentage": 0},
                ],
                failover={
                    "max_failovers": 2,
                    "failure_threshold": 3,
                    "cooldown_seconds": 1.0,
                    "probe_interval_seconds": 0.05,
                },
            )
        ]
    )
    client = _client(config)
    post_calls = {"a": 0, "b": 0}
    head_calls = {"a": 0, "b": 0}
    statuses = {"a": 503, "b": 200}
    head_statuses = {"a": 500, "b": 200}

    with (
        mock.patch(
            "aiohttp.ClientSession.post",
            side_effect=_post_side_effect(statuses, post_calls),
        ),
        mock.patch(
            "aiohttp.ClientSession.head",
            side_effect=_head_side_effect(head_statuses, head_calls),
        ),
    ):
        # Every request succeeds: requests landing on the dead target fail
        # over to the healthy one.
        responses = [
            client.post(
                f"{MLFLOW_GATEWAY_ROUTE_BASE}split/invocations",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            for _ in range(12)
        ]
        assert all(r.status_code == 200 for r in responses)

        # ta is tried exactly threshold times and then ejected.
        assert post_calls["a"] == 3
        assert post_calls["b"] == 12

        resp = _health(client, "split")
        assert resp.status_code == 200
        data = resp.json()
        assert data["failover_enabled"] is True
        by_name = {d["name"]: d for d in data["destinations"]}
        assert by_name["ta"]["state"] == "ejected"
        assert by_name["ta"]["ejected_reason"] is not None
        assert by_name["ta"]["last_failure"]["source"] == "request"
        assert by_name["ta"]["last_failure"]["status_code"] == 503
        assert by_name["ta"]["effective_weight"] == 0.0
        assert by_name["tb"]["state"] == "healthy"
        assert by_name["tb"]["effective_weight"] == 100.0

        # Still inside the cooldown: no probe has run.
        assert head_calls["a"] == 0

        # Cooldown elapsed; the first probe still fails (target not back yet).
        time.sleep(1.1)
        client.post(
            f"{MLFLOW_GATEWAY_ROUTE_BASE}split/invocations",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert head_calls["a"] == 1
        by_name = {d["name"]: d for d in _health(client, "split").json()["destinations"]}
        assert by_name["ta"]["state"] == "ejected"
        # Probe failures are counted separately from request failures.
        assert by_name["ta"]["probe_failure_count"] == 1
        assert by_name["ta"]["request_failure_count"] == 3
        assert by_name["ta"]["consecutive_failures"] == 3
        assert by_name["ta"]["last_failure"]["source"] == "probe"

        # The target recovers: the next probe succeeds and it rejoins.
        head_statuses["a"] = 200
        statuses["a"] = 200
        time.sleep(0.1)
        for _ in range(20):
            client.post(
                f"{MLFLOW_GATEWAY_ROUTE_BASE}split/invocations",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        by_name = {d["name"]: d for d in _health(client, "split").json()["destinations"]}
        assert by_name["ta"]["state"] == "healthy"
        assert by_name["ta"]["probe_success_count"] >= 1
        assert post_calls["a"] > 3  # traffic flows to the recovered target


def test_all_targets_unavailable_returns_clear_503():
    config = _config(
        [
            _route(
                "alldown",
                [
                    {"name": "ta", "traffic_percentage": 50},
                    {"name": "tb", "traffic_percentage": 50},
                ],
                # High threshold: targets stay healthy but every attempt fails.
                failover={"max_failovers": 1, "failure_threshold": 100},
            )
        ]
    )
    client = _client(config)
    with mock.patch(
        "aiohttp.ClientSession.post",
        side_effect=_post_side_effect({"a": 530, "b": 500}, {"a": 0, "b": 0}),
    ):
        resp = client.post(
            f"{MLFLOW_GATEWAY_ROUTE_BASE}alldown/invocations",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 503
    assert "alldown" in resp.json()["detail"]
    assert "no available targets" in resp.json()["detail"]


def test_single_target_degrades_to_direct_failure():
    config = _config(
        routes=[
            _route(
                "single",
                [{"name": "ta", "traffic_percentage": 100}],
                failover={"max_failovers": 2, "failure_threshold": 1},
            )
        ],
    )
    client = _client(config)
    with mock.patch(
        "aiohttp.ClientSession.post",
        side_effect=_post_side_effect({"a": 503, "b": 200}, {"a": 0, "b": 0}),
    ):
        resp = client.post(
            f"{MLFLOW_GATEWAY_ROUTE_BASE}single/invocations",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 503
    # The upstream error is passed through, not a "no targets" error.
    assert "no available targets" not in resp.json()["detail"]


def test_client_error_is_not_failed_over():
    config = _config(
        [
            _route(
                "clienterr",
                [
                    {"name": "ta", "traffic_percentage": 100},
                    {"name": "tb", "traffic_percentage": 0},
                ],
                failover={"max_failovers": 2, "failure_threshold": 1},
            )
        ]
    )
    client = _client(config)
    counters = {"a": 0, "b": 0}
    with mock.patch(
        "aiohttp.ClientSession.post",
        side_effect=_post_side_effect({"a": 404, "b": 200}, counters),
    ):
        resp = client.post(
            f"{MLFLOW_GATEWAY_ROUTE_BASE}clienterr/invocations",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 404
    assert counters == {"a": 1, "b": 0}

    by_name = {d["name"]: d for d in _health(client, "clienterr").json()["destinations"]}
    assert by_name["ta"]["state"] == "healthy"
    assert by_name["ta"]["consecutive_failures"] == 0


def test_disabled_route_health_snapshot():
    config = _config(
        [
            _route(
                "plain",
                [
                    {"name": "ta", "traffic_percentage": 80},
                    {"name": "tb", "traffic_percentage": 20},
                ],
            )
        ]
    )
    client = _client(config)
    resp = _health(client, "plain")
    assert resp.status_code == 200
    data = resp.json()
    assert data["failover_enabled"] is False
    assert {d["name"]: d["state"] for d in data["destinations"]} == {
        "ta": "healthy",
        "tb": "healthy",
    }
    by_name = {d["name"]: d for d in data["destinations"]}
    assert by_name["ta"]["effective_weight"] == 80.0
    assert by_name["tb"]["effective_weight"] == 20.0


def test_health_endpoint_unknown_route_404():
    config = _config([_route("plain", [{"name": "ta", "traffic_percentage": 100}])])
    client = _client(config)
    assert _health(client, "nope").status_code == 404


def test_hot_reload_preserves_health_by_target_identity():
    failover = {"max_failovers": 1, "failure_threshold": 1, "cooldown_seconds": 30}
    app_a = create_app_from_config(
        _config(
            [
                _route(
                    "split",
                    [
                        # All initial weight on ta so the failing target is always
                        # picked first; failover then moves the request to tb.
                        {"name": "ta", "traffic_percentage": 100},
                        {"name": "tb", "traffic_percentage": 0},
                    ],
                    failover=failover,
                )
            ]
        )
    )
    client_a = TestClient(app_a)
    counters = {"a": 0, "b": 0}
    with mock.patch(
        "aiohttp.ClientSession.post",
        side_effect=_post_side_effect({"a": 503, "b": 200}, counters),
    ):
        resp = client_a.post(
            f"{MLFLOW_GATEWAY_ROUTE_BASE}split/invocations",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert {
            d["name"]: d["state"] for d in _health(client_a, "split").json()["destinations"]
        } == {"ta": "ejected", "tb": "healthy"}

        # Simulate a config hot reload in the same process: rewritten weights
        # and brand-new provider instances.
        app_b = create_app_from_config(
            _config(
                [
                    _route(
                        "split",
                        [
                            {"name": "ta", "traffic_percentage": 10},
                            {"name": "tb", "traffic_percentage": 90},
                        ],
                        failover=failover,
                    )
                ]
            )
        )
        client_b = TestClient(app_b)
        data = _health(client_b, "split").json()
        by_name = {d["name"]: d for d in data["destinations"]}
        assert by_name["ta"]["state"] == "ejected"
        assert by_name["ta"]["consecutive_failures"] == 1

        # The rebuilt app does not send traffic to the still-ejected target.
        before = counters["a"]
        for _ in range(5):
            resp = client_b.post(
                f"{MLFLOW_GATEWAY_ROUTE_BASE}split/invocations",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 200
        assert counters["a"] == before


@pytest.mark.parametrize(
    "bad_failover",
    [
        {"failure_threshold": 0},
        {"failure_threshold": -2},
        {"max_failovers": -1},
        {"cooldown_seconds": 0},
        {"cooldown_seconds": -1},
        {"probe_interval_seconds": 0},
        {"probe_interval_seconds": "soon"},
        {"failure_threshold": True},
    ],
)
def test_invalid_failover_config_rejected_at_load(tmp_path, bad_failover):
    config_dict = {
        "endpoints": [
            {
                "name": "ta",
                "endpoint_type": "llm/v1/chat",
                "model": {
                    "name": "gpt-4o-mini",
                    "provider": "openai",
                    "config": {"openai_api_key": "k"},
                },
            }
        ],
        "routes": [
            {
                "name": "r",
                "task_type": "llm/v1/chat",
                "destinations": [{"name": "ta", "traffic_percentage": 100}],
                "failover": {"max_failovers": 1, "failure_threshold": 2, **bad_failover},
            }
        ],
    }
    path = tmp_path / "gateway.yaml"
    path.write_text(yaml.safe_dump(config_dict))
    with pytest.raises(MlflowException):
        _load_gateway_config(path)


def test_failover_config_parses_and_serializes():
    config = _config(
        [
            _route(
                "split",
                [
                    {"name": "ta", "traffic_percentage": 50},
                    {"name": "tb", "traffic_percentage": 50},
                ],
                failover={
                    "max_failovers": 3,
                    "failure_threshold": 7,
                    "cooldown_seconds": 15,
                    "probe_interval_seconds": 5,
                },
            )
        ]
    )
    route = config.routes[0]
    assert route.failover.max_failovers == 3
    assert route.failover.failure_threshold == 7
    assert route.failover.cooldown_seconds == 15.0
    assert route.failover.probe_interval_seconds == 5.0

    # The configured block is exposed via the route payload; omitted when unset.
    client = _client(config)
    payload = client.get(f"{MLFLOW_GATEWAY_CRUD_ROUTE_V3_BASE}split").json()
    assert payload["failover"] == {
        "max_failovers": 3,
        "failure_threshold": 7,
        "cooldown_seconds": 15.0,
        "probe_interval_seconds": 5.0,
    }
