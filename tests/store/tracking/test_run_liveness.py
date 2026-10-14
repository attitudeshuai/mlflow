import json
import shutil
import time

import pytest

import mlflow.store.tracking.run_liveness as run_liveness_module
from mlflow.entities import Run, RunData, RunInfo, RunStatus, RunTag, ViewType
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.environment_variables import (
    MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS,
    MLFLOW_RUN_LEGACY_STALE_TIMEOUT_SECONDS,
    MLFLOW_RUN_STALE_POLICY,
    MLFLOW_RUN_STALE_TIMEOUT_SECONDS,
)
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.run_liveness import (
    LivenessAction,
    RunLivenessConfig,
    get_run_liveness_config,
    judge_run_liveness,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.mlflow_tags import (
    MLFLOW_LAST_HEARTBEAT_TIME,
    MLFLOW_RUN_FINALIZED_REASON,
    MLFLOW_RUN_FINALIZED_REASON_HEARTBEAT_TIMEOUT,
    MLFLOW_RUN_SUSPECTED_ZOMBIE,
)
from mlflow.utils.time import get_current_time_millis

STALE_CONFIG = RunLivenessConfig(
    heartbeat_enabled=True,
    heartbeat_interval_ms=2_000,
    stale_timeout_ms=10_000,
    stale_policy="mark",
    legacy_timeout_ms=None,
)
FAIL_CONFIG = RunLivenessConfig(
    heartbeat_enabled=True,
    heartbeat_interval_ms=2_000,
    stale_timeout_ms=10_000,
    stale_policy="fail",
    legacy_timeout_ms=None,
)
LEGACY_CONFIG = RunLivenessConfig(
    heartbeat_enabled=True,
    heartbeat_interval_ms=2_000,
    stale_timeout_ms=10_000,
    stale_policy="mark",
    legacy_timeout_ms=30_000,
)


def _make_run(
    status="RUNNING",
    start_time=1_000,
    tags=None,
    lifecycle_stage=LifecycleStage.ACTIVE,
    run_id="run-id",
):
    info = RunInfo(
        run_id=run_id,
        experiment_id="0",
        user_id="user",
        status=status,
        start_time=start_time,
        end_time=None,
        lifecycle_stage=lifecycle_stage,
    )
    tags = [RunTag(key, str(value)) for key, value in (tags or {}).items()]
    return Run(info, RunData(metrics=[], params=[], tags=tags))


# --------------------------------------------------------------------------------------
# Pure judgment logic
# --------------------------------------------------------------------------------------


def test_judge_skips_non_running_and_deleted_runs():
    for status in ("SCHEDULED", "FINISHED", "FAILED", "KILLED"):
        run = _make_run(status=status, tags={MLFLOW_LAST_HEARTBEAT_TIME: 1})
        assert judge_run_liveness(run, STALE_CONFIG, now_ms=100_000).action == LivenessAction.SKIP
    deleted = _make_run(
        tags={MLFLOW_LAST_HEARTBEAT_TIME: 1}, lifecycle_stage=LifecycleStage.DELETED
    )
    assert judge_run_liveness(deleted, STALE_CONFIG, now_ms=100_000).action == LivenessAction.SKIP


def test_judge_no_heartbeat_default_legacy_disabled():
    run = _make_run(start_time=1_000)
    assert judge_run_liveness(run, STALE_CONFIG, now_ms=10_000_000).action == LivenessAction.SKIP


def test_judge_fresh_run_without_first_heartbeat_never_stale_even_with_legacy_window():
    run = _make_run(start_time=90_000)
    assert judge_run_liveness(run, LEGACY_CONFIG, now_ms=100_000).action == LivenessAction.SKIP


def test_judge_old_run_without_heartbeat_stale_with_legacy_window():
    run = _make_run(start_time=1_000)
    verdict = judge_run_liveness(run, LEGACY_CONFIG, now_ms=100_000)
    assert verdict.action == LivenessAction.MARK_STALE
    # No heartbeat exists, "now" is the only meaningful finalization timestamp.
    assert verdict.end_time_ms is None

    fail_config = RunLivenessConfig(
        True, 2_000, 10_000, "fail", legacy_timeout_ms=30_000
    )
    verdict = judge_run_liveness(run, fail_config, now_ms=100_000)
    assert verdict.action == LivenessAction.FAIL_STALE
    assert verdict.end_time_ms == 100_000


def test_judge_fresh_heartbeat_is_healthy_at_boundary():
    run = _make_run(start_time=1_000, tags={MLFLOW_LAST_HEARTBEAT_TIME: 90_000})
    assert judge_run_liveness(run, STALE_CONFIG, now_ms=100_000).action == LivenessAction.HEALTHY


def test_judge_stale_heartbeat_mark_and_fail():
    run = _make_run(start_time=1_000, tags={MLFLOW_LAST_HEARTBEAT_TIME: 80_000})
    verdict = judge_run_liveness(run, STALE_CONFIG, now_ms=100_000)
    assert verdict.action == LivenessAction.MARK_STALE
    assert verdict.end_time_ms is None

    verdict = judge_run_liveness(run, FAIL_CONFIG, now_ms=100_000)
    assert verdict.action == LivenessAction.FAIL_STALE
    # End time must be the last heartbeat time.
    assert verdict.end_time_ms == 80_000


def test_judge_malformed_heartbeat_tag_is_ignored():
    run = _make_run(start_time=1_000, tags={MLFLOW_LAST_HEARTBEAT_TIME: "not-a-timestamp"})
    # Treated as "no heartbeat" -> skipped with the default config.
    assert judge_run_liveness(run, STALE_CONFIG, now_ms=100_000).action == LivenessAction.SKIP


# --------------------------------------------------------------------------------------
# Configuration validation
# --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_liveness_env(monkeypatch):
    for env_var in (
        MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS,
        MLFLOW_RUN_STALE_TIMEOUT_SECONDS,
        MLFLOW_RUN_STALE_POLICY,
        MLFLOW_RUN_LEGACY_STALE_TIMEOUT_SECONDS,
    ):
        monkeypatch.delenv(env_var.name, raising=False)
    yield


@pytest.mark.parametrize(
    ("env_var", "bad_value"),
    [
        (MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS, "0"),
        (MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS, "-5"),
        (MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS, "abc"),
        (MLFLOW_RUN_STALE_TIMEOUT_SECONDS, "0"),
        (MLFLOW_RUN_STALE_TIMEOUT_SECONDS, "x"),
        (MLFLOW_RUN_STALE_POLICY, "nope"),
        (MLFLOW_RUN_STALE_POLICY, ""),
        (MLFLOW_RUN_LEGACY_STALE_TIMEOUT_SECONDS, "-1"),
        (MLFLOW_RUN_LEGACY_STALE_TIMEOUT_SECONDS, "not-an-int"),
    ],
)
def test_invalid_config_raises_at_parse_time(monkeypatch, env_var, bad_value):
    monkeypatch.setenv(env_var.name, bad_value)
    with pytest.raises(ValueError, match=env_var.name):
        get_run_liveness_config()


def test_stale_timeout_smaller_than_interval_rejected(monkeypatch):
    monkeypatch.setenv(MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS.name, "30")
    monkeypatch.setenv(MLFLOW_RUN_STALE_TIMEOUT_SECONDS.name, "10")
    with pytest.raises(ValueError, match=MLFLOW_RUN_STALE_TIMEOUT_SECONDS.name):
        get_run_liveness_config()


def test_legacy_window_smaller_than_stale_timeout_rejected(monkeypatch):
    monkeypatch.setenv(MLFLOW_RUN_STALE_TIMEOUT_SECONDS.name, "30")
    monkeypatch.setenv(MLFLOW_RUN_LEGACY_STALE_TIMEOUT_SECONDS.name, "10")
    with pytest.raises(ValueError, match=MLFLOW_RUN_LEGACY_STALE_TIMEOUT_SECONDS.name):
        get_run_liveness_config()


def test_valid_config_normalizes_to_millis_and_policy(monkeypatch):
    monkeypatch.setenv(MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS.name, "2")
    monkeypatch.setenv(MLFLOW_RUN_STALE_TIMEOUT_SECONDS.name, "10")
    monkeypatch.setenv(MLFLOW_RUN_STALE_POLICY.name, "FAIL")
    monkeypatch.setenv(MLFLOW_RUN_LEGACY_STALE_TIMEOUT_SECONDS.name, "60")
    config = get_run_liveness_config()
    assert config.heartbeat_interval_ms == 2_000
    assert config.stale_timeout_ms == 10_000
    assert config.legacy_timeout_ms == 60_000
    assert config.stale_policy == "fail"


# --------------------------------------------------------------------------------------
# File / SQL store integration
# --------------------------------------------------------------------------------------


@pytest.fixture
def file_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return FileStore(str(tmp_path / "mlruns"))


@pytest.fixture
def sql_store(tmp_path, cached_db):
    db_path = tmp_path / "mlflow.db"
    shutil.copy(cached_db, db_path)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = SqlAlchemyStore(f"sqlite:///{db_path}", artifact_dir.as_uri())
    yield store
    store._dispose_engine()


@pytest.fixture(params=["file", "sql"], ids=["file-store", "sql-store"])
def store(request):
    return request.getfixturevalue(f"{request.param}_store")


@pytest.fixture(autouse=False)
def fast_liveness_env(monkeypatch):
    monkeypatch.setenv(MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS.name, "2")
    monkeypatch.setenv(MLFLOW_RUN_STALE_TIMEOUT_SECONDS.name, "10")
    monkeypatch.delenv(MLFLOW_RUN_STALE_POLICY.name, raising=False)
    monkeypatch.delenv(MLFLOW_RUN_LEGACY_STALE_TIMEOUT_SECONDS.name, raising=False)
    yield


@pytest.fixture
def fake_clock(monkeypatch):
    # Controls "now" as seen by the stale-detection logic. Heartbeat timestamps are passed
    # explicitly, so they are unaffected by the patched clock.
    holder = {"now": get_current_time_millis()}
    monkeypatch.setattr(
        run_liveness_module, "get_current_time_millis", lambda: holder["now"]
    )
    return holder


def _create_active_run(store, experiment_id=None, start_time=None):
    start_time = start_time if start_time is not None else get_current_time_millis()
    exp_id = experiment_id or store.create_experiment("exp", None, [])
    run = store.create_run(exp_id, "user", start_time, [], "run")
    return exp_id, run


@pytest.mark.usefixtures("fast_liveness_env")
def test_fresh_run_is_never_suspected(store, fake_clock):
    exp_id, run = _create_active_run(store, start_time=fake_clock["now"])
    fetched = store.get_run(run.info.run_id)
    assert fetched.info.status == "RUNNING"
    assert fetched.info.suspected_zombie is False
    assert fetched.info.last_heartbeat_time is None
    # Even after some wall-clock time, a run with no heartbeat keeps legacy semantics (no mark).
    fake_clock["now"] += 600_000
    fetched = store.get_run(run.info.run_id)
    assert fetched.info.suspected_zombie is False
    assert [r.info.run_id for r in store.search_runs([exp_id], "", ViewType.ACTIVE_ONLY)] == [
        run.info.run_id
    ]


@pytest.mark.usefixtures("fast_liveness_env")
def test_heartbeat_then_stale_marks_suspected_zombie(store, fake_clock):
    exp_id, run = _create_active_run(store, start_time=fake_clock["now"])
    run_id = run.info.run_id

    heartbeat = fake_clock["now"] + 1_000
    store.record_run_heartbeat(run_id, heartbeat)
    assert store.get_run(run_id).info.last_heartbeat_time == heartbeat

    fake_clock["now"] = heartbeat + 5_000
    assert store.get_run(run_id).info.suspected_zombie is False

    fake_clock["now"] = heartbeat + 20_000
    fetched = store.get_run(run_id)
    assert fetched.info.status == "RUNNING"
    assert fetched.info.suspected_zombie is True
    assert fetched.info.last_heartbeat_time == heartbeat
    assert fetched.data.tags[MLFLOW_RUN_SUSPECTED_ZOMBIE] == "true"

    # search_runs applies the same marking.
    searched = store.search_runs([exp_id], "", ViewType.ACTIVE_ONLY)
    assert len(searched) == 1
    assert searched[0].info.suspected_zombie is True
    assert searched[0].info.last_heartbeat_time == heartbeat


@pytest.mark.usefixtures("fast_liveness_env")
def test_recovered_heartbeat_clears_suspected_marker(store, fake_clock):
    _, run = _create_active_run(store, start_time=fake_clock["now"])
    run_id = run.info.run_id
    store.record_run_heartbeat(run_id, fake_clock["now"])
    fake_clock["now"] += 20_000
    assert store.get_run(run_id).info.suspected_zombie is True

    # The process was only briefly stuck: a new heartbeat arrives and the marker disappears.
    recovered = fake_clock["now"] + 1_000
    store.record_run_heartbeat(run_id, recovered)
    fetched = store.get_run(run_id)
    assert fetched.info.suspected_zombie is False
    assert MLFLOW_RUN_SUSPECTED_ZOMBIE not in fetched.data.tags
    assert fetched.info.last_heartbeat_time == recovered


@pytest.mark.usefixtures("fast_liveness_env")
def test_detection_survives_store_restart(store, fake_clock, tmp_path):
    _, run = _create_active_run(store, start_time=fake_clock["now"])
    run_id = run.info.run_id
    heartbeat = fake_clock["now"]
    store.record_run_heartbeat(run_id, heartbeat)
    fake_clock["now"] += 20_000

    if isinstance(store, FileStore):
        restarted = FileStore(str(tmp_path / "mlruns"))
    else:
        # Reopen the same database with a fresh store instance, as the tracking server would
        # after a restart; detection must not rely on in-memory state.
        restarted = SqlAlchemyStore(
            f"sqlite:///{tmp_path / 'mlflow.db'}", (tmp_path / "artifacts").as_uri()
        )
    fetched = restarted.get_run(run_id)
    assert fetched.info.suspected_zombie is True
    assert fetched.info.last_heartbeat_time == heartbeat


@pytest.mark.usefixtures("fast_liveness_env")
def test_auto_fail_policy_finalizes_stale_run(monkeypatch, store, fake_clock):
    monkeypatch.setenv(MLFLOW_RUN_STALE_POLICY.name, "fail")
    exp_id, run = _create_active_run(store, start_time=fake_clock["now"])
    run_id = run.info.run_id
    heartbeat = fake_clock["now"] + 1_000
    store.record_run_heartbeat(run_id, heartbeat)
    fake_clock["now"] = heartbeat + 20_000

    fetched = store.get_run(run_id)
    assert fetched.info.status == "FAILED"
    assert fetched.info.end_time == heartbeat
    assert fetched.info.suspected_zombie is False
    assert fetched.data.tags[MLFLOW_RUN_FINALIZED_REASON] == (
        MLFLOW_RUN_FINALIZED_REASON_HEARTBEAT_TIMEOUT
    )

    # Idempotent: the run stays failed, the reason remains queryable in get/search.
    fetched = store.get_run(run_id)
    assert fetched.info.status == "FAILED"
    searched = store.search_runs([exp_id], "", ViewType.ACTIVE_ONLY)
    assert searched[0].info.status == "FAILED"
    assert searched[0].data.tags[MLFLOW_RUN_FINALIZED_REASON] == (
        MLFLOW_RUN_FINALIZED_REASON_HEARTBEAT_TIMEOUT
    )

    # Late heartbeats after finalization are ignored and do not resurrect the run.
    store.record_run_heartbeat(run_id, fake_clock["now"] + 10_000)
    assert store.get_run(run_id).info.status == "FAILED"


@pytest.mark.usefixtures("fast_liveness_env")
def test_legacy_window_applies_to_heartbeat_less_runs(monkeypatch, store, fake_clock):
    monkeypatch.setenv(MLFLOW_RUN_LEGACY_STALE_TIMEOUT_SECONDS.name, "30")
    exp_id = store.create_experiment("legacy-exp", None, [])

    # An old run without any heartbeat (old client / historical data) is judged.
    old_run = store.create_run(exp_id, "user", fake_clock["now"], [], "old")
    fake_clock["now"] += 60_000
    assert store.get_run(old_run.info.run_id).info.suspected_zombie is True

    # A newly created run that has not written its first heartbeat yet is not misjudged.
    new_run = store.create_run(exp_id, "user", fake_clock["now"], [], "new")
    assert store.get_run(new_run.info.run_id).info.suspected_zombie is False
    fake_clock["now"] += 10_000
    assert store.get_run(new_run.info.run_id).info.suspected_zombie is False


@pytest.mark.usefixtures("fast_liveness_env")
def test_heartbeat_on_terminated_run_is_noop(store, fake_clock):
    _, run = _create_active_run(store, start_time=fake_clock["now"])
    run_id = run.info.run_id
    end = fake_clock["now"] + 5_000
    store.update_run_info(run_id, RunStatus.FINISHED, end, None)
    store.record_run_heartbeat(run_id, end + 1_000)  # must not raise or write
    fetched = store.get_run(run_id)
    assert fetched.info.status == "FINISHED"
    assert MLFLOW_LAST_HEARTBEAT_TIME not in fetched.data.tags


# --------------------------------------------------------------------------------------
# REST store: heartbeats go over SetTag, detection is server-side (hydrate-only locally)
# --------------------------------------------------------------------------------------


def test_rest_store_record_heartbeat_uses_set_tag(monkeypatch):
    from mlflow.protos.service_pb2 import SetTag as ProtoSetTag
    from mlflow.store.tracking.rest_store import RestStore
    from mlflow.utils.rest_utils import MlflowHostCreds

    rest = RestStore(lambda: MlflowHostCreds("http://localhost:5000"))
    captured = {}

    def fake_call_endpoint(api, body):
        captured["api"] = api
        captured["body"] = json.loads(body)
        return None

    monkeypatch.setattr(rest, "_call_endpoint", fake_call_endpoint)
    rest.record_run_heartbeat("run-id", 1234)
    assert captured["api"] is ProtoSetTag
    assert captured["body"]["run_id"] == "run-id"
    assert captured["body"]["key"] == MLFLOW_LAST_HEARTBEAT_TIME
    assert captured["body"]["value"] == "1234"


def test_rest_store_get_run_hydrates_liveness_without_writing(monkeypatch):
    from mlflow.protos.service_pb2 import GetRun as ProtoGetRun
    from mlflow.protos.service_pb2 import Run as ProtoRun
    from mlflow.store.tracking.rest_store import RestStore
    from mlflow.utils.rest_utils import MlflowHostCreds

    rest = RestStore(lambda: MlflowHostCreds("http://localhost:5000"))
    now = get_current_time_millis()
    proto_run = ProtoRun()
    proto_run.info.run_id = "run-id"
    proto_run.info.run_uuid = "run-id"
    proto_run.info.experiment_id = "0"
    proto_run.info.user_id = "user"
    proto_run.info.status = RunStatus.RUNNING
    proto_run.info.start_time = now - 100_000
    proto_run.info.lifecycle_stage = LifecycleStage.ACTIVE
    proto_run.data.tags.add(key=MLFLOW_LAST_HEARTBEAT_TIME, value=str(now - 100_000))
    proto_run.data.tags.add(key=MLFLOW_RUN_SUSPECTED_ZOMBIE, value="true")

    response_proto = ProtoGetRun.Response()
    response_proto.run.CopyFrom(proto_run)
    calls = []

    def fake_call_endpoint(api, body):
        calls.append(api)
        return response_proto

    monkeypatch.setattr(rest, "_call_endpoint", fake_call_endpoint)
    run = rest.get_run("run-id")
    assert run.info.last_heartbeat_time == now - 100_000
    assert run.info.suspected_zombie is True
    # Only GetRun was issued; the client never persists detection results.
    assert calls == [ProtoGetRun]


# --------------------------------------------------------------------------------------
# Client heartbeat thread
# --------------------------------------------------------------------------------------


def test_fluent_run_heartbeat_thread_lifecycle(monkeypatch, tmp_path):
    import mlflow

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    monkeypatch.setenv("MLFLOW_ENABLE_RUN_HEARTBEAT", "true")
    monkeypatch.setenv(MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS.name, "1")
    monkeypatch.setenv(MLFLOW_RUN_STALE_TIMEOUT_SECONDS.name, "10")

    tracking_uri = tmp_path.joinpath("mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    from mlflow.tracking.fluent import run_id_to_run_heartbeat

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        assert run_id in run_id_to_run_heartbeat
        deadline = time.time() + 5
        while time.time() < deadline:
            tags = mlflow.get_run(run_id).data.tags
            if MLFLOW_LAST_HEARTBEAT_TIME in tags:
                break
            time.sleep(0.05)
        first = mlflow.get_run(run_id).data.tags[MLFLOW_LAST_HEARTBEAT_TIME]
        time.sleep(1.3)
        second = mlflow.get_run(run_id).data.tags[MLFLOW_LAST_HEARTBEAT_TIME]
        assert second > first

    assert run_id not in run_id_to_run_heartbeat
    ended = mlflow.get_run(run_id)
    assert ended.info.status == "FINISHED"
    last_value = ended.data.tags[MLFLOW_LAST_HEARTBEAT_TIME]
    time.sleep(1.2)
    # The thread was stopped on end_run, so the tag is no longer updated.
    assert mlflow.get_run(run_id).data.tags[MLFLOW_LAST_HEARTBEAT_TIME] == last_value


def test_fluent_start_run_raises_on_invalid_interval(monkeypatch, tmp_path):
    import mlflow

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    monkeypatch.setenv(MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS.name, "0")
    mlflow.set_tracking_uri(tmp_path.joinpath("mlruns-bad").as_uri())
    with pytest.raises(ValueError, match=MLFLOW_RUN_HEARTBEAT_INTERVAL_SECONDS.name):
        mlflow.start_run()
