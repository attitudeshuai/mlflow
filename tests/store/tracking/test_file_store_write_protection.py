"""Tests for optional cross-process write protection of the filesystem tracking store."""

import json
import logging
import multiprocessing as mp
import os
import threading
import time
from unittest import mock

import pytest

from mlflow.entities import Metric, Param, RunTag
from mlflow.entities.run_status import RunStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.store.tracking._file_store_lock import (
    CrossProcessLock,
    LockUnsupportedError,
)
from mlflow.store.tracking.file_store import FileStore
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

LOCK_LOGGER = "mlflow.store.tracking._file_store_lock"


@pytest.fixture(autouse=True)
def _allow_file_store(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    # Default the protection knobs so tests don't pick up ambient configuration.
    monkeypatch.delenv("MLFLOW_FILE_STORE_WRITE_PROTECTION_ENABLED", raising=False)
    monkeypatch.delenv("MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("MLFLOW_FILE_STORE_LOCK_STALE_GRACE_SECONDS", raising=False)
    monkeypatch.delenv("MLFLOW_FILE_STORE_LOCK_UNSUPPORTED_POLICY", raising=False)


def _store(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("MLFLOW_FILE_STORE_WRITE_PROTECTION_ENABLED", "true")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    return FileStore(str(tmp_path.joinpath("mlruns")))


# --------------------------------------------------------------------------------------
# Configuration validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_name", "bad_value"),
    [
        ("MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS", "abc"),
        ("MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS", "0"),
        ("MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS", "-3"),
        ("MLFLOW_FILE_STORE_LOCK_STALE_GRACE_SECONDS", "xyz"),
        ("MLFLOW_FILE_STORE_LOCK_STALE_GRACE_SECONDS", "-0.1"),
        ("MLFLOW_FILE_STORE_LOCK_UNSUPPORTED_POLICY", "bogus"),
    ],
)
def test_invalid_lock_config_fails_at_construction(tmp_path, monkeypatch, env_name, bad_value):
    monkeypatch.setenv("MLFLOW_FILE_STORE_WRITE_PROTECTION_ENABLED", "true")
    monkeypatch.setenv(env_name, bad_value)
    with pytest.raises(MlflowException) as exc:
        FileStore(str(tmp_path.joinpath("mlruns")))
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    assert env_name in exc.value.message


def test_non_numeric_bool_value_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_FILE_STORE_WRITE_PROTECTION_ENABLED", "maybe")
    with pytest.raises(ValueError, match="MLFLOW_FILE_STORE_WRITE_PROTECTION_ENABLED"):
        FileStore(str(tmp_path.joinpath("mlruns")))


# --------------------------------------------------------------------------------------
# Default-off behavior and on-disk layout
# --------------------------------------------------------------------------------------


def test_disabled_by_default_creates_no_lock_dir_and_works(tmp_path):
    store = FileStore(str(tmp_path.joinpath("mlruns")))
    assert store._lock_config.enabled is False
    exp_id = store.create_experiment("exp")
    run = store.create_run(exp_id, "u", 1, [], "rn")
    store.log_metric(run.info.run_id, Metric("m", 1.0, 1, 0))
    store.set_tag(run.info.run_id, RunTag("t", "v"))
    assert not os.path.exists(store._locks_dir)
    # The run metadata write still goes through the legacy truncating path.
    run_dir = store._get_run_dir(exp_id, run.info.run_id)
    assert os.path.exists(os.path.join(run_dir, FileStore.META_DATA_FILE_NAME))


def test_enabled_lock_dir_is_excluded_from_experiment_listing(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    exp_id = store.create_experiment("exp")
    run = store.create_run(exp_id, "u", 1, [], "rn")
    store.set_tag(run.info.run_id, RunTag("t", "v"))

    assert os.path.isdir(store._locks_dir)
    assert os.path.isdir(os.path.join(store._locks_dir, FileStore.EXPERIMENT_LOCKS_FOLDER_NAME))
    assert os.path.isdir(os.path.join(store._locks_dir, FileStore.RUN_LOCKS_FOLDER_NAME))
    active = {e.experiment_id for e in store.search_experiments()}
    assert active == {FileStore.DEFAULT_EXPERIMENT_ID, exp_id}


# --------------------------------------------------------------------------------------
# Lock mechanics: timeout, reentrancy, stale recovery
# --------------------------------------------------------------------------------------


def _hold_in_thread(lock):
    ready = threading.Event()
    release = threading.Event()

    def _run():
        lock.acquire()
        ready.set()
        release.wait(10)
        lock.release()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert ready.wait(5)
    return thread, release


def test_run_lock_timeout_message_identifies_run(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch, MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS=0.3)
    exp_id = store.create_experiment("exp")
    run = store.create_run(exp_id, "u", 1, [], "rn")
    thread, release = _hold_in_thread(store._run_lock(run.info.run_id))
    try:
        with pytest.raises(MlflowException, match="Timed out after 0\\.3s") as exc:
            store.set_tag(run.info.run_id, RunTag("x", "y"))
        message = exc.value.message
        assert f"run '{run.info.run_id}'" in message
        assert "pid" in message and "lock file" in message
    finally:
        release.set()
        thread.join(5)


def test_experiment_lock_timeout_message_identifies_experiment(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch, MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS=0.3)
    exp_id = store.create_experiment("exp")
    thread, release = _hold_in_thread(store._experiment_lock(exp_id))
    try:
        with pytest.raises(MlflowException) as exc:
            store.rename_experiment(exp_id, "new")
        assert f"experiment '{exp_id}'" in exc.value.message
        assert "0.3s" in exc.value.message
    finally:
        release.set()
        thread.join(5)


def test_lock_is_reentrant_on_same_thread(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch, MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS=0.3)
    exp_id = store.create_experiment("exp")
    run = store.create_run(exp_id, "u", 1, [RunTag("a", "b")], "rn")
    # create_run takes the experiment lock and set_tag the run lock; updating the
    # same run right afterwards must not self-deadlock.
    store.set_tag(run.info.run_id, RunTag("c", "d"))
    with store._run_lock(run.info.run_id):
        with store._run_lock(run.info.run_id):
            store.update_run_info(run.info.run_id, RunStatus.FINISHED, 9, None)
    assert store.get_run(run.info.run_id).info.end_time == 9


def test_live_lock_older_than_grace_is_not_stolen(tmp_path, monkeypatch):
    # A lock that is actually held by a live process must not be reclaimed even
    # if its metadata looks old; the contender waits and times out.
    store = _store(
        tmp_path,
        monkeypatch,
        MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS=0.3,
        MLFLOW_FILE_STORE_LOCK_STALE_GRACE_SECONDS=0,
    )
    exp_id = store.create_experiment("exp")
    run = store.create_run(exp_id, "u", 1, [], "rn")
    thread, release = _hold_in_thread(store._run_lock(run.info.run_id))
    try:
        with pytest.raises(MlflowException, match="Timed out"):
            store.log_param(run.info.run_id, Param("p", "v"))
    finally:
        release.set()
        thread.join(5)


def test_stale_lock_left_by_dead_holder_is_reclaimed_with_log(tmp_path, monkeypatch):
    store = _store(
        tmp_path,
        monkeypatch,
        MLFLOW_FILE_STORE_LOCK_STALE_GRACE_SECONDS=5,
    )
    exp_id = store.create_experiment("exp")
    run = store.create_run(exp_id, "u", 1, [], "rn")
    lock_path = store._run_lock_path(run.info.run_id)
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    stale_meta = b" " + json.dumps(
        {
            "pid": 999999,
            "host": "crashed-host",
            "acquired_at": time.time() - 60,
            "object": f"run '{run.info.run_id}'",
        }
    ).encode("utf-8")
    with open(lock_path, "wb") as f:
        f.write(stale_meta)

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    lock_logger = logging.getLogger(LOCK_LOGGER)
    handler = _Capture()
    lock_logger.addHandler(handler)
    try:
        store.set_tag(run.info.run_id, RunTag("ok", "1"))
    finally:
        lock_logger.removeHandler(handler)

    reclaim_logs = [m for m in records if "Recovering stale write lock" in m]
    assert len(reclaim_logs) == 1
    assert run.info.run_id in reclaim_logs[0]
    assert "pid 999999" in reclaim_logs[0] and "crashed-host" in reclaim_logs[0]
    assert store.get_run(run.info.run_id).data.tags["ok"] == "1"


def test_cross_process_lock_unit(tmp_path):
    path = str(tmp_path.joinpath("x.lock"))
    lock = CrossProcessLock(
        lock_file=path,
        timeout_seconds=0.3,
        stale_grace_seconds=5,
        object_description="test object",
    )
    with lock:
        assert os.path.exists(path)
    assert not os.path.exists(path)


# --------------------------------------------------------------------------------------
# Cross-process concurrency
# --------------------------------------------------------------------------------------


def _concurrent_update_worker(root, run_id, which, rounds, queue, barrier):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    store = FileStore(root)
    errors = []
    try:
        for i in range(rounds):
            # Release both processes into the round at the same time so their
            # read-modify-write sequences genuinely overlap.
            barrier.wait()
            if which == "a":
                # A owns the status and the run name. end_time is left untouched.
                store.update_run_info(run_id, RunStatus.FINISHED, None, None)
                store.set_tag(run_id, RunTag(MLFLOW_RUN_NAME, f"name{i}"))
            else:
                # B owns the end_time; status=None preserves the current status.
                store.update_run_info(run_id, None, 12345, None)
            store.set_tag(run_id, RunTag(f"tag_{which}", f"v{i}"))
            store.log_metric(run_id, Metric(f"metric_{which}", float(i), i, i))
    except Exception as e:
        errors.append(repr(e))
    queue.put(errors)


def test_two_processes_updating_same_run_both_take_effect(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_FILE_STORE_WRITE_PROTECTION_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS", "20")
    root = str(tmp_path.joinpath("mlruns"))
    store = FileStore(root)
    exp_id = store.create_experiment("exp")
    run_id = store.create_run(exp_id, "u", 1, [], "rn").info.run_id

    ctx = mp.get_context("spawn")
    rounds = 20
    queue = ctx.Queue()
    barrier = ctx.Barrier(2)
    proc_a = ctx.Process(
        target=_concurrent_update_worker, args=(root, run_id, "a", rounds, queue, barrier)
    )
    proc_b = ctx.Process(
        target=_concurrent_update_worker, args=(root, run_id, "b", rounds, queue, barrier)
    )
    proc_a.start()
    proc_b.start()
    proc_a.join(90)
    proc_b.join(90)
    assert proc_a.exitcode == 0 and proc_b.exitcode == 0
    errors_a, errors_b = queue.get(90), queue.get(90)
    assert errors_a == [] and errors_b == []

    final = store.get_run(run_id)
    # A's status/name and B's end_time all survive the interleaved updates.
    assert final.info.status == "FINISHED"
    assert final.info.end_time == 12345
    assert final.info.run_name == f"name{rounds - 1}"
    assert final.data.tags["tag_a"] == f"v{rounds - 1}"
    assert final.data.tags["tag_b"] == f"v{rounds - 1}"
    # Neither metric append was lost or torn.
    assert len(store.get_metric_history(run_id, "metric_a")) == rounds
    assert len(store.get_metric_history(run_id, "metric_b")) == rounds


def _mutex_worker(lock_path, marker_path, rounds, queue):
    lock = CrossProcessLock(
        lock_file=lock_path,
        timeout_seconds=20,
        stale_grace_seconds=300,
        object_description="cross-process mutex probe",
    )
    violations = 0
    token = f"{os.getpid()}"
    for i in range(rounds):
        with lock:
            with open(marker_path, "w") as f:
                f.write(f"{token}:{i}")
                f.flush()
            # Give the other process a chance to interfere if the lock fails
            # to exclude it.
            time.sleep(0.02)
            with open(marker_path) as f:
                if f.read() != f"{token}:{i}":
                    violations += 1
    queue.put(violations)


def test_lock_mutually_excludes_cross_process(tmp_path):
    ctx = mp.get_context("spawn")
    lock_path = str(tmp_path.joinpath("locks").joinpath("probe.lock"))
    marker_path = str(tmp_path.joinpath("marker"))
    queue = ctx.Queue()
    rounds = 25
    p1 = ctx.Process(target=_mutex_worker, args=(lock_path, marker_path, rounds, queue))
    p2 = ctx.Process(target=_mutex_worker, args=(lock_path, marker_path, rounds, queue))
    p1.start()
    p2.start()
    p1.join(60)
    p2.join(60)
    assert p1.exitcode == 0 and p2.exitcode == 0
    assert queue.get(60) == 0
    assert queue.get(60) == 0


def _delete_restore_worker(root, exp_id, cycles, queue):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    store = FileStore(root)
    failures = []
    for _ in range(cycles):
        try:
            store.delete_experiment(exp_id)
            store.restore_experiment(exp_id)
        except Exception as e:
            failures.append(f"{type(e).__name__}: {e}")
    queue.put(("deleter", failures))


def _metric_append_worker(root, run_id, count, queue):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    store = FileStore(root)
    landed = 0
    rejected = 0
    unexpected = []
    for i in range(count):
        try:
            store.log_metric(run_id, Metric("m", float(i), i, i))
            landed += 1
        except MlflowException:
            # The run/experiment is mid-delete: an explicit rejection is allowed.
            rejected += 1
        except Exception as e:
            unexpected.append(f"{type(e).__name__}: {e}")
    queue.put(("appender", landed, rejected, unexpected))


def test_metric_appends_during_experiment_delete_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_FILE_STORE_WRITE_PROTECTION_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS", "10")
    root = str(tmp_path.joinpath("mlruns"))
    store = FileStore(root)
    exp_id = store.create_experiment("exp")
    run_id = store.create_run(exp_id, "u", 1, [], "rn").info.run_id

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    deleter = ctx.Process(target=_delete_restore_worker, args=(root, exp_id, 8, queue))
    appender = ctx.Process(target=_metric_append_worker, args=(root, run_id, 80, queue))
    deleter.start()
    appender.start()
    deleter.join(120)
    appender.join(120)
    assert deleter.exitcode == 0 and appender.exitcode == 0
    results = [queue.get(120), queue.get(120)]
    deleter_failures = next(r[1] for r in results if r[0] == "deleter")
    _, landed, rejected, unexpected = next(r for r in results if r[0] == "appender")
    assert deleter_failures == [], deleter_failures
    assert unexpected == [], unexpected
    assert landed + rejected == 80

    # No half-deleted state: experiment is active, trash is empty and the metric
    # file contains only well-formed lines.
    assert store.get_experiment(exp_id).lifecycle_stage == "active"
    trash_entries = os.listdir(os.path.join(root, FileStore.TRASH_FOLDER_NAME))
    assert trash_entries == []
    history = store.get_metric_history(run_id, "m")
    assert len(history) == landed
    assert all(pt.timestamp == pt.step for pt in history)


def test_same_process_threads_also_keep_both_updates(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch, MLFLOW_FILE_STORE_LOCK_TIMEOUT_SECONDS=10)
    exp_id = store.create_experiment("exp")
    run_id = store.create_run(exp_id, "u", 1, [], "rn").info.run_id
    errors = []

    def update(tag, status):
        try:
            for i in range(10):
                store.set_tag(run_id, RunTag(tag, str(i)))
                store.update_run_info(run_id, status, None, None)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=update, args=("t1", RunStatus.FINISHED))
    t2 = threading.Thread(target=update, args=("t2", RunStatus.FINISHED))
    t1.start()
    t2.start()
    t1.join(30)
    t2.join(30)
    assert errors == []
    tags = store.get_run(run_id).data.tags
    assert tags["t1"] == "9" and tags["t2"] == "9"


# --------------------------------------------------------------------------------------
# Atomic metadata replacement and unsupported-filesystem handling
# --------------------------------------------------------------------------------------


def test_run_metadata_uses_atomic_replace_when_enabled(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    exp_id = store.create_experiment("exp")
    run = store.create_run(exp_id, "u", 1, [], "rn")
    run_dir = store._get_run_dir(exp_id, run.info.run_id)
    assert os.path.exists(os.path.join(run_dir, FileStore.META_DATA_FILE_NAME))
    store.update_run_info(run.info.run_id, RunStatus.FINISHED, 7, None)
    # No temporary yaml left behind and the metadata is intact.
    leftovers = [f for f in os.listdir(run_dir) if f != FileStore.META_DATA_FILE_NAME]
    assert set(leftovers) == {"metrics", "params", "artifacts", "tags"}
    assert store.get_run(run.info.run_id).info.end_time == 7


def test_unsupported_filesystem_policy_error_raises_at_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_FILE_STORE_WRITE_PROTECTION_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_FILE_STORE_LOCK_UNSUPPORTED_POLICY", "error")
    with (
        mock.patch(
            "mlflow.store.tracking.file_store.probe_locking_supported",
            side_effect=LockUnsupportedError("simulated unsupported filesystem"),
        ),
        pytest.raises(MlflowException, match="does not support the required file locking"),
    ):
        FileStore(str(tmp_path.joinpath("mlruns")))


def test_unsupported_filesystem_policy_disable_degrades(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_FILE_STORE_WRITE_PROTECTION_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_FILE_STORE_LOCK_UNSUPPORTED_POLICY", "disable")
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    file_store_logger = logging.getLogger("mlflow.store.tracking.file_store")
    handler = _Capture()
    file_store_logger.addHandler(handler)
    try:
        with mock.patch(
            "mlflow.store.tracking.file_store.probe_locking_supported",
            side_effect=LockUnsupportedError("simulated read-only filesystem"),
        ):
            store = FileStore(str(tmp_path.joinpath("mlruns")))
    finally:
        file_store_logger.removeHandler(handler)
    assert store._lock_config.enabled is False
    assert any("Degrading to unlocked operation" in m for m in records)
    # No lock files are created and the store is fully usable.
    exp_id = store.create_experiment("exp")
    run = store.create_run(exp_id, "u", 1, [], "rn")
    store.log_metric(run.info.run_id, Metric("m", 1.0, 1, 0))
    assert not os.path.exists(store._locks_dir)
