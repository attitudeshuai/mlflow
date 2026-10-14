import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.entities.webhook_delivery import (
    WebhookDeliveryAttemptResult,
    WebhookDeliveryStatus,
    WebhookDeliveryTrigger,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.model_registry.dbmodels.models import SqlWebhookDelivery
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore

EVENT_RM = WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)
EVENT_MV = WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)


@pytest.fixture
def store(tmp_path: Path) -> SqlAlchemyStore:
    return SqlAlchemyStore(f"sqlite:///{tmp_path / 'test.db'}")


@pytest.fixture
def second_store(tmp_path: Path) -> SqlAlchemyStore:
    # Same database file, separate store instance (separate session maker)
    return SqlAlchemyStore(f"sqlite:///{tmp_path / 'test.db'}")


def _create(store: SqlAlchemyStore, delivery_id: str = "d1", **kwargs):
    params = {
        "webhook_id": "wh-1",
        "webhook_name": "hook",
        "webhook_url": "https://example.com/hook",
        "event": EVENT_RM,
        "trigger": WebhookDeliveryTrigger.INITIAL,
        "payload_summary": '{"name": "m"}',
    }
    params.update(kwargs)
    return store.create_webhook_delivery(delivery_id=delivery_id, **params)


def _insert_delivery_row(store: SqlAlchemyStore, delivery_id: str, creation_timestamp: int, **kwargs):
    """Insert directly with an explicit creation timestamp for ordering tests."""
    params = {
        "webhook_id": "wh-1",
        "entity": "registered_model",
        "action": "created",
        "status": "FAILURE",
        "trigger": "INITIAL",
        "root_delivery_id": delivery_id,
        "redelivery_count": 0,
        "workspace": "default",
    }
    params.update(kwargs)
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlWebhookDelivery(
                delivery_id=delivery_id,
                creation_timestamp=creation_timestamp,
                **params,
            )
        )
        session.flush()


def test_create_attempt_complete_get_round_trip(store: SqlAlchemyStore):
    created = _create(store)
    assert created.status == WebhookDeliveryStatus.PENDING
    assert created.root_delivery_id == "d1"
    assert created.parent_delivery_id is None
    assert created.redelivery_count == 0
    assert created.event == EVENT_RM
    assert created.payload_summary == '{"name": "m"}'

    store.add_webhook_delivery_attempt(
        delivery_id="d1",
        attempt_number=1,
        started_timestamp=10,
        completed_timestamp=20,
        result=WebhookDeliveryAttemptResult.HTTP_ERROR,
        http_status=500,
        error_type="HTTPError",
        error_message="boom",
    )
    store.add_webhook_delivery_attempt(
        delivery_id="d1",
        attempt_number=2,
        started_timestamp=30,
        completed_timestamp=40,
        result=WebhookDeliveryAttemptResult.SUCCESS,
        http_status=200,
    )
    store.complete_webhook_delivery(
        delivery_id="d1",
        status=WebhookDeliveryStatus.SUCCESS,
        completed_timestamp=40,
    )

    fetched = store.get_webhook_delivery("d1")
    assert fetched.status == WebhookDeliveryStatus.SUCCESS
    assert fetched.completed_timestamp == 40
    assert [a.attempt_number for a in fetched.attempts] == [1, 2]
    assert fetched.attempts[0].result == WebhookDeliveryAttemptResult.HTTP_ERROR
    assert fetched.attempts[0].http_status == 500
    assert fetched.attempts[0].error_type == "HTTPError"
    assert fetched.attempts[1].result == WebhookDeliveryAttemptResult.SUCCESS


def test_get_missing_delivery_raises(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.get_webhook_delivery("missing")
    from mlflow.exceptions import ErrorCode

    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_add_attempt_to_missing_delivery_raises(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="not found"):
        store.add_webhook_delivery_attempt(
            delivery_id="missing",
            attempt_number=1,
            started_timestamp=1,
            completed_timestamp=2,
            result=WebhookDeliveryAttemptResult.SUCCESS,
            http_status=200,
        )


def test_list_filters_by_webhook_event_status_and_time(store: SqlAlchemyStore):
    _insert_delivery_row(store, "d1", 100, webhook_id="wh-a", status="SUCCESS")
    _insert_delivery_row(
        store,
        "d2",
        200,
        webhook_id="wh-a",
        entity="model_version",
        action="created",
        status="FAILURE",
        next_retry_at=None,
    )
    _insert_delivery_row(store, "d3", 300, webhook_id="wh-b", status="FAILURE")

    all_rows = store.list_webhook_deliveries()
    assert [d.delivery_id for d in all_rows] == ["d3", "d2", "d1"]

    by_webhook = store.list_webhook_deliveries(webhook_id="wh-a")
    assert {d.delivery_id for d in by_webhook} == {"d1", "d2"}

    by_event = store.list_webhook_deliveries(event=EVENT_MV)
    assert {d.delivery_id for d in by_event} == {"d2"}

    by_status = store.list_webhook_deliveries(status=WebhookDeliveryStatus.FAILURE)
    assert {d.delivery_id for d in by_status} == {"d2", "d3"}

    by_time_start = store.list_webhook_deliveries(start_timestamp=200)
    assert {d.delivery_id for d in by_time_start} == {"d2", "d3"}

    by_time_end = store.list_webhook_deliveries(end_timestamp=200)
    assert {d.delivery_id for d in by_time_end} == {"d1", "d2"}

    by_window = store.list_webhook_deliveries(start_timestamp=150, end_timestamp=250)
    assert {d.delivery_id for d in by_window} == {"d2"}

    combined = store.list_webhook_deliveries(
        webhook_id="wh-a", status=WebhookDeliveryStatus.SUCCESS
    )
    assert {d.delivery_id for d in combined} == {"d1"}


def test_list_pagination(store: SqlAlchemyStore):
    for i in range(5):
        _insert_delivery_row(store, f"d{i}", 100 + i)

    page1 = store.list_webhook_deliveries(max_results=2)
    assert [d.delivery_id for d in page1] == ["d4", "d3"]
    assert page1.token is not None

    page2 = store.list_webhook_deliveries(max_results=2, page_token=page1.token)
    assert [d.delivery_id for d in page2] == ["d2", "d1"]

    page3 = store.list_webhook_deliveries(max_results=2, page_token=page2.token)
    assert [d.delivery_id for d in page3] == ["d0"]
    assert page3.token is None


@pytest.mark.parametrize("bad_max_results", [-1, 1001])
def test_list_rejects_invalid_max_results(store: SqlAlchemyStore, bad_max_results):
    with pytest.raises(MlflowException, match="max_results"):
        store.list_webhook_deliveries(max_results=bad_max_results)


def test_list_zero_max_results_coerced_to_default(store: SqlAlchemyStore):
    # Matches the existing list_webhooks convention: 0/None coerce to the default.
    page = store.list_webhook_deliveries(max_results=0)
    assert len(page) == 0


def test_list_rejects_inverted_time_range(store: SqlAlchemyStore):
    with pytest.raises(MlflowException, match="start_timestamp"):
        store.list_webhook_deliveries(start_timestamp=200, end_timestamp=100)


def test_prune_evicts_oldest_with_attempts(store: SqlAlchemyStore):
    for i in range(8):
        _insert_delivery_row(store, f"d{i}", 100 + i)

    store.prune_webhook_deliveries(5)

    remaining = store.list_webhook_deliveries(max_results=100)
    assert {d.delivery_id for d in remaining} == {"d3", "d4", "d5", "d6", "d7"}

    # Prune again at the same size is a no-op.
    store.prune_webhook_deliveries(5)
    assert len(store.list_webhook_deliveries(max_results=100)) == 5


def test_prune_deletes_child_attempt_rows(store: SqlAlchemyStore):
    _insert_delivery_row(store, "old", 100)
    _insert_delivery_row(store, "new", 200)
    store.add_webhook_delivery_attempt(
        delivery_id="old",
        attempt_number=1,
        started_timestamp=1,
        completed_timestamp=2,
        result=WebhookDeliveryAttemptResult.SUCCESS,
        http_status=200,
    )

    store.prune_webhook_deliveries(1)

    with store.ManagedSessionMaker() as session:
        from mlflow.store.model_registry.dbmodels.models import SqlWebhookDeliveryAttempt

        assert (
            session.query(SqlWebhookDeliveryAttempt)
            .filter(SqlWebhookDeliveryAttempt.delivery_id == "old")
            .count()
            == 0
        )


def test_claim_due_deliveries_atomic_and_batched(
    store: SqlAlchemyStore, second_store: SqlAlchemyStore
):
    _insert_delivery_row(store, "due1", 100, next_retry_at=500)
    _insert_delivery_row(store, "due2", 200, next_retry_at=600)
    _insert_delivery_row(store, "due3", 300, next_retry_at=700)
    _insert_delivery_row(store, "notdue", 400, next_retry_at=9999)
    _insert_delivery_row(store, "notscheduled", 500, next_retry_at=None)
    _insert_delivery_row(store, "success", 600, status="SUCCESS", next_retry_at=100)

    claimed1 = store.claim_due_webhook_deliveries(now=1000, limit=2)
    assert {d.delivery_id for d in claimed1} == {"due1", "due2"}

    # A second (concurrent) claimant cannot re-claim the same records; it sees
    # only the remaining due row.
    claimed2 = second_store.claim_due_webhook_deliveries(now=1000, limit=10)
    assert [d.delivery_id for d in claimed2] == ["due3"]

    assert store.claim_due_webhook_deliveries(now=1000, limit=10) == []

    with store.ManagedSessionMaker() as session:
        rows = (
            session.query(SqlWebhookDelivery.delivery_id, SqlWebhookDelivery.next_retry_at)
            .filter(SqlWebhookDelivery.delivery_id.in_(["due1", "due2", "due3"]))
            .all()
        )
        assert all(row.next_retry_at is None for row in rows)
        untouched = (
            session.query(SqlWebhookDelivery)
            .filter(
                SqlWebhookDelivery.delivery_id.in_(["notdue", "notscheduled", "success"])
            )
            .all()
        )
        assert {row.delivery_id: row.next_retry_at for row in untouched} == {
            "notdue": 9999,
            "notscheduled": None,
            "success": 100,
        }


def test_claim_due_deliveries_concurrent_claimants_no_duplicates(
    store: SqlAlchemyStore, second_store: SqlAlchemyStore
):
    for i in range(20):
        _insert_delivery_row(store, f"due{i}", 100 + i, next_retry_at=500)

    results: list[list[str]] = []
    lock = threading.Lock()

    def claim(candidate_store):
        claimed = [d.delivery_id for d in candidate_store.claim_due_webhook_deliveries(now=1000, limit=50)]
        with lock:
            results.append(claimed)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(claim, store), executor.submit(claim, second_store)]
        for future in futures:
            future.result()

    all_claimed = [delivery_id for batch in results for delivery_id in batch]
    assert len(all_claimed) == len(set(all_claimed)) == 20


def test_schedule_retry_sets_and_clears_next_retry_at(store: SqlAlchemyStore):
    _create(store, "d1", trigger=WebhookDeliveryTrigger.AUTOMATIC)
    store.complete_webhook_delivery(
        delivery_id="d1",
        status=WebhookDeliveryStatus.FAILURE,
        completed_timestamp=100,
        failure_reason="bad",
    )

    store.schedule_webhook_delivery_retry(delivery_id="d1", next_retry_at=420)
    with store.ManagedSessionMaker() as session:
        row = session.query(SqlWebhookDelivery).filter_by(delivery_id="d1").one()
        assert row.next_retry_at == 420
        assert row.failure_reason == "bad"

    store.schedule_webhook_delivery_retry(delivery_id="d1", next_retry_at=None)
    with store.ManagedSessionMaker() as session:
        row = session.query(SqlWebhookDelivery).filter_by(delivery_id="d1").one()
        assert row.next_retry_at is None


def test_non_default_workspace_records_invisible_to_default_query(
    store: SqlAlchemyStore,
):
    _create(store, "team-delivery", workspace="team-a")

    # Base store has no workspace context on these threads, so request-style
    # queries resolve to the default workspace and must not see team-a rows.
    with pytest.raises(MlflowException, match="not found"):
        store.get_webhook_delivery("team-delivery")
    assert store.list_webhook_deliveries() == []

    # Writer path with explicit workspace still resolves the row.
    store.complete_webhook_delivery(
        delivery_id="team-delivery",
        status=WebhookDeliveryStatus.SUCCESS,
        completed_timestamp=1,
        workspace="team-a",
    )


def test_concurrent_writers_and_readers_never_see_torn_rows(store: SqlAlchemyStore):
    stop = threading.Event()
    errors: list[Exception] = []

    def writer(worker_id: int):
        try:
            for i in range(20):
                delivery_id = f"w{worker_id}-d{i}"
                _create(store, delivery_id)
                for attempt_number in range(1, 4):
                    store.add_webhook_delivery_attempt(
                        delivery_id=delivery_id,
                        attempt_number=attempt_number,
                        started_timestamp=i,
                        completed_timestamp=i + 1,
                        result=WebhookDeliveryAttemptResult.SUCCESS,
                        http_status=200,
                    )
                store.complete_webhook_delivery(
                    delivery_id=delivery_id,
                    status=WebhookDeliveryStatus.SUCCESS,
                    completed_timestamp=i + 1,
                )
        except Exception as e:
            errors.append(e)
            stop.set()

    def reader():
        try:
            while not stop.is_set():
                for delivery in store.list_webhook_deliveries(max_results=50):
                    # Every visible header row must have its mandatory columns populated.
                    assert delivery.delivery_id
                    assert delivery.webhook_id
                    assert delivery.event is not None
                    assert delivery.creation_timestamp > 0
                    assert delivery.status is not None
                    for attempt in delivery.attempts:
                        assert attempt.attempt_number > 0
                        assert attempt.started_timestamp >= 0
                        assert attempt.completed_timestamp >= attempt.started_timestamp
                        assert attempt.result is not None
        except Exception as e:
            errors.append(e)
            stop.set()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(writer, i) for i in range(3)]
        reader_future = executor.submit(reader)
        for future in futures:
            future.result()
        stop.set()
        reader_future.result()

    assert errors == []
    assert len(store.list_webhook_deliveries(max_results=1000)) == 60
