import pytest

from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.entities.webhook_delivery import (
    WebhookDelivery,
    WebhookDeliveryAttempt,
    WebhookDeliveryAttemptResult,
    WebhookDeliveryStatus,
    WebhookDeliveryTrigger,
)
from mlflow.protos.webhooks_pb2 import (
    WebhookDelivery as ProtoWebhookDelivery,
)
from mlflow.protos.webhooks_pb2 import (
    WebhookDeliveryAttemptResult as ProtoWebhookDeliveryAttemptResult,
)
from mlflow.protos.webhooks_pb2 import (
    WebhookDeliveryStatus as ProtoWebhookDeliveryStatus,
)
from mlflow.protos.webhooks_pb2 import (
    WebhookDeliveryTrigger as ProtoWebhookDeliveryTrigger,
)


@pytest.mark.parametrize(
    ("entity_enum", "proto_value_name"),
    [
        (WebhookDeliveryStatus.PENDING, "PENDING"),
        (WebhookDeliveryStatus.SUCCESS, "SUCCESS"),
        (WebhookDeliveryStatus.FAILURE, "FAILURE"),
        (WebhookDeliveryTrigger.INITIAL, "INITIAL"),
        (WebhookDeliveryTrigger.MANUAL, "MANUAL"),
        (WebhookDeliveryTrigger.AUTOMATIC, "AUTOMATIC"),
        (WebhookDeliveryAttemptResult.SUCCESS, "ATTEMPT_SUCCESS"),
        (WebhookDeliveryAttemptResult.HTTP_ERROR, "HTTP_ERROR"),
        (WebhookDeliveryAttemptResult.REQUEST_ERROR, "REQUEST_ERROR"),
    ],
)
def test_delivery_enum_proto_conversion(entity_enum, proto_value_name):
    assert entity_enum.to_proto()
    proto_int = type(entity_enum).from_proto(entity_enum.to_proto())
    assert proto_int == entity_enum


def test_attempt_result_success_proto_mapping():
    assert WebhookDeliveryAttemptResult.SUCCESS.value == "ATTEMPT_SUCCESS"
    assert (
        WebhookDeliveryAttemptResult.from_proto(ProtoWebhookDeliveryAttemptResult.ATTEMPT_SUCCESS)
        == WebhookDeliveryAttemptResult.SUCCESS
    )


def test_attempt_round_trip():
    attempt = WebhookDeliveryAttempt(
        attempt_number=2,
        started_timestamp=100,
        completed_timestamp=200,
        result=WebhookDeliveryAttemptResult.HTTP_ERROR,
        http_status=503,
        error_type="HTTPError",
        error_message="Service Unavailable",
    )

    proto = attempt.to_proto()
    parsed = WebhookDeliveryAttempt.from_proto(proto)

    assert parsed.attempt_number == 2
    assert parsed.started_timestamp == 100
    assert parsed.completed_timestamp == 200
    assert parsed.result == WebhookDeliveryAttemptResult.HTTP_ERROR
    assert parsed.http_status == 503
    assert parsed.error_type == "HTTPError"
    assert parsed.error_message == "Service Unavailable"


def test_delivery_round_trip_with_redelivery_chain():
    event = WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)
    delivery = WebhookDelivery(
        delivery_id="d-new",
        webhook_id="wh-1",
        webhook_name="n",
        webhook_url="https://example.com/hook",
        event=event,
        payload_summary='{"name": "m", "version": "1"}',
        status=WebhookDeliveryStatus.SUCCESS,
        trigger=WebhookDeliveryTrigger.MANUAL,
        root_delivery_id="d-root",
        parent_delivery_id="d-parent",
        redelivery_count=1,
        creation_timestamp=1000,
        completed_timestamp=1200,
        attempts=[
            WebhookDeliveryAttempt(
                attempt_number=1,
                started_timestamp=1000,
                completed_timestamp=1200,
                result=WebhookDeliveryAttemptResult.SUCCESS,
                http_status=200,
            )
        ],
        workspace="team-a",
    )

    proto = delivery.to_proto()
    assert isinstance(proto, ProtoWebhookDelivery)
    parsed = WebhookDelivery.from_proto(proto)

    assert parsed.delivery_id == "d-new"
    assert parsed.webhook_id == "wh-1"
    assert parsed.webhook_name == "n"
    assert parsed.webhook_url == "https://example.com/hook"
    assert parsed.event == event
    assert parsed.payload_summary == '{"name": "m", "version": "1"}'
    assert parsed.status == WebhookDeliveryStatus.SUCCESS
    assert parsed.trigger == WebhookDeliveryTrigger.MANUAL
    assert parsed.root_delivery_id == "d-root"
    assert parsed.parent_delivery_id == "d-parent"
    assert parsed.redelivery_count == 1
    assert parsed.creation_timestamp == 1000
    assert parsed.completed_timestamp == 1200
    assert parsed.workspace == "team-a"
    assert len(parsed.attempts) == 1
    assert parsed.attempts[0].http_status == 200


def test_initial_delivery_defaults():
    event = WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)
    delivery = WebhookDelivery(
        delivery_id="d1",
        webhook_id="wh-1",
        event=event,
        creation_timestamp=1,
    )

    assert delivery.status == WebhookDeliveryStatus.PENDING
    assert delivery.trigger == WebhookDeliveryTrigger.INITIAL
    assert delivery.root_delivery_id == "d1"
    assert delivery.parent_delivery_id is None
    assert delivery.redelivery_count == 0
    assert delivery.attempts == []
    assert delivery.status.is_terminal() is False
    assert WebhookDeliveryStatus.FAILURE.is_terminal() is True

    proto = delivery.to_proto()
    assert proto.status == ProtoWebhookDeliveryStatus.PENDING
    assert proto.trigger == ProtoWebhookDeliveryTrigger.INITIAL
