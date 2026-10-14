from mlflow.protos.webhooks_pb2 import (
    ATTEMPT_SUCCESS,
    AUTOMATIC,
    FAILURE,
    GetWebhookDelivery,
    HTTP_ERROR,
    INITIAL,
    ListWebhookDeliveries,
    MANUAL,
    PENDING,
    RedeliverWebhookDelivery,
    SUCCESS,
    WebhookDelivery,
    WebhookDeliveryAttempt,
    WebhookDeliveryAttemptResult,
    WebhookDeliveryStatus,
    WebhookDeliveryTrigger,
    WebhookService,
)
from mlflow.utils.rest_utils import extract_api_info_for_service


def test_delivery_enums_have_expected_values():
    assert WebhookDeliveryStatus.Name(PENDING) == "PENDING"
    assert WebhookDeliveryStatus.Name(SUCCESS) == "SUCCESS"
    assert WebhookDeliveryStatus.Name(FAILURE) == "FAILURE"
    assert WebhookDeliveryTrigger.Name(INITIAL) == "INITIAL"
    assert WebhookDeliveryTrigger.Name(MANUAL) == "MANUAL"
    assert WebhookDeliveryTrigger.Name(AUTOMATIC) == "AUTOMATIC"
    assert WebhookDeliveryAttemptResult.Name(ATTEMPT_SUCCESS) == "ATTEMPT_SUCCESS"
    assert WebhookDeliveryAttemptResult.Name(HTTP_ERROR) == "HTTP_ERROR"
    assert WebhookDeliveryAttemptResult.Name(3) == "REQUEST_ERROR"


def test_delivery_messages_round_trip_fields():
    attempt = WebhookDeliveryAttempt(
        attempt_number=1,
        started_timestamp=100,
        completed_timestamp=200,
        result=ATTEMPT_SUCCESS,
        http_status=200,
    )
    delivery = WebhookDelivery(
        delivery_id="d1",
        webhook_id="wh-1",
        webhook_name="name",
        webhook_url="https://example.com",
        payload_summary='{"name": "m"}',
        status=SUCCESS,
        trigger=INITIAL,
        root_delivery_id="d0",
        redelivery_count=0,
        attempts=[attempt],
    )
    serialized = delivery.SerializeToString()
    parsed = WebhookDelivery()
    parsed.ParseFromString(serialized)

    assert parsed.delivery_id == "d1"
    assert parsed.status == SUCCESS
    assert len(parsed.attempts) == 1
    assert parsed.attempts[0].http_status == 200


def test_service_exposes_delivery_endpoints():
    info = extract_api_info_for_service(WebhookService, "/api/2.0")

    assert info[ListWebhookDeliveries] == ("/api/2.0/mlflow/webhooks/deliveries", "GET")
    assert info[GetWebhookDelivery] == (
        "/api/2.0/mlflow/webhooks/deliveries/{delivery_id}",
        "GET",
    )
    assert info[RedeliverWebhookDelivery] == (
        "/api/2.0/mlflow/webhooks/deliveries/{delivery_id}/redeliver",
        "POST",
    )
