from enum import Enum

from typing_extensions import Self

from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.protos.webhooks_pb2 import (
    WebhookDelivery as ProtoWebhookDelivery,
)
from mlflow.protos.webhooks_pb2 import (
    WebhookDeliveryAttempt as ProtoWebhookDeliveryAttempt,
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


class WebhookDeliveryStatus(str, Enum):
    """Final conclusion of a webhook delivery while it is in flight."""

    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_proto(cls, proto: int) -> Self:
        return cls(ProtoWebhookDeliveryStatus.Name(proto))

    def to_proto(self) -> int:
        return ProtoWebhookDeliveryStatus.Value(self.value)

    def is_terminal(self) -> bool:
        return self in (WebhookDeliveryStatus.SUCCESS, WebhookDeliveryStatus.FAILURE)


class WebhookDeliveryTrigger(str, Enum):
    """What triggered a webhook delivery."""

    INITIAL = "INITIAL"
    MANUAL = "MANUAL"
    AUTOMATIC = "AUTOMATIC"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_proto(cls, proto: int) -> Self:
        return cls(ProtoWebhookDeliveryTrigger.Name(proto))

    def to_proto(self) -> int:
        return ProtoWebhookDeliveryTrigger.Value(self.value)


class WebhookDeliveryAttemptResult(str, Enum):
    """Outcome of a single HTTP attempt within a delivery."""

    SUCCESS = "ATTEMPT_SUCCESS"
    HTTP_ERROR = "HTTP_ERROR"
    REQUEST_ERROR = "REQUEST_ERROR"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_proto(cls, proto: int) -> Self:
        proto_name = ProtoWebhookDeliveryAttemptResult.Name(proto)
        if proto_name == "ATTEMPT_SUCCESS":
            return cls.SUCCESS
        return cls(proto_name)

    def to_proto(self) -> int:
        return ProtoWebhookDeliveryAttemptResult.Value(self.value)


class WebhookDeliveryAttempt:
    """Result of a single HTTP attempt (including in-request retries)."""

    def __init__(
        self,
        attempt_number: int,
        started_timestamp: int,
        completed_timestamp: int,
        result: WebhookDeliveryAttemptResult,
        http_status: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ):
        self._attempt_number = attempt_number
        self._started_timestamp = started_timestamp
        self._completed_timestamp = completed_timestamp
        self._result = result
        self._http_status = http_status
        self._error_type = error_type
        self._error_message = error_message

    @property
    def attempt_number(self) -> int:
        return self._attempt_number

    @property
    def started_timestamp(self) -> int:
        return self._started_timestamp

    @property
    def completed_timestamp(self) -> int:
        return self._completed_timestamp

    @property
    def result(self) -> WebhookDeliveryAttemptResult:
        return self._result

    @property
    def http_status(self) -> int | None:
        return self._http_status

    @property
    def error_type(self) -> str | None:
        return self._error_type

    @property
    def error_message(self) -> str | None:
        return self._error_message

    @classmethod
    def from_proto(cls, proto: ProtoWebhookDeliveryAttempt) -> Self:
        return cls(
            attempt_number=proto.attempt_number,
            started_timestamp=proto.started_timestamp,
            completed_timestamp=proto.completed_timestamp,
            result=WebhookDeliveryAttemptResult.from_proto(proto.result),
            http_status=proto.http_status or None,
            error_type=proto.error_type or None,
            error_message=proto.error_message or None,
        )

    def to_proto(self) -> ProtoWebhookDeliveryAttempt:
        return ProtoWebhookDeliveryAttempt(
            attempt_number=self.attempt_number,
            started_timestamp=self.started_timestamp,
            completed_timestamp=self.completed_timestamp,
            result=self.result.to_proto(),
            http_status=self.http_status,
            error_type=self.error_type,
            error_message=self.error_message,
        )

    def __repr__(self) -> str:
        return (
            f"WebhookDeliveryAttempt(attempt_number={self.attempt_number}, "
            f"result='{self.result}', http_status={self.http_status}, "
            f"error_type='{self.error_type}')"
        )


class WebhookDelivery:
    """Persisted record of one webhook delivery to one subscribed target."""

    def __init__(
        self,
        delivery_id: str,
        webhook_id: str,
        event: WebhookEvent,
        creation_timestamp: int,
        status: str | WebhookDeliveryStatus = WebhookDeliveryStatus.PENDING,
        trigger: str | WebhookDeliveryTrigger = WebhookDeliveryTrigger.INITIAL,
        webhook_name: str | None = None,
        webhook_url: str | None = None,
        payload_summary: str | None = None,
        root_delivery_id: str | None = None,
        parent_delivery_id: str | None = None,
        redelivery_count: int = 0,
        failure_reason: str | None = None,
        completed_timestamp: int | None = None,
        attempts: list[WebhookDeliveryAttempt] | None = None,
        workspace: str | None = None,
    ):
        self._delivery_id = delivery_id
        self._webhook_id = webhook_id
        self._event = event
        self._creation_timestamp = creation_timestamp
        self._status = (
            WebhookDeliveryStatus(status) if isinstance(status, str) else status
        )
        self._trigger = (
            WebhookDeliveryTrigger(trigger) if isinstance(trigger, str) else trigger
        )
        self._webhook_name = webhook_name
        self._webhook_url = webhook_url
        self._payload_summary = payload_summary
        self._root_delivery_id = root_delivery_id or delivery_id
        self._parent_delivery_id = parent_delivery_id
        self._redelivery_count = redelivery_count
        self._failure_reason = failure_reason
        self._completed_timestamp = completed_timestamp
        self._attempts = attempts or []
        self._workspace = workspace

    @property
    def delivery_id(self) -> str:
        return self._delivery_id

    @property
    def webhook_id(self) -> str:
        return self._webhook_id

    @property
    def webhook_name(self) -> str | None:
        return self._webhook_name

    @property
    def webhook_url(self) -> str | None:
        return self._webhook_url

    @property
    def event(self) -> WebhookEvent:
        return self._event

    @property
    def payload_summary(self) -> str | None:
        return self._payload_summary

    @property
    def status(self) -> WebhookDeliveryStatus:
        return self._status

    @property
    def trigger(self) -> WebhookDeliveryTrigger:
        return self._trigger

    @property
    def root_delivery_id(self) -> str:
        return self._root_delivery_id

    @property
    def parent_delivery_id(self) -> str | None:
        return self._parent_delivery_id

    @property
    def redelivery_count(self) -> int:
        return self._redelivery_count

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    @property
    def creation_timestamp(self) -> int:
        return self._creation_timestamp

    @property
    def completed_timestamp(self) -> int | None:
        return self._completed_timestamp

    @property
    def attempts(self) -> list[WebhookDeliveryAttempt]:
        return self._attempts

    @property
    def workspace(self) -> str | None:
        return self._workspace

    @classmethod
    def from_proto(cls, proto: ProtoWebhookDelivery) -> Self:
        return cls(
            delivery_id=proto.delivery_id,
            webhook_id=proto.webhook_id,
            event=WebhookEvent(
                entity=WebhookEntity.from_proto(proto.entity),
                action=WebhookAction.from_proto(proto.action),
            ),
            webhook_name=proto.webhook_name or None,
            webhook_url=proto.webhook_url or None,
            payload_summary=proto.payload_summary or None,
            status=WebhookDeliveryStatus.from_proto(proto.status),
            trigger=WebhookDeliveryTrigger.from_proto(proto.trigger),
            root_delivery_id=proto.root_delivery_id or None,
            parent_delivery_id=proto.parent_delivery_id or None,
            redelivery_count=proto.redelivery_count,
            failure_reason=proto.failure_reason or None,
            creation_timestamp=proto.creation_timestamp,
            completed_timestamp=proto.completed_timestamp or None,
            attempts=[WebhookDeliveryAttempt.from_proto(a) for a in proto.attempts],
            workspace=proto.workspace or None,
        )

    def to_proto(self) -> ProtoWebhookDelivery:
        delivery = ProtoWebhookDelivery(
            delivery_id=self.delivery_id,
            webhook_id=self.webhook_id,
            entity=self.event.entity.to_proto(),
            action=self.event.action.to_proto(),
            status=self.status.to_proto(),
            trigger=self.trigger.to_proto(),
            root_delivery_id=self.root_delivery_id,
            redelivery_count=self.redelivery_count,
            creation_timestamp=self.creation_timestamp,
            attempts=[a.to_proto() for a in self.attempts],
        )
        if self.webhook_name is not None:
            delivery.webhook_name = self.webhook_name
        if self.webhook_url is not None:
            delivery.webhook_url = self.webhook_url
        if self.payload_summary is not None:
            delivery.payload_summary = self.payload_summary
        if self.parent_delivery_id is not None:
            delivery.parent_delivery_id = self.parent_delivery_id
        if self.failure_reason is not None:
            delivery.failure_reason = self.failure_reason
        if self.completed_timestamp is not None:
            delivery.completed_timestamp = self.completed_timestamp
        if self.workspace is not None:
            delivery.workspace = self.workspace
        return delivery

    def __repr__(self) -> str:
        return (
            f"WebhookDelivery(delivery_id='{self.delivery_id}', "
            f"webhook_id='{self.webhook_id}', event={self.event}, "
            f"status='{self.status}', trigger='{self.trigger}', "
            f"root_delivery_id='{self.root_delivery_id}', "
            f"parent_delivery_id='{self.parent_delivery_id}', "
            f"redelivery_count={self.redelivery_count}, "
            f"attempts={len(self.attempts)}, workspace='{self.workspace}')"
        )
