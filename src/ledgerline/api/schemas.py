"""Request and response models.

The transaction schema is generated from the loaded model's feature list, so
the API accepts exactly the raw fields the model was trained on: a typo or an
unknown field is a 422, not a silently ignored input. Fields with a small,
fixed set of values (product, card network, device type) only accept values
seen in training. Free-text fields (email domain, device name) accept anything;
values the model never saw are treated as missing, since new domains and
devices appear all the time.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

# Always required: without these a transaction can't be scored or audited.
CORE = {
    "TransactionID": (Annotated[int, Field(ge=0)], ...),
    "TransactionDT": (Annotated[int, Field(ge=0, description="seconds since reference")], ...),
    "TransactionAmt": (Annotated[float, Field(gt=0, le=1_000_000)], ...),
    "card1": (Annotated[int, Field(ge=0)], ...),
}
# Categorical fields restricted to the values seen in training.
CLOSED_SETS = {"ProductCD", "card4", "card6", "DeviceType"}
REQUIRED_CLOSED = {"ProductCD"}
MAX_BATCH = 1_000

STRICT = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


def transaction_model(raw_features: list[str], vocab: dict[str, list[str]]) -> type[BaseModel]:
    fields: dict[str, Any] = dict(CORE)
    for name in raw_features:
        if name in fields:
            continue
        if name in CLOSED_SETS and name in vocab:
            kind = Literal[tuple(vocab[name])]  # type: ignore[valid-type]
        elif name in vocab:
            kind = Annotated[str, Field(min_length=1, max_length=200)]
        else:
            kind = float
        fields[name] = (kind, ...) if name in REQUIRED_CLOSED else (kind | None, None)
    return create_model("Transaction", __config__=STRICT, **fields)


class Reason(BaseModel):
    feature: str
    value: float | str | None
    contribution: float = Field(description="push toward fraud, in log-odds")
    text: str


class ScoreResponse(BaseModel):
    transaction_id: int
    score: float
    flagged: bool
    threshold: float
    model_version: str
    reasons: list[Reason] = Field(description="top reasons; filled in for flagged transactions")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Records stay loosely typed here so one bad record can't reject the batch;
    # each is validated on its own.
    transactions: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_BATCH)


class BatchError(BaseModel):
    index: int
    transaction_id: Any = None
    errors: list[str]


class BatchResponse(BaseModel):
    scored: int
    failed: int
    results: list[ScoreResponse]
    errors: list[BatchError]


class MerchantRequest(BaseModel):
    model_config = STRICT
    description: str = Field(min_length=1, max_length=300, examples=["PAYPAL *DATACAMP JYF7455"])
    direction: Literal["debit", "credit"] = "debit"


class MerchantResponse(BaseModel):
    category: str
    confidence: float | None = Field(
        description="TF-IDF probability; null when the LLM answered (it gives no calibrated score)"
    )
    method: Literal["llm", "tfidf"]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    database: bool
    merchant_llm: bool
