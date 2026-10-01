"""Ledgerline HTTP API.

    uvicorn ledgerline.api.app:create_app --factory

Configuration (environment variables):
    DATABASE_URL     Postgres for card history and the audit log (required)
    MODEL_VERSION    fraud model to serve, e.g. lgbm-5697035084 (default: LATEST)
    FLAG_THRESHOLD   override the model's flag threshold without retraining
    MERCHANT_LLM     "off" to always use TF-IDF for /merchant (default: on if a key exists)
    GEMINI_API_KEY   enables the LLM merchant classifier

Startup fails with a clear message if a model is missing, the database is
unreachable or the threshold is invalid. The service never starts half-working.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ledgerline.api.schemas import (
    BatchError,
    BatchRequest,
    BatchResponse,
    Health,
    MerchantRequest,
    MerchantResponse,
    ScoreResponse,
    transaction_model,
)
from ledgerline.api.scoring import Scorer
from ledgerline.api.store import MemoryStore, PostgresStore, Store, input_hash
from ledgerline.data.load import PROJECT_ROOT
from ledgerline.merchants import tfidf
from ledgerline.merchants.train_tfidf import MERCHANT_DIR, MODEL_FILE
from ledgerline.models.registry import MODELS_DIR, load_model

log = logging.getLogger("ledgerline.api")

# Gemini rejects deadlines under 10 s (400 INVALID_ARGUMENT).
LLM_TIMEOUT_MS = 10_000


class StartupError(RuntimeError):
    """Raised when the service can't start safely."""


@dataclass
class Settings:
    database_url: str | None = field(default_factory=lambda: os.environ.get("DATABASE_URL"))
    model_version: str | None = field(default_factory=lambda: os.environ.get("MODEL_VERSION"))
    flag_threshold: str | None = field(default_factory=lambda: os.environ.get("FLAG_THRESHOLD"))
    merchant_llm: bool = field(
        default_factory=lambda: os.environ.get("MERCHANT_LLM", "on").lower() != "off"
    )
    models_dir: Path = MODELS_DIR
    merchant_dir: Path = MERCHANT_DIR


class MerchantService:
    def __init__(self, model, llm=None, tfidf_version: str = "tfidf"):
        self.model = model
        self.llm = llm
        self.tfidf_version = tfidf_version

    def version(self, method: str) -> str:
        return self.llm.model if method == "llm" and self.llm else self.tfidf_version

    def classify(self, description: str, direction: str) -> MerchantResponse:
        text = f"[{direction}] {description}"
        if self.llm is not None:
            try:
                category = self.llm.predict([text])[0]
                if category:
                    return MerchantResponse(category=category, confidence=None, method="llm")
            except Exception as error:  # timeout, quota, outage: fall back, don't fail
                log.warning("merchant LLM failed (%s); falling back to TF-IDF", error)
        labels, confidence = tfidf.predict_with_confidence(self.model, [text])
        return MerchantResponse(category=labels[0], confidence=float(confidence[0]), method="tfidf")


def _load_scorer(settings: Settings) -> Scorer:
    try:
        model, meta = load_model(settings.model_version, settings.models_dir)
    except FileNotFoundError as error:
        raise StartupError(f"Fraud model unavailable: {error}") from error
    threshold = None
    if settings.flag_threshold:
        try:
            threshold = float(settings.flag_threshold)
        except ValueError as error:
            raise StartupError(
                f"FLAG_THRESHOLD must be a number, got {settings.flag_threshold!r}"
            ) from error
        if not 0 < threshold < 1:
            raise StartupError(f"FLAG_THRESHOLD must be between 0 and 1, got {threshold}")
    return Scorer(model, meta, threshold)


def _load_merchant(settings: Settings) -> MerchantService:
    path = settings.merchant_dir / MODEL_FILE
    if not path.exists():
        raise StartupError(
            f"Merchant model not found at {path}. Run `python -m ledgerline.merchants.train_tfidf`."
        )
    llm = None
    if settings.merchant_llm and os.environ.get("GEMINI_API_KEY"):
        from ledgerline.merchants.llm import GeminiClassifier

        llm = GeminiClassifier(max_retries=1, timeout_ms=LLM_TIMEOUT_MS)
    meta_path = settings.merchant_dir / "tfidf_meta.json"
    version = json.loads(meta_path.read_text())["version"] if meta_path.exists() else "tfidf"
    return MerchantService(joblib.load(path), llm, version)


def _connect(settings: Settings) -> Store:
    if not settings.database_url:
        raise StartupError("DATABASE_URL is not set. Decisions must be audited, so it's required.")
    try:
        return PostgresStore(settings.database_url)
    except Exception as error:
        raise StartupError(f"Can't reach the database: {error}") from error


def create_app(
    settings: Settings | None = None,
    *,
    scorer: Scorer | None = None,
    merchant: MerchantService | None = None,
    store: Store | None = None,
) -> FastAPI:
    """Build the app. Tests pass their own scorer/merchant/store."""
    if settings is None:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")  # real environment variables still win
        settings = Settings()
    scorer = scorer or _load_scorer(settings)
    merchant = merchant or _load_merchant(settings)
    store = store if store is not None else _connect(settings)
    Transaction = transaction_model(scorer.raw_features, scorer.model.encoder.vocab)

    app = FastAPI(
        title="Ledgerline",
        version="1.0",
        description="Card fraud scoring and merchant categorization.",
    )
    app.state.scorer, app.state.store = scorer, store

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": "invalid request", "errors": _messages(exc.errors())},
        )

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    def audit(endpoint: str, payload: dict, output: dict, **fields) -> None:
        try:
            store.record(
                {
                    "endpoint": endpoint,
                    "input": payload,
                    "input_hash": input_hash(payload),
                    "output": jsonable_encoder(output),
                    **fields,
                }
            )
        except Exception as error:
            # No decision leaves the service without an audit record.
            log.error("audit write failed: %s", error)
            raise HTTPException(503, "decision could not be recorded; try again") from error

    def score_one(payload: dict) -> ScoreResponse:
        tx = Transaction.model_validate(payload).model_dump()
        try:
            history = store.history(tx["card1"], tx["TransactionDT"])
        except Exception as error:
            log.error("history read failed: %s", error)
            raise HTTPException(503, "card history unavailable; try again") from error
        score, flagged, reasons = scorer.score(tx, history)
        result = ScoreResponse(
            transaction_id=tx["TransactionID"],
            score=score,
            flagged=flagged,
            threshold=scorer.threshold,
            model_version=scorer.version,
            reasons=reasons,
        )
        audit(
            "/score",
            payload,
            result.model_dump(),
            transaction_id=tx["TransactionID"],
            score=score,
            threshold=scorer.threshold,
            flagged=flagged,
            model_version=scorer.version,
        )
        try:
            store.add_transaction(tx)
        except Exception:  # decision is recorded; a missed history row only affects later features
            log.warning("could not add transaction %s to history", tx["TransactionID"])
        return result

    @app.post("/score", response_model=ScoreResponse)
    def score(payload: dict) -> ScoreResponse:
        try:
            return score_one(payload)
        except ValidationError as error:
            raise RequestValidationError(error.errors()) from error

    @app.post("/score/batch", response_model=BatchResponse)
    def score_batch(batch: BatchRequest) -> BatchResponse:
        results, errors = [], []
        # Oldest first, so earlier records in the batch count as history for later ones.
        order = sorted(
            range(len(batch.transactions)),
            key=lambda i: _sort_time(batch.transactions[i]),
        )
        for i in order:
            record = batch.transactions[i]
            try:
                results.append(score_one(record))
            except ValidationError as error:
                log.info("batch record %d rejected", i)
                errors.append(
                    BatchError(
                        index=i,
                        transaction_id=record.get("TransactionID"),
                        errors=_messages(error.errors()),
                    )
                )
        return BatchResponse(
            scored=len(results),
            failed=len(errors),
            results=results,
            errors=sorted(errors, key=lambda e: e.index),
        )

    @app.post("/merchant", response_model=MerchantResponse)
    def classify_merchant(request: MerchantRequest) -> MerchantResponse:
        result = merchant.classify(request.description, request.direction)
        audit(
            "/merchant",
            request.model_dump(),
            result.model_dump(),
            model_version=merchant.version(result.method),
        )
        return result

    @app.get("/health", response_model=Health)
    def health():
        database = store.healthy()
        # From the audit log, so it covers every worker and survives restarts.
        # With the LLM enabled, TF-IDF answers mean the LLM failed: a fallback
        # keeps requests working but can hide a broken setup.
        try:
            recent = store.merchant_methods(since_minutes=60) if database else {}
        except Exception:
            recent = {}
        body = Health(
            status="ok" if database else "degraded",
            model_loaded=True,
            model_version=scorer.version,
            database=database,
            merchant_llm=merchant.llm is not None,
            merchant_last_hour=recent,
        )
        return JSONResponse(status_code=200 if database else 503, content=body.model_dump())

    return app


def _sort_time(record: dict) -> int:
    value = record.get("TransactionDT") if isinstance(record, dict) else None
    return value if isinstance(value, int) else 0


def _messages(errors) -> list[str]:
    """Pydantic errors as short readable strings: 'TransactionAmt: must be > 0'."""
    out = []
    for e in errors:
        where = ".".join(str(p) for p in e.get("loc", ()) if p not in ("body",))
        out.append(f"{where}: {e.get('msg')}" if where else e.get("msg"))
    return out


def memory_app(**kwargs) -> FastAPI:
    """App backed by an in-memory store, for local experiments without Postgres."""
    return create_app(store=MemoryStore(), **kwargs)
