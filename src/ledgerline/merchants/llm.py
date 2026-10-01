"""Gemini classifier with structured output.

The response schema limits the model to the 17 category names, so it can't
invent a category or answer in prose. Descriptions go in batches (one request
classifies many rows), answers are cached on disk keyed by model + prompt
version + description, and every request's token counts are recorded so cost
is measured rather than estimated.

Needs GEMINI_API_KEY in .env. GEMINI_MODEL overrides the default model.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from ledgerline.data.load import PROJECT_ROOT
from ledgerline.merchants.data import CATEGORIES

DEFAULT_MODEL = "gemini-3.1-flash-lite"
# USD per 1M tokens (input, output), from ai.google.dev/gemini-api/docs/pricing on 2026-09-30.
PRICES = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
}
BATCH_SIZE = 40
MAX_RETRIES = 5
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "llm_cache"

PROMPT_VERSION = "v1"
INSTRUCTIONS = """You categorize raw US bank and card transaction descriptors.
Each descriptor starts with [debit] (money out) or [credit] (money in), then the
raw text from the statement: merchant names, payment-processor prefixes like
PAYPAL *, SQ *, TST*, store numbers, addresses and reference codes.

Pick exactly one category per descriptor:
- Education: schools, tuition, online courses and learning platforms
- Entertainment: movies, events, games, streaming purchases, theme parks, betting
- Fees: bank fees and charges (overdraft, ATM, wire, late, annual, foreign transaction)
- Groceries: supermarkets and grocery stores, grocery delivery
- Healthcare: doctors, dentists, hospitals, pharmacies, labs
- Income: payroll, salary, benefits, interest and other money earned
- Insurance: auto, home, health, life insurance premiums
- Mortgage: home loan payments, principal, escrow
- Personal Care: salons, barbers, spas, gyms, cosmetics
- Rent: rent paid to landlords and property managers
- Restaurants: restaurants, cafes, bars, fast food, food delivery
- Shopping: retail and online stores, marketplaces, electronics, clothing
- Subscription: recurring software, apps, media and membership subscriptions
- Transfer: moving money between people or accounts (Zelle, Venmo, wires, ACH transfers)
- Transportation: rideshare, fuel, parking, transit, tolls, car services
- Travel: airlines, hotels, rentals, cruises, travel booking
- Utilities: electricity, water, gas, internet, phone, trash

Use what you know about the merchant. Return one answer for every id."""


def _schema(n: int) -> dict:
    return {
        "type": "ARRAY",
        "minItems": n,
        "maxItems": n,
        "items": {
            "type": "OBJECT",
            "properties": {
                "id": {"type": "INTEGER"},
                "category": {"type": "STRING", "enum": CATEGORIES},
            },
            "required": ["id", "category"],
        },
    }


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    latencies: list[float] = field(default_factory=list)

    def cost_usd(self, model: str) -> float:
        price_in, price_out = PRICES[model]
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1_000_000


class GeminiClassifier:
    def __init__(self, model: str | None = None, cache_dir: Path = CACHE_DIR, client=None):
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        if self.model not in PRICES:
            raise ValueError(f"no price on file for {self.model}; add it to PRICES")
        self._client = client
        self.cache_path = cache_dir / f"{self.model}-{PROMPT_VERSION}.jsonl"
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row["category"]
        self.usage = Usage()

    @property
    def cost_rates(self) -> tuple[float, float]:
        """USD per 1M (input, output) tokens for this model."""
        return PRICES[self.model]

    @property
    def client(self):
        if self._client is None:
            from dotenv import load_dotenv
            from google import genai

            load_dotenv(PROJECT_ROOT / ".env")
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env (see .env.example).")
            self._client = genai.Client(api_key=key)
        return self._client

    def _key(self, description: str) -> str:
        raw = f"{self.model}|{PROMPT_VERSION}|{description}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _call(self, descriptions: list[str]) -> list[str | None]:
        from google.genai import types

        numbered = "\n".join(f"{i}: {d}" for i, d in enumerate(descriptions))
        config = types.GenerateContentConfig(
            system_instruction=INSTRUCTIONS,
            temperature=0,
            response_mime_type="application/json",
            response_schema=_schema(len(descriptions)),
        )
        for attempt in range(MAX_RETRIES):
            try:
                start = time.perf_counter()
                response = self.client.models.generate_content(
                    model=self.model, contents=numbered, config=config
                )
                elapsed = time.perf_counter() - start
                break
            except Exception as error:  # rate limits and transient server errors
                if attempt == MAX_RETRIES - 1:
                    raise
                print(f"  retrying after {type(error).__name__}", flush=True)
                time.sleep(2**attempt * 5)

        meta = response.usage_metadata
        self.usage.requests += 1
        self.usage.input_tokens += meta.prompt_token_count or 0
        self.usage.output_tokens += (meta.candidates_token_count or 0) + (
            getattr(meta, "thoughts_token_count", None) or 0
        )
        self.usage.seconds += elapsed
        self.usage.latencies.append(elapsed)

        answers: list[str | None] = [None] * len(descriptions)
        for item in json.loads(response.text):
            if 0 <= item.get("id", -1) < len(descriptions) and item.get("category") in CATEGORIES:
                answers[item["id"]] = item["category"]
        return answers

    def _store(self, description: str, category: str) -> None:
        key = self._key(description)
        self.cache[key] = category
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "category": category}) + "\n")

    def predict(self, descriptions: list[str], batch_size: int = BATCH_SIZE) -> list[str | None]:
        """Category per description; None only if the model never gave a valid answer."""
        todo = list(dict.fromkeys(d for d in descriptions if self._key(d) not in self.cache))
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            answers = self._call(batch)
            # Anything the batch dropped gets one more try on its own.
            for description, answer in zip(batch, answers, strict=True):
                if answer is None:
                    answer = self._call([description])[0]
                if answer is not None:
                    self._store(description, answer)
        return [self.cache.get(self._key(d)) for d in descriptions]

    def time_single_calls(self, descriptions: list[str]) -> list[float]:
        """Latency of one-description requests (what a live API call would see). Not cached."""
        before = len(self.usage.latencies)
        for d in descriptions:
            self._call([d])
        return self.usage.latencies[before:]
