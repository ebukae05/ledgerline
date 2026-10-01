"""Regex cleanup for raw bank descriptors.

"[debit] PYPL *TRADER JOE'S #552 1234 MAIN ST SEATTLE 98101 WA USA"
    -> "trader joe's main st seattle wa"

Removes the sign prefix, payment-processor tags, reference codes, numbers and
country suffixes, which carry no category information.
"""

import re

SIGN = re.compile(r"^\[(debit|credit)\]\s*")
PROCESSOR = re.compile(
    r"^(paypal inst xfer|paypal|pypl|pp|sq|tst|clv|google|apple\.com/bill)\s*\*?\s*", re.I
)
REFERENCE = re.compile(r"\b(ppd|web|ccd|tel)\s*id:?\s*\S+|\*\w+", re.I)
DIGITS = re.compile(r"[#\d][\w-]*")
COUNTRY = re.compile(r"\busa?\b$")
SPACES = re.compile(r"\s+")


def sign(description: str) -> str:
    m = SIGN.match(description.strip().lower())
    return m.group(1) if m else "unknown"


def clean(description: str) -> str:
    text = SIGN.sub("", description.strip().lower())
    text = PROCESSOR.sub("", text)
    text = REFERENCE.sub(" ", text)
    text = DIGITS.sub(" ", text)
    text = SPACES.sub(" ", text).strip()
    return COUNTRY.sub("", text).strip()
