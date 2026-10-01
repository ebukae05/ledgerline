"""Hand-written keyword rules: what an engineer writes before any ML.

Rules are checked in order and the first match wins, so specific ones
(fees, payroll) come before broad ones (shopping words). Anything unmatched
falls back to the most common category.
"""

import re

from ledgerline.merchants.clean import clean, sign

FALLBACK = "Shopping"

RULES: list[tuple[str, str]] = [
    ("Fees", r"\bfee\b|\bfees\b|overdraft|service charge|\bnsf\b|finance charge"),
    ("Income", r"payroll|salary|direct dep|dir dep|\bbenefit|pension|dividend"),
    ("Mortgage", r"mortgage|\bmtg\b|home loan|escrow|principal"),
    ("Rent", r"\brent\b|apartment|property mgmt|properties|residential"),
    ("Insurance", r"insurance|\binsur|premium|geico|state farm|allstate|progressive|usaa"),
    ("Transfer", r"zelle|venmo|cash app|transfer|\bxfer\b|\bwire\b"),
    (
        "Healthcare",
        r"pharmacy|\bcvs\b|walgreens|dental|medical|hospital|clinic|health|pediatric|\bdr\b",
    ),
    (
        "Utilities",
        r"electric|\bwater\b|\bgas co|utilit|internet|comcast|xfinity|verizon|at&t|energy|waste",
    ),
    ("Subscription", r"subscription|netflix|spotify|hulu|disney|\bprime\b|youtube|icloud"),
    ("Education", r"school|tuition|university|college|course|academy|learning|udemy|coursera"),
    ("Travel", r"airline|airlines|airbnb|hotel|marriott|hilton|cruise|expedia|flight|delta|united"),
    ("Transportation", r"\buber\b|\blyft\b|parking|\bfuel|transit|shell|chevron|exxon|\bbp\b|toll"),
    ("Personal Care", r"salon|\bspa\b|barber|nail|beauty|sephora|ulta|massage|fitness|gym"),
    ("Entertainment", r"cinema|theater|theatre|ticket|movie|concert|\bgame|amc|fandango|park"),
    (
        "Restaurants",
        r"restaurant|pizza|cafe|grill|kitchen|burger|coffee|starbucks|diner|taco|sushi|bar\b",
    ),
    ("Groceries", r"grocery|market|foods|safeway|kroger|trader joe|whole foods|aldi|costco"),
    ("Shopping", r"amazon|amzn|target|walmart|store|shop|best buy|ebay|etsy"),
]
COMPILED = [(category, re.compile(pattern)) for category, pattern in RULES]


def predict_one(description: str) -> str:
    text = clean(description)
    for category, pattern in COMPILED:
        if pattern.search(text):
            return category
    # Unmatched money coming in is far more likely income than shopping.
    return "Income" if sign(description) == "credit" else FALLBACK


def predict(descriptions: list[str]) -> list[str]:
    return [predict_one(d) for d in descriptions]
