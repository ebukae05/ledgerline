"""Hand-written fraud rules: the baseline a bank analyst might write on day one.

Each rule is a yes/no check. The score is how many rules fire, so a
transaction that trips 4 rules ranks above one that trips 1. Anything learned
from data (which email domains count as risky) comes from the training split;
the amount cutoff is tuned on validation.
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ledgerline.eval.metrics import pr_auc

AMOUNT_CUTOFFS = (250.0, 500.0, 1000.0, 2000.0)
RISKY_DOMAIN_MIN_COUNT = 1000
RISKY_DOMAIN_LIFT = 2.0


@dataclass
class RulesBaseline:
    high_amount: float = 1000.0
    risky_domains: list[str] = field(default_factory=list)

    def rules(self) -> dict[str, pl.Expr]:
        amount = pl.col("TransactionAmt")
        return {
            "product_c": pl.col("ProductCD") == "C",
            "credit_card": pl.col("card6") == "credit",
            "mobile_device": pl.col("DeviceType") == "mobile",
            "risky_email_domain": pl.col("P_emaildomain").is_in(self.risky_domains),
            "small_amount": (amount >= 5) & (amount < 20),
            "high_amount": amount >= self.high_amount,
        }

    def fired(self, df: pl.DataFrame) -> pl.DataFrame:
        """One 0/1 column per rule, for explaining which rules fired."""
        return df.select(
            [expr.fill_null(False).cast(pl.Int8).alias(name) for name, expr in self.rules().items()]
        )

    def score(self, df: pl.DataFrame) -> np.ndarray:
        return self.fired(df).sum_horizontal().to_numpy().astype(float)

    def fit(self, train: pl.DataFrame, val: pl.DataFrame) -> "RulesBaseline":
        base_rate = train["isFraud"].mean()
        domains = (
            train.group_by("P_emaildomain")
            .agg(pl.len().alias("n"), pl.col("isFraud").mean().alias("rate"))
            .filter(
                pl.col("P_emaildomain").is_not_null()
                & (pl.col("n") >= RISKY_DOMAIN_MIN_COUNT)
                & (pl.col("rate") >= RISKY_DOMAIN_LIFT * base_rate)
            )
        )
        self.risky_domains = sorted(domains["P_emaildomain"].to_list())

        def val_pr_auc(cutoff: float) -> float:
            self.high_amount = cutoff
            return pr_auc(val["isFraud"], self.score(val))

        self.high_amount = max(AMOUNT_CUTOFFS, key=val_pr_auc)
        return self
