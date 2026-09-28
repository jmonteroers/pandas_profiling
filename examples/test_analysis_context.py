from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_log

from analysis_context import (
    AnalysisContext,
    create_report,
)


N_CUSTOMERS = 100_000
N_ORDERS = 1_000_000


def make_customers(
    n_customers=N_CUSTOMERS,
    seed=42,
):
    rng = np.random.default_rng(seed)

    customer_id = np.arange(
        1,
        n_customers + 1,
        dtype=np.int64,
    )

    countries = np.array([
        "UK", "US", "DE", "FR", "ES",
        "IT", "NL", "AU", "CA", "IE",
    ])

    segments = np.array([
        "Consumer",
        "Business",
        "Enterprise",
    ])

    signup_days = rng.integers(
        0,
        2000,
        size=n_customers,
    )

    return pd.DataFrame({
        "customer_id": customer_id,
        "country": rng.choice(
            countries,
            size=n_customers,
        ),
        "segment": rng.choice(
            segments,
            size=n_customers,
            p=[0.70, 0.25, 0.05],
        ),
        "status": rng.choice(
            ["active", "inactive"],
            size=n_customers,
            p=[0.90, 0.10],
        ),
        "signup_date": (
            np.datetime64("2020-01-01")
            + signup_days.astype("timedelta64[D]")
        ),
        "credit_limit": (
            rng.lognormal(
                mean=8.5,
                sigma=0.7,
                size=n_customers,
            ).round(2)
        ),
    })


def make_orders(
    n_orders=N_ORDERS,
    n_customers=N_CUSTOMERS,
    seed=43,
):
    rng = np.random.default_rng(seed)

    customer_id = (
        rng.zipf(
            a=1.4,
            size=n_orders,
        )
        % n_customers
    ) + 1

    order_days = rng.integers(
        0,
        730,
        size=n_orders,
    )

    return pd.DataFrame({
        "order_id": np.arange(
            1,
            n_orders + 1,
            dtype=np.int64,
        ),
        "customer_id": customer_id,
        "order_date": (
            np.datetime64("2024-01-01")
            + order_days.astype("timedelta64[D]")
        ),
        "category": rng.choice(
            [
                "Electronics",
                "Home",
                "Clothing",
                "Sports",
                "Books",
                "Beauty",
            ],
            size=n_orders,
        ),
        "payment_method": rng.choice(
            [
                "Card",
                "PayPal",
                "Bank Transfer",
            ],
            size=n_orders,
            p=[0.65, 0.25, 0.10],
        ),
        "status": rng.choice(
            [
                "completed",
                "cancelled",
                "pending",
                "returned",
            ],
            size=n_orders,
            p=[0.82, 0.05, 0.08, 0.05],
        ),
        "quantity": rng.integers(
            1,
            6,
            size=n_orders,
            dtype=np.int16,
        ),
        "unit_price": (
            rng.lognormal(
                mean=3.5,
                sigma=1.0,
                size=n_orders,
            ).round(2)
        ),
    })


def run_analysis():

    context = AnalysisContext()

    customers = make_customers()
    orders = make_orders()

    with pandas_log.enable(
        silent=True,
        copy_ok=False,
    ):

        context.capture(
            "customers",
            "01 — Loaded customers",
            customers,
        )

        customers = customers.query(
            "status == 'active'"
        )

        context.capture(
            "customers",
            "02 — Active customers",
            customers,
        )

        customers = customers.assign(
            customer_age_days=lambda df: (
                pd.Timestamp("2026-01-01")
                - df["signup_date"]
            ).dt.days
        )

        context.capture(
            "customers",
            "03 — Added customer age",
            customers,
        )

        customers = customers.sort_values(
            "customer_id"
        )

        context.capture(
            "customers",
            "04 — Sorted customers",
            customers,
        )

        context.capture(
            "orders",
            "01 — Loaded orders",
            orders,
        )

        orders = orders.query(
            "status == 'completed'"
        )

        context.capture(
            "orders",
            "02 — Completed orders",
            orders,
        )

        orders = orders.assign(
            revenue=lambda df:
                df["quantity"] * df["unit_price"]
        )

        context.capture(
            "orders",
            "03 — Calculated revenue",
            orders,
        )

        orders = orders.query(
            "revenue > 100"
        )

        context.capture(
            "orders",
            "04 — Orders over £100",
            orders,
        )

        customer_orders = orders.merge(
            customers[
                [
                    "customer_id",
                    "country",
                    "segment",
                    "credit_limit",
                ]
            ],
            on="customer_id",
            how="inner",
        )

        context.capture(
            "customer_orders",
            "01 — Joined orders to customers",
            customer_orders,
        )

        customer_orders = customer_orders.assign(
            credit_utilisation=lambda df:
                df["revenue"] / df["credit_limit"]
        )

        context.capture(
            "customer_orders",
            "02 — Calculated credit utilisation",
            customer_orders,
        )

        customer_summary = (
            customer_orders
            .groupby(
                [
                    "customer_id",
                    "country",
                    "segment",
                ],
                as_index=False,
            )
            .agg(
                order_count=("order_id", "count"),
                total_revenue=("revenue", "sum"),
                average_order_value=(
                    "revenue",
                    "mean",
                ),
            )
        )

        context.capture(
            "customer_summary",
            "01 — Customer order summary",
            customer_summary,
            code=(
            """
            customer_summary = (
                        customer_orders
                        .groupby(
                            [
                                "customer_id",
                                "country",
                                "segment",
                            ],
                            as_index=False,
                        )
                        .agg(
                            order_count=("order_id", "count"),
                            total_revenue=("revenue", "sum"),
                            average_order_value=(
                                "revenue",
                                "mean",
                            ),
                        )
                    )
            """).strip()
        )

    return context


def main():

    context = run_analysis()

    create_report(context)


if __name__ == "__main__":
    main()