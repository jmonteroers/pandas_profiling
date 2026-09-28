from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pandas_log
import pyreball as pb


# ============================================================
# Configuration
# ============================================================

IGNORED_OPERATIONS = {
    "copy",
}


# ============================================================
# Snapshot
# ============================================================

@dataclass
class Snapshot:
    """
    A DataFrame state captured at a particular point in time.
    """

    name: str
    dataframe: pd.DataFrame
    operations: list[Any] = field(default_factory=list)


# ============================================================
# DataFrame tracking
# ============================================================

@dataclass
class DataFrameTracker:
    """
    Maintains the history cursor for one logical DataFrame.
    """

    name: str
    snapshots: list[Snapshot] = field(default_factory=list)

    # Position in pandas-log execution_history already consumed
    history_position: int = 0


# ============================================================
# AnalysisContext
# ============================================================

class AnalysisContext:
    """
    Tracks multiple DataFrames independently.

    It knows nothing about Pyreball and performs no transformations.
    """

    def __init__(self):
        self.dataframes: dict[str, DataFrameTracker] = {}

    def capture(
        self,
        dataframe_name: str,
        snapshot_name: str,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Capture a DataFrame state.

        Only pandas-log operations since the previous capture
        of THIS DataFrame are associated with the snapshot.
        """

        # Create a tracker the first time we see this DataFrame.
        if dataframe_name not in self.dataframes:
            self.dataframes[dataframe_name] = DataFrameTracker(
                name=dataframe_name
            )

        tracker = self.dataframes[dataframe_name]

        history = list(
            getattr(df, "execution_history", [])
        )

        # Only operations since the previous capture of this
        # particular DataFrame.
        new_operations = history[
            tracker.history_position:
        ]

        tracker.history_position = len(history)

        tracker.snapshots.append(
            Snapshot(
                name=snapshot_name,
                dataframe=df.copy(),
                operations=new_operations,
            )
        )

        return df

    def get(
        self,
        dataframe_name: str,
    ) -> DataFrameTracker:
        return self.dataframes[dataframe_name]


# ============================================================
# Example source data
# ============================================================

def load_customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 5],
            "name": [
                "Alice",
                "Bob",
                "Charlie",
                "David",
                "Eva",
                "Eva",
            ],
            "country": [
                "UK",
                "UK",
                "US",
                "UK",
                "UK",
                "UK",
            ],
            "status": [
                "active",
                "inactive",
                "active",
                "active",
                "active",
                "active",
            ],
        }
    )


def load_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": range(101, 108),
            "customer_id": [
                1, 2, 3, 4, 5, 5, 99
            ],
            "amount": [
                100, 50, 500, 250, 1200, 800, 75
            ],
            "status": [
                "completed",
                "cancelled",
                "completed",
                "completed",
                "completed",
                "completed",
                "completed",
            ],
        }
    )


# ============================================================
# Analysis
# ============================================================

def run_analysis() -> AnalysisContext:

    context = AnalysisContext()

    with pandas_log.enable(
        silent=True,
        copy_ok=False,
    ):

        # ====================================================
        # CUSTOMERS
        # ====================================================

        customers = load_customers()

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
            "02 — Filtered active customers",
            customers,
        )

        customers = customers.drop_duplicates(
            subset=["customer_id"]
        )

        context.capture(
            "customers",
            "03 — Removed duplicate customers",
            customers,
        )

        # ====================================================
        # ORDERS
        # ====================================================

        orders = load_orders()

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
            "02 — Filtered completed orders",
            orders,
        )

        orders = orders.assign(
            total=lambda df: df["amount"] * 1.20
        )

        context.capture(
            "orders",
            "03 — Added order total",
            orders,
        )

        # ====================================================
        # CUSTOMER + ORDERS
        # ====================================================

        customer_orders = customers.merge(
            orders,
            on="customer_id",
            how="inner",
        )

        context.capture(
            "customer_orders",
            "01 — Joined customers and orders",
            customer_orders,
        )

        customer_orders = (
            customer_orders
            .groupby(
                ["customer_id", "name"],
                as_index=False,
            )
            .agg(
                order_count=("order_id", "count"),
                total_spend=("total", "sum"),
            )
        )

        context.capture(
            "customer_orders_agg",
            "02 — Aggregated customer orders",
            customer_orders,
        )

    return context


# ============================================================
# Reporting
# ============================================================

def meaningful_operations(
    operations: list[Any],
) -> list[Any]:

    return [
        operation
        for operation in operations
        if (
            not operation.fn.__name__.startswith("__")
            and operation.fn.__name__ not in IGNORED_OPERATIONS
        )
    ]


def render_snapshot(
    snapshot: Snapshot,
) -> None:

    pb.print_h3(
        snapshot.name
    )

    # --------------------------------------------------------
    # Operations
    # --------------------------------------------------------

    operations = meaningful_operations(
        snapshot.operations
    )

    if operations:

        rows = []

        for operation in operations:
            rows.append(
                {
                    "Operation": operation.fn.__name__,
                    "Arguments": str(operation.fn_args),
                    "Keyword arguments": str(
                        operation.fn_kwargs
                    ),
                }
            )

        pb.print_table(
            pd.DataFrame(rows)
        )

    else:
        pb.print(
            "No pandas transformations recorded for this dataframe (possibly newly generated)."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    df = snapshot.dataframe

    summary = pd.DataFrame(
        {
            "Value": [
                len(df),
                len(df.columns),
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
            ]
        },
        index=[
            "Rows",
            "Columns",
            "Missing values",
            "Duplicate rows",
        ],
    )

    pb.print_h4("Summary")

    pb.print_table(
        summary
    )

    # --------------------------------------------------------
    # Dtypes
    # --------------------------------------------------------

    dtypes = pd.DataFrame(
        {
            "Column": df.columns,
            "Dtype": [
                str(dtype)
                for dtype in df.dtypes
            ],
        }
    )

    pb.print_h4("Dtypes")

    pb.print_table(
        dtypes
    )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    pb.print_h4("Sample")

    pb.print_table(
        df.head(10)
    )


def create_report(
    context: AnalysisContext,
) -> None:

    pb.print_h1(
        "Data Analysis Audit Report"
    )

    for dataframe_name, tracker in (
        context.dataframes.items()
    ):

        pb.print_h2(
            dataframe_name
        )

        for snapshot in tracker.snapshots:

            render_snapshot(
                snapshot
            )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Phase 1: execute analysis
    # --------------------------------------------------------

    context = run_analysis()

    # --------------------------------------------------------
    # Phase 2: create report
    # --------------------------------------------------------

    create_report(context)


if __name__ == "__main__":
    main()