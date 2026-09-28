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

DEFAULT_SAMPLE_SIZE = 10


# ============================================================
# Captured metadata
# ============================================================

@dataclass
class Snapshot:
    """
    A DataFrame state captured at one point during analysis.
    """

    name: str

    # Independent snapshot of the DataFrame at this point.
    dataframe: pd.DataFrame

    # pandas-log operations generated since the previous
    # capture of this logical DataFrame.
    operations: list[Any] = field(default_factory=list)


@dataclass
class DataFrameTracker:
    """
    Maintains the history and snapshots for one logical DataFrame.
    """

    name: str

    snapshots: list[Snapshot] = field(
        default_factory=list
    )

    # Position in pandas-log execution_history already consumed.
    history_position: int = 0


# ============================================================
# AnalysisContext
# ============================================================

class AnalysisContext:
    """
    Collects DataFrame snapshots and pandas-log metadata.

    This class deliberately has no dependency on Pyreball.
    It only records what happened during analysis.
    """

    def __init__(self):
        self.dataframes: dict[
            str,
            DataFrameTracker,
        ] = {}

    def capture(
        self,
        dataframe_name: str,
        snapshot_name: str,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Capture the current state of a logical DataFrame.

        Only pandas-log operations since the previous capture
        of THIS DataFrame are associated with this snapshot.

        The DataFrame is copied so later transformations cannot
        modify the captured report state.
        """

        # Create tracker for a new logical DataFrame.
        if dataframe_name not in self.dataframes:
            self.dataframes[dataframe_name] = (
                DataFrameTracker(
                    name=dataframe_name
                )
            )

        tracker = self.dataframes[dataframe_name]

        # Get cumulative pandas-log history.
        history = list(
            getattr(
                df,
                "execution_history",
                [],
            )
        )

        # Extract only operations since the previous
        # capture of this particular DataFrame.
        new_operations = history[
            tracker.history_position:
        ]

        # Advance the cursor.
        tracker.history_position = len(history)

        # Store an independent snapshot.
        tracker.snapshots.append(
            Snapshot(
                name=snapshot_name,
                dataframe=df.copy(),
                operations=new_operations,
            )
        )

        # Returning df makes this possible:
        #
        # df = context.capture("customers", "Cleaned", df)
        #
        return df

    def get(
        self,
        dataframe_name: str,
    ) -> DataFrameTracker:
        return self.dataframes[dataframe_name]


# ============================================================
# Example data sources
# ============================================================

def load_customers() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "customer_id": [
                1, 2, 3, 4, 5, 5
            ],
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
            "spend": [
                100,
                50,
                500,
                250,
                1200,
                1200,
            ],
        }
    )


def load_orders() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "order_id": [
                101, 102, 103, 104,
                105, 106, 107,
            ],
            "customer_id": [
                1, 2, 3, 4, 5, 5, 99
            ],
            "amount": [
                100,
                50,
                500,
                250,
                1200,
                800,
                75,
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
# ANALYSIS
# ============================================================

def run_analysis() -> AnalysisContext:

    context = AnalysisContext()

    # ========================================================
    # pandas-log observes the analysis.
    #
    # There is NO Pyreball code inside this block.
    # ========================================================

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

        # ----------------------------------------------------
        # Transformation
        # ----------------------------------------------------

        customers = customers.query(
            "status == 'active'"
        )

        context.capture(
            "customers",
            "02 — Filtered active customers",
            customers,
        )

        # ----------------------------------------------------
        # Transformation
        # ----------------------------------------------------

        customers = customers.drop_duplicates(
            subset=["customer_id"]
        )

        context.capture(
            "customers",
            "03 — Removed duplicate customers",
            customers,
        )

        # ----------------------------------------------------
        # Transformation
        # ----------------------------------------------------

        customers = customers.assign(
            segment=lambda df: pd.cut(
                df["spend"],
                bins=[
                    -float("inf"),
                    100,
                    500,
                    float("inf"),
                ],
                labels=[
                    "Low",
                    "Medium",
                    "High",
                ],
            )
        )

        context.capture(
            "customers",
            "04 — Added customer segment",
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

        # ----------------------------------------------------
        # Transformation
        # ----------------------------------------------------

        orders = orders.query(
            "status == 'completed'"
        )

        context.capture(
            "orders",
            "02 — Filtered completed orders",
            orders,
        )

        # ----------------------------------------------------
        # Transformation
        # ----------------------------------------------------

        orders = orders.assign(
            total=lambda df:
                df["amount"] * 1.20
        )

        context.capture(
            "orders",
            "03 — Added order total",
            orders,
        )

        # ====================================================
        # CUSTOMER ORDERS
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

        # ----------------------------------------------------
        # Transformation
        # ----------------------------------------------------

        customer_orders = (
            customer_orders
            .groupby(
                ["customer_id", "name"],
                as_index=False,
            )
            .agg(
                order_count=(
                    "order_id",
                    "count",
                ),
                total_spend=(
                    "total",
                    "sum",
                ),
            )
        )

        context.capture(
            "customer_orders",
            "02 — Aggregated customer orders",
            customer_orders,
        )

    # ========================================================
    # pandas-log context has finished.
    #
    # All analysis is complete.
    # ========================================================

    return context


# ============================================================
# REPORTING HELPERS
# ============================================================

def meaningful_operations(
    operations: list[Any],
) -> list[Any]:

    return [
        operation
        for operation in operations
        if (
            not operation.fn.__name__.startswith("__")
            and operation.fn.__name__
            not in IGNORED_OPERATIONS
        )
    ]


def render_operations(
    operations: list[Any],
) -> None:

    operations = meaningful_operations(
        operations
    )

    pb.print_h4("Operations")

    if not operations:

        pb.print(
            "No pandas transformations recorded."
        )

        return

    rows = []

    for operation in operations:

        rows.append(
            {
                "Operation":
                    operation.fn.__name__,

                "Arguments":
                    str(operation.fn_args),

                "Keyword arguments":
                    str(operation.fn_kwargs),
            }
        )

    pb.print_table(
        pd.DataFrame(rows),
        sortable=True,
    )


def render_summary(
    df: pd.DataFrame,
) -> None:

    summary = pd.DataFrame(
        {
            "Value": [
                len(df),
                len(df.columns),
                int(
                    df.isna()
                    .sum()
                    .sum()
                ),
                int(
                    df.duplicated()
                    .sum()
                ),
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


def render_dtypes(
    df: pd.DataFrame,
) -> None:

    dtypes = pd.DataFrame(
        {
            "Column": df.columns,

            "Dtype": [
                str(dtype)
                for dtype in df.dtypes
            ],

            "Non-null": [
                int(
                    df[column]
                    .notna()
                    .sum()
                )
                for column in df.columns
            ],

            "Missing": [
                int(
                    df[column]
                    .isna()
                    .sum()
                )
                for column in df.columns
            ],

            "Missing %": [
                round(
                    df[column]
                    .isna()
                    .mean()
                    * 100,
                    2,
                )
                for column in df.columns
            ],
        }
    )

    pb.print_h4("Dtypes")

    pb.print_table(
        dtypes,
        sortable=True,
    )


def render_sample(
    df: pd.DataFrame,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> None:

    pb.print_h4(
        f"Sample — first {sample_size} rows"
    )

    pb.print_table(
        df.head(sample_size)
    )


# ============================================================
# SNAPSHOT REPORT
# ============================================================

def render_snapshot(
    snapshot: Snapshot,
    back_reference: pb.Reference,
) -> None:

    df = snapshot.dataframe

    # --------------------------------------------------------
    # Operations
    # --------------------------------------------------------

    render_operations(
        snapshot.operations
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    render_summary(
        df
    )

    # --------------------------------------------------------
    # Dtypes
    # --------------------------------------------------------

    render_dtypes(
        df
    )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    render_sample(
        df
    )

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    pb.print(
        back_reference(
            "↑ Back to snapshot index"
        )
    )


# ============================================================
# PYREBALL REPORT
# ============================================================

def create_report(
    context: AnalysisContext,
) -> None:

    pb.set_title(
        "Data Analysis Audit Report"
    )

    # ========================================================
    # Create references before rendering.
    #
    # This lets us use references in the navigation index
    # before the corresponding headings are rendered.
    # ========================================================

    snapshot_references: dict[
        str,
        list[pb.Reference],
    ] = {}

    dataframe_references: dict[
        str,
        pb.Reference,
    ] = {}

    for dataframe_name, tracker in (
        context.dataframes.items()
    ):

        dataframe_references[
            dataframe_name
        ] = pb.Reference(
            dataframe_name
        )

        snapshot_references[
            dataframe_name
        ] = [
            pb.Reference(
                snapshot.name
            )
            for snapshot in tracker.snapshots
        ]

    # Reference for the snapshot index itself.
    snapshot_index_reference = pb.Reference(
        "Snapshot index"
    )

    # ========================================================
    # SNAPSHOT INDEX
    # ========================================================

    pb.print_h1(
        "Snapshot index",
        reference=snapshot_index_reference,
    )

    pb.print(
        "Jump directly to any captured DataFrame state."
    )

    for dataframe_name, tracker in (
        context.dataframes.items()
    ):

        pb.print_h2(
            dataframe_name
        )

        links = []

        for snapshot, reference in zip(
            tracker.snapshots,
            snapshot_references[dataframe_name],
        ):

            links.append(
                reference(
                    snapshot.name
                )
            )

        pb.print(
            pb.ulist(*links)
        )

    # ========================================================
    # DATAFRAME REPORTS
    # ========================================================

    for dataframe_name, tracker in (
        context.dataframes.items()
    ):

        # ----------------------------------------------------
        # DataFrame heading
        # ----------------------------------------------------

        pb.print_h1(
            dataframe_name,
            reference=dataframe_references[
                dataframe_name
            ],
        )

        # ----------------------------------------------------
        # Snapshot navigation for this DataFrame
        # ----------------------------------------------------

        pb.print(
            "Snapshots: ",
            sep="",
        )

        snapshot_links = []

        for snapshot, reference in zip(
            tracker.snapshots,
            snapshot_references[dataframe_name],
        ):

            snapshot_links.append(
                reference(
                    snapshot.name
                )
            )

        pb.print(
            pb.ulist(
                *snapshot_links
            )
        )

        # ----------------------------------------------------
        # Individual snapshots
        # ----------------------------------------------------

        for snapshot, reference in zip(
            tracker.snapshots,
            snapshot_references[dataframe_name],
        ):

            pb.print_h2(
                snapshot.name,
                reference=reference,
            )

            render_snapshot(
                snapshot,
                snapshot_index_reference,
            )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # PHASE 1 — EXECUTION
    #
    # No Pyreball.
    # No reporting.
    # Only analysis + metadata capture.
    # ========================================================

    context = run_analysis()

    # ========================================================
    # PHASE 2 — REPORTING
    #
    # No transformations.
    # Uses only captured state.
    # ========================================================

    create_report(
        context
    )


if __name__ == "__main__":
    main()