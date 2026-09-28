from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pyreball as pb
import inspect


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

    code: str | None = None


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
        code: str | None = None,
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
                code=code
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
                "Input shape":
                    operation.input_df.shape,
                "Output shape":
                    operation.output_df.shape,
                "Arguments":
                    str(operation.fn_args),
                "Keyword arguments":
                    str(replace_functions_with_source(operation.fn_kwargs)),
                "Execution": str(operation.execution_stats.exec_time)
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


def render_code(code: str | None) -> None:
    if not code:
        return

    pb.print_h4("Transformation code")
    pb.print_code_block(code)


def replace_functions_with_source(data):
    """
    Recursively inspects a dictionary, list, or nested structure and replaces
    function types/callables with their source code string representation.
    """
    if isinstance(data, dict):
        return {k: replace_functions_with_source(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_functions_with_source(item) for item in data]
    elif callable(data):
        try:
            # inspect.getsource retrieves the original Python code
            return inspect.getsource(data).strip()
        except (TypeError, OSError):
            # Fallback for C-extensions/built-ins (e.g., len) or dynamically executed code
            return f"<Source unavailable for {getattr(data, '__name__', str(data))}>"
    return data



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
    # Code (optional)
    # --------------------------------------------------------
    render_code(
        snapshot.code
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
            f"Table {dataframe_name}",
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