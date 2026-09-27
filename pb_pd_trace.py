import io
import re
import html
import logging
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
import pandas_log
import pyreball as pb
import inspect
import html
from data_profiling import ProfileReport
from pandas.api.types import is_numeric_dtype
from typing import Callable, Dict, Any
from datetime import datetime

EXEC_DATE = f"{datetime.now():%Y%m%d_%H%M%S}"

def safe_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Sample up to n only if at least n rows available
    """
    if len(df) > n:
        return df.sample(n).sort_index()
    return df


def ansi_to_html(text: str) -> str:
    """
    Converts ANSI formatting codes into HTML tags while escaping raw text.
    """
    # 1. Escape HTML special characters in the captured text
    escaped = html.escape(text)
    
    # 2. Replace ANSI bold, underline, and reset sequences
    escaped = re.sub(r'\x1b?\[1m', '<b>', escaped)
    escaped = re.sub(r'\x1b?\[4m', '<u>', escaped)
    escaped = re.sub(r'\x1b?\[0m', '</b></u>', escaped)
    
    # 3. Strip any other unhandled ANSI color or positioning codes
    escaped = re.sub(r'\x1b?\[[0-9;]*[a-zA-Z]', '', escaped)
    
    # 4. Wrap in a preformatted monospace box for Pyreball
    return f"""
    <div style="font-family: monospace; background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 12px; border-radius: 6px; white-space: pre-wrap; line-height: 1.5;">
        {escaped}
    </div>
    """


def audit_pandas_transform(
    inputs: Dict[str, pd.DataFrame],
    transform_func: Callable[..., Any],
    report_title: str = "DataFrame Transformation Audit",
    render_html_formatting: bool = True,
    max_rows: int = 100,
    incl_profiling: bool = True,
    profiling_frac: float = 0.01
) -> Any:
    # Initialize Pyreball Header
    pb.print_h1(report_title)

    # Print source code
    pb.print_h2("Source Code")
    pb.print_code_block(inspect.getsource(transform_func))

    # Render Inputs
    pb.print_h2("Original Input DataFrames")
    for name, df in inputs.items():
        pb.print_h3(f"Input Dataset: '{name}' ({df.shape[0]} rows × {df.shape[1]} columns)")
        pb.print_table(
            safe_sample(df, max_rows), display_option="scrolling", search_box=True, numbered=False
            )
        if incl_profiling:
            pb.print_h4(f"Data profiling report for '{name}'")
            if len(df) * profiling_frac > 1:
                sampled_df = df if profiling_frac == 1.0 else df.sample(frac=profiling_frac)
            else:
                sampled_df = df
            add_profiling_report(sampled_df, filename=f"outputs/dp_input_{name}_{EXEC_DATE}.html")
    
    # Capture stdout during transformation
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    
    logger = logging.getLogger("pandas_log")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    try:
        with redirect_stdout(log_stream), redirect_stderr(log_stream), pandas_log.enable(full_signature=False):
            result = transform_func(**inputs)
    finally:
        captured_logs = log_stream.getvalue()
        logger.removeHandler(handler)
    
    # Render Transformed Output
    pb.print_h2("Transformed Output DataFrames")
    if isinstance(result, pd.DataFrame):
        pb.print_h3(f"Output Dataset ({result.shape[0]} rows × {result.shape[1]} columns)")
        pb.print_table(
            safe_sample(result, max_rows), display_option="scrolling", search_box=True, numbered=False
            )
        if incl_profiling:
            pb.print_h4("Data profiling report for output dataset")
            if len(result) * profiling_frac > 1:
                sampled_result = result if profiling_frac == 1.0 else result.sample(frac=profiling_frac)
            else:
                sampled_result = result
            add_profiling_report(sampled_result, filename=f"outputs/dp_output_{EXEC_DATE}.html")
    elif isinstance(result, tuple):
        for idx, res_df in enumerate(result):
            if isinstance(res_df, pd.DataFrame):
                pb.print_h3(f"Output Dataset {idx+1} ({res_df.shape[0]} rows × {res_df.shape[1]} columns)")
                pb.print_table(
                    safe_sample(res_df, max_rows), display_option="scrolling", search_box=True, numbered=False
                    )
                if incl_profiling:
                    pb.print_h4(f"Data profiling report for output dataset {idx+1}")
                    if len(res_df) * profiling_frac > 1:
                        sampled_res_df = res_df if profiling_frac == 1.0 else res_df.sample(frac=profiling_frac)
                    else:
                        sampled_res_df = res_df
                    add_profiling_report(sampled_res_df, filename=f"outputs/dp_output_{idx+1}_{EXEC_DATE}.html")

    # Render Formatted Audit Trace
    pb.print_h2("Step-by-Step Transformation Trace")
    if captured_logs.strip():
        if render_html_formatting:
            # Renders preserved bold/underline formatting in Pyreball
            pb.print(ansi_to_html(captured_logs))
        else:
            # Strips ANSI entirely and renders inside a standard code block
            clean_text = re.sub(r'\x1b?\[[0-9;]*[a-zA-Z]', '', captured_logs)
            pb.print_code_block(clean_text)
    else:
        pb.print("No pandas_log transformation logs were captured during execution.")
    
    return result


def add_profiling_report(
    df,
    filename="profiling_report.html",
    title="Data Profiling Report", 
    minimal=True,
    type_schema=None
):
    """Saves the profiling report as an external HTML file and links it inside Pyreball."""
    if type_schema is None: type_schema = {
    col: "numeric" if is_numeric_dtype(df[col]) else "categorical"
    for col in df.columns
}

    # 1. Save profiling report to file
    profile = ProfileReport(df, title=title, minimal=minimal, type_schema=type_schema)
    profile.to_file(filename)

    # 2. Create a link/button styled with inline CSS
    # target="_blank" ensures it opens in a separate browser tab
    link_html = f"""
    <div style="margin: 15px 0;">
        <a href="{filename}" target="_blank" rel="noopener noreferrer" style="
            display: inline-block;
            padding: 10px 18px;
            background-color: #2b5797;
            color: #ffffff;
            text-decoration: none;
            border-radius: 4px;
            font-weight: 600;
            font-family: system-ui, -apple-system, sans-serif;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            📊 {title} ↗
        </a>
    </div>
    """
    pb.print(link_html)


"""
MRE
"""
if __name__ == "__main__":
    # Sample Data
    df_customers = pd.DataFrame({
        "customer_id": [101, 102, 103, 104],
        "name": ["Alice", "Bob", "Charlie", "David"],
        "region": ["North", "South", "East", "West"]
    })

    df_orders = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "customer_id": [101, 102, 101, 105, 102],
        "amount": [250.0, 120.5, 300.0, 450.0, 80.0]
    })

    # Define your transformation logic as a clean function
    def my_merge_transform(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
        return (
            orders
            .query("amount > 100")
            .merge(customers, on="customer_id", how="left")
            .assign(discount_amount=lambda x: x["amount"] * 0.1)
            .drop(columns=["region"])
        )

    # Run the audit helper
    final_df = audit_pandas_transform(
        inputs={
            "orders": df_orders,
            "customers": df_customers
        },
        transform_func=my_merge_transform,
        report_title="Customer Order Merge Audit"
    )