"""Stage 0 exploration script.

Read-only inspection of the two raw data sources so later stages can be
built against real data instead of assumptions:
  - data/raw/Customer_Data_.csv          (one row per LOAN, not per customer)
  - data/raw/Factsheet_July-2026.pdf     (mutual fund factsheet, ~65 pages)

Prints structure, distributions, and data-quality findings to stdout only.
Writes nothing to disk.
"""

import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Customer_Data_.csv"
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "Factsheet_July-2026.pdf"


def section(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def inspect_csv() -> None:
    section("CSV SHAPE AND DTYPES")
    df = pd.read_csv(CSV_PATH)
    print(f"shape: {df.shape}")
    print()
    print("dtypes:")
    print(df.dtypes)

    section("FIRST 5 ROWS")
    print(df.head())

    section("DESCRIBE (NUMERIC COLUMNS)")
    print(df.describe())

    section("UNIQUE VALUE COUNTS: Loan_status, Loan_type, "
            "Acquiring_channel, Employment_type, Gender")
    for col in ["Loan_status", "Loan_type", "Acquiring_channel",
                "Employment_type", "Gender"]:
        if col in df.columns:
            print(f"--- {col} ---")
            print(df[col].value_counts(dropna=False))
            print()
        else:
            print(f"--- {col} --- COLUMN NOT FOUND")

    section("NULL COUNTS PER COLUMN")
    print(df.isnull().sum())

    section("MULTI-LOAN CUSTOMERS (MobileNo)")
    total_rows = len(df)
    unique_customers = df["MobileNo"].nunique(dropna=False)
    print(f"total rows (loans): {total_rows}")
    print(f"unique MobileNo values (customers): {unique_customers}")
    print(f"implied multi-loan rows: {total_rows - unique_customers}")

    loan_counts = df.groupby("MobileNo").size()
    multi_loan_customers = loan_counts[loan_counts > 1]
    print(f"customers with more than one loan: {len(multi_loan_customers)}")
    print()
    print("sample loan rows for up to 3 multi-loan customers:")
    sample_mobiles = multi_loan_customers.index[:3]
    for mobile in sample_mobiles:
        print(f"--- MobileNo: {mobile} ---")
        print(df[df["MobileNo"] == mobile])
        print()

    section("MIN / MAX / MEAN: App_score, Imputed_Income, Loan_amount, "
            "Tenure, EMI_amount")
    for col in ["App_score", "Imputed_Income", "Loan_amount", "Tenure",
                "EMI_amount"]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            print(f"--- {col} ---")
            print(f"min:  {series.min()}")
            print(f"max:  {series.max()}")
            print(f"mean: {series.mean()}")
            print()
        else:
            print(f"--- {col} --- COLUMN NOT FOUND")

    section("Disbursement_date PARSE CHECK")
    rows_before = len(df)
    parsed_dates = pd.to_datetime(df["Disbursement_date"], format="%d-%m-%Y",
                                   errors="coerce")
    rows_after = parsed_dates.notna().sum()
    print(f"row count before parsing: {rows_before}")
    print(f"row count successfully parsed (non-null after parse): {rows_after}")
    if rows_before != rows_after:
        print("WARNING: parse dropped rows — some dates did not match "
              "format '%d-%m-%Y'. Rows that failed to parse:")
        print(df.loc[parsed_dates.isna(), "Disbursement_date"])
    print(f"min date: {parsed_dates.min()}")
    print(f"max date: {parsed_dates.max()}")


def inspect_pdf() -> None:
    section("PDF PAGE COUNT")
    reader = PdfReader(str(PDF_PATH))
    print(f"total pages: {len(reader.pages)}")

    section("PDF PAGES 12-16 TEXT EXTRACTION")
    # Pages are 0-indexed in pypdf; "pages 12-16" refers to 1-indexed
    # page numbers as seen in a PDF viewer.
    for page_num in range(12, 17):
        idx = page_num - 1
        if idx >= len(reader.pages):
            print(f"--- page {page_num} --- OUT OF RANGE")
            continue
        text = reader.pages[idx].extract_text()
        print(f"--- page {page_num} (extracted length: {len(text) if text else 0}) ---")
        if page_num == 12:
            print("RAW TEXT:")
            print(text)
        print()

    section("DISTINCT FUND NAMES (pages 1-45, lines matching "
            "'Bajaj Finserv ... Fund/ETF')")
    fund_name_pattern = re.compile(
        r"^Bajaj Finserv\s+.*?(?:Fund|ETF)\b", re.IGNORECASE
    )
    found_names = set()
    max_page = min(45, len(reader.pages))
    for idx in range(max_page):
        text = reader.pages[idx].extract_text()
        if not text:
            continue
        for line in text.splitlines():
            line = line.strip()
            match = fund_name_pattern.match(line)
            if match:
                found_names.add(match.group(0).strip())

    print(f"distinct fund names found: {len(found_names)}")
    for name in sorted(found_names):
        print(f"  - {name}")


def main() -> None:
    inspect_csv()
    inspect_pdf()


if __name__ == "__main__":
    main()

# NOTE: Confirm after running whether `DPD_30`/`DPD_90` are mostly blank vs
# populated with numeric day counts, and whether `Loan_status` /
# `Loan_type` contain any values beyond the expected sets — the printed
# value_counts() sections above will surface this; no assumption is baked
# into the logic here.
# NOTE: Fund-name extraction assumes each fund's name appears on its own
# line starting with "Bajaj Finserv" and ending in "Fund" or "ETF" (e.g.
# "Bajaj Finserv Large Cap Fund"). If pypdf extraction garbles table/heading
# layout on the factsheet pages, this regex may under- or over-match —
# verify against the printed page 12 raw text before trusting the list.
