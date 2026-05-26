"""Preprocess the raw MyInvestor mule-detection datasets.

This module takes the two raw files shipped in ``data/raw``:

* ``mulas.csv``        – one row per account (KYC + device snapshot + balance)
* ``movimientos.csv``  – one row per transaction posted to those accounts

and produces three tidy tables under ``data/processed/``:

* ``accounts.parquet``       – cleaned account table (one row per CUENTA)
* ``transactions.parquet``   – cleaned transaction table
* ``master.parquet``         – account table left-joined with per-account
                               transaction aggregates, ready for downstream
                               feature engineering / modelling.

Run it as a script from the project root::

    python -m src.data.preprocess
    # or
    python src/data/preprocess.py --raw-dir data/raw --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ACCOUNTS_FILE = "mulas.csv"
TRANSACTIONS_FILE = "movimientos.csv"

# Reference date used to compute age and "balance as of" features. The raw
# balance column is labelled SALDO_31/12/2025, so we anchor on that.
REFERENCE_DATE = pd.Timestamp("2025-12-31")

OutputFormat = Literal["parquet", "csv", "auto"]


# Tokens that effectively mean "missing" across the device-info columns.
UNKNOWN_TOKENS = {
    "",
    "no info",
    "no info ",
    "noinfo",
    "n/a",
    "na",
    "none",
    "null",
    "nan",
    "-",
    "?",
}


@dataclass(frozen=True)
class PreprocessOutputs:
    """Paths to the artefacts produced by :func:`run`."""

    accounts: Path
    transactions: Path
    master: Path


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _fix_mojibake(series: pd.Series) -> pd.Series:
    """Repair UTF-8 strings that were once decoded as cp1252.

    The transactions file contains values such as ``SUSCRIPCIÃ"N`` which
    should read ``SUSCRIPCIÓN``. This is the classic ``utf-8 → cp1252 →
    utf-8`` mojibake; we reverse it by re-encoding to cp1252 and decoding
    back as utf-8. Strings that cannot be re-encoded cleanly are returned
    unchanged so the function is safe to call indiscriminately.
    """

    def _fix(value: object) -> object:
        if not isinstance(value, str):
            return value
        try:
            return value.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return value

    return series.map(_fix)


def _normalize_whitespace(series: pd.Series) -> pd.Series:
    """Trim and collapse internal whitespace on string columns."""
    return (
        series.astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _to_unknown(series: pd.Series, extra_tokens: Iterable[str] = ()) -> pd.Series:
    """Replace placeholder tokens (``"No Info"``, empty strings, ...) with NA."""
    tokens = {t.lower() for t in UNKNOWN_TOKENS} | {t.lower() for t in extra_tokens}
    cleaned = series.where(~series.str.lower().isin(tokens), other=pd.NA)
    return cleaned


def _parse_yyyymmdd(series: pd.Series) -> pd.Series:
    """Parse an integer/string ``YYYYMMDD`` column into ``datetime64[ns]``."""
    return pd.to_datetime(series.astype("string"), format="%Y%m%d", errors="coerce")


# ---------------------------------------------------------------------------
# Accounts (mulas.csv)
# ---------------------------------------------------------------------------


def load_accounts(path: Path | str) -> pd.DataFrame:
    """Read ``mulas.csv`` with all strings kept as-is for further cleaning."""
    path = Path(path)
    logger.info("Loading accounts from %s", path)
    df = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        encoding="utf-8",
        keep_default_na=False,
        na_values=[""],
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy the account snapshot table.

    Steps applied:

    * Rename the typo'd ``TIPO_DE_PERSONA_TIULAR`` to ``TIPO_DE_PERSONA_TITULAR``
      and the encoded balance column to ``balance_2025_12_31``.
    * Lower-snake-case all column names.
    * Trim/collapse whitespace and normalise unknown markers to NA.
    * Fix the ``NF-DNI`` typo in ``tipo_de_documento``.
    * Parse ``fecha_de_nacimiento`` (YYYYMMDD) to a ``datetime`` and derive
      ``age_years`` w.r.t. :data:`REFERENCE_DATE`.
    * Cast the balance to float.
    * Drop exact duplicates and keep the first record per ``cuenta``.
    """
    df = df.copy()

    rename = {
        "TIPO_DE_PERSONA_TIULAR": "tipo_de_persona_titular",
        "SALDO_31/12/2025": "balance_2025_12_31",
    }
    df = df.rename(columns=rename)
    df.columns = [c.strip().lower().replace("/", "_") for c in df.columns]

    string_cols = [
        "cuenta",
        "dato_persona_titular",
        "tipo_de_persona_titular",
        "pais_de_nacimiento",
        "pais_de_nacionalidad_titular",
        "pais_residencia_ok",
        "provincia_ok",
        "tipo_de_documento",
        "os",
        "device_name",
        "device_brand",
        "device_model",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = _normalize_whitespace(df[col])

    df["tipo_de_documento"] = df["tipo_de_documento"].replace({"NF-DNI": "NIF-DNI"})

    device_cols = ["os", "device_name", "device_brand", "device_model"]
    for col in device_cols:
        if col in df.columns:
            df[col] = _to_unknown(df[col])

    df["pais_de_nacimiento"] = _to_unknown(df["pais_de_nacimiento"])
    df["pais_de_nacionalidad_titular"] = _to_unknown(df["pais_de_nacionalidad_titular"])
    df["pais_residencia_ok"] = _to_unknown(df["pais_residencia_ok"])
    df["provincia_ok"] = _to_unknown(df["provincia_ok"])

    df["fecha_de_nacimiento"] = _parse_yyyymmdd(df["fecha_de_nacimiento"])
    df["age_years"] = (
        (REFERENCE_DATE - df["fecha_de_nacimiento"]).dt.days / 365.25
    ).round(2)
    df.loc[(df["age_years"] < 14) | (df["age_years"] > 110), "age_years"] = np.nan

    df["balance_2025_12_31"] = pd.to_numeric(df["balance_2025_12_31"], errors="coerce")

    before = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["cuenta"], keep="first")
    logger.info("accounts: %d → %d rows after deduplication", before, len(df))

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Transactions (movimientos.csv)
# ---------------------------------------------------------------------------


def load_transactions(path: Path | str) -> pd.DataFrame:
    """Read ``movimientos.csv`` keeping every column as string for cleaning."""
    path = Path(path)
    logger.info("Loading transactions from %s", path)
    df = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        encoding="utf-8",
        keep_default_na=False,
        na_values=[""],
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_transactions(
    df: pd.DataFrame, valid_accounts: Iterable[str] | None = None
) -> pd.DataFrame:
    """Tidy the movements table.

    Steps applied:

    * Snake-case column names.
    * Trim whitespace and repair mojibake on text columns.
    * Parse ``fecoper`` and ``fecvalor`` (YYYYMMDD) as datetimes.
    * Cast ``efecbruto`` to float and derive a ``signed_amount`` based on
      ``indcarabo`` (``+`` credit, ``-`` debit).
    * Bucket descriptions into a coarse ``tx_category`` (transfer, bizum,
      interest, salary, opening, fund, seizure, card, fee, other).
    * Optionally restrict to a known set of accounts.
    * Drop rows with no operation date or no amount.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    text_cols = ["indcarabo", "desmovefec", "tipoper", "cuenta", "observac", "tipprod"]
    for col in text_cols:
        if col in df.columns:
            df[col] = _normalize_whitespace(df[col])
            df[col] = _fix_mojibake(df[col])

    df["fecoper"] = _parse_yyyymmdd(df["fecoper"])
    df["fecvalor"] = _parse_yyyymmdd(df["fecvalor"])

    df["efecbruto"] = pd.to_numeric(df["efecbruto"], errors="coerce")

    sign = df["indcarabo"].map({"+": 1, "-": -1}).fillna(0).astype(int)
    df["signed_amount"] = sign * df["efecbruto"].abs()
    df["is_credit"] = sign == 1
    df["is_debit"] = sign == -1

    df["tx_category"] = _categorise_description(df["desmovefec"])

    if valid_accounts is not None:
        valid = set(valid_accounts)
        before = len(df)
        df = df[df["cuenta"].isin(valid)]
        logger.info(
            "transactions: dropped %d rows whose account is not in the accounts table",
            before - len(df),
        )

    before = len(df)
    df = df.dropna(subset=["fecoper", "efecbruto"])
    logger.info("transactions: dropped %d rows with missing date/amount", before - len(df))

    return df.reset_index(drop=True)


def _categorise_description(series: pd.Series) -> pd.Series:
    """Map free-text descriptions into a small set of coarse categories."""
    s = series.fillna("").str.upper()

    conditions = [
        s.str.contains("BIZUM", na=False),
        s.str.contains("TRANSFER", na=False),
        s.str.contains("INTERES", na=False),
        s.str.contains("NOMINA|SALARIO", regex=True, na=False),
        s.str.contains("APERTURA", na=False),
        s.str.contains("FONDOS|INVERSI", regex=True, na=False),
        s.str.contains("EMBARGO", na=False),
        s.str.contains("TARJETA|VENTA|COMPRA", regex=True, na=False),
        s.str.contains("COMISION|CARGO", regex=True, na=False),
    ]
    choices = [
        "bizum",
        "transfer",
        "interest",
        "salary",
        "opening",
        "investment",
        "seizure",
        "card",
        "fee",
    ]
    return pd.Series(np.select(conditions, choices, default="other"), index=series.index)


# ---------------------------------------------------------------------------
# Aggregates & master table
# ---------------------------------------------------------------------------


def build_account_aggregates(tx_df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic per-account transaction statistics.

    These are intentionally lightweight aggregates (counts, sums, date
    ranges, category mix). Richer feature engineering lives in
    ``src/features`` and consumes this table.
    """
    if tx_df.empty:
        return pd.DataFrame(columns=["cuenta"]).set_index("cuenta")

    grouped = tx_df.groupby("cuenta", sort=False)

    agg = grouped.agg(
        n_transactions=("efecbruto", "size"),
        n_credits=("is_credit", "sum"),
        n_debits=("is_debit", "sum"),
        total_credit=("signed_amount", lambda s: s[s > 0].sum()),
        total_debit=("signed_amount", lambda s: s[s < 0].sum()),
        gross_volume=("efecbruto", lambda s: s.abs().sum()),
        mean_abs_amount=("efecbruto", lambda s: s.abs().mean()),
        max_abs_amount=("efecbruto", lambda s: s.abs().max()),
        first_tx_date=("fecoper", "min"),
        last_tx_date=("fecoper", "max"),
        n_unique_descriptions=("desmovefec", "nunique"),
    )

    agg["net_flow"] = agg["total_credit"] + agg["total_debit"]
    agg["tx_window_days"] = (agg["last_tx_date"] - agg["first_tx_date"]).dt.days
    agg["days_since_last_tx"] = (REFERENCE_DATE - agg["last_tx_date"]).dt.days

    cat_counts = (
        tx_df.groupby(["cuenta", "tx_category"], sort=False)
        .size()
        .unstack(fill_value=0)
        .add_prefix("n_tx_")
    )
    agg = agg.join(cat_counts, how="left").fillna({c: 0 for c in cat_counts.columns})

    return agg.reset_index()


def build_master_dataset(
    accounts_df: pd.DataFrame, aggregates_df: pd.DataFrame
) -> pd.DataFrame:
    """Left-join the cleaned account table with per-account aggregates."""
    master = accounts_df.merge(aggregates_df, on="cuenta", how="left")

    count_cols = [c for c in master.columns if c.startswith("n_tx_") or c.startswith("n_")]
    for col in count_cols:
        if col in master.columns:
            master[col] = master[col].fillna(0).astype("Int64")

    money_cols = ["total_credit", "total_debit", "net_flow", "gross_volume"]
    for col in money_cols:
        if col in master.columns:
            master[col] = master[col].fillna(0.0)

    return master


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_format(fmt: OutputFormat) -> Literal["parquet", "csv"]:
    """Pick a concrete output format. ``"auto"`` prefers parquet when possible."""
    if fmt != "auto":
        return fmt
    if importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"):
        return "parquet"
    logger.warning(
        "Neither pyarrow nor fastparquet is installed; falling back to CSV. "
        "Install pyarrow for faster, typed I/O: pip install pyarrow"
    )
    return "csv"


def _write_table(df: pd.DataFrame, base_path: Path, fmt: Literal["parquet", "csv"]) -> Path:
    """Persist ``df`` next to ``base_path`` using the requested format."""
    out_path = base_path.with_suffix(f".{fmt}")
    if fmt == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path


def run(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    output_format: OutputFormat = "auto",
) -> PreprocessOutputs:
    """End-to-end preprocessing: load, clean, aggregate, and persist to disk."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    fmt = _resolve_format(output_format)

    accounts = clean_accounts(load_accounts(raw_dir / ACCOUNTS_FILE))
    transactions = clean_transactions(
        load_transactions(raw_dir / TRANSACTIONS_FILE),
        valid_accounts=accounts["cuenta"],
    )
    aggregates = build_account_aggregates(transactions)
    master = build_master_dataset(accounts, aggregates)

    accounts_path = _write_table(accounts, processed_dir / "accounts", fmt)
    transactions_path = _write_table(transactions, processed_dir / "transactions", fmt)
    master_path = _write_table(master, processed_dir / "master", fmt)

    logger.info("Wrote %s (%d rows)", accounts_path, len(accounts))
    logger.info("Wrote %s (%d rows)", transactions_path, len(transactions))
    logger.info("Wrote %s (%d rows)", master_path, len(master))

    return PreprocessOutputs(
        accounts=accounts_path,
        transactions=transactions_path,
        master=master_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--format",
        dest="output_format",
        default="auto",
        choices=["auto", "parquet", "csv"],
        help="Output format. 'auto' uses parquet when pyarrow/fastparquet is installed.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    outputs = run(args.raw_dir, args.processed_dir, args.output_format)
    print("Preprocessing complete:")
    print(f"  accounts     -> {outputs.accounts}")
    print(f"  transactions -> {outputs.transactions}")
    print(f"  master       -> {outputs.master}")


if __name__ == "__main__":
    main()
