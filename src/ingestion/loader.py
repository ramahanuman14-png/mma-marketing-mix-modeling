"""
loader.py — Phase 2: Data Ingestion
Loads all 7 raw source files into clean, typed DataFrames.
No transformations. No feature engineering. Load and validate only.
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

RAW_DIR = Path("data/raw")


# ── Helpers ────────────────────────────────────────────────────────────────

def _excel_serial_to_date(serial):
    """Convert Excel serial number (e.g. 42186) to Python datetime."""
    try:
        return datetime(1899, 12, 30) + timedelta(days=int(float(serial)))
    except Exception:
        return None


def _parse_sales_xlsb(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Sales.xlsb is tab-separated inside a single column.
    Split it into proper columns.
    """
    col = df_raw.columns[0]
    df = df_raw[col].str.split("\t", expand=True)
    df.columns = col.split("\t")
    # Drop duplicate header rows that appear mid-file
    df = df[df["ID"] != "ID"].reset_index(drop=True)
    return df


# ── Individual Loaders ──────────────────────────────────────────────────────

def load_sales_transactions() -> pd.DataFrame:
    """
    Sales.xlsb — Raw transaction-level data.
    ~1M rows. Columns: ID, Date, GMV, Units_sold, Product_Category, etc.
    """
    path = RAW_DIR / "Sales.xlsb"
    logger.info(f"Loading {path}")
    df_raw = pd.read_excel(path, engine="pyxlsb", header=0)
    df = _parse_sales_xlsb(df_raw)

    # Type casting
    df["GMV"]        = pd.to_numeric(df["GMV"], errors="coerce")
    df["Units_sold"] = pd.to_numeric(df["Units_sold"], errors="coerce")
    df["MRP"]        = pd.to_numeric(df["MRP"], errors="coerce")
    df["SLA"]        = pd.to_numeric(df["SLA"], errors="coerce")
    df["Date"]       = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    logger.info(f"sales_transactions loaded: {df.shape}")
    return df


def load_firstfile() -> pd.DataFrame:
    """
    firstfile.xlsb — Sales with promotion/event labels.
    Date stored as Excel serial number — converted to datetime.
    """
    path = RAW_DIR / "firstfile.xlsb"
    logger.info(f"Loading {path}")
    df = pd.read_excel(path, engine="pyxlsb")

    # Remove duplicate headers mid-file
    df = df[df["Date"] != "Date"].reset_index(drop=True)

    # Convert Excel serial date
    df["Date"] = df["Date"].apply(_excel_serial_to_date)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Type casting
    df["gmv_new"]     = pd.to_numeric(df["gmv_new"], errors="coerce")
    df["units"]       = pd.to_numeric(df["units"], errors="coerce")
    df["product_mrp"] = pd.to_numeric(df["product_mrp"], errors="coerce")
    df["discount"]    = pd.to_numeric(df["discount"], errors="coerce")

    # Drop unnamed index column
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    logger.info(f"firstfile loaded: {df.shape}")
    return df


def load_secondfile() -> pd.DataFrame:
    """
    Secondfile.csv — Monthly aggregated master table.
    Contains: Revenue, Units, MRP, Discount, Media Investment, NPS.
    This is the PRIMARY modeling table.
    """
    path = RAW_DIR / "Secondfile.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)

    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Rename dot-notation columns to underscore (Python-safe)
    df = df.rename(columns={
        "Total.Investment":  "Total_Investment",
        "Content.Marketing": "Content_Marketing",
        "Online.marketing":  "Online_Marketing",
    })

    logger.info(f"secondfile loaded: {df.shape}")
    return df


def load_media_investment() -> pd.DataFrame:
    """
    MediaInvestment.csv — Monthly spend per channel.
    Radio and Other have 9 nulls — handled in preprocessing.
    Note: 'Affiliates' column has a leading space — stripped.
    """
    path = RAW_DIR / "MediaInvestment.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)

    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Rename spaces to underscores
    df = df.rename(columns={"Total Investment": "Total_Investment",
                             "Content Marketing": "Content_Marketing",
                             "Online marketing": "Online_Marketing"})

    logger.info(f"media_investment loaded: {df.shape}")
    return df


def load_nps() -> pd.DataFrame:
    """
    MonthlyNPSscore.csv — Net Promoter Score by month.
    Will be used as an exogenous variable in MMM.
    """
    path = RAW_DIR / "MonthlyNPSscore.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    logger.info(f"nps loaded: {df.shape}")
    return df


def load_special_sales() -> pd.DataFrame:
    """
    SpecialSale.csv — Calendar of promotional events.
    Used to create event/promo dummy variables in feature engineering.
    """
    path = RAW_DIR / "SpecialSale.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    logger.info(f"special_sales loaded: {df.shape}")
    return df


def load_product_list() -> pd.DataFrame:
    """
    ProductList.csv — Product frequency distribution.
    Note: Contains '\\N' null markers — treated as NaN.
    """
    path = RAW_DIR / "ProductList.csv"
    logger.info(f"Loading {path}")
    df = pd.read_csv(path, na_values=["\\N", "N/A", ""])
    df["Frequency"] = pd.to_numeric(df["Frequency"], errors="coerce")
    logger.info(f"product_list loaded: {df.shape}")
    return df


# ── Master Loader ───────────────────────────────────────────────────────────

def load_all() -> dict:
    """
    Load all 7 source files and return as a named dictionary.
    Entry point for the pipeline.
    """
    logger.info("Starting full data ingestion...")
    data = {
        "sales_transactions": load_sales_transactions(),
        "firstfile":          load_firstfile(),
        "secondfile":         load_secondfile(),
        "media_investment":   load_media_investment(),
        "nps":                load_nps(),
        "special_sales":      load_special_sales(),
        "product_list":       load_product_list(),
    }
    logger.info(f"Ingestion complete. {len(data)} datasets loaded.")
    return data
