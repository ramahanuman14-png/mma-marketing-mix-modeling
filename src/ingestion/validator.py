"""
validator.py — Phase 2: Schema Validation
Pandera schema contracts for every dataset.
If bad data enters the pipeline, this raises an error LOUDLY.
The pipeline should FAIL on bad data — never silently produce wrong results.
"""

import pandera as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd
from loguru import logger


# ── Schema Definitions ──────────────────────────────────────────────────────

SALES_TRANSACTIONS_SCHEMA = DataFrameSchema(
    columns={
        "ID":               Column(str,   nullable=False),
        "Date":             Column("datetime64[ns]", nullable=True),
        "GMV":              Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "Units_sold":       Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "MRP":              Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "Product_Category": Column(str,   nullable=True),
        "Analytic_Category":Column(str,   nullable=True),
    },
    coerce=True,
    name="sales_transactions",
)


SECONDFILE_SCHEMA = DataFrameSchema(
    columns={
        "Date":             Column("datetime64[ns]", nullable=False),
        "total_gmv":        Column(float, Check.greater_than(0), nullable=False),
        "total_Units":      Column(int,   Check.greater_than(0), nullable=False),
        "Total_Investment": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        "TV":               Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        "Digital":          Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        "NPS":              Column(float, Check.in_range(0, 100), nullable=False),
        "Year":             Column(int,   Check.in_range(2010, 2030), nullable=False),
        "Month":            Column(int,   Check.in_range(1, 12), nullable=False),
    },
    coerce=True,
    name="secondfile",
)


MEDIA_INVESTMENT_SCHEMA = DataFrameSchema(
    columns={
        "Year":              Column(int,   Check.in_range(2010, 2030), nullable=False),
        "Month":             Column(int,   Check.in_range(1, 12), nullable=False),
        "Total_Investment":  Column(float, Check.greater_than_or_equal_to(0), nullable=False),
        "TV":                Column(float, nullable=False),
        "Digital":           Column(float, nullable=False),
        "Sponsorship":       Column(float, nullable=False),
        "Radio":             Column(float, nullable=True),   # Has 9 known nulls
        "Other":             Column(float, nullable=True),   # Has 9 known nulls
    },
    coerce=True,
    name="media_investment",
)


NPS_SCHEMA = DataFrameSchema(
    columns={
        "Date": Column("datetime64[ns]", nullable=False),
        "NPS":  Column(float, Check.in_range(0, 100), nullable=False),
    },
    coerce=True,
    name="nps",
)


SPECIAL_SALES_SCHEMA = DataFrameSchema(
    columns={
        "Date":       Column("datetime64[ns]", nullable=False),
        "Sales Name": Column(str, nullable=False),
    },
    coerce=True,
    name="special_sales",
)


FIRSTFILE_SCHEMA = DataFrameSchema(
    columns={
        "Date":                Column("datetime64[ns]", nullable=False),
        "Sales_name":          Column(str,   nullable=False),
        "gmv_new":             Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "units":               Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "product_mrp":         Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "product_category":    Column(str,   nullable=True),
        "product_subcategory": Column(str,   nullable=True),
    },
    coerce=True,
    name="firstfile",
)


# ── Schema Registry ─────────────────────────────────────────────────────────

SCHEMA_REGISTRY = {
    "sales_transactions": SALES_TRANSACTIONS_SCHEMA,
    "secondfile":         SECONDFILE_SCHEMA,
    "media_investment":   MEDIA_INVESTMENT_SCHEMA,
    "nps":                NPS_SCHEMA,
    "special_sales":      SPECIAL_SALES_SCHEMA,
    "firstfile":          FIRSTFILE_SCHEMA,
}


# ── Validation Runner ────────────────────────────────────────────────────────

def validate(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a single DataFrame against its registered schema.
    Returns the validated DataFrame on success.
    Raises SchemaError on failure — pipeline stops immediately.
    """
    if name not in SCHEMA_REGISTRY:
        logger.warning(f"No schema registered for '{name}' — skipping validation.")
        return df

    schema = SCHEMA_REGISTRY[name]
    logger.info(f"Validating schema: {name} ({df.shape[0]} rows)")

    try:
        validated = schema.validate(df, lazy=True)
        logger.info(f"✅ {name} passed schema validation")
        return validated
    except pa.errors.SchemaErrors as e:
        logger.error(f"❌ Schema validation FAILED for {name}")
        logger.error(f"Failure summary:\n{e.failure_cases}")
        raise


def validate_all(datasets: dict) -> dict:
    """
    Validate all datasets in the pipeline.
    Returns dict of validated DataFrames.
    """
    logger.info("Running schema validation on all datasets...")
    validated = {}
    errors = []

    for name, df in datasets.items():
        try:
            validated[name] = validate(name, df)
        except pa.errors.SchemaErrors as e:
            errors.append(name)
            logger.error(f"FAILED: {name}")

    if errors:
        raise RuntimeError(
            f"Schema validation failed for: {errors}. "
            f"Fix data quality issues before proceeding."
        )

    logger.info(f"✅ All {len(validated)} datasets passed validation.")
    return validated
