"""
Production data validation: schema enforcement, quality checks,
statistical profiling, and distribution alignment verification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.common.enums import DatasetStatus
from src.common.exceptions import (
    DataQualityError,
    DataValidationError,
    SchemaViolationError,
)

logger = logging.getLogger("ml_platform.data.validation")

# Defines the "rules" for each column in your dataset.
@dataclass
class ColumnSchema:
    name: str
    dtype: str                       # "float64", "int64", "object", "category", "datetime64"
    nullable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[list[Any]] = None  # For categoricals
    regex_pattern: Optional[str] = None         # For string columns

# Defines the complete structure of your expected dataset.
@dataclass
class DatasetSchema:
    """When to use:
    Define this once for each dataset type (training data, inference input, batch prediction),
    Set min_rows to prevent empty or too-small datasets from corrupting your model,
    Set max_rows to catch accidental data duplication,
    get_column() method helps you look up individual column rules quickly."""
    columns: list[ColumnSchema]
    min_rows: int = 100
    max_rows: Optional[int] = None
    version: str = "1.0"
    # Helps you look up individual column rules quickly
    def get_column(self, name: str) -> Optional[ColumnSchema]:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

#  Standardizes how each quality check reports its findings i.e. Every validation check returns this format, making it easy to aggregate results and generate reports.
@dataclass
class QualityCheckResult:
    """When to use:
    1. Every validation check returns this format
    2. The severity field determines if the issue blocks processing or just warns
    3. error severity = stop processing, data is bad
    4. warning severity = log it, but continue processing"""
    check_name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # "error" | "warning"

# The final summary of all validation efforts.
@dataclass
class ValidationReport:
    """When to use:
    Return this to your pipeline orchestrator,
    Use is_valid property to decide whether to proceed with model training/inference,
    Use to_dict() for logging, monitoring dashboards, or alerting systems."""
    dataset_id: str
    status: DatasetStatus
    schema_valid: bool
    quality_checks: list[QualityCheckResult]
    profile: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == DatasetStatus.VALIDATED

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "status": self.status.value,
            "schema_valid": self.schema_valid,
            "quality_checks": [
                {"check": qc.check_name, "passed": qc.passed, "details": qc.details}
                for qc in self.quality_checks
            ],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
        }


class DataValidator:
    """
    Comprehensive data validation engine for production ML pipelines.

    Validates:
    - Schema conformance (columns, types, constraints)
    - Data quality (nulls, duplicates, outliers, value ranges)
    - Statistical profile (distributions, correlations)
    - Alignment with reference dataset distribution
    """

    def __init__(self, schema: DatasetSchema):
        self.schema = schema

    def validate(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        reference_df: Optional[pd.DataFrame] = None,
    ) -> ValidationReport:
        """
        Run full validation suite on a dataframe.

        Args:
            df: The dataframe to validate.
            dataset_id: Unique identifier for tracking.
            reference_df: Reference (training) data for distribution alignment checks.

        Returns:
            ValidationReport with all check results.
        """
        logger.info(f"Starting validation for dataset '{dataset_id}' with {len(df)} rows")

        errors: list[str] = []
        warnings: list[str] = []
        quality_checks: list[QualityCheckResult] = []

        # 1. Schema validation
        schema_valid = True
        try:
            self._validate_schema(df)
        except SchemaViolationError as e:
            schema_valid = False
            errors.append(f"Schema violation: {e.message}")

        # 2. Quality checks
        quality_checks.extend(self._check_null_rates(df))
        quality_checks.extend(self._check_duplicates(df))
        quality_checks.extend(self._check_value_ranges(df))
        quality_checks.extend(self._check_cardinality(df))
        quality_checks.extend(self._check_outliers(df))

        # 3. Row count checks
        row_check = self._check_row_count(df)
        quality_checks.append(row_check)
        if not row_check.passed:
            errors.append(f"Row count check failed: {row_check.details}")

        # 4. Distribution alignment with reference
        if reference_df is not None:
            alignment_checks = self._check_distribution_alignment(df, reference_df)
            quality_checks.extend(alignment_checks)
            for check in alignment_checks:
                if not check.passed and check.severity == "warning":
                    warnings.append(f"Distribution drift in '{check.details.get('column', '?')}': {check.details}")

        # 5. Statistical profile
        profile = self._compute_profile(df)

        # Collect errors from failed checks
        for qc in quality_checks:
            if not qc.passed and qc.severity == "error":
                errors.append(f"{qc.check_name}: {qc.details}")

        # Determine overall status
        has_blocking_errors = any(
            not qc.passed and qc.severity == "error" for qc in quality_checks
        ) or not schema_valid

        status = DatasetStatus.REJECTED if has_blocking_errors else DatasetStatus.VALIDATED

        report = ValidationReport(
            dataset_id=dataset_id,
            status=status,
            schema_valid=schema_valid,
            quality_checks=quality_checks,
            profile=profile,
            errors=errors,
            warnings=warnings,
        )

        logger.info(
            f"Validation complete for '{dataset_id}': status={status.value}, "
            f"errors={len(errors)}, warnings={len(warnings)}"
        )
        return report

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Checks if columns exist and have correct types.
        When to use:
        Missing columns detection: Catches when your API changes and drops fields,
        Extra columns detection: Identifies new features that need schema updates,
        Type compatibility: Uses _dtypes_compatible() to allow safe type conversions."""
        expected_cols = set(self.schema.column_names)
        actual_cols = set(df.columns)

        missing = expected_cols - actual_cols
        if missing:
            raise SchemaViolationError(
                f"Missing columns: {missing}",
                details={"missing_columns": list(missing)},
            )

        extra = actual_cols - expected_cols
        if extra:
            logger.warning(f"Extra columns found and will be ignored: {extra}")

        for col_schema in self.schema.columns:
            actual_dtype = str(df[col_schema.name].dtype)
            # Allow compatible types
            if not self._dtypes_compatible(col_schema.dtype, actual_dtype):
                raise SchemaViolationError(
                    f"Column '{col_schema.name}' expected dtype '{col_schema.dtype}', "
                    f"got '{actual_dtype}'",
                    details={"column": col_schema.name, "expected": col_schema.dtype, "actual": actual_dtype},
                )

    @staticmethod
    def _dtypes_compatible(expected: str, actual: str) -> bool:
        """When to use:
        Allows int32 to pass for int64 (minor precision difference),
        Prevents string data in numeric columns (catastrophic for ML models),
        The compatibility map is your "type safety net."""
        compatibility_map = {
            "float64": {"float64", "float32", "int64", "int32"},
            "int64": {"int64", "int32", "int16", "int8"},
            "object": {"object", "string", "str"},
            "category": {"category", "object"},
        }
        return actual in compatibility_map.get(expected, {expected})

    def _check_null_rates(self, df: pd.DataFrame) -> list[QualityCheckResult]:
        """What it does: Counts missing values per column.
        When to use:
        Critical fields (like user_id): Set nullable=False → errors on any null,
        Optional fields (like middle_name): Allow some nulls,
        30% threshold: Flags columns that are mostly empty (probably broken data pipeline).
        """
        results = []
        for col_schema in self.schema.columns:
            if col_schema.name not in df.columns:
                continue
            null_rate = df[col_schema.name].isnull().mean()
            passed = True
            severity = "warning"

            if not col_schema.nullable and null_rate > 0:
                passed = False
                severity = "error"
            elif null_rate > 0.3:
                passed = False
                severity = "warning"

            results.append(QualityCheckResult(
                check_name=f"null_rate_{col_schema.name}",
                passed=passed,
                details={"column": col_schema.name, "null_rate": round(null_rate, 4)},
                severity=severity,
            ))
        return results

    def _check_duplicates(self, df: pd.DataFrame) -> list[QualityCheckResult]:
        """When to use:
        Training data: Duplicates can cause overfitting,
        Inference data: Duplicates waste compute resources,
        10% threshold is conservative; adjust based on your domain.
        """
        dup_rate = df.duplicated().mean()
        passed = dup_rate < 0.1  # More than 10% duplicates is suspicious
        return [QualityCheckResult(
            check_name="duplicate_rows",
            passed=passed,
            details={"duplicate_rate": round(dup_rate, 4), "duplicate_count": int(df.duplicated().sum())},
            severity="warning",
        )]

    def _check_value_ranges(self, df: pd.DataFrame) -> list[QualityCheckResult]:
        """When to use:
        Physical constraints: Age 0-150, temperatures within reasonable ranges,
        Business rules: Price > 0, ratings 1-5,
        Allowed values: Catches typos in categorical data (e.g., "Male" vs "male")."""
        results = []
        for col_schema in self.schema.columns:
            if col_schema.name not in df.columns:
                continue
            col_data = df[col_schema.name].dropna()

            if col_schema.min_value is not None and pd.api.types.is_numeric_dtype(col_data):
                violations = (col_data < col_schema.min_value).sum()
                results.append(QualityCheckResult(
                    check_name=f"min_value_{col_schema.name}",
                    passed=violations == 0,
                    details={"column": col_schema.name, "min_allowed": col_schema.min_value, "violations": int(violations)},
                    severity="error",
                ))

            if col_schema.max_value is not None and pd.api.types.is_numeric_dtype(col_data):
                violations = (col_data > col_schema.max_value).sum()
                results.append(QualityCheckResult(
                    check_name=f"max_value_{col_schema.name}",
                    passed=violations == 0,
                    details={"column": col_schema.name, "max_allowed": col_schema.max_value, "violations": int(violations)},
                    severity="error",
                ))

            if col_schema.allowed_values is not None:
                invalid = ~col_data.isin(col_schema.allowed_values)
                results.append(QualityCheckResult(
                    check_name=f"allowed_values_{col_schema.name}",
                    passed=invalid.sum() == 0,
                    details={
                        "column": col_schema.name,
                        "invalid_count": int(invalid.sum()),
                        "sample_invalid": col_data[invalid].head(5).tolist() if invalid.any() else [],
                    },
                    severity="error",
                ))

        return results

    def _check_cardinality(self, df: pd.DataFrame) -> list[QualityCheckResult]:
        """ Detects columns with too many unique values.
        When to use:
        Prevent ID leakage: If a categorical column has >50% unique values, it might be an ID masquerading as a category,
        Model health: High cardinality categoricals can cause dimensionality problems,
        Set threshold based on your dataset size."""
        results = []
        for col_schema in self.schema.columns:
            if col_schema.name not in df.columns:
                continue
            if col_schema.dtype in ("object", "category"):
                nunique = df[col_schema.name].nunique()
                # Flag if cardinality is suspiciously high (possibly IDs leaking)
                high_cardinality = nunique > 0.5 * len(df) and len(df) > 100
                results.append(QualityCheckResult(
                    check_name=f"cardinality_{col_schema.name}",
                    passed=not high_cardinality,
                    details={"column": col_schema.name, "unique_values": nunique, "row_count": len(df)},
                    severity="warning",
                ))
        return results

    def _check_outliers(self, df: pd.DataFrame) -> list[QualityCheckResult]:
        """Uses IQR method to find extreme values.
        When to use:
        Training data: Outliers can skew your model,
        Production monitoring: Sudden increase in outliers indicates data drift,
        Uses 3×IQR (more conservative than standard 1.5×IQR),
        5% threshold: More than 5% outliers is suspicious."""
        results = []
        for col_schema in self.schema.columns:
            if col_schema.name not in df.columns:
                continue
            if col_schema.dtype not in ("float64", "int64"):
                continue
            col_data = df[col_schema.name].dropna()
            if len(col_data) < 10:
                continue

            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue

            lower = q1 - 3.0 * iqr
            upper = q3 + 3.0 * iqr
            outlier_rate = ((col_data < lower) | (col_data > upper)).mean()

            results.append(QualityCheckResult(
                check_name=f"outliers_{col_schema.name}",
                passed=outlier_rate < 0.05,
                details={"column": col_schema.name, "outlier_rate": round(outlier_rate, 4)},
                severity="warning",
            ))
        return results

    def _check_row_count(self, df: pd.DataFrame) -> QualityCheckResult:
        """Ensures dataset size is within expected range.
        When to use:
        Minimum check: Catch empty or truncated files,
        Maximum check: Detect accidental data duplication or aggregation errors."""
        passed = len(df) >= self.schema.min_rows
        details = {"row_count": len(df), "min_required": self.schema.min_rows}
        if self.schema.max_rows:
            passed = passed and len(df) <= self.schema.max_rows
            details["max_allowed"] = self.schema.max_rows
        return QualityCheckResult(
            check_name="row_count",
            passed=passed,
            details=details,
            severity="error",
        )

    def _check_distribution_alignment(
        self, new_df: pd.DataFrame, reference_df: pd.DataFrame
    ) -> list[QualityCheckResult]:
        """
        Compares new data against training data distribution.
        When to use:
        Before inference: Ensure your model is making predictions on familiar data,
        Continuous monitoring: Set up daily/weekly comparison jobs,
        Numeric columns: Uses Kolmogorov-Smirnov test (sensitive to any distribution change),
        Categorical columns: Uses Population Stability Index (industry standard in finance).
        
        Thresholds:
        KS test: p-value > 0.01 (lenient because production data varies),
        PSI: < 0.2 (standard industry threshold).
        """
        results = []
        for col_schema in self.schema.columns:
            col = col_schema.name
            if col not in new_df.columns or col not in reference_df.columns:
                continue

            if col_schema.dtype in ("float64", "int64"):
                ref_vals = reference_df[col].dropna().values
                new_vals = new_df[col].dropna().values
                if len(ref_vals) < 20 or len(new_vals) < 20:
                    continue

                statistic, pvalue = stats.ks_2samp(ref_vals, new_vals)
                passed = pvalue > 0.05  # Relaxed in distribution alignment check
                results.append(QualityCheckResult(
                    check_name=f"distribution_alignment_{col}",
                    passed=passed,
                    details={
                        "column": col,
                        "test": "ks_2samp",
                        "ks_statistic": round(float(statistic), 4),
                        "p_value": round(float(pvalue), 6),
                    },
                    severity="warning",
                ))
            elif col_schema.dtype in ("object", "category"):
                ref_counts = reference_df[col].value_counts(normalize=True)
                new_counts = new_df[col].value_counts(normalize=True)

                # Align categories
                all_cats = set(ref_counts.index) | set(new_counts.index)
                ref_aligned = np.array([ref_counts.get(c, 0) for c in all_cats])
                new_aligned = np.array([new_counts.get(c, 0) for c in all_cats])

                # Add small epsilon to avoid division by zero
                eps = 1e-8
                ref_aligned = ref_aligned + eps
                new_aligned = new_aligned + eps

                # Normalize
                ref_aligned = ref_aligned / ref_aligned.sum()
                new_aligned = new_aligned / new_aligned.sum()

                # PSI
                psi = np.sum((new_aligned - ref_aligned) * np.log(new_aligned / ref_aligned))

                results.append(QualityCheckResult(
                    check_name=f"distribution_alignment_{col}",
                    passed=psi < 0.2,
                    details={
                        "column": col,
                        "test": "psi",
                        "psi_value": round(float(psi), 4),
                        "new_categories_unseen": list(set(new_counts.index) - set(ref_counts.index)),
                    },
                    severity="warning",
                ))
        return results

    def _compute_profile(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute a statistical profile summary of the dataset."""
        profile: dict[str, Any] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "columns": {},
        }

        for col in df.columns:
            col_profile: dict[str, Any] = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_rate": round(df[col].isnull().mean(), 4),
                "unique_count": int(df[col].nunique()),
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                desc = df[col].describe()
                col_profile.update({
                    "mean": round(float(desc["mean"]), 4),
                    "std": round(float(desc["std"]), 4),
                    "min": round(float(desc["min"]), 4),
                    "p25": round(float(desc["25%"]), 4),
                    "median": round(float(desc["50%"]), 4),
                    "p75": round(float(desc["75%"]), 4),
                    "max": round(float(desc["max"]), 4),
                })

            profile["columns"][col] = col_profile

        return profile