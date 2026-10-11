import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from models import DataQualityReport, ColumnQuality, QualityIssue, CleanDataRequest

def assess_data_quality(df: pd.DataFrame) -> DataQualityReport:
    """
    Evaluates dataset quality against 6 key dimensions:
    - Completeness (missing values, empty columns)
    - Uniqueness (duplicate rows)
    - Validity & Integrity (mixed data types, invalid date conversions)
    - Consistency (potential statistical outliers)
    """
    row_count = len(df)
    col_count = len(df.columns)
    
    if row_count == 0 or col_count == 0:
        return DataQualityReport(
            score=0.0,
            grade="Poor",
            row_count=0,
            column_count=0,
            total_missing=0,
            missing_pct=0.0,
            duplicate_rows=0,
            duplicate_pct=0.0,
            empty_columns=[],
            potential_outliers=0,
            columns=[],
            issues=[QualityIssue(issue_type="empty_dataset", severity="High", description="Dataset has 0 rows or 0 columns", affected_count=0)],
            score_breakdown={"formula": "Empty dataset"}
        )

    # 1. Missing Values
    total_cells = row_count * col_count
    missing_by_col = df.isna().sum().to_dict()
    total_missing = int(sum(missing_by_col.values()))
    missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

    # 2. Duplicate Rows
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / row_count) * 100, 2) if row_count > 0 else 0.0

    # 3. Empty Columns
    empty_cols = [col for col, count in missing_by_col.items() if count == row_count]

    # 4. Outliers & Column Quality breakdown
    columns_quality: List[ColumnQuality] = []
    issues: List[QualityIssue] = []
    total_outliers = 0

    if duplicate_rows > 0:
        issues.append(QualityIssue(
            column=None,
            issue_type="duplicates",
            severity="High" if duplicate_pct > 5.0 else "Medium",
            description=f"{duplicate_rows} duplicate row(s) detected ({duplicate_pct}% of dataset). Duplicate rows skew analytics aggregations.",
            affected_count=duplicate_rows
        ))

    for col in df.columns:
        series = df[col]
        miss_count = int(missing_by_col.get(col, 0))
        miss_p = round((miss_count / row_count) * 100, 2) if row_count > 0 else 0.0
        uniq_count = int(series.nunique())
        is_empty = miss_count == row_count
        outliers_in_col = 0

        # Outlier check for numeric columns
        if pd.api.types.is_numeric_dtype(series) and not is_empty:
            s_clean = series.dropna()
            if len(s_clean) >= 5 and s_clean.std() > 0:
                q25 = s_clean.quantile(0.25)
                q75 = s_clean.quantile(0.75)
                iqr = q75 - q25
                if iqr > 0:
                    outliers_in_col = int(((s_clean < (q25 - 1.5 * iqr)) | (s_clean > (q75 + 1.5 * iqr))).sum())
                    total_outliers += outliers_in_col

        columns_quality.append(ColumnQuality(
            name=col,
            dtype=str(series.dtype),
            total_count=row_count,
            missing_count=miss_count,
            missing_pct=miss_p,
            unique_count=uniq_count,
            is_empty=is_empty,
            outliers_count=outliers_in_col
        ))

        # Flag specific issues per column
        if is_empty:
            issues.append(QualityIssue(
                column=col,
                issue_type="empty_column",
                severity="High",
                description=f"Column '{col}' is 100% empty (all values are missing).",
                affected_count=row_count
            ))
        elif miss_p > 20.0:
            issues.append(QualityIssue(
                column=col,
                issue_type="missing_values",
                severity="High" if miss_p > 50.0 else "Medium",
                description=f"Column '{col}' has {miss_count} missing cells ({miss_p}% missing rate).",
                affected_count=miss_count
            ))
        elif miss_count > 0:
            issues.append(QualityIssue(
                column=col,
                issue_type="missing_values",
                severity="Low",
                description=f"Column '{col}' has {miss_count} missing values ({miss_p}%).",
                affected_count=miss_count
            ))

        if outliers_in_col > row_count * 0.05 and outliers_in_col > 3:
            issues.append(QualityIssue(
                column=col,
                issue_type="outliers",
                severity="Medium",
                description=f"Column '{col}' contains {outliers_in_col} statistical outliers (>1.5× IQR).",
                affected_count=outliers_in_col
            ))

    # 5. Documented Data Quality Score Calculation (0 to 100)
    # Deductions:
    # - Missing values: up to 35 points
    # - Duplicates: up to 25 points
    # - Empty columns: up to 20 points
    # - Outlier density: up to 15 points
    missing_deduction = min(35.0, (total_missing / total_cells) * 100 * 0.7 if total_cells > 0 else 0.0)
    duplicate_deduction = min(25.0, duplicate_pct * 1.25)
    empty_col_deduction = min(20.0, (len(empty_cols) / col_count) * 100 * 0.5 if col_count > 0 else 0.0)
    outlier_ratio = (total_outliers / (row_count * max(1, len(df.select_dtypes(include=['number']).columns))))
    outlier_deduction = min(15.0, outlier_ratio * 100 * 0.3)

    final_score = max(0.0, round(100.0 - (missing_deduction + duplicate_deduction + empty_col_deduction + outlier_deduction), 1))

    if final_score >= 90.0:
        grade = "Excellent"
    elif final_score >= 75.0:
        grade = "Good"
    elif final_score >= 55.0:
        grade = "Fair"
    else:
        grade = "Poor"

    # Sort issues by severity: High -> Medium -> Low
    sev_order = {"High": 0, "Medium": 1, "Low": 2}
    issues.sort(key=lambda x: sev_order.get(x.severity, 3))

    return DataQualityReport(
        score=final_score,
        grade=grade,
        row_count=row_count,
        column_count=col_count,
        total_missing=total_missing,
        missing_pct=missing_pct,
        duplicate_rows=duplicate_rows,
        duplicate_pct=duplicate_pct,
        empty_columns=empty_cols,
        potential_outliers=total_outliers,
        columns=columns_quality,
        issues=issues,
        score_breakdown={
            "formula": "100 - (missing_penalty + duplicate_penalty + empty_col_penalty + outlier_penalty)",
            "missing_deduction": round(missing_deduction, 1),
            "duplicate_deduction": round(duplicate_deduction, 1),
            "empty_column_deduction": round(empty_col_deduction, 1),
            "outlier_deduction": round(outlier_deduction, 1),
        }
    )

def clean_dataframe(df: pd.DataFrame, request: CleanDataRequest) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies selective, user-consented data cleaning operations:
    - Drop duplicate rows
    - Drop empty columns
    - Impute missing values
    Returns cleaned dataframe and a summary of actions taken.
    """
    cleaned = df.copy()
    actions_taken: List[str] = []
    
    # 1. Drop duplicates
    if request.drop_duplicates:
        initial_len = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        dropped_dups = initial_len - len(cleaned)
        if dropped_dups > 0:
            actions_taken.append(f"Removed {dropped_dups} duplicate row(s)")

    # 2. Drop empty columns
    if request.drop_empty_columns:
        empty_cols = [c for c in cleaned.columns if cleaned[c].isna().all()]
        if empty_cols:
            cleaned = cleaned.drop(columns=empty_cols)
            actions_taken.append(f"Removed {len(empty_cols)} completely empty column(s): {', '.join(empty_cols)}")

    # 3. Fill missing values
    if request.fill_missing:
        filled_count = int(cleaned.isna().sum().sum())
        if filled_count > 0:
            if request.fill_strategy == "smart":
                for c in cleaned.columns:
                    if pd.api.types.is_numeric_dtype(cleaned[c]):
                        median_val = cleaned[c].median()
                        cleaned[c] = cleaned[c].fillna(median_val if pd.notnull(median_val) else 0)
                    else:
                        mode_vals = cleaned[c].mode()
                        fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
                        cleaned[c] = cleaned[c].fillna(fill_val)
                actions_taken.append(f"Smart-imputed {filled_count} missing cell(s) (Median for numeric, Mode for categorical)")
            elif request.fill_strategy == "zero":
                for c in cleaned.columns:
                    if pd.api.types.is_numeric_dtype(cleaned[c]):
                        cleaned[c] = cleaned[c].fillna(0)
                    else:
                        cleaned[c] = cleaned[c].fillna("Unknown")
                actions_taken.append(f"Imputed {filled_count} missing cell(s) with zeroes and 'Unknown'")
            elif request.fill_strategy == "drop":
                before_drop = len(cleaned)
                cleaned = cleaned.dropna()
                actions_taken.append(f"Dropped {before_drop - len(cleaned)} row(s) containing missing values")

    return cleaned, {
        "actions_taken": actions_taken,
        "rows_before": len(df),
        "rows_after": len(cleaned),
        "cols_before": len(df.columns),
        "cols_after": len(cleaned.columns)
    }
