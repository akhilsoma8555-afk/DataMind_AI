import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from .models import AnomalyItem, AnomalyReport

ID_PATTERNS = ['id', 'postal', 'zip', 'code', 'phone', 'ssn']

def _is_numeric_metric_candidate(col_name: str, series: pd.Series) -> bool:
    name_lower = col_name.lower().strip()
    if any(pat in name_lower for pat in ID_PATTERNS):
        return False
    if not pd.api.types.is_numeric_dtype(series):
        return False
    # Must have more than 2 distinct values to avoid binary flags
    if series.nunique() <= 2:
        return False
    return True

def detect_anomalies(df: pd.DataFrame) -> AnomalyReport:
    """
    Identifies unusual patterns and outliers in the dataset using:
    1. Scikit-learn Isolation Forest for multivariate / high-dimensional patterns
    2. Statistical IQR (Interquartile Range) & Z-score for univariate metrics
    3. Temporal trend anomaly detection when a date column is present
    """
    anomalies: List[AnomalyItem] = []
    
    if df.empty or len(df) < 5:
        return AnomalyReport(anomalies=[], columns_analyzed=[], total_anomalies=0)

    # 1. Select suitable numeric columns
    numeric_cols = [c for c in df.columns if _is_numeric_metric_candidate(c, df[c])]
    categorical_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if df[c].nunique() < 500]

    # Primary entity column (for reporting which product/customer/store was affected)
    entity_col = next((c for c in categorical_cols if any(k in c.lower() for k in ['product', 'customer', 'item', 'make', 'model', 'store', 'region', 'state', 'category'])), None)
    if not entity_col and categorical_cols:
        entity_col = categorical_cols[0]

    # 2. Univariate Outlier Detection using IQR & Z-score
    for col in numeric_cols[:6]: # Analyze up to top 6 numerical columns
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(series) < 5 or series.std() == 0:
            continue
            
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        median_val = float(series.median())
        mean_val = float(series.mean())
        std_val = float(series.std())

        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        extreme_upper = q75 + 3.0 * iqr
        extreme_lower = q25 - 3.0 * iqr

        outliers_df = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers_df)

        if outlier_count > 0:
            # Pick the most extreme outlier
            if (df[col] > upper_bound).any():
                extreme_idx = df[col].idxmax()
                extreme_row = df.loc[extreme_idx]
                obs_val = float(extreme_row[col]) if pd.notnull(extreme_row[col]) else 0.0
                direction = "high"
            else:
                extreme_idx = df[col].idxmin()
                extreme_row = df.loc[extreme_idx]
                obs_val = float(extreme_row[col]) if pd.notnull(extreme_row[col]) else 0.0
                direction = "low"

            # Compute severity based on distance from IQR boundaries & Z-score
            z_score = abs((obs_val - mean_val) / std_val) if std_val > 0 else 0
            if obs_val > extreme_upper or obs_val < extreme_lower or z_score >= 3.5:
                severity = "High"
            elif z_score >= 2.5 or outlier_count > len(df) * 0.05:
                severity = "Medium"
            else:
                severity = "Low"

            entity_val = str(extreme_row[entity_col]) if entity_col and entity_col in extreme_row and pd.notnull(extreme_row[entity_col]) else None
            
            # Format expected range and human-readable explanation
            expected_str = f"{max(0, lower_bound):,.2f} – {upper_bound:,.2f}"
            explanation = (
                f"Observed value of {obs_val:,.2f} deviates significantly from the normal expected range "
                f"({expected_str}, median {median_val:,.2f}). {outlier_count} record(s) ({outlier_count/len(df)*100:.1f}% of dataset) exceed statistical bounds."
            )
            if entity_val:
                explanation = f"Record '{entity_val}' shows an unusually {direction} {col} of {obs_val:,.2f}. {explanation}"

            # Prepare sample underlying records
            sample_records = outliers_df.head(5).to_dict(orient="records")
            # Clean records for JSON
            cleaned_samples = []
            for rec in sample_records:
                cleaned_samples.append({k: (v if pd.notnull(v) else None) for k, v in rec.items()})

            anomalies.append(AnomalyItem(
                id=f"iqr-{col}",
                metric=col,
                entity_column=entity_col,
                entity_value=entity_val,
                observed_value=round(obs_val, 2),
                expected_range=expected_str,
                severity=severity,
                method="IQR Outlier Analysis",
                explanation=explanation,
                affected_rows_count=outlier_count,
                sample_records=cleaned_samples
            ))

    # 3. Multivariate Anomaly Detection using Scikit-Learn Isolation Forest
    if len(numeric_cols) >= 2 and len(df) >= 20:
        try:
            feat_df = df[numeric_cols[:5]].apply(pd.to_numeric, errors='coerce').fillna(0)
            iso = IsolationForest(n_estimators=100, contamination=0.03, random_state=42)
            preds = iso.fit_predict(feat_df)
            scores = iso.decision_function(feat_df)

            anomaly_indices = np.where(preds == -1)[0]
            if len(anomaly_indices) > 0:
                # Find most anomalous sample (lowest decision score)
                most_anomalous_idx = df.index[anomaly_indices[np.argmin(scores[anomaly_indices])]]
                most_row = df.loc[most_anomalous_idx]
                min_score = float(scores[anomaly_indices].min())

                multi_severity = "High" if min_score < -0.15 else "Medium"
                primary_metric = numeric_cols[0]
                obs_val = most_row[primary_metric]
                
                ent_val = str(most_row[entity_col]) if entity_col and entity_col in most_row and pd.notnull(most_row[entity_col]) else None

                multi_samples = df.iloc[anomaly_indices[:5]].to_dict(orient="records")
                cleaned_multi = [{k: (v if pd.notnull(v) else None) for k, v in r.items()} for r in multi_samples]

                anomalies.append(AnomalyItem(
                    id="iso-forest-multivariate",
                    metric=f"Multivariate Pattern ({', '.join(numeric_cols[:3])})",
                    entity_column=entity_col,
                    entity_value=ent_val,
                    observed_value=f"Anomaly Score: {min_score:.3f}",
                    expected_range="Score > 0.00 (Normal cluster)",
                    severity=multi_severity,
                    method="Isolation Forest",
                    explanation=(
                        f"Detected {len(anomaly_indices)} records with abnormal joint combinations across "
                        f"{', '.join(numeric_cols[:3])} that deviate from typical multivariate clustering."
                        + (f" Extreme outlier identified: '{ent_val}'." if ent_val else "")
                    ),
                    affected_rows_count=len(anomaly_indices),
                    sample_records=cleaned_multi
                ))
        except Exception:
            pass

    # 4. Temporal Anomaly Detection (if date column exists)
    date_col = next((c for c in df.columns if any(k in c.lower() for k in ['date', 'time', 'timestamp'])), None)
    if date_col and numeric_cols:
        try:
            temp_df = df[[date_col, numeric_cols[0]]].copy()
            temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
            temp_df = temp_df.dropna().sort_values(date_col)
            if len(temp_df) >= 12:
                # Group by month or period
                temp_df.set_index(date_col, inplace=True)
                monthly = temp_df[numeric_cols[0]].resample('ME').sum()
                if len(monthly) >= 4:
                    pct_change = monthly.pct_change().dropna()
                    # Check for sudden drop (> 40% decline) or sudden surge (> 100% spike)
                    extreme_changes = pct_change[(pct_change < -0.40) | (pct_change > 1.0)]
                    if not extreme_changes.empty:
                        worst_period = extreme_changes.index[extreme_changes.abs().argmax()]
                        change_val = float(extreme_changes.loc[worst_period])
                        period_str = worst_period.strftime('%B %Y')
                        severity = "High" if abs(change_val) > 0.6 else "Medium"
                        direction = "drop" if change_val < 0 else "surge"

                        anomalies.append(AnomalyItem(
                            id=f"temporal-{numeric_cols[0]}",
                            metric=f"Monthly {numeric_cols[0]}",
                            entity_column=date_col,
                            entity_value=period_str,
                            observed_value=f"{change_val*100:+.1f}% vs prior month",
                            expected_range="±25% Monthly Variance",
                            severity=severity,
                            method="Time-Series Variance Analysis",
                            explanation=(
                                f"A sudden {direction} of {abs(change_val)*100:.1f}% in {numeric_cols[0]} occurred in {period_str}. "
                                f"Requires operational review to determine if caused by seasonal factors, disruption, or reporting changes."
                            ),
                            affected_rows_count=1,
                            sample_records=[]
                        ))
        except Exception:
            pass

    # Sort anomalies: High severity first, then Medium, then Low
    severity_rank = {"High": 0, "Medium": 1, "Low": 2}
    anomalies.sort(key=lambda a: severity_rank.get(a.severity, 3))

    return AnomalyReport(
        anomalies=anomalies,
        columns_analyzed=numeric_cols,
        total_anomalies=len(anomalies)
    )
