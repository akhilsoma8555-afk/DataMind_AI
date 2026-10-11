import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from fastapi import HTTPException
from .models import AnalyticsRequest, AnalyticsResponse

def execute_analytics(df: pd.DataFrame, request: AnalyticsRequest) -> AnalyticsResponse:
    op = (request.operation or "").lower().strip()
    col = request.column
    group_col = request.group_by
    
    # Work on a copy of dataframe
    work_df = df.copy()

    # Step 1: Apply all filters before any calculations
    filters_to_apply = []
    if request.filters:
        filters_to_apply.extend(request.filters)
    if request.filter_column and request.filter_value is not None:
        filters_to_apply.append({"column": request.filter_column, "value": request.filter_value, "operator": "=="})

    for f in filters_to_apply:
        f_col = f.column if hasattr(f, 'column') else f.get('column')
        f_val = f.value if hasattr(f, 'value') else f.get('value')
        f_op = getattr(f, 'operator', '==') if hasattr(f, 'operator') else f.get('operator', '==')

        if f_col not in work_df.columns:
            raise HTTPException(status_code=400, detail=f"Filter column '{f_col}' does not exist in dataset")

        f_val_str = str(f_val).strip()
        series_str = work_df[f_col].astype(str).str.strip()

        if f_op in ["==", "equals", "=", "eq"]:
            mask = series_str.str.lower() == f_val_str.lower()
        elif f_op in ["!=", "not_equals", "<>", "neq"]:
            mask = series_str.str.lower() != f_val_str.lower()
        elif f_op in ["contains", "like"]:
            mask = series_str.str.lower().str.contains(f_val_str.lower(), na=False, regex=False)
        else:
            mask = series_str.str.lower() == f_val_str.lower()

        work_df = work_df[mask]

    if work_df.empty:
        return AnalyticsResponse(
            operation=op,
            result=0 if op in ["sum", "count", "count_rows", "count_unique", "count_distinct"] else None,
            chart_data=None,
            explanation="No matching records found for the applied filter(s)."
        )

    try:
        # 1. list_unique: List distinct categories/values
        if op == "list_unique":
            target_c = col or group_col
            if not target_c:
                raise ValueError("list_unique requires a target column")
            if target_c not in work_df.columns:
                raise ValueError(f"Column '{target_c}' not found in dataset")
            
            uniques = [str(x) for x in work_df[target_c].dropna().unique().tolist()]
            # Sort for clean display
            uniques.sort()
            return AnalyticsResponse(
                operation=op,
                result={
                    "column": target_c,
                    "count": len(uniques),
                    "values": uniques,
                    "unique_values": uniques
                },
                chart_data=None,
                explanation=f"Found {len(uniques)} distinct values for {target_c}: {', '.join(uniques[:8])}{'...' if len(uniques) > 8 else ''}"
            )

        # 2. count_unique / count_distinct
        elif op in ["count_unique", "count_distinct"]:
            target_c = col or group_col
            if not target_c:
                raise ValueError("count_unique requires a target column")
            if target_c not in work_df.columns:
                raise ValueError(f"Column '{target_c}' not found in dataset")
            
            distinct_count = int(work_df[target_c].nunique())
            sample_uniques = [str(x) for x in work_df[target_c].dropna().unique()[:30].tolist()]
            return AnalyticsResponse(
                operation=op,
                result={
                    "column": target_c,
                    "count": distinct_count,
                    "unique_values": sample_uniques
                },
                chart_data=None,
                explanation=f"Found {distinct_count} unique values for {target_c}."
            )

        # 3. count / count_rows
        elif op in ["count", "count_rows"]:
            if group_col:
                agg = work_df.groupby(group_col).size()
                if request.sort == "descending":
                    agg = agg.sort_values(ascending=False)
                elif request.sort == "ascending":
                    agg = agg.sort_values(ascending=True)
                if request.limit:
                    agg = agg.head(request.limit)
                result = agg.to_dict()
                chart_data = {"type": "bar", "x": list(result.keys()), "y": list(result.values()), "title": f"Count by {group_col}"}
                return AnalyticsResponse(operation=op, result=result, chart_data=chart_data)
            else:
                return AnalyticsResponse(operation=op, result=len(work_df), explanation=f"Dataset contains {len(work_df)} records.")

        # 4. sum
        elif op in ["sum", "total"]:
            if not col:
                raise ValueError("sum operation requires a target numeric column")
            if col not in work_df.columns:
                raise ValueError(f"Column '{col}' does not exist in dataset")
            if group_col:
                agg = work_df.groupby(group_col)[col].sum()
                if request.sort == "descending" or (request.sort is None and not request.ascending):
                    agg = agg.sort_values(ascending=False)
                elif request.sort == "ascending" or request.ascending:
                    agg = agg.sort_values(ascending=True)
                if request.limit:
                    agg = agg.head(request.limit)
                result = agg.to_dict()
                chart_data = {"type": "bar", "x": list(result.keys()), "y": list(result.values()), "title": f"Sum of {col} by {group_col}"}
                return AnalyticsResponse(operation=op, result=result, chart_data=chart_data)
            else:
                return AnalyticsResponse(operation=op, result=float(work_df[col].sum()), explanation=f"Total {col}: {float(work_df[col].sum()):,.2f}")

        # 5. average / mean
        elif op in ["average", "mean", "avg"]:
            if not col:
                raise ValueError("average operation requires a target numeric column")
            if col not in work_df.columns:
                raise ValueError(f"Column '{col}' does not exist in dataset")
            if group_col:
                agg = work_df.groupby(group_col)[col].mean()
                if request.sort == "descending" or (request.sort is None and not request.ascending):
                    agg = agg.sort_values(ascending=False)
                elif request.sort == "ascending" or request.ascending:
                    agg = agg.sort_values(ascending=True)
                if request.limit:
                    agg = agg.head(request.limit)
                result = agg.to_dict()
                chart_data = {"type": "bar", "x": list(result.keys()), "y": list(result.values()), "title": f"Average of {col} by {group_col}"}
                return AnalyticsResponse(operation=op, result=result, chart_data=chart_data)
            else:
                return AnalyticsResponse(operation=op, result=float(work_df[col].mean()), explanation=f"Average {col}: {float(work_df[col].mean()):,.2f}")

        # 6. min
        elif op == "min":
            if not col:
                raise ValueError("min operation requires a target numeric column")
            if group_col:
                agg = work_df.groupby(group_col)[col].min()
                result = agg.to_dict()
                chart_data = {"type": "bar", "x": list(result.keys()), "y": list(result.values()), "title": f"Min of {col} by {group_col}"}
                return AnalyticsResponse(operation=op, result=result, chart_data=chart_data)
            else:
                return AnalyticsResponse(operation=op, result=float(work_df[col].min()))

        # 7. max
        elif op == "max":
            if not col:
                raise ValueError("max operation requires a target numeric column")
            if group_col:
                agg = work_df.groupby(group_col)[col].max()
                result = agg.to_dict()
                chart_data = {"type": "bar", "x": list(result.keys()), "y": list(result.values()), "title": f"Max of {col} by {group_col}"}
                return AnalyticsResponse(operation=op, result=result, chart_data=chart_data)
            else:
                return AnalyticsResponse(operation=op, result=float(work_df[col].max()))

        # 8. median
        elif op == "median":
            if not col:
                raise ValueError("median operation requires a target numeric column")
            if group_col:
                agg = work_df.groupby(group_col)[col].median()
                result = agg.to_dict()
                chart_data = {"type": "bar", "x": list(result.keys()), "y": list(result.values()), "title": f"Median of {col} by {group_col}"}
                return AnalyticsResponse(operation=op, result=result, chart_data=chart_data)
            else:
                return AnalyticsResponse(operation=op, result=float(work_df[col].median()))

        # 9. group_by / group_aggregate / compare
        elif op in ["group_by", "group_aggregate", "compare"]:
            val_c = request.value_column or col
            if not group_col or not val_c:
                raise ValueError("group aggregation requires group_by and value column")
            if val_c not in work_df.columns:
                raise ValueError(f"Column '{val_c}' not found in dataset")
            
            is_mean = getattr(request, 'metric_agg', None) in ["mean", "avg", "average"]
            agg = work_df.groupby(group_col)[val_c].mean() if is_mean else work_df.groupby(group_col)[val_c].sum()
            agg_label = f"Average {val_c}" if is_mean else f"Total {val_c}"

            if request.sort == "descending" or (request.sort is None and not request.ascending):
                agg = agg.sort_values(ascending=False)
            elif request.sort == "ascending" or request.ascending:
                agg = agg.sort_values(ascending=True)
            if request.limit:
                agg = agg.head(request.limit)
            result = agg.to_dict()
            chart_data = {"type": "bar", "x": list(result.keys()), "y": list(result.values()), "title": f"{agg_label} by {group_col}"}
            return AnalyticsResponse(operation=op, result=result, chart_data=chart_data, explanation=f"Comparison of {val_c} across {len(result)} {group_col} segments.")

        # 10. top_n / bottom_n / top_performer
        elif op in ["top_n", "bottom_n", "top_performer"]:
            val_c = request.value_column or col
            if not group_col or not val_c:
                raise ValueError("top/bottom ranking requires group_by and value_column")
            if val_c not in work_df.columns:
                raise ValueError(f"Column '{val_c}' not found in dataset")
            
            is_mean = getattr(request, 'metric_agg', None) in ["mean", "avg", "average"]
            agg = work_df.groupby(group_col)[val_c].mean() if is_mean else work_df.groupby(group_col)[val_c].sum()
            agg_word = "average" if is_mean else "total"
            if agg.empty:
                return AnalyticsResponse(operation=op, result=None)
            
            is_ascending = (op == "bottom_n") or (request.sort == "ascending") or request.ascending
            agg_sorted = agg.sort_values(ascending=is_ascending)
            
            limit_n = request.limit if request.limit else (1 if op == "top_performer" else 5)
            sliced = agg_sorted.head(limit_n)
            
            ranking_dict = {str(k): float(v) for k, v in sliced.items()}
            top_entity = str(sliced.index[0])
            top_val = float(sliced.iloc[0])
            
            direction_word = "Lowest" if is_ascending else "Top"
            chart_data = {
                "type": "bar",
                "x": list(sliced.index),
                "y": [float(v) for v in sliced.values],
                "title": f"{direction_word} {limit_n} {group_col} by {val_c}"
            }
            
            if limit_n == 1:
                explanation = f"'{top_entity}' had the {'lowest' if is_ascending else 'highest'} {agg_word} {val_c} ({top_val:,.2f})."
            else:
                top_items_summary = ", ".join([f"{k} ({v:,.2f})" for k, v in list(ranking_dict.items())[:3]])
                explanation = f"{direction_word} {limit_n} {group_col} by {val_c}: {top_items_summary}..."

            return AnalyticsResponse(
                operation=op,
                result={
                    "entity": top_entity,
                    "value": top_val,
                    "ranking": ranking_dict,
                    "limit": limit_n
                },
                chart_data=chart_data,
                explanation=explanation
            )

        # 11. trend / time_series
        elif op in ["trend", "time_series"]:
            d_col = request.date_column
            val_c = request.value_column or col
            if not d_col or not val_c:
                raise ValueError("Trend analysis requires both a Date Column and a Value Column.")
            if d_col not in work_df.columns:
                raise ValueError(f"Date column '{d_col}' not found in dataset.")
            if val_c not in work_df.columns:
                raise ValueError(f"Value column '{val_c}' not found in dataset.")
            if not pd.api.types.is_numeric_dtype(work_df[val_c]):
                raise ValueError(f"Selected Value Column '{val_c}' is non-numeric ({work_df[val_c].dtype}). Trend analysis requires a numeric metric column such as Sales or Profit.")
                
            temp_df = work_df.copy()
            if pd.api.types.is_numeric_dtype(temp_df[d_col]):
                if temp_df[d_col].dropna().min() >= 1900 and temp_df[d_col].dropna().max() <= 2100:
                    temp_df[d_col] = pd.to_datetime(temp_df[d_col].astype(str), format='%Y', errors='coerce')
                else:
                    temp_df[d_col] = pd.to_datetime(temp_df[d_col], errors='coerce')
            else:
                temp_df[d_col] = pd.to_datetime(temp_df[d_col], errors='coerce')

            valid_dates = temp_df.dropna(subset=[d_col]).copy()
            if valid_dates.empty:
                return AnalyticsResponse(
                    operation=op,
                    result={},
                    chart_data=None,
                    explanation=f"Column '{d_col}' contains no valid or parseable date values."
                )
            
            interval = (request.interval or ("year" if 'year' in d_col.lower() else "month")).lower()
            if interval in ["day", "daily", "d"]:
                period_freq = 'D'
                interval_title = "Daily"
            elif interval in ["year", "yearly", "annual", "y"] or 'year' in d_col.lower():
                period_freq = 'Y'
                interval_title = "Yearly"
            else:
                period_freq = 'M'
                interval_title = "Monthly"
            
            valid_dates['Period'] = valid_dates[d_col].dt.to_period(period_freq)
            is_mean = getattr(request, 'metric_agg', None) in ["mean", "avg", "average"]
            grouped = valid_dates.groupby('Period')[val_c].mean().sort_index() if is_mean else valid_dates.groupby('Period')[val_c].sum().sort_index()
            agg_word = "Average" if is_mean else "Total"
            
            labels = [str(p) for p in grouped.index]
            values = [float(v) for v in grouped.values]
            
            trend_data = dict(zip(labels, values))
            chart_data = {
                "type": "line",
                "x": labels,
                "y": values,
                "title": f"{interval_title} {agg_word} of {val_c}"
            }
            return AnalyticsResponse(
                operation=op,
                result=trend_data,
                chart_data=chart_data,
                explanation=f"{interval_title} chronological trend of {val_c} across {len(labels)} periods."
            )

        # 12. correlation
        elif op == "correlation":
            num_cols = list(work_df.select_dtypes(include=['number']).columns)
            if len(num_cols) < 2:
                return AnalyticsResponse(operation=op, result=None, explanation="Dataset has fewer than 2 numeric columns for correlation.")
            corr_matrix = work_df[num_cols].corr().round(3).to_dict()
            return AnalyticsResponse(operation=op, result=corr_matrix, explanation="Computed pairwise Pearson correlation matrix for numeric columns.")

        # 13. describe_dataset
        elif op == "describe_dataset":
            desc = {
                "total_rows": len(work_df),
                "total_columns": len(work_df.columns),
                "columns": list(work_df.columns),
                "numeric_columns": list(work_df.select_dtypes(include=['number']).columns),
                "categorical_columns": list(work_df.select_dtypes(include=['object', 'category', 'string']).columns),
                "missing_values": work_df.isna().sum().to_dict()
            }
            return AnalyticsResponse(operation=op, result=desc, explanation=f"Dataset contains {len(work_df)} rows and {len(work_df.columns)} columns.")

        # 14. profit_margin
        elif op in ["profit_margin", "margin"]:
            profit_c = next((c for c in work_df.columns if 'profit' in c.lower()), None)
            sales_c = next((c for c in work_df.columns if 'sale' in c.lower() or 'revenue' in c.lower() or 'price' in c.lower()), None)
            if profit_c and sales_c:
                tot_profit = float(work_df[profit_c].sum())
                tot_sales = float(work_df[sales_c].sum())
                margin_pct = (tot_profit / tot_sales * 100) if tot_sales != 0 else 0.0
                filter_entity = filters_to_apply[0].get('value') if filters_to_apply and isinstance(filters_to_apply[0], dict) else None
                expl = f"Profit margin for '{filter_entity}' is {margin_pct:.2f}% (Total Profit: ${tot_profit:,.2f}, Total Sales: ${tot_sales:,.2f})." if filter_entity else f"Overall profit margin is {margin_pct:.2f}% (Total Profit: ${tot_profit:,.2f}, Total Sales: ${tot_sales:,.2f})."
                return AnalyticsResponse(
                    operation=op,
                    result={
                        "profit_margin_pct": round(margin_pct, 2),
                        "total_profit": round(tot_profit, 2),
                        "total_sales": round(tot_sales, 2),
                        "entity": filter_entity
                    },
                    chart_data={"type": "bar", "x": ["Sales", "Profit"], "y": [tot_sales, tot_profit], "title": f"Sales vs Profit ({filter_entity or 'Total'})"},
                    explanation=expl
                )
            elif profit_c:
                tot_profit = float(work_df[profit_c].sum())
                return AnalyticsResponse(operation=op, result=tot_profit, explanation=f"Total profit: ${tot_profit:,.2f}")
            else:
                raise ValueError("Profit column not found in dataset for margin calculation")

        # 15. scatter / relationship
        elif op in ["scatter", "relationship"]:
            x_c = request.column
            y_c = request.value_column
            num_cols = list(work_df.select_dtypes(include=['number']).columns)
            if not x_c or x_c not in work_df.columns:
                x_c = num_cols[0] if len(num_cols) > 0 else None
            if not y_c or y_c not in work_df.columns:
                x_c_idx = num_cols.index(x_c) if x_c in num_cols else -1
                candidates = [c for c in num_cols if c != x_c]
                y_c = candidates[0] if candidates else x_c
            if x_c and y_c:
                clean_df = work_df[[x_c, y_c]].dropna().head(100)
                x_vals = [float(v) for v in clean_df[x_c].values]
                y_vals = [float(v) for v in clean_df[y_c].values]
                corr = float(work_df[x_c].corr(work_df[y_c])) if x_c != y_c else 1.0
                chart_data = {
                    "type": "scatter",
                    "x": x_vals,
                    "y": y_vals,
                    "title": f"Relationship: {y_c} vs {x_c}"
                }
                return AnalyticsResponse(
                    operation=op,
                    result={"correlation": round(corr, 3), "x_column": x_c, "y_column": y_c},
                    chart_data=chart_data,
                    explanation=f"Relationship between {x_c} and {y_c} (Pearson correlation: {corr:.3f})."
                )

        # 16. distribution / histogram
        elif op in ["distribution", "histogram"]:
            target_c = col or request.value_column
            if not target_c or target_c not in work_df.columns:
                num_cols = list(work_df.select_dtypes(include=['number']).columns)
                target_c = num_cols[0] if num_cols else None
            if target_c and target_c in work_df.columns:
                clean_s = pd.to_numeric(work_df[target_c], errors='coerce').dropna()
                counts, bin_edges = np.histogram(clean_s, bins=10)
                bin_labels = [f"{bin_edges[i]:,.1f} - {bin_edges[i+1]:,.1f}" for i in range(len(counts))]
                chart_data = {
                    "type": "bar",
                    "x": bin_labels,
                    "y": [int(c) for c in counts],
                    "title": f"Distribution of {target_c}"
                }
                return AnalyticsResponse(
                    operation=op,
                    result=dict(zip(bin_labels, [int(c) for c in counts])),
                    chart_data=chart_data,
                    explanation=f"Distribution of {target_c} across {len(clean_s):,} records (Mean: {clean_s.mean():,.2f}, Std: {clean_s.std():,.2f})."
                )

        # 17. proportion / share / pie
        elif op in ["proportion", "share", "pie"]:
            group_c = group_col or col
            val_c = request.value_column or col
            if group_c in work_df.columns:
                if val_c and val_c in work_df.columns and pd.api.types.is_numeric_dtype(work_df[val_c]):
                    agg = work_df.groupby(group_c)[val_c].sum().sort_values(ascending=False).head(8)
                else:
                    agg = work_df[group_c].value_counts().head(8)
                chart_data = {
                    "type": "pie",
                    "x": [str(k) for k in agg.index],
                    "y": [float(v) for v in agg.values],
                    "title": f"Proportion of {val_c or 'Records'} by {group_c}"
                }
                return AnalyticsResponse(
                    operation=op,
                    result=agg.to_dict(),
                    chart_data=chart_data,
                    explanation=f"Categorical share and proportions across top {len(agg)} {group_c} segments."
                )

        # 18. business_report
        elif op == "business_report":
            # Synthesize key findings
            num_cols = list(work_df.select_dtypes(include=['number']).columns)
            cat_cols = list(work_df.select_dtypes(include=['object', 'category', 'string']).columns)
            
            report = {
                "report_title": "Executive Business Performance Report",
                "summary": f"Business analysis of {len(work_df):,} rows across {len(work_df.columns)} dimensions.",
                "metrics": {},
                "top_performers": {},
                "recommendations": []
            }
            
            # Primary sales metric
            sales_col = next((c for c in num_cols if 'sale' in c.lower() or 'revenue' in c.lower()), (num_cols[0] if num_cols else None))
            if sales_col:
                total_sales = float(work_df[sales_col].sum())
                avg_sales = float(work_df[sales_col].mean())
                report["metrics"][f"Total {sales_col}"] = f"${total_sales:,.2f}"
                report["metrics"][f"Average {sales_col}"] = f"${avg_sales:,.2f}"
            
            # Primary category breakdown
            cat_col = next((c for c in cat_cols if 'category' in c.lower()), (cat_cols[0] if cat_cols else None))
            chart_data = None
            if cat_col and sales_col:
                agg = work_df.groupby(cat_col)[sales_col].sum().sort_values(ascending=False)
                top_cat = agg.index[0]
                top_cat_val = float(agg.iloc[0])
                lowest_cat = agg.index[-1]
                lowest_cat_val = float(agg.iloc[-1])
                report["top_performers"][f"Leading {cat_col}"] = f"{top_cat} (${top_cat_val:,.2f})"
                report["top_performers"][f"Lowest {cat_col}"] = f"{lowest_cat} (${lowest_cat_val:,.2f})"
                chart_data = {"type": "bar", "x": list(agg.index), "y": [float(v) for v in agg.values], "title": f"{sales_col} by {cat_col}"}
                
                report["recommendations"].append(f"Focus sales incentive programs on '{top_cat}' which drives the highest volume ({top_cat_val:,.2f}).")
                report["recommendations"].append(f"Review marketing and pricing strategy for '{lowest_cat}' to address underperformance.")

            report["kpis"] = [{"label": k, "value": v} for k, v in report["metrics"].items()]
            report["performers"] = [{"title": k, "entity": v.split(" (")[0], "metric": sales_col or "Metric", "value": v} for k, v in report["top_performers"].items()]

            return AnalyticsResponse(operation=op, result=report, chart_data=chart_data, explanation="Comprehensive business performance report generated.")

        elif op == "filter":
            return AnalyticsResponse(operation=op, result={"filtered_rows": len(work_df), "total_rows": len(df)}, chart_data=None)

        else:
            raise ValueError(f"Unsupported operation: {op}")
            
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Column not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analytics calculation failed: {str(e)}")
