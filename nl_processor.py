import json
import re
import pandas as pd
from typing import Any, Dict, List, Optional
from fastapi import HTTPException
from app.models import NLQueryResponse, AnalyticsRequest, FilterItem
from app.analytics import execute_analytics

ID_COLUMN_PATTERNS = [r'id$', r'_id$', r'order_id', r'row_id', r'postal_code', r'zip']

def _is_id_column(col_name: str) -> bool:
    clean = col_name.lower().strip()
    return any(re.search(pat, clean) for pat in ID_COLUMN_PATTERNS)

def _find_column_by_name_or_synonym(name: Optional[str], columns: List[str]) -> Optional[str]:
    if not name:
        return None
    name_clean = name.lower().strip().replace('_', ' ').replace('-', ' ')
    
    # Exact match after punctuation/case normalization
    for col in columns:
        col_clean = col.lower().strip().replace('_', ' ').replace('-', ' ')
        if name_clean == col_clean:
            return col
            
    # Specific known domain aliases
    synonym_map = {
        "customer": "Customer Name",
        "customers": "Customer Name",
        "client": "Customer Name",
        "clients": "Customer Name",
        "product": "Product Name",
        "products": "Product Name",
        "item": "Product Name",
        "items": "Product Name",
        "sub category": "Sub-Category",
        "subcategory": "Sub-Category",
        "sub categories": "Sub-Category",
        "subcategories": "Sub-Category",
        "revenue": "Sales",
        "sales amount": "Sales",
        "turnover": "Sales",
        "margin": "Profit",
        "earnings": "Profit",
        "date": "Order Date",
        "order date": "Order Date",
        "order": "Order ID",
        "orders": "Order ID",
        "country": "Country",
        "countries": "Country",
        "city": "City",
        "cities": "City",
        "state": "State",
        "states": "State",
        "region": "Region",
        "regions": "Region",
        "segment": "Segment",
        "segments": "Segment",
        # Automotive domain
        "car make": "Car Make",
        "make": "Car Make",
        "makes": "Car Make",
        "manufacturer": "Car Make",
        "manufacturers": "Car Make",
        "brand": "Car Make",
        "brands": "Car Make",
        "car model": "Car Model",
        "model": "Car Model",
        "models": "Car Model",
        "car models": "Car Model",
        "fuel": "Fuel Type",
        "fuel type": "Fuel Type",
        "fuel types": "Fuel Type",
        "transmission": "Transmission",
        "transmissions": "Transmission",
        "gearbox": "Transmission",
        "mileage": "Mileage",
        "miles": "Mileage",
        "price": "Price",
        "prices": "Price",
        "car price": "Price",
        "car prices": "Price",
        "cost": "Price",
        "year": "Year",
        "years": "Year",
        "color": "Color",
        "colors": "Color",
        "condition": "Condition",
        "accident": "Accident",
        "accidents": "Accident",
    }
    
    if name_clean in synonym_map:
        target = synonym_map[name_clean]
        for col in columns:
            if col.lower().strip().replace('_', ' ').replace('-', ' ') == target.lower().replace('-', ' '):
                return col
            # Direct partial check for target
            if target.lower() in col.lower() or col.lower() in target.lower():
                return col
                
    # Substring match with boundary check
    for col in columns:
        col_clean = col.lower().strip().replace('_', ' ').replace('-', ' ')
        if name_clean == col_clean or f" {name_clean} " in f" {col_clean} " or (len(name_clean) >= 4 and name_clean in col_clean):
            return col

    return None

def _get_dataset_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect actual columns, data types, and unique values for categorical columns."""
    num_cols = [c for c in df.select_dtypes(include=['number']).columns if not _is_id_column(c)]
    cat_cols = list(df.select_dtypes(include=['object', 'category', 'string']).columns)
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'month' in c.lower() or 'year' in c.lower()]

    cat_values = {}
    for c in cat_cols:
        uniques = df[c].dropna().unique()
        if len(uniques) <= 100:
            cat_values[c] = [str(x) for x in uniques.tolist()]
        else:
            cat_values[c] = [str(x) for x in uniques[:30].tolist()]

    return {
        "all_columns": list(df.columns),
        "numeric_metric_columns": num_cols,
        "categorical_columns": cat_cols,
        "date_columns": date_cols,
        "categorical_values_sample": cat_values
    }

def _get_primary_metric_column(num_cols: List[str]) -> Optional[str]:
    """Dynamically determine the most meaningful business metric (e.g. Sales, Price, Revenue, Profit, Mileage)."""
    priority_keywords = ['sales', 'revenue', 'price', 'profit', 'amount', 'total', 'cost', 'mileage', 'value', 'salary', 'income', 'spend', 'quantity']
    for kw in priority_keywords:
        for col in num_cols:
            if kw in col.lower():
                return col
    # Prefer columns that are not year/date
    non_temporal = [c for c in num_cols if not any(t in c.lower() for t in ['year', 'yr', 'date', 'month', 'day', 'quarter'])]
    if non_temporal:
        return non_temporal[0]
    return num_cols[0] if num_cols else None

def _heuristic_parse_question(question: str, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
    q_lower = question.lower().strip()
    num_cols = context["numeric_metric_columns"]
    cat_cols = context["categorical_columns"]
    cat_values = context["categorical_values_sample"]
    date_cols = context["date_columns"]
    all_cols = context["all_columns"]

    # 1. Report queries
    if any(k in q_lower for k in ["business report", "performance report", "summary report", "executive report", "overview report"]):
        return {
            "operation": "business_report"
        }

    # 2. Dataset overview
    if any(k in q_lower for k in ["describe dataset", "dataset overview", "dataset summary", "show dataset info", "columns in dataset", "overview of this dataset"]):
        return {
            "operation": "describe_dataset"
        }

    # 3. Correlation queries
    if any(k in q_lower for k in ["correlation", "correlated", "correlations"]):
        return {
            "operation": "correlation"
        }

    # 4. Detect categorical filters in question
    filters = []
    for col in cat_cols:
        unique_vals = cat_values.get(col, [])
        for val in unique_vals:
            val_lower = val.lower().strip()
            if len(val_lower) >= 3:
                pattern = rf"\b{re.escape(val_lower)}\b"
                if re.search(pattern, q_lower):
                    if not any(f["column"] == col and f["value"].lower() == val.lower() for f in filters):
                        filters.append({"column": col, "value": val, "operator": "=="})

    # 5. Extract top/bottom limit
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    limit = None
    limit_match = re.search(r'\b(top|bottom)\s+(\d+)\b', q_lower)
    if limit_match:
        limit = int(limit_match.group(2))
    else:
        word_match = re.search(r'\b(top|bottom)\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b', q_lower)
        if word_match:
            limit = word_to_num.get(word_match.group(2), 5)

    is_bottom = "bottom" in q_lower or "worst" in q_lower or "least" in q_lower or "lowest" in q_lower
    is_average_requested = any(k in q_lower for k in ["average", "mean", "avg"])

    # 6. Detect grouping column
    group_col = None
    detected_cand = None
    filtered_cols = [f["column"] for f in filters]

    dimension_checks = [
        ("Sub-Category", ["sub-category", "sub category", "subcategory", "sub-categories", "sub categories", "subcategories"]),
        ("Category", ["product category", "product categories", "category", "categories"]),
        ("Customer Name", ["customer name", "customer", "customers", "client", "clients"]),
        ("Product Name", ["product name", "item", "items"]),
        ("Car Make", ["car make", "make", "makes", "manufacturer", "manufacturers", "brand", "brands"]),
        ("Car Model", ["car model", "car models", "model", "models"]),
        ("Fuel Type", ["fuel type", "fuel types", "fuel"]),
        ("Transmission", ["transmission", "transmissions", "gearbox"]),
        ("Condition", ["condition", "conditions"]),
        ("Accident", ["accident", "accidents"]),
        ("Color", ["color", "colors", "colour", "colours"]),
        ("State", ["state", "states"]),
        ("Region", ["region", "regions"]),
        ("City", ["city", "cities"]),
        ("Segment", ["segment", "segments"]),
        ("Ship Mode", ["ship mode", "ship modes", "shipping mode", "shipping modes", "shipping"]),
    ]

    for dim_name, keywords in dimension_checks:
        if any(k in q_lower for k in keywords):
            cand = _find_column_by_name_or_synonym(dim_name, all_cols)
            if cand:
                detected_cand = cand
                break

    # If no predefined dimension matched, dynamically check actual categorical columns
    if not detected_cand:
        for col in cat_cols:
            col_c = col.lower().replace('-', ' ').replace('_', ' ')
            col_sing = col_c.rstrip('s')
            if col_c in q_lower or (len(col_sing) >= 3 and col_sing in q_lower):
                detected_cand = col
                break

    if detected_cand:
        if detected_cand in filtered_cols:
            cand_clean = detected_cand.lower().replace('_', ' ').replace('-', ' ')
            if f"by {cand_clean}" in q_lower or f"per {cand_clean}" in q_lower or f"each {cand_clean}" in q_lower:
                group_col = detected_cand
        else:
            group_col = detected_cand

    if not group_col:
        for col in cat_cols:
            col_clean = col.lower().replace('_', ' ').replace('-', ' ')
            col_sing = col_clean.rstrip('s')
            if any(f"{prep} {col_clean}" in q_lower or f"{prep} {col_sing}" in q_lower for prep in ["by", "per", "for each", "each", "across", "between"]):
                group_col = col
                break

    # 7. Check for list_unique: "What are the different...", "List all...", "Which categories are present"
    is_comparison = any(k in q_lower for k in ["compare", "comparison", "breakdown", "versus", "vs.", "vs "])
    has_metric_word = any(k in q_lower for k in ["sales", "profit", "revenue", "price", "prices", "mileage", "quantity", "discount", "amount", "total", "sum", "average", "mean"])
    if not is_comparison and not has_metric_word and any(k in q_lower for k in ["different", "list all", "list of", "show unique", "which are present", "what are the categories", "what are the shipping", "what categories", "what shipping", "what distinct", "list distinct"]):
        target_list_col = group_col or detected_cand
        if not target_list_col:
            for col in cat_cols:
                col_c = col.lower().replace('-', ' ').replace('_', ' ')
                if col_c in q_lower or (col_c + 's') in q_lower or (col_c + 'es') in q_lower:
                    target_list_col = col
                    break
        if target_list_col:
            return {
                "operation": "list_unique",
                "column": target_list_col,
                "filters": filters
            }

    # 7b. Check for two-metric comparison (e.g. "Compare sales and profit")
    mentioned_metrics = []
    for kw, col_target in [("sales", "Sales"), ("profit", "Profit"), ("revenue", "Sales"), ("price", "Price"), ("cost", "Price"), ("mileage", "Mileage"), ("quantity", "Quantity"), ("discount", "Discount")]:
        if kw in q_lower:
            found_c = _find_column_by_name_or_synonym(col_target, all_cols)
            if found_c and found_c not in mentioned_metrics:
                mentioned_metrics.append(found_c)

    if len(mentioned_metrics) >= 2 and any(k in q_lower for k in ["compare", "vs", "versus", "relationship", "and", "against"]):
        return {
            "operation": "scatter",
            "column": mentioned_metrics[0],
            "value_column": mentioned_metrics[1],
            "filters": filters
        }

    # 7c. Average order value (AOV)
    if "average order value" in q_lower or "aov" in q_lower:
        sales_c = _find_column_by_name_or_synonym("Sales", all_cols) or _get_primary_metric_column(num_cols)
        return {
            "operation": "average",
            "column": sales_c,
            "filters": filters
        }

    # 8. Detect target metric column
    target_metric = None
    if "profit" in q_lower or "margin" in q_lower:
        target_metric = _find_column_by_name_or_synonym("Profit", all_cols)
        if not target_metric:
            return {
                "operation": "missing_column",
                "missing_column": "Profit",
                "message": "The uploaded dataset does not contain a 'Profit' column."
            }
    elif any(k in q_lower for k in ["price", "prices", "cost", "pricing"]):
        target_metric = _find_column_by_name_or_synonym("Price", all_cols)
    elif any(k in q_lower for k in ["mileage", "miles", "odometer"]):
        target_metric = _find_column_by_name_or_synonym("Mileage", all_cols)
    elif "quantity" in q_lower or "volume" in q_lower:
        target_metric = _find_column_by_name_or_synonym("Quantity", all_cols)
    elif "discount" in q_lower:
        target_metric = _find_column_by_name_or_synonym("Discount", all_cols)
    elif any(k in q_lower for k in ["sales", "sale", "revenue", "turnover"]):
        target_metric = _find_column_by_name_or_synonym("Sales", all_cols)

    # Check if any other numeric column is directly mentioned
    if not target_metric:
        for c in num_cols:
            c_clean = c.lower().replace('_', ' ').replace('-', ' ')
            if c_clean in q_lower:
                target_metric = c
                break

    # If still none and spending/buying keywords present, look for primary metric
    if not target_metric:
        if any(w in q_lower for w in ["sold", "bought", "spending", "spend", "purchases", "order value"]):
            target_metric = _find_column_by_name_or_synonym("Sales", all_cols) or _get_primary_metric_column(num_cols)

    # 9. Distinct count queries: "How many unique...", "How many models...", "How many orders"
    if any(k in q_lower for k in ["how many", "number of", "count of", "unique", "distinct", "represented"]):
        # Car models
        if any(w in q_lower for w in ["car model", "car models", "model", "models"]):
            model_col = _find_column_by_name_or_synonym("Car Model", all_cols)
            if model_col:
                return {
                    "operation": "count_unique",
                    "column": model_col,
                    "distinct": True,
                    "filters": filters
                }
        # Car makes / manufacturers
        if any(w in q_lower for w in ["car make", "manufacturer", "manufacturers", "brand", "brands", "make", "makes"]):
            make_col = _find_column_by_name_or_synonym("Car Make", all_cols)
            if make_col:
                return {
                    "operation": "count_unique",
                    "column": make_col,
                    "distinct": True,
                    "filters": filters
                }
        # Products
        if "product" in q_lower or "products" in q_lower:
            prod_col = _find_column_by_name_or_synonym("Product ID", all_cols) or _find_column_by_name_or_synonym("Product Name", all_cols)
            if prod_col:
                return {
                    "operation": "count_unique",
                    "column": prod_col,
                    "distinct": True,
                    "filters": filters
                }
        # Orders
        if "order" in q_lower or "orders" in q_lower:
            order_col = _find_column_by_name_or_synonym("Order ID", all_cols)
            if order_col:
                return {
                    "operation": "count_unique",
                    "column": order_col,
                    "distinct": True,
                    "filters": filters
                }
        # Customers
        if "customer" in q_lower or "customers" in q_lower:
            cust_col = _find_column_by_name_or_synonym("Customer Name", all_cols) or _find_column_by_name_or_synonym("Customer ID", all_cols)
            if cust_col:
                return {
                    "operation": "count_unique",
                    "column": cust_col,
                    "distinct": True,
                    "filters": filters
                }

        # Any other categorical dimension in dataset
        for col in cat_cols:
            col_clean = col.lower().replace('-', ' ').replace('_', ' ')
            col_sing = col_clean.rstrip('s')
            if col_clean in q_lower or (len(col_sing) >= 3 and col_sing in q_lower):
                return {
                    "operation": "count_unique",
                    "column": col,
                    "distinct": True,
                    "filters": filters
                }

        if any(w in q_lower for w in ["row", "rows", "record", "records", "transaction", "transactions", "cars", "items", "entries"]):
            return {
                "operation": "count_rows",
                "filters": filters
            }

    # 10. Trend / time_series
    if any(k in q_lower for k in ["trend", "over time", "monthly", "by month", "by year", "yearly", "timeline", "trend of", "over the years"]):
        d_col = date_cols[0] if date_cols else _find_column_by_name_or_synonym("Order Date", all_cols)
        metric = target_metric or _get_primary_metric_column(num_cols)
        return {
            "operation": "time_series",
            "date_column": d_col,
            "value_column": metric,
            "column": metric,
            "metric_agg": "mean" if is_average_requested else "sum",
            "filters": filters
        }

    # 11. Top N / Bottom N / Highest / Lowest ranking
    if limit or any(k in q_lower for k in ["top", "bottom", "highest", "lowest", "best", "worst", "most", "major", "largest", "maximum sales"]):
        # Check if an explicit entity was requested that is not present in this dataset
        for ent_kw, ent_col in [("customer", "Customer Name"), ("client", "Customer Name"), ("sales rep", "Sales Rep"), ("vendor", "Vendor"), ("store", "Store")]:
            if ent_kw in q_lower and not _find_column_by_name_or_synonym(ent_col, all_cols):
                return {
                    "operation": "missing_column",
                    "missing_column": ent_col,
                    "message": f"The dataset does not contain a '{ent_col}' column. Available dimensions: {', '.join(cat_cols)}."
                }

        metric = target_metric or _get_primary_metric_column(num_cols)
        if group_col and metric:
            if limit:
                effective_limit = limit
            elif any(pl in q_lower for pl in ["sub-categories", "categories", "customers", "products", "states", "regions", "cities", "makes", "manufacturers", "models", "brands"]):
                effective_limit = 5
            else:
                effective_limit = 1

            op_name = "bottom_n" if is_bottom else "top_n"
            return {
                "operation": op_name,
                "column": metric,
                "value_column": metric,
                "group_by": group_col,
                "limit": effective_limit,
                "ascending": is_bottom,
                "sort": "ascending" if is_bottom else "descending",
                "metric_agg": "mean" if is_average_requested else "sum",
                "filters": filters
            }

    # 12. Median
    if any(k in q_lower for k in ["median"]):
        metric = target_metric or _get_primary_metric_column(num_cols)
        if group_col:
            return {"operation": "median", "column": metric, "group_by": group_col, "filters": filters}
        return {"operation": "median", "column": metric, "filters": filters}

    # 13. Average / Mean
    if is_average_requested:
        metric = target_metric or _get_primary_metric_column(num_cols)
        if group_col:
            return {"operation": "average", "column": metric, "group_by": group_col, "filters": filters}
        return {"operation": "average", "column": metric, "filters": filters}

    # 14. Grouped comparison
    if group_col and ("by" in q_lower or "per" in q_lower or "compare" in q_lower or "between" in q_lower or "across" in q_lower):
        metric = target_metric or _get_primary_metric_column(num_cols)
        return {
            "operation": "group_aggregate",
            "group_by": group_col,
            "value_column": metric,
            "column": metric,
            "metric_agg": "mean" if is_average_requested else "sum",
            "filters": filters
        }

    # 15. Total / Sum aggregation
    if target_metric or any(k in q_lower for k in ["total", "sum", "sales", "revenue", "how much"]):
        metric = target_metric or _get_primary_metric_column(num_cols)
        if metric:
            if group_col:
                return {"operation": "sum", "column": metric, "group_by": group_col, "filters": filters}
            return {"operation": "sum", "column": metric, "filters": filters}

    return {"operation": "unknown"}

def _extract_recent_context(history: Optional[List[Any]]) -> Dict[str, Any]:
    if not history:
        return {}
    ctx = {"entity": None, "group_by": None, "metric": None, "operation": None}
    for item in reversed(history):
        if not item:
            continue
        res = getattr(item, 'result', None) if hasattr(item, 'result') else (item.get('result') if isinstance(item, dict) else None)
        if isinstance(res, dict):
            if res.get('entity') and not ctx["entity"]:
                ctx["entity"] = str(res.get('entity'))
            elif res.get('ranking') and not ctx["entity"]:
                ctx["entity"] = list(res.get('ranking').keys())[0]

        interpreted = getattr(item, 'interpreted_as', None) if hasattr(item, 'interpreted_as') else (item.get('interpreted_as') if isinstance(item, dict) else None)
        if interpreted and isinstance(interpreted, str) and interpreted.startswith("{"):
            try:
                p = json.loads(interpreted)
                if p.get('group_by') and not ctx["group_by"]:
                    ctx["group_by"] = p.get('group_by')
                if (p.get('column') or p.get('value_column')) and not ctx["metric"]:
                    ctx["metric"] = p.get('column') or p.get('value_column')
                if p.get('operation') and not ctx["operation"]:
                    ctx["operation"] = p.get('operation')
            except Exception:
                pass
        if ctx["entity"] and ctx["group_by"]:
            break
    return ctx

def process_nl_query(df: pd.DataFrame, filename: str, question: str, gemini_client: Any, history: Optional[List[Any]] = None) -> NLQueryResponse:
    dataset_ctx = _get_dataset_context(df)
    all_cols = dataset_ctx["all_columns"]
    num_cols = dataset_ctx["numeric_metric_columns"]
    op_dict = None
    q_lower = question.lower().strip()

    # Extract conversational context from previous turns
    recent_ctx = _extract_recent_context(history)
    has_pronoun = any(re.search(r'\b' + re.escape(w) + r'\b', q_lower) for w in ["it", "its", "that", "this"])

    # 1. Deterministic conversational reference resolution
    if recent_ctx.get("entity") and recent_ctx.get("group_by") and has_pronoun:
        last_ent = recent_ctx["entity"]
        last_grp = recent_ctx["group_by"]
        
        # Follow-up: "What is its profit margin?" / "profit margin"
        if "profit margin" in q_lower or "margin" in q_lower:
            op_dict = {
                "operation": "profit_margin",
                "group_by": last_grp,
                "column": "Profit",
                "value_column": "Sales",
                "filters": [{"column": last_grp, "value": last_ent}]
            }
        # Follow-up: "Compare it with the second-highest product"
        elif "second" in q_lower or "compare it with" in q_lower or "compare with second" in q_lower:
            metric = recent_ctx.get("metric") or _get_primary_metric_column(num_cols)
            op_dict = {
                "operation": "top_n",
                "group_by": last_grp,
                "column": metric,
                "value_column": metric,
                "limit": 2,
                "sort": "descending"
            }
        # Follow-up: "What is its profit?" / "What is its sales?"
        elif any(k in q_lower for k in ["profit", "sales", "revenue", "price", "cost", "quantity"]):
            target_m = _find_column_by_name_or_synonym(
                next((k for k in ["profit", "sales", "revenue", "price", "cost", "quantity"] if k in q_lower), None),
                all_cols
            ) or recent_ctx.get("metric")
            if target_m:
                op_dict = {
                    "operation": "sum",
                    "column": target_m,
                    "filters": [{"column": last_grp, "value": last_ent}]
                }

    # 2. LLM interpretation with conversation history
    if not op_dict and gemini_client:
        history_context_str = ""
        if history:
            history_context_str = "\nRECENT CONVERSATION HISTORY:\n"
            for h in history[-4:]:
                r = getattr(h, 'role', 'user') if hasattr(h, 'role') else (h.get('role', 'user') if isinstance(h, dict) else 'user')
                c = getattr(h, 'content', '') if hasattr(h, 'content') else (h.get('content', '') if isinstance(h, dict) else '')
                history_context_str += f"{r.upper()}: {c}\n"

        prompt = f"""
You are an expert business analytics SQL/Pandas translator.
Dataset Columns: {dataset_ctx['all_columns']}
Numeric Metrics: {dataset_ctx['numeric_metric_columns']}
Categorical Columns: {dataset_ctx['categorical_columns']}
Sample Categorical Values: {json.dumps(dataset_ctx['categorical_values_sample'])}
Date Columns: {dataset_ctx['date_columns']}
{history_context_str}
USER QUESTION: "{question}"

Translate this question into a strict JSON query specification:
{{
  "operation": "top_n" | "bottom_n" | "sum" | "average" | "min" | "max" | "median" | "count_rows" | "count_unique" | "list_unique" | "group_aggregate" | "time_series" | "correlation" | "business_report" | "describe_dataset" | "profit_margin" | "scatter",
  "metric": string | null (column name to measure, e.g. "Sales", "Profit"),
  "group_by": string | null (dimension column to group by, e.g. "Customer Name", "Category", "Sub-Category", "State", "Ship Mode"),
  "limit": integer | null (e.g. 5 for "top 5"),
  "filters": [{{"column": string, "value": string, "operator": "equals"}}],
  "date_column": string | null,
  "sort": "descending" | "ascending" | null
}}

IMPORTANT GUIDELINES:
1. If the question contains pronouns like "it", "its", "that product", resolve the entity from the conversation history and place it into "filters".
2. "What are the different product categories?" -> operation="list_unique", column="Category".
3. "What are the different shipping modes?" -> operation="list_unique", column="Ship Mode".
4. "How many unique products are there?" -> operation="count_unique", column="Product Name".
5. "Who are the top 5 customers by total sales?" -> operation="top_n", metric="Sales", group_by="Customer Name", limit=5, sort="descending".
6. "Which sub-categories have the lowest total sales?" -> operation="bottom_n", metric="Sales", group_by="Sub-Category", limit=5, sort="ascending".
7. "Compare sales between different shipping modes." -> operation="group_aggregate", metric="Sales", group_by="Ship Mode".
8. "Compare sales and profit" -> operation="scatter", metric="Sales", column="Sales", value_column="Profit".
9. "Generate a business performance report..." -> operation="business_report".
10. Output ONLY raw JSON without code fences.
"""
        try:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            text_resp = response.text.strip()
            if text_resp.startswith("```json"):
                text_resp = text_resp[7:]
            if text_resp.endswith("```"):
                text_resp = text_resp[:-3]
            parsed = json.loads(text_resp.strip())
            if isinstance(parsed, dict) and parsed.get("operation") != "unknown":
                op_dict = {
                    "operation": parsed.get("operation"),
                    "column": parsed.get("metric") or parsed.get("column"),
                    "value_column": parsed.get("metric") or parsed.get("value_column"),
                    "group_by": parsed.get("group_by"),
                    "limit": parsed.get("limit"),
                    "sort": parsed.get("sort"),
                    "date_column": parsed.get("date_column"),
                    "distinct": parsed.get("distinct", False) or (parsed.get("operation") in ["count_unique", "count_distinct"]),
                    "filters": parsed.get("filters", []),
                    "missing_column": parsed.get("missing_column")
                }
        except Exception:
            op_dict = None

    if not op_dict:
        op_dict = _heuristic_parse_question(question, df, dataset_ctx)

    if op_dict.get("operation") == "missing_column":
        missing_c = op_dict.get("missing_column", "requested column")
        return NLQueryResponse(
            question=question,
            interpreted_as=json.dumps(op_dict),
            result=None,
            chart_data=None,
            explanation=f"The uploaded dataset '{filename}' does not contain a '{missing_c}' column."
        )

    if not op_dict or op_dict.get("operation") == "unknown":
        return NLQueryResponse(
            question=question,
            interpreted_as="unknown",
            result=None,
            chart_data=None,
            explanation="I could not map this question to a supported operation. Please specify the metric (e.g. Sales) and dimension (e.g. Category, State, Customer Name, Ship Mode)."
        )

    # Resolve and validate columns against actual dataframe schema
    target_metric = _find_column_by_name_or_synonym(op_dict.get("column"), all_cols) or _find_column_by_name_or_synonym(op_dict.get("value_column"), all_cols)
    value_col = _find_column_by_name_or_synonym(op_dict.get("value_column"), all_cols) or target_metric
    group_col = _find_column_by_name_or_synonym(op_dict.get("group_by"), all_cols)
    date_col = _find_column_by_name_or_synonym(op_dict.get("date_column"), all_cols)

    # Validate filters
    filters_list = []
    for f in op_dict.get("filters", []):
        raw_col = f.get("column")
        f_val = f.get("value")
        matched_f_col = _find_column_by_name_or_synonym(raw_col, all_cols)
        if matched_f_col:
            filters_list.append(FilterItem(column=matched_f_col, value=f_val, operator=f.get("operator", "==")))
        else:
            return NLQueryResponse(
                question=question,
                interpreted_as=json.dumps(op_dict),
                result=None,
                chart_data=None,
                explanation=f"Filter column '{raw_col}' was requested but does not exist in dataset '{filename}'."
            )

    op_type = op_dict.get("operation")

    # If list_unique or count_unique without target column, use group_col
    if op_type in ["list_unique", "count_unique", "count_distinct"]:
        col_target = target_metric or group_col or _find_column_by_name_or_synonym(op_dict.get("column"), all_cols)
        if not col_target:
            return NLQueryResponse(
                question=question,
                interpreted_as=json.dumps(op_dict),
                result=None,
                explanation="Please specify which field to inspect (e.g. Category, Ship Mode, Customer Name)."
            )
        target_metric = col_target

    req = AnalyticsRequest(
        filename=filename,
        operation=op_type,
        column=target_metric,
        group_by=group_col,
        filters=filters_list,
        date_column=date_col,
        value_column=value_col,
        metric_agg=op_dict.get("metric_agg", "sum"),
        ascending=(op_dict.get("sort") == "ascending") or op_dict.get("ascending", False),
        distinct=op_dict.get("distinct", False) or (op_type in ["count_unique", "count_distinct"]),
        limit=op_dict.get("limit"),
        sort=op_dict.get("sort")
    )

    analytics_result = execute_analytics(df, req)

    explanation = analytics_result.explanation
    if not explanation:
        filter_str = ""
        if filters_list:
            filter_str = " with " + ", ".join([f"{f.column} = '{f.value}'" for f in filters_list])
        
        if req.operation in ["top_n", "top_performer"]:
            explanation = f"Calculated top {req.limit or 1} {req.group_by} by {req.column}{filter_str}."
        elif req.operation == "bottom_n":
            explanation = f"Calculated lowest {req.limit or 5} {req.group_by} by {req.column}{filter_str}."
        elif req.operation == "sum":
            explanation = f"Calculated total {req.column}{filter_str}."
        elif req.operation in ["group_by", "group_aggregate", "compare"]:
            explanation = f"Calculated {req.column} grouped by {req.group_by}{filter_str}."
        elif req.operation in ["count_unique", "count_distinct"]:
            explanation = f"Counted unique {req.column}{filter_str}."
        elif req.operation == "list_unique":
            explanation = f"Retrieved distinct values for {req.column}{filter_str}."
        else:
            explanation = f"Calculated {req.operation} on {req.column or 'data'}{filter_str}."

    if gemini_client and analytics_result.result is not None and op_type not in ["business_report"]:
        try:
            insight_prompt = f"""
Given this exact calculation result for the user's question, provide a concise 1-2 sentence business statement.
Question: "{question}"
Result: {analytics_result.result}
Explanation: {explanation}
Do not invent any new numbers; only explain the calculated result accurately.
"""
            insight_resp = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=insight_prompt)
            if insight_resp and insight_resp.text:
                explanation = insight_resp.text.strip()
        except Exception:
            pass

    return NLQueryResponse(
        question=question,
        interpreted_as=json.dumps(op_dict),
        result=analytics_result.result,
        chart_data=analytics_result.chart_data,
        explanation=explanation,
        insights=None
    )
