import pandas as pd
import json
from typing import List, Any, Dict
from app.models import (
    ComprehensiveInsightsResponse,
    DatasetOverview,
    KPIItem,
    PerformerItem,
    InsightItem
)

from app.anomaly_detector import detect_anomalies

def generate_insights(df: pd.DataFrame, gemini_client: Any) -> ComprehensiveInsightsResponse:
    num_cols = list(df.select_dtypes(include=['number']).columns)
    cat_cols = list(df.select_dtypes(include=['object', 'category', 'string']).columns)
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'month' in c.lower() or 'year' in c.lower()]
    
    # Run Automated Anomaly Detection
    anom_report = detect_anomalies(df)
    anomalies = anom_report.anomalies
    
    # 1. Dataset Overview
    date_range_str = None
    if date_cols:
        d_col = date_cols[0]
        try:
            temp_dates = pd.to_datetime(df[d_col], errors='coerce').dropna()
            if not temp_dates.empty:
                date_range_str = f"{temp_dates.min().strftime('%Y-%m-%d')} to {temp_dates.max().strftime('%Y-%m-%d')}"
        except Exception:
            pass

    overview = DatasetOverview(
        row_count=len(df),
        column_count=len(df.columns),
        numeric_columns=num_cols,
        categorical_columns=cat_cols,
        date_columns=date_cols,
        missing_values=df.isna().sum().to_dict(),
        date_range=date_range_str
    )

    # 2. Key Performance Indicators
    kpis: List[KPIItem] = []
    
    priority_metrics = ['sale', 'revenue', 'price', 'profit', 'amount', 'total', 'cost', 'mileage', 'value']
    primary_metric = None
    for kw in priority_metrics:
        for c in num_cols:
            if kw in c.lower():
                primary_metric = c
                break
        if primary_metric:
            break
    if not primary_metric:
        non_temp = [c for c in num_cols if not any(t in c.lower() for t in ['year', 'id', 'zip', 'code'])]
        primary_metric = non_temp[0] if non_temp else (num_cols[0] if num_cols else None)

    if primary_metric:
        tot_val = float(df[primary_metric].sum())
        avg_val = float(df[primary_metric].mean())
        prefix = "$" if any(k in primary_metric.lower() for k in ["sale", "revenue", "price", "profit", "cost", "amount"]) else ""
        kpis.append(KPIItem(
            label=f"Total {primary_metric}",
            value=f"{prefix}{tot_val:,.2f}" if prefix else f"{tot_val:,.0f}",
            detail=f"Sum of {primary_metric}"
        ))
        kpis.append(KPIItem(
            label=f"Average {primary_metric}",
            value=f"{prefix}{avg_val:,.2f}" if prefix else f"{avg_val:,.1f}",
            detail=f"Mean per record"
        ))

    # Secondary metric (e.g. Profit if primary is Sales, or Mileage if primary is Price)
    sec_metric = None
    for kw in ['profit', 'mileage', 'quantity', 'discount']:
        for c in num_cols:
            if kw in c.lower() and c != primary_metric:
                sec_metric = c
                break
        if sec_metric:
            break

    if sec_metric:
        tot_sec = float(df[sec_metric].sum())
        avg_sec = float(df[sec_metric].mean())
        sec_prefix = "$" if any(k in sec_metric.lower() for k in ["profit", "cost"]) else ""
        kpis.append(KPIItem(
            label=f"Average {sec_metric}",
            value=f"{sec_prefix}{avg_sec:,.2f}" if sec_prefix else f"{avg_sec:,.0f}",
            detail=f"Mean {sec_metric}"
        ))

    # Entity counts
    car_make_col = next((c for c in cat_cols if 'make' in c.lower() or 'brand' in c.lower() or 'manufacturer' in c.lower()), None)
    cust_col = next((c for c in cat_cols if 'customer' in c.lower()), None)
    order_col = next((c for c in df.columns if 'order' in c.lower() and 'id' in c.lower()), None)

    if car_make_col:
        kpis.append(KPIItem(label="Total Vehicles", value=f"{len(df):,}", detail="Active records"))
        kpis.append(KPIItem(label="Car Brands", value=f"{df[car_make_col].nunique():,}", detail=f"Distinct {car_make_col}"))
    elif cust_col:
        kpis.append(KPIItem(label="Unique Customers", value=f"{df[cust_col].nunique():,}", detail=f"Distinct {cust_col}"))
        if order_col:
            kpis.append(KPIItem(label="Total Orders", value=f"{df[order_col].nunique():,}", detail=f"Distinct {order_col}"))
    else:
        kpis.append(KPIItem(label="Total Records", value=f"{len(df):,}", detail="Row count"))

    # 3. Top and Bottom Performers + Charts
    performers: List[PerformerItem] = []
    charts: List[Dict[str, Any]] = []

    if primary_metric and cat_cols:
        # Prioritize key dimensions
        dim_candidates = []
        for kw in ['category', 'make', 'brand', 'model', 'fuel', 'transmission', 'region', 'state', 'segment', 'ship']:
            for c in cat_cols:
                if kw in c.lower() and c not in dim_candidates:
                    dim_candidates.append(c)
        
        # Fall back to first categorical columns if none matched
        for c in cat_cols:
            if c not in dim_candidates and len(dim_candidates) < 3:
                dim_candidates.append(c)

        for dim in dim_candidates[:2]:
            is_price_metric = any(k in primary_metric.lower() for k in ["price", "mileage", "rate", "cost"])
            if is_price_metric:
                agg = df.groupby(dim)[primary_metric].mean().sort_values(ascending=False)
                metric_label = f"Avg {primary_metric}"
            else:
                agg = df.groupby(dim)[primary_metric].sum().sort_values(ascending=False)
                metric_label = f"Total {primary_metric}"

            if not agg.empty:
                top_entity = agg.index[0]
                top_val = float(agg.iloc[0])
                low_entity = agg.index[-1]
                low_val = float(agg.iloc[-1])
                prefix = "$" if any(k in primary_metric.lower() for k in ["sale", "revenue", "price", "profit", "cost"]) else ""

                performers.append(PerformerItem(
                    title=f"Top {dim}",
                    entity=str(top_entity),
                    metric=metric_label,
                    value=f"{prefix}{top_val:,.2f}"
                ))
                if len(agg) > 1:
                    performers.append(PerformerItem(
                        title=f"Lowest {dim}",
                        entity=str(low_entity),
                        metric=metric_label,
                        value=f"{prefix}{low_val:,.2f}"
                    ))

                # Chart
                display_agg = agg.head(10)
                charts.append({
                    "type": "bar",
                    "title": f"{metric_label} by {dim}",
                    "x": [str(x) for x in display_agg.index],
                    "y": [float(v) for v in display_agg.values]
                })

    # 4. Trend chart
    if date_cols and primary_metric:
        d_col = date_cols[0]
        try:
            temp_df = df.copy()
            if pd.api.types.is_numeric_dtype(temp_df[d_col]) and temp_df[d_col].dropna().min() >= 1900 and temp_df[d_col].dropna().max() <= 2100:
                is_price = any(k in primary_metric.lower() for k in ["price", "mileage"])
                grouped_trend = temp_df.groupby(d_col)[primary_metric].mean().sort_index() if is_price else temp_df.groupby(d_col)[primary_metric].sum().sort_index()
                charts.append({
                    "type": "line",
                    "title": f"Yearly {'Average' if is_price else 'Total'} {primary_metric} Trend",
                    "x": [str(x) for x in grouped_trend.index],
                    "y": [float(v) for v in grouped_trend.values]
                })
            else:
                temp_df[d_col] = pd.to_datetime(temp_df[d_col], errors='coerce')
                temp_df = temp_df.dropna(subset=[d_col])
                temp_df['Period'] = temp_df[d_col].dt.to_period('M')
                monthly_trend = temp_df.groupby('Period')[primary_metric].sum().sort_index()
                charts.append({
                    "type": "line",
                    "title": f"Monthly {primary_metric} Trend",
                    "x": [str(p) for p in monthly_trend.index],
                    "y": [float(v) for v in monthly_trend.values]
                })
        except Exception:
            pass

    # 5. Data-Driven Evidence-Based Recommendations
    recommendations: List[InsightItem] = []
    
    # A. Performer-driven recommendations
    if performers:
        for p in performers:
            if "Top" in p.title or "Leading" in p.title:
                recommendations.append(InsightItem(
                    category="recommendation",
                    finding=f"'{p.entity}' is the leading segment with {p.value} in {p.metric}.",
                    recommendation=f"Prioritize supply-chain and inventory allocation for '{p.entity}' to maximize revenue momentum.",
                    priority="High",
                    evidence=f"Verified aggregate metric: {p.metric} = {p.value} computed across all dataset rows.",
                    next_steps=[
                        f"Audit supply chain lead times specifically for {p.entity}.",
                        f"Formulate dedicated promotional bundles focused around {p.entity}.",
                        f"Protect healthy profit margins on {p.entity} against raw material inflation."
                    ]
                ))
            elif "Lowest" in p.title:
                recommendations.append(InsightItem(
                    category="anomaly",
                    finding=f"'{p.entity}' represents the lowest tier in {p.metric} ({p.value}).",
                    recommendation=f"Conduct a pricing and conversion review for '{p.entity}' to identify root causes of low demand.",
                    priority="Medium",
                    evidence=f"Underperforming bottom quartile ranking in {p.metric} at {p.value}.",
                    next_steps=[
                        f"Review unit economics and gross margin viability for {p.entity}.",
                        f"Survey regional customers regarding satisfaction with {p.entity}.",
                        f"Test localized promotional discounts to stimulate trial."
                    ]
                ))

    # B. Profit margin / loss detection
    profit_col = next((c for c in num_cols if 'profit' in c.lower()), None)
    sales_col = next((c for c in num_cols if 'sale' in c.lower() or 'revenue' in c.lower()), None)
    if profit_col and sales_col and cat_cols:
        group_dim = cat_cols[0]
        cat_margins = df.groupby(group_dim)[[sales_col, profit_col]].sum()
        negative_margin_entities = cat_margins[cat_margins[profit_col] < 0]
        if not negative_margin_entities.empty:
            neg_names = list(negative_margin_entities.index)
            neg_losses = negative_margin_entities[profit_col].sum()
            recommendations.append(InsightItem(
                category="anomaly",
                finding=f"Loss-making segments detected: {', '.join([str(n) for n in neg_names[:3]])} generated total cumulative net loss of ${abs(neg_losses):,.2f}.",
                recommendation="Review product bundling and discount policies on loss-making categories to halt margin erosion.",
                priority="Critical",
                evidence=f"Cumulative net loss of ${abs(neg_losses):,.2f} recorded in {profit_col} across {len(negative_margin_entities)} entities.",
                next_steps=[
                    f"Freeze maximum discount thresholds on {', '.join([str(n) for n in neg_names[:2]])}.",
                    "Analyze fulfillment and return rate costs associated with these lines.",
                    "Reprice baseline SKUs to enforce at least a 12% gross floor margin."
                ]
            ))

    # C. Pareto Concentration Check
    if primary_metric and cat_cols:
        prim_cat = cat_cols[0]
        entity_sums = df.groupby(prim_cat)[primary_metric].sum().sort_values(ascending=False)
        if len(entity_sums) >= 5:
            top_20_count = max(1, int(len(entity_sums) * 0.2))
            top_20_share = entity_sums.head(top_20_count).sum() / entity_sums.sum()
            if top_20_share > 0.60:
                recommendations.append(InsightItem(
                    category="recommendation",
                    finding=f"High Pareto concentration: Top {top_20_count} {prim_cat} entities generate {top_20_share*100:.1f}% of total {primary_metric}.",
                    recommendation="Diversify revenue streams and establish defensive accounts to mitigate customer/product concentration risk.",
                    priority="High",
                    evidence=f"Calculated Gini / Pareto ratio: top 20% entities control {top_20_share*100:.1f}% of aggregate {primary_metric}.",
                    next_steps=[
                        "Implement key-account retention agreements with top purchasers.",
                        f"Incentivize commercial teams to cross-sell into mid-tier {prim_cat} accounts.",
                        "Establish SLA guarantees for the primary revenue drivers."
                    ]
                ))

    # D. Missing values data quality finding
    if overview.missing_values and any(v > 0 for v in overview.missing_values.values()):
        cols_with_null = [f"{k} ({v} nulls)" for k, v in overview.missing_values.items() if v > 0]
        recommendations.append(InsightItem(
            category="trend",
            finding=f"Missing records detected in: {', '.join(cols_with_null[:3])}.",
            recommendation="Use Data Quality Inspector to clean or impute missing cells and eliminate downstream analytics bias.",
            priority="Medium",
            evidence=f"Total null count across affected columns: {sum(v for v in overview.missing_values.values() if v > 0)} empty cells.",
            next_steps=[
                "Navigate to Data Quality Center to inspect affected row samples.",
                "Execute 1-click smart imputation (median for numeric, mode for categorical).",
                "Validate downstream reporting outputs post-cleaning."
            ]
        ))

    # If Gemini is available, refine recommendations with AI narrative
    if gemini_client and recommendations:
        try:
            prompt = f"""
You are a senior business intelligence consultant. Given these verified empirical facts:
KPIs: {[k.dict() for k in kpis]}
Performers: {[p.dict() for p in performers]}
Anomalies Detected: {len(anomalies)}

Synthesize 3-4 strategic business recommendations in JSON format:
[
  {{"category": "recommendation"|"anomaly"|"trend", "finding": "exact empirical fact", "recommendation": "specific actionable recommendation"}}
]
Return ONLY raw JSON array.
"""
            resp = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            text_r = resp.text.strip()
            if text_r.startswith("```json"):
                text_r = text_r[7:]
            if text_r.endswith("```"):
                text_r = text_r[:-3]
            ai_items = json.loads(text_r.strip())
            recommendations = [InsightItem(**item) for item in ai_items]
        except Exception:
            pass

    return ComprehensiveInsightsResponse(
        overview=overview,
        kpis=kpis,
        performers=performers,
        charts=charts,
        recommendations=recommendations,
        anomalies=anomalies,
        insights=recommendations
    )

