import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import io
from typing import List
from google import genai

from app.models import (
    UploadResponse, AnalyticsRequest, AnalyticsResponse,
    NLQueryResponse, ChartRequest, InsightsRequest, InsightsResponse, ExportRequest,
    ContextualNLQueryRequest, AnomalyReport, DataQualityReport, CleanDataRequest,
    DatasetListResponse, DatasetSummary, SetActiveDatasetRequest
)
from app.data_manager import (
    save_upload, get_dataframe, get_raw_dataframe, get_sample_data,
    get_active_dataset, set_active_dataset, list_datasets,
    update_dataframe, generate_suggested_questions
)
from app.analytics import execute_analytics
from app.nl_processor import process_nl_query
from app.charts import generate_chart
from app.insights import generate_insights
from app.anomaly_detector import detect_anomalies
from app.data_quality import assess_data_quality, clean_dataframe
from app.pdf_exporter import generate_pdf_report

router = APIRouter()

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key and api_key != "your_api_key_here":
        try:
            return genai.Client(api_key=api_key)
        except Exception:
            return None
    return None

# ==================== DATASET MANAGEMENT ====================

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    return await save_upload(file)

@router.get("/sample-data", response_model=UploadResponse)
def load_sample_data():
    return get_sample_data()

@router.get("/datasets", response_model=DatasetListResponse)
def get_datasets():
    datasets = list_datasets()
    active = get_active_dataset()
    return DatasetListResponse(
        datasets=[DatasetSummary(**d) for d in datasets],
        active_dataset=active
    )

@router.post("/datasets/active")
def switch_active_dataset(request: SetActiveDatasetRequest):
    set_active_dataset(request.filename)
    df = get_dataframe(request.filename)
    return {
        "status": "success",
        "active_dataset": request.filename,
        "row_count": len(df),
        "column_count": len(df.columns),
        "suggested_questions": generate_suggested_questions(df, request.filename)
    }

@router.get("/data/{filename}/preview")
def get_data_preview(filename: str):
    df = get_dataframe(filename)
    preview = df.head(50).to_dict(orient="records")
    for row in preview:
        for k, v in row.items():
            if pd.isna(v):
                row[k] = None
    return {"preview": preview}

# ==================== ANALYTICS & NLP ====================

@router.post("/analytics", response_model=AnalyticsResponse)
def run_analytics(request: AnalyticsRequest):
    df = get_dataframe(request.filename)
    return execute_analytics(df, request)

@router.post("/nl-query", response_model=NLQueryResponse)
def nl_query(request: ContextualNLQueryRequest):
    client = get_gemini_client()
    df = get_dataframe(request.filename)
    return process_nl_query(df, request.filename, request.question, client, request.history)

@router.post("/chart")
def create_chart(request: ChartRequest):
    df = get_dataframe(request.filename)
    return generate_chart(
        df=df,
        chart_type=request.chart_type,
        x_column=request.x_column,
        y_column=request.y_column,
        group_by=request.group_by,
        title=request.title
    )

# ==================== INSIGHTS & ANOMALIES ====================

@router.post("/insights")
def get_insights(request: InsightsRequest):
    client = get_gemini_client()
    df = get_dataframe(request.filename)
    return generate_insights(df, client)

@router.get("/anomalies/{filename}", response_model=AnomalyReport)
@router.post("/anomalies/{filename}", response_model=AnomalyReport)
def get_anomalies(filename: str):
    df = get_dataframe(filename)
    return detect_anomalies(df)

# ==================== DATA QUALITY INSPECTOR ====================

@router.get("/data-quality/{filename}", response_model=DataQualityReport)
def get_data_quality(filename: str):
    # Use raw dataframe so missing values and duplicates are preserved accurately
    raw_df = get_raw_dataframe(filename)
    return assess_data_quality(raw_df)

@router.post("/data-quality/clean")
def clean_dataset(request: CleanDataRequest):
    raw_df = get_raw_dataframe(request.filename)
    cleaned_df, summary = clean_dataframe(raw_df, request)
    update_dataframe(request.filename, cleaned_df)
    new_report = assess_data_quality(cleaned_df)
    return {
        "status": "success",
        "summary": summary,
        "quality_report": new_report,
        "new_row_count": len(cleaned_df),
        "new_column_count": len(cleaned_df.columns)
    }

@router.get("/data-quality/{filename}/problematic")
def get_problematic_records(filename: str, issue_type: str = "missing", column: str = None, limit: int = 25):
    raw_df = get_raw_dataframe(filename)
    results = []
    
    if issue_type == "duplicates":
        dups = raw_df[raw_df.duplicated(keep=False)]
        records = dups.head(limit).to_dict(orient="records")
        results = records
    elif issue_type == "outliers" and column and column in raw_df.columns and pd.api.types.is_numeric_dtype(raw_df[column]):
        col_s = raw_df[column].dropna()
        q25 = col_s.quantile(0.25)
        q75 = col_s.quantile(0.75)
        iqr = q75 - q25
        if iqr > 0:
            outlier_mask = (raw_df[column] < (q25 - 1.5 * iqr)) | (raw_df[column] > (q75 + 1.5 * iqr))
            records = raw_df[outlier_mask].head(limit).to_dict(orient="records")
            results = records
    else: # missing values
        if column and column in raw_df.columns:
            missing_df = raw_df[raw_df[column].isna()]
        else:
            missing_df = raw_df[raw_df.isna().any(axis=1)]
        records = missing_df.head(limit).to_dict(orient="records")
        results = records
        
    # sanitize NaN
    for r in results:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None
                
    return {
        "filename": filename,
        "issue_type": issue_type,
        "column": column,
        "records": results,
        "total_returned": len(results)
    }

# In-memory storage for report history
_report_history: List[Dict[str, Any]] = []

@router.get("/reports/history")
def get_report_history():
    return {"reports": _report_history}

@router.get("/export-report/{filename}")
@router.post("/export-report/{filename}")
def export_pdf_report(filename: str):
    df = get_dataframe(filename)
    raw_df = get_raw_dataframe(filename)
    
    # Gather comprehensive metrics for PDF
    client = get_gemini_client()
    insights_res = generate_insights(df, client)
    quality_res = assess_data_quality(raw_df)
    anomalies_res = detect_anomalies(df)
    
    pdf_buffer = generate_pdf_report(
        filename=filename,
        overview=insights_res.overview.dict() if insights_res.overview else {"row_count": len(df), "column_count": len(df.columns)},
        kpis=[k.dict() for k in insights_res.kpis] if insights_res.kpis else [],
        performers=[p.dict() for p in insights_res.performers] if insights_res.performers else [],
        anomalies=[a.dict() for a in anomalies_res.anomalies],
        recommendations=[r.dict() for r in insights_res.recommendations] if insights_res.recommendations else [],
        quality=quality_res.dict()
    )
    
    import datetime
    report_entry = {
        "id": f"rep-{int(datetime.datetime.now().timestamp())}",
        "filename": filename,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kpi_count": len(insights_res.kpis or []),
        "anomalies_detected": len(anomalies_res.anomalies or []),
        "quality_score": quality_res.score,
        "status": "Generated"
    }
    _report_history.insert(0, report_entry)
    
    response = StreamingResponse(pdf_buffer, media_type="application/pdf")
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}_Executive_Report.pdf"'
    return response

@router.get("/export/{filename}")
def export_data(filename: str):
    df = get_dataframe(filename)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}_export.csv"
    return response

@router.post("/export")
def export_filtered_data(request: ExportRequest):
    df = get_dataframe(request.filename)
    if request.filter_column and request.filter_value:
        df = df[df[request.filter_column].astype(str) == request.filter_value]
    if request.columns:
        df = df[request.columns]
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={request.filename}_filtered.csv"
    return response

@router.get("/data/{filename}/columns")
def get_columns(filename: str):
    df = get_dataframe(filename)
    return [{"name": col, "dtype": str(dtype)} for col, dtype in zip(df.columns, df.dtypes)]

@router.get("/data/{filename}/summary")
def get_summary(filename: str):
    df = get_dataframe(filename)
    summary = df.describe(include='all').to_dict()
    for col in summary:
        for stat in summary[col]:
            if pd.isna(summary[col][stat]):
                summary[col][stat] = None
    return summary

@router.get("/data/{filename}/suggestions")
def get_dataset_suggestions(filename: str):
    df = get_dataframe(filename)
    actual_filename = filename if filename != "active" else (get_active_dataset() or "active_dataset.csv")
    suggestions = generate_suggested_questions(df, actual_filename)
    return {"filename": actual_filename, "suggested_questions": suggestions}
