from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ColumnInfo(BaseModel):
    name: str
    dtype: str

class UploadResponse(BaseModel):
    filename: str
    row_count: int
    column_count: int
    columns: List[ColumnInfo]
    preview: List[Dict[str, Any]]
    missing_values: Dict[str, int]
    suggested_questions: Optional[List[str]] = None

class FilterItem(BaseModel):
    column: str
    value: Any
    operator: Optional[str] = "=="  # "==", "!=", "in", "contains", "equals"

class AnalyticsRequest(BaseModel):
    filename: str
    operation: str # sum/average/mean/min/max/median/count/count_rows/count_unique/count_distinct/list_unique/group_by/group_aggregate/top_n/bottom_n/filter/trend/time_series/describe_dataset/correlation/business_report
    column: Optional[str] = None
    group_by: Optional[str] = None
    filter_column: Optional[str] = None
    filter_value: Optional[str] = None
    filters: Optional[List[FilterItem]] = None
    date_column: Optional[str] = None
    value_column: Optional[str] = None
    interval: Optional[str] = "month"
    distinct: Optional[bool] = False
    ascending: Optional[bool] = False
    limit: Optional[int] = None
    sort: Optional[str] = None
    metric_agg: Optional[str] = "sum"

class AnalyticsResponse(BaseModel):
    operation: str
    result: Any
    chart_data: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None

class NLQueryRequest(BaseModel):
    filename: str
    question: str

class NLQueryResponse(BaseModel):
    question: str
    interpreted_as: str
    result: Any
    chart_data: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    insights: Optional[str] = None

class ChartRequest(BaseModel):
    filename: str
    chart_type: str # bar/line/pie
    x_column: str
    y_column: str
    group_by: Optional[str] = None
    title: Optional[str] = None

class InsightItem(BaseModel):
    category: str # 'trend'|'anomaly'|'recommendation'|'kpi'|'performer'
    finding: str
    recommendation: str
    priority: Optional[str] = "High" # "Critical" | "High" | "Medium" | "Low"
    evidence: Optional[str] = None
    next_steps: Optional[List[str]] = None

class KPIItem(BaseModel):
    label: str
    value: str
    detail: Optional[str] = None

class PerformerItem(BaseModel):
    title: str
    entity: str
    metric: str
    value: str

class DatasetOverview(BaseModel):
    row_count: int
    column_count: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    date_columns: List[str]
    missing_values: Dict[str, int]
    date_range: Optional[str] = None

class AnomalyItem(BaseModel):
    id: str
    metric: str
    entity_column: Optional[str] = None
    entity_value: Optional[str] = None
    observed_value: Any
    expected_range: str
    severity: str # "High" | "Medium" | "Low"
    method: str   # "Isolation Forest" | "IQR Outlier Analysis" | "Z-Score Analysis"
    explanation: str
    affected_rows_count: int
    sample_records: Optional[List[Dict[str, Any]]] = None

class AnomalyReport(BaseModel):
    anomalies: List[AnomalyItem]
    columns_analyzed: List[str]
    total_anomalies: int

class ComprehensiveInsightsResponse(BaseModel):
    overview: DatasetOverview
    kpis: List[KPIItem]
    performers: List[PerformerItem]
    charts: List[Dict[str, Any]]
    recommendations: List[InsightItem]
    anomalies: Optional[List[AnomalyItem]] = None
    insights: List[InsightItem] # For backwards compatibility with existing UI

class InsightsRequest(BaseModel):
    filename: str

class InsightsResponse(BaseModel):
    overview: Optional[DatasetOverview] = None
    kpis: Optional[List[KPIItem]] = None
    performers: Optional[List[PerformerItem]] = None
    charts: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[InsightItem]] = None
    anomalies: Optional[List[AnomalyItem]] = None
    insights: List[InsightItem]

class ExportRequest(BaseModel):
    filename: str
    filter_column: Optional[str] = None
    filter_value: Optional[str] = None
    columns: Optional[List[str]] = None

class ChatHistoryEntry(BaseModel):
    role: str # "user" | "assistant"
    content: str
    interpreted_as: Optional[str] = None
    result: Optional[Any] = None

class ContextualNLQueryRequest(BaseModel):
    filename: str
    question: str
    history: Optional[List[ChatHistoryEntry]] = None

class QualityIssue(BaseModel):
    column: Optional[str] = None
    issue_type: str # "missing_values" | "duplicates" | "outliers" | "empty_column" | "mixed_types"
    severity: str   # "High" | "Medium" | "Low"
    description: str
    affected_count: int

class ColumnQuality(BaseModel):
    name: str
    dtype: str
    total_count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    is_empty: bool
    outliers_count: int

class DataQualityReport(BaseModel):
    score: float # 0 to 100
    grade: str   # "Excellent" | "Good" | "Fair" | "Poor"
    row_count: int
    column_count: int
    total_missing: int
    missing_pct: float
    duplicate_rows: int
    duplicate_pct: float
    empty_columns: List[str]
    potential_outliers: int
    columns: List[ColumnQuality]
    issues: List[QualityIssue]
    score_breakdown: Dict[str, Any]

class CleanDataRequest(BaseModel):
    filename: str
    drop_duplicates: Optional[bool] = False
    drop_empty_columns: Optional[bool] = False
    fill_missing: Optional[bool] = False
    fill_strategy: Optional[str] = "smart" # "smart" (median for num, mode for cat) | "zero" | "drop"

class DatasetSummary(BaseModel):
    filename: str
    row_count: int
    column_count: int
    is_active: bool
    upload_time: Optional[str] = None

class DatasetListResponse(BaseModel):
    datasets: List[DatasetSummary]
    active_dataset: Optional[str] = None

class SetActiveDatasetRequest(BaseModel):
    filename: str

