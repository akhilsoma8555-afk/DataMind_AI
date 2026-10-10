export interface ColumnInfo {
  name: string;
  dtype: string;
}

export interface UploadResponse {
  filename: string;
  row_count: number;
  column_count: number;
  columns: ColumnInfo[];
  preview: Record<string, any>[];
  missing_values: Record<string, number>;
  suggested_questions?: string[];
}

export interface FilterItem {
  column: string;
  value: any;
  operator?: string;
}

export interface AnalyticsRequest {
  filename: string;
  operation: string;
  column?: string;
  group_by?: string;
  filter_column?: string;
  filter_value?: string;
  filters?: FilterItem[];
  date_column?: string;
  value_column?: string;
  interval?: string;
  distinct?: boolean;
  ascending?: boolean;
  limit?: number;
  sort?: string;
}

export interface AnalyticsResponse {
  operation: string;
  result: any;
  chart_data?: {
    type: string;
    x: any[];
    y: any[];
    title: string;
  };
  explanation?: string;
}

export interface NLQueryResponse {
  question: string;
  interpreted_as: string;
  result: any;
  chart_data?: {
    type: string;
    x: any[];
    y: any[];
    title: string;
  };
  explanation?: string;
  insights?: string;
}

export interface InsightItem {
  category: string;
  finding: string;
  recommendation: string;
  priority?: 'Critical' | 'High' | 'Medium' | 'Low';
  evidence?: string;
  next_steps?: string[];
  reviewed?: boolean;
}

export interface ReportRecord {
  id: string;
  filename: string;
  created_at: string;
  kpi_count: number;
  anomalies_detected: number;
  quality_score: number;
  status: string;
}

export interface KPIItem {
  label: string;
  value: string;
  detail?: string;
}

export interface PerformerItem {
  title: string;
  entity: string;
  metric: string;
  value: string;
}

export interface DatasetOverview {
  row_count: number;
  column_count: number;
  numeric_columns: string[];
  categorical_columns: string[];
  date_columns: string[];
  missing_values: Record<string, number>;
  date_range?: string;
}

export interface AnomalyItem {
  id: string;
  metric: string;
  entity_column?: string;
  entity_value?: string;
  observed_value: any;
  expected_range: string;
  severity: 'High' | 'Medium' | 'Low';
  method: string;
  explanation: string;
  affected_rows_count: number;
  sample_records?: Record<string, any>[];
}

export interface AnomalyReport {
  anomalies: AnomalyItem[];
  columns_analyzed: string[];
  total_anomalies: number;
}

export interface QualityIssue {
  column?: string;
  issue_type: string;
  severity: 'High' | 'Medium' | 'Low';
  description: string;
  affected_count: number;
}

export interface ColumnQuality {
  name: string;
  dtype: string;
  total_count: number;
  missing_count: number;
  missing_pct: number;
  unique_count: number;
  is_empty: boolean;
  outliers_count: number;
}

export interface DataQualityReport {
  score: number;
  grade: 'Excellent' | 'Good' | 'Fair' | 'Poor';
  row_count: number;
  column_count: number;
  total_missing: number;
  missing_pct: number;
  duplicate_rows: number;
  duplicate_pct: number;
  empty_columns: string[];
  potential_outliers: number;
  columns: ColumnQuality[];
  issues: QualityIssue[];
  score_breakdown: Record<string, any>;
}

export interface CleanDataRequest {
  filename: string;
  drop_duplicates?: boolean;
  drop_empty_columns?: boolean;
  fill_missing?: boolean;
  fill_strategy?: 'smart' | 'zero' | 'drop';
}

export interface DatasetSummary {
  filename: string;
  row_count: number;
  column_count: number;
  is_active: boolean;
  upload_time?: string;
}

export interface DatasetListResponse {
  datasets: DatasetSummary[];
  active_dataset?: string;
}

export interface InsightsResponse {
  overview?: DatasetOverview;
  kpis?: KPIItem[];
  performers?: PerformerItem[];
  charts?: any[];
  recommendations?: InsightItem[];
  anomalies?: AnomalyItem[];
  insights: InsightItem[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  interpreted_as?: string;
  result?: any;
  chart_data?: any;
  viewMode?: 'chart' | 'table';
  timestamp: Date;
}

