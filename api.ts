import axios from 'axios';
import type {
  UploadResponse, AnalyticsRequest, AnalyticsResponse, NLQueryResponse,
  InsightsResponse, ColumnInfo, DatasetListResponse, DataQualityReport,
  CleanDataRequest, AnomalyReport
} from './types';

const backendHost = typeof window !== 'undefined' && window.location.hostname === '127.0.0.1' 
  ? 'http://127.0.0.1:8000/api' 
  : 'http://localhost:8000/api';

const api = axios.create({
  baseURL: backendHost,
});

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post<UploadResponse>('/upload', formData);
  return data;
}

export async function loadSampleData(): Promise<UploadResponse> {
  const { data } = await api.get<UploadResponse>('/sample-data');
  return data;
}

export async function getDatasets(): Promise<DatasetListResponse> {
  const { data } = await api.get<DatasetListResponse>('/datasets');
  return data;
}

export async function switchActiveDataset(filename: string): Promise<any> {
  const { data } = await api.post('/datasets/active', { filename });
  return data;
}

export async function getPreview(filename: string): Promise<any[]> {
  const { data } = await api.get(`/data/${filename}/preview`);
  return data.preview;
}

export async function runAnalytics(params: AnalyticsRequest): Promise<AnalyticsResponse> {
  const { data } = await api.post<AnalyticsResponse>('/analytics', params);
  return data;
}

export async function nlQuery(filename: string, question: string, history?: any[]): Promise<NLQueryResponse> {
  const { data } = await api.post<NLQueryResponse>('/nl-query', { filename, question, history });
  return data;
}

export async function getChart(params: { filename: string; chart_type: string; x_column: string; y_column: string; group_by?: string; title?: string }): Promise<any> {
  const { data } = await api.post('/chart', params);
  return data;
}

export async function getInsights(filename: string): Promise<InsightsResponse> {
  const { data } = await api.post<InsightsResponse>('/insights', { filename });
  return data;
}

export async function getAnomalies(filename: string): Promise<AnomalyReport> {
  const { data } = await api.get<AnomalyReport>(`/anomalies/${filename}`);
  return data;
}

export async function getDataQuality(filename: string): Promise<DataQualityReport> {
  const { data } = await api.get<DataQualityReport>(`/data-quality/${filename}`);
  return data;
}

export async function cleanDataset(params: CleanDataRequest): Promise<any> {
  const { data } = await api.post('/data-quality/clean', params);
  return data;
}

export async function exportPdfReport(filename: string): Promise<void> {
  const response = await api.get(`/export-report/${filename}`, { responseType: 'blob' });
  const url = window.URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', `${filename}_Executive_Report.pdf`);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

export async function exportData(filename: string): Promise<void> {
  const response = await api.get(`/export/${filename}`, { responseType: 'blob' });
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', `${filename}_export.csv`);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

export async function getProblematicRecords(filename: string, issueType: string = 'missing', column?: string, limit: number = 25): Promise<any> {
  const params: any = { issue_type: issueType, limit };
  if (column) params.column = column;
  const { data } = await api.get(`/data-quality/${filename}/problematic`, { params });
  return data;
}

export async function getReportHistory(): Promise<any> {
  const { data } = await api.get('/reports/history');
  return data;
}

export async function getColumns(filename: string): Promise<ColumnInfo[]> {
  const { data } = await api.get(`/data/${filename}/columns`);
  return data;
}

export async function getSummary(filename: string): Promise<any> {
  const { data } = await api.get(`/data/${filename}/summary`);
  return data;
}
