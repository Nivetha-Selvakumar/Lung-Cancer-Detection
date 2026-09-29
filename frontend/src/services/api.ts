import { HealthResponse, MetricsResponse, PredictResponse } from '../types/api';

const API_BASE = 'http://localhost:5000/api';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error('Health check failed');
  }
  return res.json();
}

export async function fetchMetrics(): Promise<MetricsResponse> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) {
    throw new Error('Metrics unavailable');
  }
  return res.json();
}

export async function predictImage(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });

  const data = await res.json();
  if (!res.ok || data.error) {
    throw new Error(data.error || 'Prediction failed');
  }

  return data;
}
