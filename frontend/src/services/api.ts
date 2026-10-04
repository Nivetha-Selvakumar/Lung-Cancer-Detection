import { HealthResponse, PredictResponse, AuthResponse, UserProfile, DbStatusResponse } from '../types/api';

const API_BASE = '';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) {
    throw new Error('Health check failed');
  }
  return res.json();
}

export async function fetchCurrentUser(token: string): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/api/auth/me`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  return res.json();
}

export async function predictImage(file: File, username?: string): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (username) {
    formData.append('username', username);
  }

  const token = localStorage.getItem('lung_ai_token');
  const headers: Record<string, string> = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers,
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || 'Prediction failed');
  }
  return data;
}

export async function loginUser(username: string, password: string): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  return res.json();
}

export async function registerUser(user: Partial<UserProfile> & { password: string }): Promise<AuthResponse> {
  const token = localStorage.getItem('lung_ai_token');
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/api/auth/register`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...user, doctor_token: token }),
  });
  return res.json();
}

export async function checkDbStatus(): Promise<DbStatusResponse> {
  const res = await fetch(`${API_BASE}/api/auth/db-status`);
  return res.json();
}

export async function configureDbPassword(password: string): Promise<{ success: boolean; message?: string; error?: string }> {
  const res = await fetch(`${API_BASE}/api/auth/configure-db`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password }),
  });
  return res.json();
}
