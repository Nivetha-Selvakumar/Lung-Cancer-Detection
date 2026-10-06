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

export async function hashPasswordClient(password: string): Promise<string> {
  if (!password) return '';
  // If already a 64-character SHA-256 hex string, return as is
  if (/^[a-fA-F0-9]{64}$/.test(password)) {
    return password;
  }
  try {
    const encoder = new TextEncoder();
    const data = encoder.encode(password);
    const hashBuffer = await window.crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  } catch (e) {
    return password;
  }
}

export async function loginUser(username: string, password: string): Promise<AuthResponse> {
  const encryptedPassword = await hashPasswordClient(password);
  const res = await fetch(`${API_BASE}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password: encryptedPassword }),
  });
  return res.json();
}

export async function logoutUser(token?: string): Promise<{ success: boolean }> {
  const authToken = token || localStorage.getItem('lung_ai_token') || '';
  const res = await fetch(`${API_BASE}/api/auth/logout`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${authToken}`
    },
    body: JSON.stringify({ token: authToken }),
  });
  return res.json();
}

export async function requestForgotPassword(identity: string): Promise<{ success: boolean; message?: string; error?: string; reset_token?: string }> {
  const res = await fetch(`${API_BASE}/api/auth/forgot-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: identity, username: identity }),
  });
  return res.json();
}

export async function resetPassword(identity: string, resetToken: string, newPassword: string): Promise<{ success: boolean; message?: string; error?: string }> {
  const encryptedPassword = await hashPasswordClient(newPassword);
  const res = await fetch(`${API_BASE}/api/auth/reset-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email: identity, username: identity, reset_token: resetToken, new_password: encryptedPassword }),
  });
  return res.json();
}

export async function registerUser(user: Partial<UserProfile> & { password: string }): Promise<AuthResponse> {
  const encryptedPassword = await hashPasswordClient(user.password);
  const token = localStorage.getItem('lung_ai_token');
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/api/auth/register`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...user, password: encryptedPassword, doctor_token: token }),
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

export interface DatasetImageSample {
  id: string;
  dataset: string;
  category: string;
  filename: string;
  image_url: string;
}

export async function fetchDatasetImages(dataset: string, category: string, limit: number = 12): Promise<{ success: boolean; images: DatasetImageSample[] }> {
  const res = await fetch(`${API_BASE}/api/dataset/images?dataset=${encodeURIComponent(dataset)}&category=${encodeURIComponent(category)}&limit=${limit}`);
  if (!res.ok) {
    return { success: false, images: [] };
  }
  return res.json();
}


