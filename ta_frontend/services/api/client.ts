import { API_BASE_URL } from './config';
import { getAuthToken } from './storage';
import type { ApiError, ApiErrorResponse } from './types';

type JsonRequestBody = Record<string, unknown>;

type RequestOptions = {
  body?: BodyInit | null;
  headers?: Record<string, string>;
  method: 'GET' | 'POST';
  requiresAuth?: boolean;
};

function getErrorMessage(data: ApiErrorResponse | null) {
  const firstValidationMessage = data?.errors
    ?.map((error) => error?.message?.trim())
    .find((message): message is string => Boolean(message));

  if (firstValidationMessage) {
    return firstValidationMessage;
  }

  if (data?.message?.trim()) {
    return data.message.trim();
  }

  return 'Request failed. Please try again.';
}

async function request<T>(path: string, options: RequestOptions): Promise<T> {
  const headers = { ...(options.headers ?? {}) };

  if (options.requiresAuth) {
    const token = await getAuthToken();
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: options.method,
    headers,
    body: options.body,
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw {
      message: getErrorMessage(data),
      status: response.status,
    } as ApiError;
  }

  return data as T;
}

export function getJson<T>(path: string, requiresAuth: boolean = false) {
  return request<T>(path, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
    requiresAuth,
  });
}

export function postJson<T>(path: string, body: JsonRequestBody, requiresAuth: boolean = false) {
  return request<T>(path, {
    method: 'POST',
    body: JSON.stringify(body),
    headers: {
      'Content-Type': 'application/json',
    },
    requiresAuth,
  });
}

export function postFormData<T>(path: string, formData: FormData, requiresAuth: boolean = false) {
  return request<T>(path, {
    method: 'POST',
    body: formData,
    requiresAuth,
  });
}

