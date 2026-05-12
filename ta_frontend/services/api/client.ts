import { API_BASE_URL } from './config';
import { getAuthToken } from './storage';
import type { ApiError, ApiErrorResponse } from './types';

type JsonRequestBody = Record<string, unknown>;
type ProgressCallback = (progressPercent: number) => void;

type RequestOptions = {
  body?: BodyInit | null;
  headers?: Record<string, string>;
  method: 'DELETE' | 'GET' | 'POST' | 'PUT';
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

export function putJson<T>(path: string, body: JsonRequestBody, requiresAuth: boolean = false) {
  return request<T>(path, {
    method: 'PUT',
    body: JSON.stringify(body),
    headers: {
      'Content-Type': 'application/json',
    },
    requiresAuth,
  });
}

export function deleteJson<T>(path: string, requiresAuth: boolean = false) {
  return request<T>(path, {
    method: 'DELETE',
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

export async function postFormDataWithProgress<T>(
  path: string,
  formData: FormData,
  onUploadProgress: ProgressCallback,
  requiresAuth: boolean = false,
) {
  const token = requiresAuth ? await getAuthToken() : null;

  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE_URL}${path}`);
    if (token) {
      xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    }

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) {
        return;
      }
      const progressPercent = Math.max(0, Math.min(100, Math.round((event.loaded / event.total) * 100)));
      onUploadProgress(progressPercent);
    };

    xhr.onerror = () => {
      reject({
        message: 'Network request failed. Please try again.',
      } as ApiError);
    };

    xhr.onload = () => {
      const data = xhr.responseText ? safelyParseJson(xhr.responseText) : null;
      if (xhr.status >= 200 && xhr.status < 300) {
        onUploadProgress(100);
        resolve(data as T);
        return;
      }
      reject({
        message: getErrorMessage(data as ApiErrorResponse | null),
        status: xhr.status,
      } as ApiError);
    };

    xhr.send(formData);
  });
}

function safelyParseJson(rawValue: string) {
  try {
    return JSON.parse(rawValue) as ApiErrorResponse | unknown;
  } catch {
    return null;
  }
}
