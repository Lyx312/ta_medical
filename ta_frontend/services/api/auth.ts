import { getJson, postJson } from './client';
import { clearAuthToken, setAuthToken } from './storage';
import type { AuthUser, LoginPayload, LoginResponse, RegisterPayload, RegisterResponse } from './types';

export async function loginUser(payload: LoginPayload): Promise<LoginResponse> {
  const response = await postJson<LoginResponse>('/auth/login', payload);
  await setAuthToken(response.access_token);
  return response;
}

export async function registerUser(payload: RegisterPayload): Promise<RegisterResponse> {
  const response = await postJson<RegisterResponse>('/auth/register', payload);
  await setAuthToken(response.access_token);
  return response;
}

export function getCurrentUser(): Promise<AuthUser> {
  return getJson<AuthUser>('/auth/me', true);
}

export function logoutUser() {
  return clearAuthToken();
}
