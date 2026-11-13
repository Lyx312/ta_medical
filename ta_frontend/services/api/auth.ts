import { getJson, postFormData, postJson, putJson } from './client';
import { clearAuthToken, setAuthToken } from './storage';
import type { AuthUser, LoginPayload, LoginResponse, RegisterPayload, RegisterResponse, UpdateUserProfilePayload } from './types';

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

export function updateCurrentUserProfile(payload: UpdateUserProfilePayload): Promise<AuthUser> {
  return putJson<AuthUser>('/auth/me/profile', payload, true);
}

export function uploadCurrentUserInsuranceCard(uri: string): Promise<AuthUser> {
  const formData = new FormData();
  const filename = uri.split('/').pop() ?? `insurance_card_${Date.now()}.jpg`;
  const type = filename.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';

  formData.append('file', {
    uri,
    name: filename,
    type,
  } as never);

  return postFormData<AuthUser>('/auth/me/insurance-card', formData, true);
}

export function logoutUser() {
  return clearAuthToken();
}
