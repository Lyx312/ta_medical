import AsyncStorage from '@react-native-async-storage/async-storage';

export const TOKEN_KEY = 'auth_token';
export const THEME_PREFERENCE_KEY = 'theme_preference';

export type StoredThemePreference = 'light' | 'dark';

export async function getAuthToken() {
  return AsyncStorage.getItem(TOKEN_KEY);
}

export async function setAuthToken(token: string) {
  await AsyncStorage.setItem(TOKEN_KEY, token);
}

export async function clearAuthToken() {
  await AsyncStorage.removeItem(TOKEN_KEY);
}

export async function getThemePreference() {
  const value = await AsyncStorage.getItem(THEME_PREFERENCE_KEY);
  if (value === 'light' || value === 'dark') {
    return value as StoredThemePreference;
  }
  return null;
}

export async function setThemePreference(preference: StoredThemePreference) {
  await AsyncStorage.setItem(THEME_PREFERENCE_KEY, preference);
}
