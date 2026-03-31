import Constants from 'expo-constants';
import { Platform } from 'react-native';

const LOCAL_API_FALLBACK_ANDROID = 'http://10.0.2.2:8000/api';
const LOCAL_API_FALLBACK_OTHER = 'http://127.0.0.1:8000/api';

export const API_BASE_URL =
  (Constants.expoConfig?.extra?.apiBaseUrl as string | undefined) ??
  (Platform.OS === 'android' ? LOCAL_API_FALLBACK_ANDROID : LOCAL_API_FALLBACK_OTHER);

