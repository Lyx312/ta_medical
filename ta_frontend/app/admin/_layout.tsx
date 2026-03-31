import { useEffect } from 'react';
import { router, Stack } from 'expo-router';

import { useAuth } from '@/contexts/auth-context';
import { getDefaultProtectedRoute } from '@/features/auth/utils/auth-routing';

export default function AdminLayout() {
  const { isAuthenticated, isLoading, user } = useAuth();

  useEffect(() => {
    if (isLoading) {
      return;
    }

    if (!isAuthenticated || !user) {
      router.replace('/login');
      return;
    }

    if (user.role !== 'admin') {
      router.replace(getDefaultProtectedRoute(user));
    }
  }, [isAuthenticated, isLoading, user]);

  if (isLoading || !isAuthenticated || !user || user.role !== 'admin') {
    return null;
  }

  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen name="(tabs)" />
    </Stack>
  );
}
