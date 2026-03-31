import type { Href } from 'expo-router';

import type { AuthUser } from '@/services/api/types';

export function getDefaultProtectedRoute(user: Pick<AuthUser, 'role'> | null | undefined): Href {
  return user?.role === 'admin' ? '/admin/settings' : '/user/new-claim';
}

