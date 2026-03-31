import { useEffect } from 'react';
import { router, Href } from 'expo-router';

export default function AdminIndex() {
  useEffect(() => {
    router.replace('/admin/settings' as Href);
  }, []);
  return null;
}
