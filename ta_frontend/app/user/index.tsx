import { useEffect } from 'react';
import { router, Href } from 'expo-router';

export default function UserIndex() {
  useEffect(() => {
    router.replace('/user/new-claim' as Href);
  }, []);
  return null;
}
