import { useEffect, useState } from 'react';
import { useContext } from 'react';
import { useColorScheme as useRNColorScheme } from 'react-native';

import { ThemeContext } from '@/contexts/theme-context';

/**
 * To support static rendering, this value needs to be re-calculated on the client side for web
 */
export function useColorScheme() {
  const [hasHydrated, setHasHydrated] = useState(false);
  const themeContext = useContext(ThemeContext);

  useEffect(() => {
    setHasHydrated(true);
  }, []);

  const colorScheme = useRNColorScheme();

  if (themeContext) {
    return hasHydrated ? themeContext.colorScheme : 'light';
  }

  if (hasHydrated) {
    return colorScheme === 'dark' ? 'dark' : 'light';
  }

  return 'light';
}
