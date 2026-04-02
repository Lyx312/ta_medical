import { useColorScheme as useRNColorScheme } from 'react-native';
import { useContext } from 'react';

import { ThemeContext } from '@/contexts/theme-context';

export function useColorScheme() {
  const themeContext = useContext(ThemeContext);
  const systemColorScheme = useRNColorScheme();

  return themeContext?.colorScheme ?? (systemColorScheme === 'dark' ? 'dark' : 'light');
}
