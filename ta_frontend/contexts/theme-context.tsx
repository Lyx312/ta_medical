import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { useColorScheme as useRNColorScheme } from 'react-native';

import {
  getThemePreference as getStoredThemePreference,
  setThemePreference as storeThemePreference,
  type StoredThemePreference,
} from '@/services/api/storage';

type ThemeContextType = {
  colorScheme: 'light' | 'dark';
  preference: StoredThemePreference | null;
  isLoading: boolean;
  setPreference: (preference: StoredThemePreference) => Promise<void>;
  toggleColorScheme: () => Promise<void>;
};

export const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const systemColorScheme = useRNColorScheme() === 'dark' ? 'dark' : 'light';
  const [preference, setPreferenceState] = useState<StoredThemePreference | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    const loadThemePreference = async () => {
      try {
        const storedPreference = await getStoredThemePreference();
        if (isMounted) {
          setPreferenceState(storedPreference);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    void loadThemePreference();

    return () => {
      isMounted = false;
    };
  }, []);

  const colorScheme = preference ?? systemColorScheme;

  const setPreference = useCallback(async (nextPreference: StoredThemePreference) => {
    setPreferenceState(nextPreference);
    await storeThemePreference(nextPreference);
  }, []);

  const toggleColorScheme = useCallback(async () => {
    const nextPreference: StoredThemePreference = colorScheme === 'dark' ? 'light' : 'dark';
    setPreferenceState(nextPreference);
    await storeThemePreference(nextPreference);
  }, [colorScheme]);

  const value = useMemo(
    () => ({
      colorScheme,
      preference,
      isLoading,
      setPreference,
      toggleColorScheme,
    }),
    [colorScheme, isLoading, preference, setPreference, toggleColorScheme],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useThemePreference() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useThemePreference must be used within a ThemeProvider');
  }
  return context;
}
