import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { router } from 'expo-router';

import { getCurrentUser, loginUser, logoutUser, registerUser } from '@/services/api/auth';
import { getAuthToken } from '@/services/api/storage';
import type { AuthUser } from '@/services/api/types';

type AuthState = {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
};

type AuthContextType = AuthState & {
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (fullName: string, email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const checkAuthState = useCallback(async () => {
    try {
      const token = await getAuthToken();
      if (!token) {
        setUser(null);
        return;
      }

      const userData = await getCurrentUser();
      setUser(userData);
    } catch {
      await logoutUser();
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    checkAuthState();
  }, [checkAuthState]);

  const signIn = async (email: string, password: string) => {
    const response = await loginUser({ email, password });
    setUser(response.user);
  };

  const signUp = async (fullName: string, email: string, password: string) => {
    const response = await registerUser({
      full_name: fullName,
      email,
      password,
      confirm_password: password,
    });
    setUser(response.user);
  };

  const signOut = async () => {
    await logoutUser();
    setUser(null);
    router.replace('/login');
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: user !== null,
        isLoading,
        signIn,
        signUp,
        signOut,
      }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
