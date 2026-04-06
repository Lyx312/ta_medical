import { Ionicons } from '@expo/vector-icons';
import { useMemo, useState } from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { AppButton } from '@/components/ui/app-button';
import { FieldError, FieldLabel, TextField } from '@/components/ui/form-field';
import { SurfaceCard } from '@/components/ui/surface-card';
import { ThemedText } from '@/components/themed-text';
import { Colors, Opacity, Spacing, Typography } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { useAuth } from '@/contexts/auth-context';
import type { ApiError } from '@/services/api/types';

type AuthMode = 'login' | 'register';

type AuthScreenProps = {
  mode: AuthMode;
};

export function AuthScreen({ mode }: AuthScreenProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const { signIn, signUp } = useAuth();
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const isLogin = mode === 'login';

  const pageCopy = useMemo(
    () =>
      isLogin
        ? {
            action: 'Login',
            heading: 'MedInsure',
            sectionTitle: 'Sign In',
            subtitle: 'Access your medical coverage in seconds',
          }
        : {
            action: 'Create Account',
            heading: 'Create Account',
            sectionTitle: 'Create Account',
            subtitle: 'Start managing your medical insurance claims',
          },
    [isLogin],
  );

  const handleLogin = async () => {
    if (!email.trim() || !password.trim()) {
      setError('Please fill in email and password.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      await signIn(email.trim(), password);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.message ?? 'Unable to login at the moment.');
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async () => {
    if (!fullName.trim() || !email.trim() || !password.trim() || !confirmPassword.trim()) {
      setError('Please complete all fields.');
      return;
    }

    if (password !== confirmPassword) {
      setError('Password confirmation does not match.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      await signUp(fullName.trim(), email.trim(), password);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.message ?? 'Unable to register at the moment.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: colors.background }]}>
      <KeyboardAvoidingView
        style={styles.keyboardContainer}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
        <View style={styles.screen}>
          <ScrollView
            contentContainerStyle={styles.scrollContent}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}>
            <View style={styles.container}>
              <View style={styles.headerContainer}>
                <ThemedText type="title" style={[styles.title, { color: colors.primary }]}>
                  {pageCopy.heading}
                </ThemedText>
                <ThemedText style={styles.subtitle}>{pageCopy.subtitle}</ThemedText>
              </View>

              <SurfaceCard style={styles.card}>
                <ThemedText type="subtitle" style={styles.sectionTitle}>
                  {pageCopy.sectionTitle}
                </ThemedText>

                {!isLogin ? (
                  <View style={styles.fieldGroup}>
                    <FieldLabel>Full Name</FieldLabel>
                    <TextField
                      value={fullName}
                      onChangeText={setFullName}
                      placeholder="Enter your full name"
                      autoCapitalize="words"
                    />
                  </View>
                ) : null}

                <View style={styles.fieldGroup}>
                  <FieldLabel>Email</FieldLabel>
                  <TextField
                    value={email}
                    onChangeText={setEmail}
                    placeholder="Enter your email"
                    keyboardType="email-address"
                    autoCapitalize="none"
                  />
                </View>

                <View style={styles.fieldGroup}>
                  <FieldLabel>Password</FieldLabel>
                  <View style={styles.passwordInputWrapper}>
                    <TextField
                      value={password}
                      onChangeText={setPassword}
                      placeholder="Enter your password"
                      autoCapitalize="none"
                      autoCorrect={false}
                      secureTextEntry={!showPassword}
                      style={styles.passwordInput}
                    />
                    <Pressable
                      onPress={() => setShowPassword((current) => !current)}
                      style={styles.eyeButton}
                      hitSlop={8}>
                      <Ionicons
                        name={showPassword ? 'eye-off-outline' : 'eye-outline'}
                        size={20}
                        color={colors.icon}
                      />
                    </Pressable>
                  </View>
                </View>

                {!isLogin ? (
                  <View style={styles.fieldGroup}>
                    <FieldLabel>Confirm Password</FieldLabel>
                    <View style={styles.passwordInputWrapper}>
                      <TextField
                        value={confirmPassword}
                        onChangeText={setConfirmPassword}
                        placeholder="Confirm your password"
                        autoCapitalize="none"
                        autoCorrect={false}
                        secureTextEntry={!showConfirmPassword}
                        style={styles.passwordInput}
                      />
                      <Pressable
                        onPress={() => setShowConfirmPassword((current) => !current)}
                        style={styles.eyeButton}
                        hitSlop={8}>
                        <Ionicons
                          name={showConfirmPassword ? 'eye-off-outline' : 'eye-outline'}
                          size={20}
                          color={colors.icon}
                        />
                      </Pressable>
                    </View>
                  </View>
                ) : null}

                <FieldError message={error} style={styles.errorText} />

                <AppButton
                  label={pageCopy.action}
                  tone="cta"
                  loading={loading}
                  onPress={isLogin ? handleLogin : handleRegister}
                />
              </SurfaceCard>
            </View>
          </ScrollView>

        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
  },
  keyboardContainer: {
    flex: 1,
  },
  screen: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },
  container: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingHorizontal: Spacing.xl,
    paddingVertical: Spacing.xxl,
    gap: Spacing.xl,
  },
  headerContainer: {
    gap: Spacing.sm,
  },
  title: {
    fontSize: 36,
    lineHeight: 40,
  },
  subtitle: {
    ...Typography.subtitle,
    opacity: 0.85,
  },
  card: {
    borderRadius: 18,
    gap: Spacing.md,
  },
  sectionTitle: {
    fontSize: 24,
  },
  fieldGroup: {
    gap: 6,
  },
  passwordInputWrapper: {
    position: 'relative',
    justifyContent: 'center',
  },
  passwordInput: {
    paddingRight: 46,
  },
  eyeButton: {
    position: 'absolute',
    right: 14,
    height: '100%',
    justifyContent: 'center',
  },
  errorText: {
    ...Typography.fieldLabel,
    opacity: Opacity.subtle,
  },
});
