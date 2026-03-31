import { Ionicons } from '@expo/vector-icons';
import { useMemo, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { Colors } from '@/constants/theme';
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

              <View style={[styles.card, { backgroundColor: colors.cardBackground, borderColor: colors.divider }]}>
                <ThemedText type="subtitle" style={styles.sectionTitle}>
                  {pageCopy.sectionTitle}
                </ThemedText>

                {!isLogin ? (
                  <View style={styles.fieldGroup}>
                    <ThemedText style={styles.fieldLabel}>Full Name</ThemedText>
                    <TextInput
                      value={fullName}
                      onChangeText={setFullName}
                      placeholder="Enter your full name"
                      placeholderTextColor={colors.icon}
                      autoCapitalize="words"
                      style={[
                        styles.input,
                        {
                          borderColor: colors.divider,
                          color: colors.text,
                        },
                      ]}
                    />
                  </View>
                ) : null}

                <View style={styles.fieldGroup}>
                  <ThemedText style={styles.fieldLabel}>Email</ThemedText>
                  <TextInput
                    value={email}
                    onChangeText={setEmail}
                    placeholder="Enter your email"
                    placeholderTextColor={colors.icon}
                    keyboardType="email-address"
                    autoCapitalize="none"
                    style={[
                      styles.input,
                      {
                        borderColor: colors.divider,
                        color: colors.text,
                      },
                    ]}
                  />
                </View>

                <View style={styles.fieldGroup}>
                  <ThemedText style={styles.fieldLabel}>Password</ThemedText>
                  <View style={styles.passwordInputWrapper}>
                    <TextInput
                      value={password}
                      onChangeText={setPassword}
                      placeholder="Enter your password"
                      placeholderTextColor={colors.icon}
                      autoCapitalize="none"
                      autoCorrect={false}
                      secureTextEntry={!showPassword}
                      style={[
                        styles.input,
                        styles.passwordInput,
                        {
                          borderColor: colors.divider,
                          color: colors.text,
                        },
                      ]}
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
                    <ThemedText style={styles.fieldLabel}>Confirm Password</ThemedText>
                    <View style={styles.passwordInputWrapper}>
                      <TextInput
                        value={confirmPassword}
                        onChangeText={setConfirmPassword}
                        placeholder="Confirm your password"
                        placeholderTextColor={colors.icon}
                        autoCapitalize="none"
                        autoCorrect={false}
                        secureTextEntry={!showConfirmPassword}
                        style={[
                          styles.input,
                          styles.passwordInput,
                          {
                            borderColor: colors.divider,
                            color: colors.text,
                          },
                        ]}
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

                {error ? <ThemedText style={[styles.errorText, { color: colors.error }]}>{error}</ThemedText> : null}

                <Pressable
                  onPress={isLogin ? handleLogin : handleRegister}
                  disabled={loading}
                  style={[
                    styles.primaryButton,
                    { backgroundColor: colors.cta, opacity: loading ? 0.7 : 1 },
                  ]}>
                  {loading ? (
                    <ActivityIndicator color={colors.cardBackground} />
                  ) : (
                    <ThemedText style={[styles.primaryButtonLabel, { color: colors.cardBackground }]}>
                      {pageCopy.action}
                    </ThemedText>
                  )}
                </Pressable>
              </View>
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
    paddingHorizontal: 20,
    paddingVertical: 24,
    gap: 20,
  },
  headerContainer: {
    gap: 8,
  },
  title: {
    fontSize: 36,
    lineHeight: 40,
  },
  subtitle: {
    fontSize: 16,
    opacity: 0.85,
  },
  card: {
    borderWidth: 1,
    borderRadius: 18,
    padding: 16,
    gap: 12,
  },
  sectionTitle: {
    fontSize: 24,
  },
  fieldGroup: {
    gap: 6,
  },
  fieldLabel: {
    fontSize: 14,
    fontWeight: '600',
  },
  input: {
    borderWidth: 1,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 16,
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
    fontSize: 14,
  },
  primaryButton: {
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  primaryButtonLabel: {
    fontSize: 16,
    fontWeight: '700',
  },
});
