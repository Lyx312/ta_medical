import { ActivityIndicator, StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';

import { AppButton } from '@/components/ui/app-button';
import { ThemedText } from '@/components/themed-text';
import { Colors, Spacing } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';

export type CenteredStateProps = {
  loading?: boolean;
  message?: string;
  tone?: 'default' | 'error';
  actionLabel?: string;
  onAction?: () => void;
  minHeight?: number;
  style?: StyleProp<ViewStyle>;
};

export function CenteredState({
  loading = false,
  message,
  tone = 'default',
  actionLabel,
  onAction,
  minHeight,
  style,
}: CenteredStateProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];

  return (
    <View style={[styles.container, minHeight ? { minHeight } : null, style]}>
      {loading ? <ActivityIndicator color={colors.primary} /> : null}
      {message ? (
        <ThemedText style={[styles.message, tone === 'error' ? { color: colors.error } : null]}>{message}</ThemedText>
      ) : null}
      {actionLabel && onAction ? <AppButton label={actionLabel} onPress={onAction} style={styles.action} /> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  action: {
    marginTop: Spacing.lg,
  },
  container: {
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  message: {
    marginTop: Spacing.sm,
    textAlign: 'center',
  },
});
