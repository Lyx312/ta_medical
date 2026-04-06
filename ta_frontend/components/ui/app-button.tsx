import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  View,
  type PressableProps,
  type StyleProp,
  type TextStyle,
  type ViewStyle,
} from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Colors, Radii, Spacing, Typography } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';

type ButtonVariant = 'filled' | 'outline';
type ButtonTone = 'primary' | 'secondary' | 'cta' | 'error';

export type AppButtonProps = Omit<PressableProps, 'style'> & {
  label?: string;
  children?: React.ReactNode;
  variant?: ButtonVariant;
  tone?: ButtonTone;
  loading?: boolean;
  leftIcon?: React.ReactNode;
  style?: StyleProp<ViewStyle>;
  textStyle?: StyleProp<TextStyle>;
};

export function AppButton({
  label,
  children,
  variant = 'filled',
  tone = 'primary',
  loading = false,
  leftIcon,
  disabled,
  style,
  textStyle,
  ...rest
}: AppButtonProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const toneColor = colors[tone];
  const isDisabled = disabled || loading;
  const textColor = variant === 'filled' ? colors.cardBackground : toneColor;

  return (
    <Pressable
      disabled={isDisabled}
      style={[
        styles.button,
        variant === 'filled'
          ? { backgroundColor: toneColor }
          : {
              backgroundColor: 'transparent',
              borderColor: toneColor,
              borderWidth: 1,
            },
        isDisabled && styles.disabled,
        style,
      ]}
      {...rest}>
      {loading ? (
        <ActivityIndicator color={textColor} />
      ) : (
        <View style={styles.content}>
          {leftIcon ? leftIcon : null}
          {label ? (
            <ThemedText style={[styles.label, { color: textColor }, textStyle]}>{label}</ThemedText>
          ) : (
            children
          )}
        </View>
      )}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    alignItems: 'center',
    borderRadius: Radii.md,
    justifyContent: 'center',
    minHeight: 48,
    paddingHorizontal: 18,
    paddingVertical: 12,
  },
  content: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: Spacing.sm,
    justifyContent: 'center',
  },
  disabled: {
    opacity: 0.7,
  },
  label: {
    ...Typography.button,
  },
});
