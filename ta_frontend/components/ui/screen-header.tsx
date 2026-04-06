import { StyleSheet, View, type StyleProp, type ViewStyle } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Opacity, Spacing, Typography } from '@/constants/theme';

export type ScreenHeaderProps = {
  title: string;
  subtitle?: string;
  helperText?: string;
  style?: StyleProp<ViewStyle>;
};

export function ScreenHeader({ title, subtitle, helperText, style }: ScreenHeaderProps) {
  return (
    <View style={style}>
      <ThemedText type="title">{title}</ThemedText>
      {subtitle ? <ThemedText style={styles.subtitle}>{subtitle}</ThemedText> : null}
      {helperText ? <ThemedText style={styles.helperText}>{helperText}</ThemedText> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  helperText: {
    ...Typography.helper,
    lineHeight: 20,
    marginTop: Spacing.md,
    opacity: Opacity.subtle,
  },
  subtitle: {
    ...Typography.subtitle,
    marginTop: Spacing.sm,
  },
});
