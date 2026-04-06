import { StyleSheet, View, type StyleProp, type TextStyle, type ViewStyle } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Radii, Typography } from '@/constants/theme';

export type StatusBadgeProps = {
  label: string;
  textColor: string;
  backgroundColor: string;
  style?: StyleProp<ViewStyle>;
  textStyle?: StyleProp<TextStyle>;
};

export function StatusBadge({ label, textColor, backgroundColor, style, textStyle }: StatusBadgeProps) {
  return (
    <View style={[styles.badge, { backgroundColor }, style]}>
      <ThemedText style={[styles.label, { color: textColor }, textStyle]}>{label}</ThemedText>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    alignItems: 'center',
    alignSelf: 'flex-start',
    borderRadius: Radii.pill,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  label: {
    ...Typography.badge,
  },
});
