import { Ionicons } from '@expo/vector-icons';
import { Pressable, StyleSheet, TextInput, View, type TextInputProps } from 'react-native';

import { Colors, Opacity, Radii, Spacing } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';

export type SearchInputProps = TextInputProps & {
  onClear?: () => void;
  compact?: boolean;
};

export function SearchInput({ onClear, value, style, compact = false, ...rest }: SearchInputProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const hasValue = typeof value === 'string' && value.length > 0;

  return (
    <View
      style={[
        styles.container,
        compact && styles.containerCompact,
        { backgroundColor: colors.cardBackground, borderColor: colors.divider },
      ]}>
      <Ionicons name="search-outline" size={18} color={colors.icon} />
      <TextInput
        value={value}
        placeholderTextColor={colors.icon}
        style={[styles.input, compact && styles.inputCompact, { color: colors.text }, style]}
        {...rest}
      />
      {hasValue ? (
        <Pressable onPress={onClear} hitSlop={8}>
          <Ionicons name="close-circle" size={18} color={colors.icon} />
        </Pressable>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    borderRadius: Radii.md,
    borderWidth: 1,
    flexDirection: 'row',
    gap: Spacing.sm,
    minHeight: 48,
    paddingHorizontal: 14,
  },
  containerCompact: {
    minHeight: 42,
    paddingHorizontal: 12,
  },
  input: {
    flex: 1,
    fontSize: 16,
    opacity: Opacity.secondary,
    paddingVertical: 12,
  },
  inputCompact: {
    fontSize: 15,
    paddingVertical: 10,
  },
});
