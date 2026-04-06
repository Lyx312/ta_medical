import { Pressable, ScrollView, StyleSheet } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Colors, Radii, Spacing } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';

export type QueueFilterChip = {
  key: string;
  label: string;
};

export type QueueFilterChipRowProps = {
  chips: QueueFilterChip[];
  selectedKey: string;
  onSelect: (key: string) => void;
  compact?: boolean;
};

export function QueueFilterChipRow({ chips, selectedKey, onSelect, compact = false }: QueueFilterChipRowProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];

  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={[styles.content, compact && styles.contentCompact]}>
      {chips.map((chip) => {
        const isSelected = chip.key === selectedKey;

        return (
          <Pressable
            key={chip.key}
            onPress={() => onSelect(chip.key)}
            style={[
              styles.chip,
              compact && styles.chipCompact,
              {
                backgroundColor: isSelected ? `${colors.primary}18` : colors.cardBackground,
                borderColor: isSelected ? colors.primary : colors.divider,
              },
            ]}>
            <ThemedText
              style={[
                styles.label,
                compact && styles.labelCompact,
                { color: isSelected ? colors.primary : colors.text },
              ]}>
              {chip.label}
            </ThemedText>
          </Pressable>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  chip: {
    borderRadius: Radii.pill,
    borderWidth: 1,
    minHeight: 38,
    justifyContent: 'center',
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  chipCompact: {
    minHeight: 32,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  content: {
    gap: Spacing.sm,
  },
  contentCompact: {
    gap: 6,
  },
  label: {
    fontSize: 13,
    fontWeight: '700',
  },
  labelCompact: {
    fontSize: 12,
  },
});
