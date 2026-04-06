import { ScrollView, StyleSheet, View } from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { Colors, Opacity, Radii, Spacing } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';

export type QueueSummaryItem = {
  key: string;
  label: string;
  value: string;
  tone?: 'primary' | 'secondary' | 'cta' | 'warning' | 'error';
  helper?: string;
};

export type QueueSummaryStripProps = {
  items: QueueSummaryItem[];
  compact?: boolean;
};

export function QueueSummaryStrip({ items, compact = false }: QueueSummaryStripProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];

  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={[styles.content, compact && styles.contentCompact]}>
      {items.map((item) => {
        const toneColor =
          item.tone === 'secondary'
            ? colors.secondary
            : item.tone === 'cta'
              ? colors.cta
              : item.tone === 'warning'
                ? colors.warning
                : item.tone === 'error'
                  ? colors.error
                  : colors.primary;

        return (
          <View
            key={item.key}
            style={[
              styles.card,
              compact && styles.cardCompact,
              {
                backgroundColor: colors.cardBackground,
                borderColor: colors.divider,
              },
            ]}>
            <ThemedText style={[styles.label, compact && styles.labelCompact]}>{item.label}</ThemedText>
            <ThemedText style={[styles.value, compact && styles.valueCompact, { color: toneColor }]}>
              {item.value}
            </ThemedText>
            {!compact && item.helper ? <ThemedText style={styles.helper}>{item.helper}</ThemedText> : null}
          </View>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: Radii.lg,
    borderWidth: 1,
    minWidth: 150,
    padding: Spacing.lg,
  },
  cardCompact: {
    minWidth: 112,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  content: {
    gap: Spacing.md,
  },
  contentCompact: {
    gap: Spacing.sm,
  },
  helper: {
    fontSize: 12,
    marginTop: Spacing.xs,
    opacity: Opacity.subtle,
  },
  label: {
    fontSize: 12,
    fontWeight: '700',
    opacity: Opacity.subtle,
    textTransform: 'uppercase',
  },
  labelCompact: {
    fontSize: 10,
  },
  value: {
    fontSize: 22,
    fontWeight: '800',
    marginTop: Spacing.sm,
  },
  valueCompact: {
    fontSize: 17,
    marginTop: 4,
  },
});
