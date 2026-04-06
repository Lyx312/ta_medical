import type { ReactNode } from 'react';
import { Pressable, StyleSheet, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { StatusBadge } from '@/components/ui/status-badge';
import { SurfaceCard } from '@/components/ui/surface-card';
import { ThemedText } from '@/components/themed-text';
import { Opacity, Spacing, Typography, Colors } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import type { ClaimResponse } from '@/services/api/types';
import {
  formatClaimCurrency,
  formatClaimServiceDate,
  getClaimStatusPresentation,
} from '@/features/claims/utils/claim-display';

export type ClaimSummaryCardProps = {
  claim: ClaimResponse;
  onPress?: () => void;
  footer?: ReactNode;
  metadata?: { label: string; value: string }[];
  statusVariant?: 'badge' | 'text';
  showPressHint?: boolean;
};

export function ClaimSummaryCard({
  claim,
  onPress,
  footer,
  metadata,
  statusVariant = 'badge',
  showPressHint = false,
}: ClaimSummaryCardProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const status = getClaimStatusPresentation(claim.status, colors.text);

  const content = (
    <SurfaceCard style={styles.card}>
      <View style={styles.row}>
        <ThemedText style={styles.amount}>{formatClaimCurrency(claim.nominal_klaim)}</ThemedText>
        <View style={styles.rowEnd}>
          {statusVariant === 'badge' ? (
            <StatusBadge
              label={claim.status}
              backgroundColor={status.backgroundColor}
              textColor={status.textColor}
            />
          ) : (
            <ThemedText style={[styles.statusText, { color: status.textColor }]}>{claim.status}</ThemedText>
          )}
          {showPressHint ? <Ionicons name="chevron-forward" size={18} color={colors.icon} /> : null}
        </View>
      </View>
      <ThemedText style={styles.patientName}>{claim.nama_pasien}</ThemedText>
      <ThemedText style={styles.description}>{claim.keterangan_penyakit}</ThemedText>
      <ThemedText style={styles.date}>{formatClaimServiceDate(claim.tanggal_pelayanan)}</ThemedText>
      {metadata?.length ? (
        <View style={styles.metadataList}>
          {metadata.map((item) => (
            <View key={`${item.label}-${item.value}`} style={styles.metadataRow}>
              <ThemedText style={styles.metadataLabel}>{item.label}</ThemedText>
              <ThemedText style={styles.metadataValue}>{item.value}</ThemedText>
            </View>
          ))}
        </View>
      ) : null}
      {footer ? <View style={styles.footer}>{footer}</View> : null}
    </SurfaceCard>
  );

  if (!onPress) {
    return content;
  }

  return <Pressable onPress={onPress}>{content}</Pressable>;
}

const styles = StyleSheet.create({
  amount: {
    fontSize: 18,
    fontWeight: '700',
  },
  card: {
    gap: Spacing.sm,
  },
  date: {
    ...Typography.meta,
    marginTop: Spacing.xs,
    opacity: Opacity.faint,
  },
  description: {
    fontSize: 16,
    marginTop: Spacing.sm,
    opacity: Opacity.secondary,
  },
  footer: {
    marginTop: Spacing.md,
  },
  metadataLabel: {
    ...Typography.meta,
    flex: 1,
  },
  metadataList: {
    gap: Spacing.xs,
    marginTop: Spacing.sm,
  },
  metadataRow: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: Spacing.md,
  },
  metadataValue: {
    ...Typography.meta,
    flex: 1.4,
    opacity: Opacity.secondary,
    textAlign: 'right',
  },
  patientName: {
    fontSize: 15,
    fontWeight: '600',
    marginTop: Spacing.sm,
  },
  row: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  rowEnd: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  statusText: {
    fontSize: 14,
    fontWeight: '600',
  },
});
