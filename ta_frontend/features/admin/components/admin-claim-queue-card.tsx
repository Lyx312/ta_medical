import { Pressable, StyleSheet, View } from 'react-native';

import { AppButton } from '@/components/ui/app-button';
import { StatusBadge } from '@/components/ui/status-badge';
import { SurfaceCard } from '@/components/ui/surface-card';
import { ThemedText } from '@/components/themed-text';
import { Colors, Opacity, Radii, Spacing } from '@/constants/theme';
import { formatClaimCurrency, formatClaimServiceDate, getClaimStatusPresentation } from '@/features/claims/utils/claim-display';
import { useColorScheme } from '@/hooks/use-color-scheme';
import type { ClaimResponse } from '@/services/api/types';
import {
  formatAdminTimestamp,
  formatBenefitTypeLabel,
  getClaimSourcePresentation,
  isHighValueClaim,
} from '@/features/admin/utils/admin-queue';

export type AdminClaimQueueCardProps = {
  claim: ClaimResponse;
  onPress: () => void;
  onOpenScan?: () => void;
};

export function AdminClaimQueueCard({ claim, onPress, onOpenScan }: AdminClaimQueueCardProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const status = getClaimStatusPresentation(claim.status, colors.text);
  const source = getClaimSourcePresentation(claim, colors);
  const highValue = isHighValueClaim(claim.nominal_klaim);

  return (
    <Pressable onPress={onPress}>
      <SurfaceCard
        style={[
          styles.card,
          highValue && {
            borderColor: colors.warning,
            borderLeftWidth: 4,
          },
        ]}>
        <View style={styles.topRow}>
          <View style={styles.amountBlock}>
            <ThemedText style={styles.amount}>{formatClaimCurrency(claim.nominal_klaim)}</ThemedText>
            <ThemedText style={styles.timestamp}>{formatAdminTimestamp(claim.created_at)}</ThemedText>
          </View>
          <StatusBadge label={claim.status} backgroundColor={status.backgroundColor} textColor={status.textColor} />
        </View>

        <View style={styles.badgeRow}>
          <StatusBadge
            label={source.label}
            backgroundColor={source.backgroundColor}
            textColor={source.textColor}
          />
          {highValue ? (
            <StatusBadge
              label="High Value"
              backgroundColor={`${colors.warning}20`}
              textColor={colors.warning}
            />
          ) : null}
        </View>

        <View style={styles.section}>
          <ThemedText style={styles.patientName}>{claim.nama_pasien}</ThemedText>
          <ThemedText style={styles.metaText}>{claim.user_email}</ThemedText>
        </View>

        <View style={styles.grid}>
          <View style={styles.gridItem}>
            <ThemedText style={styles.gridLabel}>Benefit</ThemedText>
            <ThemedText style={styles.gridValue}>{formatBenefitTypeLabel(claim.jenis_manfaat)}</ThemedText>
          </View>
          <View style={styles.gridItem}>
            <ThemedText style={styles.gridLabel}>Service Date</ThemedText>
            <ThemedText style={styles.gridValue}>{formatClaimServiceDate(claim.tanggal_pelayanan)}</ThemedText>
          </View>
          <View style={styles.gridItemWide}>
            <ThemedText style={styles.gridLabel}>Facility</ThemedText>
            <ThemedText style={styles.gridValue}>{claim.fasilitas_kesehatan}</ThemedText>
          </View>
        </View>

        <View style={styles.footer}>
          <ThemedText style={styles.description} numberOfLines={2}>
            {claim.keterangan_penyakit}
          </ThemedText>
          {claim.source_scan_log_id && onOpenScan ? (
            <AppButton
              label="Open Related Scan"
              variant="outline"
              tone="primary"
              onPress={(event) => {
                event.stopPropagation();
                onOpenScan();
              }}
              style={styles.actionButton}
            />
          ) : null}
        </View>
      </SurfaceCard>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  actionButton: {
    marginTop: Spacing.md,
  },
  amount: {
    fontSize: 22,
    fontWeight: '800',
  },
  amountBlock: {
    flex: 1,
    gap: Spacing.xs,
  },
  badgeRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.sm,
  },
  card: {
    gap: Spacing.md,
  },
  description: {
    fontSize: 14,
    lineHeight: 20,
    opacity: Opacity.secondary,
  },
  footer: {
    marginTop: Spacing.xs,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.md,
  },
  gridItem: {
    flexBasis: '47%',
    gap: Spacing.xs,
  },
  gridItemWide: {
    gap: Spacing.xs,
    width: '100%',
  },
  gridLabel: {
    fontSize: 11,
    fontWeight: '700',
    opacity: Opacity.subtle,
    textTransform: 'uppercase',
  },
  gridValue: {
    fontSize: 14,
    fontWeight: '600',
  },
  metaText: {
    fontSize: 13,
    opacity: Opacity.subtle,
  },
  patientName: {
    fontSize: 17,
    fontWeight: '700',
  },
  section: {
    gap: Spacing.xs,
  },
  timestamp: {
    fontSize: 12,
    opacity: Opacity.subtle,
  },
  topRow: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: Spacing.md,
    justifyContent: 'space-between',
  },
});
