import { Pressable, StyleSheet, View } from 'react-native';

import { AppButton } from '@/components/ui/app-button';
import { StatusBadge } from '@/components/ui/status-badge';
import { SurfaceCard } from '@/components/ui/surface-card';
import { ThemedText } from '@/components/themed-text';
import { Colors, Opacity, Spacing } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import type { AdminScanLogSummary } from '@/services/api/types';
import {
  formatAdminTimestamp,
  formatScanStatusLabel,
  getScanLinkPresentation,
  getScanStatusPresentation,
  isFailedScanStatus,
} from '@/features/admin/utils/admin-queue';

export type AdminScanLogQueueCardProps = {
  log: AdminScanLogSummary;
  onPress: () => void;
  onOpenClaim?: () => void;
};

export function AdminScanLogQueueCard({ log, onPress, onOpenClaim }: AdminScanLogQueueCardProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const status = getScanStatusPresentation(log.status, colors);
  const linkState = getScanLinkPresentation(log, colors);
  const needsReview = isFailedScanStatus(log.status) || log.linked_claim_ids.length === 0;

  return (
    <Pressable onPress={onPress}>
      <SurfaceCard
        style={[
          styles.card,
          needsReview && {
            borderColor: isFailedScanStatus(log.status) ? colors.error : colors.warning,
            borderLeftWidth: 4,
          },
        ]}>
        <View style={styles.topRow}>
          <View style={styles.titleBlock}>
            <ThemedText style={styles.title}>Scan #{log.id}</ThemedText>
            <ThemedText style={styles.metaText}>{formatAdminTimestamp(log.created_at)}</ThemedText>
          </View>
          <StatusBadge
            label={formatScanStatusLabel(log.status)}
            backgroundColor={status.backgroundColor}
            textColor={status.textColor}
          />
        </View>

        <View style={styles.badgeRow}>
          <StatusBadge
            label={linkState.label}
            backgroundColor={linkState.backgroundColor}
            textColor={linkState.textColor}
          />
          {needsReview ? (
            <StatusBadge
              label="Needs Review"
              backgroundColor={`${colors.warning}20`}
              textColor={colors.warning}
            />
          ) : null}
        </View>

        <View style={styles.section}>
          <ThemedText style={styles.name}>{log.user_full_name}</ThemedText>
          <ThemedText style={styles.metaText}>{log.user_email}</ThemedText>
        </View>

        <View style={styles.grid}>
          <View style={styles.gridItem}>
            <ThemedText style={styles.gridLabel}>Files</ThemedText>
            <ThemedText style={styles.gridValue}>{log.file_count}</ThemedText>
          </View>
          <View style={styles.gridItem}>
            <ThemedText style={styles.gridLabel}>Linked Claims</ThemedText>
            <ThemedText style={styles.gridValue}>
              {log.linked_claim_ids.length > 0 ? log.linked_claim_ids.join(', ') : 'Not linked'}
            </ThemedText>
          </View>
          <View style={styles.gridItemWide}>
            <ThemedText style={styles.gridLabel}>Model</ThemedText>
            <ThemedText style={styles.gridValue}>
              {log.ai_model_name ? `${log.ai_model_provider}/${log.ai_model_name}` : 'Unavailable'}
            </ThemedText>
          </View>
        </View>

        {log.linked_claim_ids.length === 1 && onOpenClaim ? (
          <AppButton
            label={`Open Claim #${log.linked_claim_ids[0]}`}
            variant="outline"
            tone="primary"
            onPress={(event) => {
              event.stopPropagation();
              onOpenClaim();
            }}
            style={styles.actionButton}
          />
        ) : null}
      </SurfaceCard>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  actionButton: {
    marginTop: Spacing.sm,
  },
  badgeRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.sm,
  },
  card: {
    gap: Spacing.md,
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
  name: {
    fontSize: 17,
    fontWeight: '700',
  },
  section: {
    gap: Spacing.xs,
  },
  title: {
    fontSize: 19,
    fontWeight: '800',
  },
  titleBlock: {
    flex: 1,
    gap: Spacing.xs,
  },
  topRow: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: Spacing.md,
    justifyContent: 'space-between',
  },
});
