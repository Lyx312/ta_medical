import { StyleSheet, View, ActivityIndicator, Pressable } from 'react-native';

import { SurfaceCard } from '@/components/ui/surface-card';
import { ThemedText } from '@/components/themed-text';
import { Colors, Opacity, Radii, Spacing, Typography } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import type { ScanProgressPayload } from '@/services/api/types';

type ClaimScanProgressCardProps = {
  progress: ScanProgressPayload;
  onCancel?: () => Promise<void>;
  isCancelling?: boolean;
};

const STAGE_ORDER: ScanProgressPayload['stage'][] = [
  'queued',
  'preprocessing',
  'validating',
  'extracting',
  'fusing',
  'prioritizing_medicines',
  'completed',
];

const STAGE_LABELS: Record<ScanProgressPayload['stage'], string> = {
  queued: 'Uploading files',
  preprocessing: 'Checking documents',
  validating: 'Checking documents',
  extracting: 'Extracting medical data',
  fusing: 'Combining AI results',
  prioritizing_medicines: 'Prioritizing medicines',
  completed: 'Ready to review',
  failed: 'Scan failed',
};

export function ClaimScanProgressCard({ progress, onCancel, isCancelling }: ClaimScanProgressCardProps) {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const activeStageIndex = progress.stage === 'failed' ? STAGE_ORDER.length - 1 : STAGE_ORDER.indexOf(progress.stage);
  const isProcessing = progress.status === 'running' || progress.status === 'pending';
  const canCancel = isProcessing && onCancel && !isCancelling;

  return (
    <SurfaceCard style={styles.card}>
      <View style={styles.header}>
        <View style={{ flex: 1 }}>
          <ThemedText style={[styles.title, { color: colors.primary }]}>{progress.title}</ThemedText>
        </View>
        <ThemedText style={styles.percent}>{progress.percent}%</ThemedText>
        {canCancel ? (
          <Pressable
            onPress={onCancel}
            style={[styles.cancelButton, { borderColor: colors.error }]}
            disabled={isCancelling}
          >
            {isCancelling ? (
              <ActivityIndicator size="small" color={colors.error} />
            ) : (
              <ThemedText style={[styles.cancelButtonText, { color: colors.error }]}>Cancel</ThemedText>
            )}
          </Pressable>
        ) : null}
      </View>
      {progress.detail ? <ThemedText style={styles.detail}>{progress.detail}</ThemedText> : null}

      <View style={[styles.progressTrack, { backgroundColor: colors.divider }]}>
        <View
          style={[
            styles.progressFill,
            {
              width: `${Math.max(0, Math.min(progress.percent, 100))}%`,
              backgroundColor: progress.status === 'failed' ? colors.error : colors.secondary,
            },
          ]}
        />
      </View>

      <View style={styles.timeline}>
        {STAGE_ORDER.map((stage, index) => {
          const isDone = progress.status !== 'failed' && index < activeStageIndex;
          const isActive = progress.stage === stage;
          return (
            <View key={stage} style={styles.timelineRow}>
              <View
                style={[
                  styles.timelineDot,
                  {
                    backgroundColor: isDone || isActive ? colors.secondary : colors.divider,
                    borderColor: isActive ? colors.primary : 'transparent',
                  },
                ]}
              />
              <ThemedText
                style={[
                  styles.timelineText,
                  {
                    color: isActive ? colors.primary : colors.text,
                    opacity: isDone || isActive ? 1 : Opacity.subtle,
                  },
                ]}
              >
                {STAGE_LABELS[stage]}
              </ThemedText>
            </View>
          );
        })}
      </View>
    </SurfaceCard>
  );
}

const styles = StyleSheet.create({
  card: {
    gap: Spacing.md,
  },
  cancelButton: {
    borderRadius: Radii.sm,
    borderWidth: 1,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cancelButtonText: {
    ...Typography.meta,
    fontWeight: '600',
  },
  detail: {
    ...Typography.helper,
  },
  header: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: Spacing.md,
  },
  percent: {
    ...Typography.bodyStrong,
  },
  progressFill: {
    borderRadius: Radii.pill,
    height: '100%',
  },
  progressTrack: {
    borderRadius: Radii.pill,
    height: 8,
    overflow: 'hidden',
  },
  timeline: {
    gap: Spacing.sm,
  },
  timelineDot: {
    borderRadius: Radii.pill,
    borderWidth: 1,
    height: 10,
    width: 10,
  },
  timelineRow: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: Spacing.sm,
  },
  timelineText: {
    ...Typography.meta,
    flex: 1,
  },
  title: {
    ...Typography.sectionTitle,
  },
});
