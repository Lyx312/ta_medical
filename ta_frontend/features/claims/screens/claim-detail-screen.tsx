import { useCallback, useMemo, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { router, useLocalSearchParams } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

import { CenteredState } from '@/components/ui/centered-state';
import { ScreenHeader } from '@/components/ui/screen-header';
import { StatusBadge } from '@/components/ui/status-badge';
import { SurfaceCard } from '@/components/ui/surface-card';
import { ThemedText } from '@/components/themed-text';
import { Colors, Opacity, Spacing, Typography } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import {
  formatBenefitTypeLabel,
  formatClaimCurrency,
  formatClaimServiceDate,
  formatClaimSourceLabel,
  formatClaimTimestamp,
  getClaimStatusPresentation,
} from '@/features/claims/utils/claim-display';
import { getClaim } from '@/services/api/claims';
import type { ApiError, ClaimDetailResponse } from '@/services/api/types';

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.detailRow}>
      <ThemedText style={styles.detailLabel}>{label}</ThemedText>
      <ThemedText style={styles.detailValue}>{value}</ThemedText>
    </View>
  );
}

function formatMedicineItemValue(claimMedicine: ClaimDetailResponse['medicines'][number]) {
  const details: string[] = [];
  if (claimMedicine.quantity !== undefined) {
    details.push(`Qty ${claimMedicine.quantity}${claimMedicine.unit ? ` ${claimMedicine.unit}` : ''}`);
  } else if (claimMedicine.unit) {
    details.push(claimMedicine.unit);
  }
  if (claimMedicine.dosage_instruction) {
    details.push(claimMedicine.dosage_instruction);
  }

  const evaluation = claimMedicine.evaluation;
  if (!evaluation) {
    return [claimMedicine.name, details.join(' | ')].filter((item) => item).join('\n');
  }

  const relevanceLabel =
    evaluation.relevance === 'not_relevant'
      ? 'Not Relevant'
      : evaluation.relevance === 'uncertain'
        ? 'Uncertain'
        : 'Relevant';

  const confidencePct = Math.round(evaluation.confidence * 100);
  const evaluationLine = `Relevance: ${relevanceLabel} (${confidencePct}%)`;

  return [claimMedicine.name, details.join(' | '), evaluationLine, evaluation.reason]
    .filter((item) => item)
    .join('\n');
}

export function ClaimDetailScreen() {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const params = useLocalSearchParams<{ claimId?: string }>();
  const [claim, setClaim] = useState<ClaimDetailResponse | null>(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  const loadClaim = useCallback(async () => {
    if (!params.claimId) {
      setError('No claim ID provided.');
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      setClaim(await getClaim(Number(params.claimId)));
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.message ?? 'Unable to load this claim.');
    } finally {
      setIsLoading(false);
    }
  }, [params.claimId]);

  useFocusEffect(
    useCallback(() => {
      void loadClaim();
    }, [loadClaim]),
  );

  const content = useMemo(() => {
    if (isLoading) {
      return <CenteredState loading minHeight={260} />;
    }

    if (error) {
      return (
        <CenteredState
          message={error}
          tone="error"
          minHeight={260}
          actionLabel="Try Again"
          onAction={() => void loadClaim()}
        />
      );
    }

    if (!claim) {
      return <CenteredState message="Claim data was not found." minHeight={260} />;
    }

    const status = getClaimStatusPresentation(claim.status, colors.text);

    return (
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ScreenHeader
          title={`Claim #${claim.id}`}
          subtitle="Review your submitted claim details and current status."
        />

        <SurfaceCard style={styles.heroCard}>
          <View style={styles.heroTopRow}>
            <View style={styles.heroTextBlock}>
              <ThemedText style={styles.heroLabel}>Claim Amount</ThemedText>
              <ThemedText style={styles.heroAmount}>{formatClaimCurrency(claim.nominal_klaim)}</ThemedText>
            </View>
            <StatusBadge
              label={claim.status}
              backgroundColor={status.backgroundColor}
              textColor={status.textColor}
            />
          </View>
          <ThemedText style={styles.heroPatient}>{claim.nama_pasien}</ThemedText>
          <ThemedText style={styles.heroMeta}>
            {formatClaimServiceDate(claim.tanggal_pelayanan)} | {formatBenefitTypeLabel(claim.jenis_manfaat)}
          </ThemedText>
        </SurfaceCard>

        <SurfaceCard style={styles.sectionCard}>
          <ThemedText style={styles.sectionTitle}>Patient & Service</ThemedText>
          <DetailRow label="Patient" value={claim.nama_pasien} />
          <DetailRow label="Service Date" value={formatClaimServiceDate(claim.tanggal_pelayanan)} />
          <DetailRow label="Facility" value={claim.fasilitas_kesehatan} />
        </SurfaceCard>

        <SurfaceCard style={styles.sectionCard}>
          <ThemedText style={styles.sectionTitle}>Claim Information</ThemedText>
          <DetailRow label="Benefit Type" value={formatBenefitTypeLabel(claim.jenis_manfaat)} />
          <DetailRow label="Diagnosis" value={claim.keterangan_penyakit} />
          {claim.medicines.length > 0
            ? claim.medicines.map((medicine, index) => (
                <DetailRow
                  key={`${medicine.name}-${index}`}
                  label={`Medicine #${index + 1}`}
                  value={formatMedicineItemValue(medicine)}
                />
              ))
            : null}
          <DetailRow label="Chronology" value={claim.kronologi} />
        </SurfaceCard>

        <SurfaceCard style={styles.sectionCard}>
          <ThemedText style={styles.sectionTitle}>Submission Info</ThemedText>
          <DetailRow label="Submitted At" value={formatClaimTimestamp(claim.created_at)} />
          <DetailRow label="Files Uploaded" value={`${claim.file_count}`} />
          <DetailRow label="Source" value={formatClaimSourceLabel(claim.source_scan_log_id)} />
          <DetailRow label="Additional Review" value={claim.requires_additional_review ? 'Required' : 'Not required'} />
          {claim.review_note ? <DetailRow label="Review Note" value={claim.review_note} /> : null}
        </SurfaceCard>
      </ScrollView>
    );
  }, [claim, colors.text, error, isLoading, loadClaim]);

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { backgroundColor: colors.background }]}>
        <Pressable style={styles.backButton} onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={18} color={colors.primary} />
          <ThemedText style={[styles.backButtonText, { color: colors.primary }]}>Back</ThemedText>
        </Pressable>
      </View>
      {content}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  backButton: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: Spacing.xs,
  },
  backButtonText: {
    ...Typography.button,
    fontSize: 15,
  },
  container: {
    flex: 1,
  },
  content: {
    gap: Spacing.md,
    padding: Spacing.xl,
    paddingBottom: Spacing.xxxl,
  },
  detailLabel: {
    ...Typography.meta,
    flex: 0.95,
  },
  detailRow: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: Spacing.md,
    paddingTop: Spacing.sm,
  },
  detailValue: {
    flex: 1.45,
    fontSize: 14,
    lineHeight: 20,
  },
  header: {
    paddingHorizontal: Spacing.xl,
    paddingTop: Spacing.sm,
  },
  heroAmount: {
    fontSize: 24,
    fontWeight: '700',
    lineHeight: 28,
  },
  heroCard: {
    gap: Spacing.sm,
  },
  heroLabel: {
    ...Typography.meta,
    textTransform: 'uppercase',
  },
  heroMeta: {
    ...Typography.helper,
    opacity: Opacity.subtle,
  },
  heroPatient: {
    ...Typography.bodyStrong,
  },
  heroTextBlock: {
    flex: 1,
    gap: Spacing.xs,
  },
  heroTopRow: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: Spacing.md,
    justifyContent: 'space-between',
  },
  sectionCard: {
    gap: Spacing.xs,
  },
  sectionTitle: {
    ...Typography.sectionTitle,
    marginBottom: Spacing.xs,
  },
});
