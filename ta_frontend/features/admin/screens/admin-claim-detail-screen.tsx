import React, { useCallback, useMemo, useState } from 'react';
import { ActivityIndicator, Image, LayoutAnimation, Modal, Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { Href, useLocalSearchParams, router } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

import { ThemedText } from '@/components/themed-text';
import { Colors, getStatusColor, getStatusBadgeBg } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { getAdminClaim } from '@/services/api/admin-claims';
import type { AdminClaimDetail, AdminClaimFile, ApiError } from '@/services/api/types';

function formatCurrency(amount: number) {
  return new Intl.NumberFormat('id-ID', {
    style: 'currency',
    currency: 'IDR',
    maximumFractionDigits: 0,
  }).format(amount);
}

function formatDate(dateString: string) {
  const dmyMatch = dateString.match(/^(\d{2})-(\d{2})-(\d{4})$/);
  if (dmyMatch) {
    const [, day, month, year] = dmyMatch;
    const date = new Date(Number(year), Number(month) - 1, Number(day));
    return Number.isNaN(date.getTime()) ? dateString : date.toLocaleDateString('id-ID');
  }

  const date = new Date(dateString);
  return Number.isNaN(date.getTime()) ? dateString : date.toLocaleDateString('id-ID');
}

function formatTimestamp(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

function formatByteSize(byteSize: number) {
  if (byteSize < 1024) {
    return `${byteSize} B`;
  }

  const units = ['KB', 'MB', 'GB'];
  let value = byteSize / 1024;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}

type CollapsibleSectionProps = {
  title: string;
  children: React.ReactNode;
  initiallyExpanded?: boolean;
  rightElement?: React.ReactNode;
  cardBg?: string;
  borderColor?: string;
};

function CollapsibleSection({ title, children, initiallyExpanded = true, rightElement, cardBg, borderColor }: CollapsibleSectionProps) {
  const [isExpanded, setIsExpanded] = useState(initiallyExpanded);

  const toggleExpanded = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setIsExpanded(!isExpanded);
  };

  return (
    <View style={[styles.section, cardBg && { backgroundColor: cardBg }, borderColor && { borderColor }]}>
      <Pressable style={styles.sectionHeader} onPress={toggleExpanded}>
        <View style={styles.sectionHeaderLeft}>
          <ThemedText style={styles.sectionTitle}>{title}</ThemedText>
          {rightElement}
        </View>
        <Ionicons
          name="chevron-down"
          size={20}
          color="#718096"
          style={{ transform: [{ rotate: isExpanded ? '0deg' : '-90deg' }] }}
        />
      </Pressable>
      {isExpanded && <View style={styles.sectionContent}>{children}</View>}
    </View>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.detailRow}>
      <ThemedText style={styles.detailLabel}>{label}</ThemedText>
      <ThemedText style={styles.detailValue}>{value}</ThemedText>
    </View>
  );
}

function ClaimDocumentCard({
  file,
  colors,
  onPress,
}: {
  file: AdminClaimFile;
  colors: typeof Colors.light;
  onPress: () => void;
}) {
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);

  return (
    <View style={[styles.documentCard, { borderColor: colors.divider, backgroundColor: colors.cardBackground }]}>
      <ThemedText style={[styles.documentFilename, { color: colors.text }]}>{file.filename}</ThemedText>
      <Pressable onPress={onPress}>
        <Image
          source={{ uri: file.file_data_url }}
          style={styles.documentPreview}
          onLoad={(event) => {
            const { width, height } = event.nativeEvent.source;
            if (width && height) {
              setImageSize({ width, height });
            }
          }}
        />
      </Pressable>
      <View style={styles.documentMeta}>
        {imageSize ? (
          <ThemedText style={[styles.documentMetaText, { color: colors.text }]}>
            {imageSize.width} x {imageSize.height}
          </ThemedText>
        ) : null}
        <ThemedText style={[styles.documentMetaText, { color: colors.text }]}>
          {formatByteSize(file.byte_size)}
        </ThemedText>
        <ThemedText style={[styles.documentMetaText, { color: colors.text }]}>{file.mime_type}</ThemedText>
      </View>
    </View>
  );
}

export function AdminClaimDetailScreen() {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const params = useLocalSearchParams<{ claimId?: string }>();
  const [claim, setClaim] = useState<AdminClaimDetail | null>(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [viewerImageUri, setViewerImageUri] = useState<string | null>(null);
  const [viewerImageTitle, setViewerImageTitle] = useState('');

  const loadClaim = useCallback(async () => {
    if (!params.claimId) {
      setError('No claim ID provided.');
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      setClaim(await getAdminClaim(Number(params.claimId)));
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.message ?? 'Unable to load claim detail.');
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
      return (
        <View style={styles.centerState}>
          <ActivityIndicator color={colors.primary} />
        </View>
      );
    }

    if (error) {
      return (
        <View style={styles.centerState}>
          <ThemedText style={[styles.errorText, { color: colors.error }]}>{error}</ThemedText>
        </View>
      );
    }

    if (!claim) {
      return (
        <View style={styles.centerState}>
          <ThemedText>No claim data found.</ThemedText>
        </View>
      );
    }

    const statusColor = getStatusColor(claim.status, colors.text);
    const statusBg = getStatusBadgeBg(claim.status);
    const linkedScanLogId = claim.source_scan_log_id;

    return (
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <CollapsibleSection title="Patient Details" cardBg={colors.cardBackground} borderColor={colors.divider}>
          <DetailRow label="Nama Pasien" value={claim.nama_pasien} />
          <DetailRow label="Tanggal Pelayanan" value={formatDate(claim.tanggal_pelayanan)} />
          <DetailRow label="Fasilitas Kesehatan" value={claim.fasilitas_kesehatan} />
        </CollapsibleSection>

        <CollapsibleSection title="Claim Information" cardBg={colors.cardBackground} borderColor={colors.divider}>
          <DetailRow label="Jenis Manfaat" value={claim.jenis_manfaat} />
          <DetailRow label="Nominal Klaim" value={formatCurrency(claim.nominal_klaim)} />
          <DetailRow label="Keterangan Penyakit" value={claim.keterangan_penyakit} />
          <DetailRow label="Kronologi" value={claim.kronologi} />
        </CollapsibleSection>

        <CollapsibleSection title="Submission Info" cardBg={colors.cardBackground} borderColor={colors.divider}>
          <View style={[styles.statusRow, { backgroundColor: statusBg, borderColor: statusColor }]}>
            <ThemedText style={[styles.statusLabel, { color: statusColor }]}>Status</ThemedText>
            <ThemedText style={[styles.statusValue, { color: statusColor }]}>{claim.status}</ThemedText>
          </View>
          <DetailRow label="Submitted" value={formatTimestamp(claim.created_at)} />
          <DetailRow
            label="Linked Scan Log"
            value={linkedScanLogId ? `#${linkedScanLogId}` : 'Not linked'}
          />
          {linkedScanLogId ? (
            <Pressable
              style={[styles.linkButton, { borderColor: colors.primary }]}
              onPress={() => router.push(`/admin/logs/${linkedScanLogId}` as Href)}>
              <ThemedText style={[styles.linkButtonText, { color: colors.primary }]}>Open Related Scan</ThemedText>
            </Pressable>
          ) : null}
        </CollapsibleSection>

        <CollapsibleSection title="Documents" cardBg={colors.cardBackground} borderColor={colors.divider}>
          {claim.files.length > 0 ? (
            <View style={styles.documentList}>
              {claim.files.map((file) => (
                <ClaimDocumentCard
                  key={file.id}
                  file={file}
                  colors={colors}
                  onPress={() => {
                    setViewerImageUri(file.file_data_url);
                    setViewerImageTitle(file.filename);
                  }}
                />
              ))}
            </View>
          ) : (
            <ThemedText style={styles.placeholderText}>No documents were uploaded for this claim.</ThemedText>
          )}
        </CollapsibleSection>
      </ScrollView>
    );
  }, [claim, colors, error, isLoading]);

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { backgroundColor: colors.cardBackground, borderColor: colors.divider }]}>
        <Pressable style={styles.backButton} onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={18} color={colors.primary} />
          <ThemedText style={[styles.backButtonText, { color: colors.primary }]}>Back</ThemedText>
        </Pressable>
        {claim && (
          <View style={[styles.statusPill, { backgroundColor: getStatusBadgeBg(claim.status) }]}>
            <ThemedText style={[styles.statusPillText, { color: getStatusColor(claim.status, colors.text) }]}>
              {claim.status}
            </ThemedText>
          </View>
        )}
      </View>
      {content}

      <Modal animationType="fade" transparent visible={viewerImageUri !== null} onRequestClose={() => setViewerImageUri(null)}>
        <View style={styles.viewerOverlay}>
          <SafeAreaView style={styles.viewerContainer}>
            <Pressable style={styles.viewerCloseButton} onPress={() => setViewerImageUri(null)}>
              <Ionicons name="close" size={22} color="#FFFFFF" />
            </Pressable>
            <ThemedText style={styles.viewerTitle}>{viewerImageTitle}</ThemedText>
            <View style={styles.viewerStage}>
              {viewerImageUri ? <Image source={{ uri: viewerImageUri }} style={styles.viewerImage} /> : null}
            </View>
          </SafeAreaView>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  backButton: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: 4,
  },
  backButtonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  centerState: {
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center',
    minHeight: 240,
    paddingHorizontal: 24,
  },
  container: {
    flex: 1,
  },
  content: {
    padding: 20,
    paddingBottom: 32,
  },
  documentCard: {
    borderRadius: 12,
    borderWidth: 1,
    overflow: 'hidden',
    padding: 12,
  },
  documentFilename: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  documentList: {
    gap: 12,
  },
  documentMeta: {
    gap: 2,
    marginTop: 8,
  },
  documentMetaText: {
    fontSize: 12,
    opacity: 0.7,
  },
  documentPreview: {
    borderRadius: 10,
    height: 180,
    resizeMode: 'cover',
    width: '100%',
  },
  detailLabel: {
    fontSize: 14,
    opacity: 0.6,
    width: 140,
  },
  detailRow: {
    flexDirection: 'row',
    paddingVertical: 8,
  },
  detailValue: {
    flex: 1,
    fontSize: 14,
    fontWeight: '500',
  },
  errorText: {
    textAlign: 'center',
  },
  header: {
    alignItems: 'center',
    borderBottomWidth: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  placeholderText: {
    fontSize: 14,
    opacity: 0.6,
    fontStyle: 'italic',
  },
  linkButton: {
    alignItems: 'center',
    borderRadius: 10,
    borderWidth: 1,
    marginTop: 8,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  linkButtonText: {
    fontSize: 14,
    fontWeight: '700',
  },
  section: {
    backgroundColor: '#FFFFFF',
    borderColor: '#E2E8F0',
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: 16,
    overflow: 'hidden',
  },
  sectionContent: {
    paddingBottom: 16,
    paddingHorizontal: 16,
    paddingTop: 4,
  },
  sectionHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  sectionHeaderLeft: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '700',
  },
  statusLabel: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
  },
  statusPill: {
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  statusPillText: {
    fontSize: 14,
    fontWeight: '600',
  },
  statusRow: {
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  statusValue: {
    fontSize: 14,
    fontWeight: '700',
  },
  viewerCloseButton: {
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 22,
    height: 44,
    justifyContent: 'center',
    position: 'absolute',
    right: 16,
    top: 16,
    width: 44,
    zIndex: 10,
  },
  viewerContainer: {
    flex: 1,
  },
  viewerImage: {
    height: '100%',
    resizeMode: 'contain',
    width: '100%',
  },
  viewerOverlay: {
    backgroundColor: 'rgba(10, 16, 24, 0.96)',
    flex: 1,
  },
  viewerStage: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 16,
    paddingVertical: 24,
  },
  viewerTitle: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    paddingTop: 24,
    textAlign: 'center',
  },
});
