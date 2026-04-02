import { useCallback, useState } from 'react';
import { ActivityIndicator, FlatList, Pressable, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { Href, router } from 'expo-router';

import { ThemedText } from '@/components/themed-text';
import { Colors, getStatusColor, getStatusBadgeBg } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { getAdminClaims } from '@/services/api/admin-claims';
import type { ApiError, ClaimResponse } from '@/services/api/types';

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

export function AdminClaimsScreen() {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const [claims, setClaims] = useState<ClaimResponse[]>([]);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const loadClaims = useCallback(async (refresh: boolean = false) => {
    setError('');

    if (refresh) {
      setIsRefreshing(true);
    } else {
      setIsLoading(true);
    }

    try {
      setClaims(await getAdminClaims());
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.message ?? 'Unable to load claims right now.');
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void loadClaims();
    }, [loadClaims]),
  );

  const renderItem = ({ item }: { item: ClaimResponse }) => {
    const statusColor = getStatusColor(item.status, colors.text);
    const statusBg = getStatusBadgeBg(item.status);

    return (
      <Pressable
        style={[styles.card, { backgroundColor: colors.cardBackground, borderColor: colors.divider }]}
        onPress={() => router.push(`/admin/claims/${item.id}` as Href)}>
        <View style={styles.row}>
          <ThemedText style={styles.amount}>{formatCurrency(item.nominal_klaim)}</ThemedText>
          <View style={[styles.statusPill, { backgroundColor: statusBg }]}>
            <ThemedText style={[styles.status, { color: statusColor }]}>{item.status}</ThemedText>
          </View>
        </View>
        <ThemedText style={styles.patientName}>{item.nama_pasien}</ThemedText>
        <ThemedText style={styles.description}>{item.keterangan_penyakit}</ThemedText>
        <ThemedText style={styles.date}>{formatDate(item.tanggal_pelayanan)}</ThemedText>
        <View style={styles.actionRow}>
          <ThemedText style={[styles.metaText, { color: colors.text }]}>{item.user_email}</ThemedText>
          {item.source_scan_log_id ? (
            <Pressable
              onPress={(event) => {
                event.stopPropagation();
                router.push(`/admin/logs/${item.source_scan_log_id}` as Href);
              }}
              style={[styles.linkButton, { borderColor: colors.primary }]}>
              <ThemedText style={[styles.linkButtonText, { color: colors.primary }]}>View Scan</ThemedText>
            </Pressable>
          ) : null}
        </View>
      </Pressable>
    );
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={styles.content}>
        <ThemedText type="title">Claims</ThemedText>
        <ThemedText style={styles.subtitle}>View and manage all submitted claims</ThemedText>

        {isLoading ? (
          <View style={styles.centerState}>
            <ActivityIndicator color={colors.primary} />
          </View>
        ) : error ? (
          <View style={styles.centerState}>
            <ThemedText style={[styles.errorText, { color: colors.error }]}>{error}</ThemedText>
            <Pressable style={[styles.retryButton, { backgroundColor: colors.primary }]} onPress={() => void loadClaims()}>
              <ThemedText style={[styles.retryLabel, { color: colors.cardBackground }]}>Try Again</ThemedText>
            </Pressable>
          </View>
        ) : (
          <FlatList
            data={claims}
            keyExtractor={(item) => item.id.toString()}
            renderItem={renderItem}
            contentContainerStyle={claims.length === 0 ? styles.emptyList : styles.list}
            refreshing={isRefreshing}
            onRefresh={() => void loadClaims(true)}
            showsVerticalScrollIndicator={false}
            ListEmptyComponent={
              <View style={styles.centerState}>
                <ThemedText style={styles.emptyText}>No claims have been submitted yet.</ThemedText>
              </View>
            }
          />
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  amount: {
    fontSize: 18,
    fontWeight: '700',
  },
  card: {
    borderRadius: 12,
    borderWidth: 1,
    padding: 16,
  },
  centerState: {
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  date: {
    fontSize: 12,
    marginTop: 4,
    opacity: 0.6,
  },
  description: {
    fontSize: 16,
    marginTop: 8,
    opacity: 0.8,
  },
  emptyList: {
    flexGrow: 1,
  },
  emptyText: {
    opacity: 0.75,
    textAlign: 'center',
  },
  errorText: {
    textAlign: 'center',
  },
  list: {
    gap: 12,
    marginTop: 20,
    paddingBottom: 12,
  },
  linkButton: {
    borderRadius: 999,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  linkButtonText: {
    fontSize: 12,
    fontWeight: '700',
  },
  metaText: {
    flex: 1,
    fontSize: 12,
    opacity: 0.7,
  },
  patientName: {
    fontSize: 15,
    fontWeight: '600',
    marginTop: 8,
  },
  retryButton: {
    borderRadius: 12,
    marginTop: 16,
    paddingHorizontal: 18,
    paddingVertical: 12,
  },
  retryLabel: {
    fontWeight: '600',
  },
  row: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  actionRow: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: 12,
    marginTop: 12,
  },
  status: {
    fontSize: 14,
    fontWeight: '600',
  },
  statusPill: {
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  subtitle: {
    marginTop: 8,
    opacity: 0.7,
  },
});
