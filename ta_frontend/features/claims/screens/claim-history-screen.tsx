import { useCallback, useState } from 'react';
import { ActivityIndicator, FlatList, Pressable, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';

import { ThemedText } from '@/components/themed-text';
import { Colors } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { getClaims } from '@/services/api/claims';
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

function getStatusColor(status: string, colors: typeof Colors.light) {
  const normalizedStatus = status.toLowerCase();

  if (normalizedStatus.includes('approve')) {
    return '#22c55e';
  }

  if (normalizedStatus.includes('pending') || normalizedStatus.includes('review')) {
    return '#eab308';
  }

  if (normalizedStatus.includes('reject')) {
    return colors.error;
  }

  return colors.text;
}

export function ClaimHistoryScreen() {
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
      setClaims(await getClaims());
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.message ?? 'Unable to load claim history right now.');
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

  const renderItem = ({ item }: { item: ClaimResponse }) => (
    <View style={[styles.card, { backgroundColor: colors.cardBackground, borderColor: colors.divider }]}>
      <View style={styles.row}>
        <ThemedText style={styles.amount}>{formatCurrency(item.nominal_klaim)}</ThemedText>
        <ThemedText style={[styles.status, { color: getStatusColor(item.status, colors) }]}>{item.status}</ThemedText>
      </View>
      <ThemedText style={styles.patientName}>{item.nama_pasien}</ThemedText>
      <ThemedText style={styles.description}>{item.keterangan_penyakit}</ThemedText>
      <ThemedText style={styles.date}>{formatDate(item.tanggal_pelayanan)}</ThemedText>
    </View>
  );

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={styles.content}>
        <ThemedText type="title">Claim History</ThemedText>
        <ThemedText style={styles.subtitle}>View your past claims</ThemedText>

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
  status: {
    fontSize: 14,
    fontWeight: '600',
  },
  subtitle: {
    marginTop: 8,
    opacity: 0.7,
  },
});
