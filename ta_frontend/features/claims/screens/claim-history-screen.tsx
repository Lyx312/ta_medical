import { useCallback, useState } from 'react';
import { FlatList, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { Href, router } from 'expo-router';

import { CenteredState } from '@/components/ui/centered-state';
import { ScreenHeader } from '@/components/ui/screen-header';
import { ClaimSummaryCard } from '@/features/claims/components/claim-summary-card';
import { Colors, Spacing } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import {
  formatBenefitTypeLabel,
  formatClaimSourceLabel,
} from '@/features/claims/utils/claim-display';
import { getClaims } from '@/services/api/claims';
import type { ApiError, ClaimResponse } from '@/services/api/types';

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

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={styles.content}>
        <ScreenHeader title="Claim History" subtitle="View your past claims" />

        {isLoading ? (
          <CenteredState loading minHeight={220} />
        ) : error ? (
          <CenteredState
            message={error}
            tone="error"
            actionLabel="Try Again"
            onAction={() => void loadClaims()}
          />
        ) : (
          <FlatList
            data={claims}
            keyExtractor={(item) => item.id.toString()}
            renderItem={({ item }) => (
              <ClaimSummaryCard
                claim={item}
                statusVariant="text"
                showPressHint
                onPress={() => router.push(`/user/claims/${item.id}` as Href)}
                metadata={[
                  { label: 'Facility', value: item.fasilitas_kesehatan },
                  { label: 'Benefit', value: formatBenefitTypeLabel(item.jenis_manfaat) },
                  { label: 'Files', value: `${item.file_count} uploaded` },
                  { label: 'Source', value: formatClaimSourceLabel(item.source_scan_log_id) },
                ]}
              />
            )}
            contentContainerStyle={claims.length === 0 ? styles.emptyList : styles.list}
            refreshing={isRefreshing}
            onRefresh={() => void loadClaims(true)}
            showsVerticalScrollIndicator={false}
            ListEmptyComponent={<CenteredState message="No claims have been submitted yet." minHeight={220} />}
          />
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
    padding: Spacing.xl,
  },
  emptyList: {
    flexGrow: 1,
  },
  list: {
    gap: Spacing.md,
    marginTop: Spacing.xl,
    paddingBottom: Spacing.md,
  },
});
