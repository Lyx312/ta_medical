import { useCallback, useDeferredValue, useMemo, useState } from 'react';
import { FlatList, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { Href, router } from 'expo-router';

import { CenteredState } from '@/components/ui/centered-state';
import { SearchInput } from '@/components/ui/search-input';
import { Colors, Spacing } from '@/constants/theme';
import { AdminClaimQueueCard } from '@/features/admin/components/admin-claim-queue-card';
import { QueueFilterChipRow } from '@/features/admin/components/queue-filter-chip-row';
import { QueueSummaryStrip } from '@/features/admin/components/queue-summary-strip';
import {
  isRecentTimestamp,
  summarizeClaims,
} from '@/features/admin/utils/admin-queue';
import { ThemedText } from '@/components/themed-text';
import { formatClaimCurrency } from '@/features/claims/utils/claim-display';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { getAdminClaims, getAdminClaimsSummary } from '@/services/api/admin-claims';
import type { AdminClaimSummary, ApiError, ClaimResponse } from '@/services/api/types';

type ClaimQueueFilter = 'all' | 'linked' | 'manual' | 'recent';
type ClaimQueueRow =
  | { key: 'controls'; type: 'controls' }
  | { key: string; type: 'claim'; claim: ClaimResponse };

const CLAIM_QUEUE_FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'linked', label: 'Linked Scan' },
  { key: 'manual', label: 'Manual Entry' },
  { key: 'recent', label: 'Recent' },
] as const;

function buildEmptyMessage(filter: ClaimQueueFilter, query: string) {
  if (query.trim()) {
    return 'No claims matched this search.';
  }

  if (filter === 'linked') {
    return 'No AI-linked claims are available right now.';
  }

  if (filter === 'manual') {
    return 'No manual claim submissions are available right now.';
  }

  if (filter === 'recent') {
    return 'No recent claims were submitted in the last 7 days.';
  }

  return 'No claims have been submitted yet.';
}

export function AdminClaimsScreen() {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const [claims, setClaims] = useState<ClaimResponse[]>([]);
  const [summary, setSummary] = useState<AdminClaimSummary | null>(null);
  const [query, setQuery] = useState('');
  const deferredQuery = useDeferredValue(query);
  const [activeFilter, setActiveFilter] = useState<ClaimQueueFilter>('all');
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

    const trimmedQuery = deferredQuery.trim();
    const linkedFilter = activeFilter === 'linked' ? true : activeFilter === 'manual' ? false : undefined;

    try {
      const [claimList, serverSummary] = await Promise.all([
        getAdminClaims({
          q: trimmedQuery || undefined,
          linked: linkedFilter,
          limit: 200,
          sort: 'newest',
        }),
        getAdminClaimsSummary({
          q: trimmedQuery || undefined,
          linked: linkedFilter,
        }),
      ]);

      const visibleClaims = activeFilter === 'recent'
        ? claimList.filter((claim) => isRecentTimestamp(claim.created_at))
        : claimList;

      setClaims(visibleClaims);
      setSummary(activeFilter === 'recent' ? summarizeClaims(visibleClaims) : serverSummary);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.message ?? 'Unable to load claims right now.');
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [activeFilter, deferredQuery]);

  useFocusEffect(
    useCallback(() => {
      void loadClaims();
    }, [loadClaims]),
  );

  const summaryItems = useMemo(
    () => [
      {
        key: 'total',
        label: 'Claims',
        value: `${summary?.total_claims ?? 0}`,
        tone: 'primary' as const,
      },
      {
        key: 'amount',
        label: 'Claim Amount',
        value: formatClaimCurrency(summary?.total_claim_amount ?? 0),
        tone: 'secondary' as const,
      },
      {
        key: 'linked',
        label: 'Linked Scan',
        value: `${summary?.linked_scan_count ?? 0}`,
        tone: 'cta' as const,
      },
      {
        key: 'manual',
        label: 'Manual Entry',
        value: `${summary?.manual_submission_count ?? 0}`,
        tone: 'warning' as const,
      },
    ],
    [summary],
  );

  const headerContent = useMemo(
    () => (
      <View>
        <View style={styles.introBlock}>
          <ThemedText type="title" style={styles.introTitle}>
            Claims Queue
          </ThemedText>
          <ThemedText style={styles.introSubtitle}>
            Triage claims quickly by source, recency, and claimant context.
          </ThemedText>
        </View>

        <View style={styles.summaryBlock}>
          <QueueSummaryStrip items={summaryItems} compact />
        </View>
      </View>
    ),
    [summaryItems],
  );

  const queueRows = useMemo<ClaimQueueRow[]>(
    () => [
      { key: 'controls', type: 'controls' },
      ...claims.map((claim) => ({
        key: `claim-${claim.id}`,
        type: 'claim' as const,
        claim,
      })),
    ],
    [claims],
  );

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={styles.content}>
        {isLoading ? (
          <CenteredState loading minHeight={260} />
        ) : error ? (
          <CenteredState
            message={error}
            tone="error"
            actionLabel="Try Again"
            onAction={() => void loadClaims()}
          />
        ) : (
          <FlatList
            data={queueRows}
            ListHeaderComponent={headerContent}
            stickyHeaderIndices={[1]}
            keyExtractor={(item) => item.key}
            renderItem={({ item }) => {
              if (item.type === 'controls') {
                return (
                  <View style={[styles.controlsBlock, styles.stickyControls, { backgroundColor: colors.background }]}>
                    <SearchInput
                      compact
                      value={query}
                      onChangeText={setQuery}
                      onClear={() => setQuery('')}
                      placeholder="Search by claim ID, patient, or email"
                      returnKeyType="search"
                    />
                    <QueueFilterChipRow
                      compact
                      chips={[...CLAIM_QUEUE_FILTERS]}
                      selectedKey={activeFilter}
                      onSelect={(key) => setActiveFilter(key as ClaimQueueFilter)}
                    />
                  </View>
                );
              }

              return (
                <AdminClaimQueueCard
                  claim={item.claim}
                  onPress={() => router.push(`/admin/claims/${item.claim.id}` as Href)}
                  onOpenScan={
                    item.claim.source_scan_log_id
                      ? () => router.push(`/admin/logs/${item.claim.source_scan_log_id}` as Href)
                      : undefined
                  }
                />
              );
            }}
            contentContainerStyle={styles.list}
            ItemSeparatorComponent={() => <View style={styles.listSeparator} />}
            refreshing={isRefreshing}
            onRefresh={() => void loadClaims(true)}
            showsVerticalScrollIndicator={false}
            ListFooterComponent={
              claims.length === 0
                ? <CenteredState message={buildEmptyMessage(activeFilter, deferredQuery)} minHeight={260} />
                : null
            }
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
    paddingHorizontal: Spacing.xl,
    paddingTop: Spacing.lg,
  },
  controlsBlock: {
    gap: Spacing.sm,
    marginBottom: Spacing.md,
    paddingBottom: Spacing.sm,
  },
  introBlock: {
    gap: Spacing.xs,
    marginBottom: Spacing.sm,
  },
  introSubtitle: {
    fontSize: 14,
    opacity: 0.72,
  },
  introTitle: {
    fontSize: 28,
    lineHeight: 30,
  },
  list: {
    flexGrow: 1,
    paddingBottom: Spacing.lg,
  },
  listSeparator: {
    height: Spacing.md,
  },
  stickyControls: {
    zIndex: 1,
  },
  summaryBlock: {
    marginBottom: Spacing.sm,
  },
});
