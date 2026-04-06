import { useCallback, useDeferredValue, useMemo, useState } from 'react';
import { FlatList, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { Href, router } from 'expo-router';

import { CenteredState } from '@/components/ui/centered-state';
import { SearchInput } from '@/components/ui/search-input';
import { Colors, Spacing } from '@/constants/theme';
import { AdminScanLogQueueCard } from '@/features/admin/components/admin-scan-log-queue-card';
import { QueueFilterChipRow } from '@/features/admin/components/queue-filter-chip-row';
import { QueueSummaryStrip } from '@/features/admin/components/queue-summary-strip';
import {
  isFailedScanStatus,
  summarizeScanLogs,
} from '@/features/admin/utils/admin-queue';
import { ThemedText } from '@/components/themed-text';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { getAdminScanLogs, getAdminScanLogsSummary } from '@/services/api/admin-scan-logs';
import type { AdminScanLogQueueSummary, AdminScanLogSummary, ApiError } from '@/services/api/types';

type ScanQueueFilter = 'all' | 'pending' | 'succeeded' | 'failed' | 'unlinked';
type ScanQueueRow =
  | { key: 'controls'; type: 'controls' }
  | { key: string; type: 'log'; log: AdminScanLogSummary };

const SCAN_QUEUE_FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'pending', label: 'Pending' },
  { key: 'succeeded', label: 'Succeeded' },
  { key: 'failed', label: 'Failed' },
  { key: 'unlinked', label: 'Unlinked' },
] as const;

function buildEmptyMessage(filter: ScanQueueFilter, query: string) {
  if (query.trim()) {
    return 'No scan logs matched this search.';
  }

  if (filter === 'pending') {
    return 'No pending scan logs need attention right now.';
  }

  if (filter === 'succeeded') {
    return 'No successful scan logs were found.';
  }

  if (filter === 'failed') {
    return 'No failed scan logs are waiting for review.';
  }

  if (filter === 'unlinked') {
    return 'No unlinked scan logs are waiting to be connected to a claim.';
  }

  return 'No scan logs have been recorded yet.';
}

export function AdminScanLogsScreen() {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const [logs, setLogs] = useState<AdminScanLogSummary[]>([]);
  const [summary, setSummary] = useState<AdminScanLogQueueSummary | null>(null);
  const [query, setQuery] = useState('');
  const deferredQuery = useDeferredValue(query);
  const [activeFilter, setActiveFilter] = useState<ScanQueueFilter>('all');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const loadLogs = useCallback(async (refresh: boolean = false) => {
    if (refresh) {
      setIsRefreshing(true);
    } else {
      setIsLoading(true);
    }
    setError('');

    const trimmedQuery = deferredQuery.trim();
    const statusFilter = activeFilter === 'pending' || activeFilter === 'succeeded' ? activeFilter : undefined;
    const linkedFilter = activeFilter === 'unlinked' ? false : undefined;

    try {
      const [logList, serverSummary] = await Promise.all([
        getAdminScanLogs({
          status: statusFilter,
          q: trimmedQuery || undefined,
          linked: linkedFilter,
          limit: 100,
        }),
        getAdminScanLogsSummary({
          status: statusFilter,
          q: trimmedQuery || undefined,
          linked: linkedFilter,
        }),
      ]);

      const visibleLogs = activeFilter === 'failed'
        ? logList.filter((log) => isFailedScanStatus(log.status))
        : logList;

      setLogs(visibleLogs);
      setSummary(activeFilter === 'failed' ? summarizeScanLogs(visibleLogs) : serverSummary);
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.message ?? 'Unable to load scan logs right now.');
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [activeFilter, deferredQuery]);

  useFocusEffect(
    useCallback(() => {
      void loadLogs();
    }, [loadLogs]),
  );

  const summaryItems = useMemo(
    () => [
      {
        key: 'total',
        label: 'Logs',
        value: `${summary?.total_logs ?? 0}`,
        tone: 'primary' as const,
      },
      {
        key: 'succeeded',
        label: 'Succeeded',
        value: `${summary?.succeeded_count ?? 0}`,
        tone: 'cta' as const,
      },
      {
        key: 'failed',
        label: 'Failed',
        value: `${summary?.failed_count ?? 0}`,
        tone: 'error' as const,
      },
      {
        key: 'unlinked',
        label: 'Unlinked',
        value: `${summary?.unlinked_count ?? 0}`,
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
            Scan Queue
          </ThemedText>
          <ThemedText style={styles.introSubtitle}>
            Review AI scan attempts and jump into the right claim faster.
          </ThemedText>
        </View>

        <View style={styles.summaryBlock}>
          <QueueSummaryStrip items={summaryItems} compact />
        </View>
      </View>
    ),
    [summaryItems],
  );

  const queueRows = useMemo<ScanQueueRow[]>(
    () => [
      { key: 'controls', type: 'controls' },
      ...logs.map((log) => ({
        key: `log-${log.id}`,
        type: 'log' as const,
        log,
      })),
    ],
    [logs],
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
            minHeight={260}
            actionLabel="Try Again"
            onAction={() => void loadLogs()}
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
                      placeholder="Search by scan ID, user, or email"
                      returnKeyType="search"
                    />
                    <QueueFilterChipRow
                      compact
                      chips={[...SCAN_QUEUE_FILTERS]}
                      selectedKey={activeFilter}
                      onSelect={(key) => setActiveFilter(key as ScanQueueFilter)}
                    />
                  </View>
                );
              }

              return (
                <AdminScanLogQueueCard
                  log={item.log}
                  onPress={() => router.push(`/admin/logs/${item.log.id}` as Href)}
                  onOpenClaim={
                    item.log.linked_claim_ids.length === 1
                      ? () => router.push(`/admin/claims/${item.log.linked_claim_ids[0]}` as Href)
                      : undefined
                  }
                />
              );
            }}
            contentContainerStyle={styles.list}
            ItemSeparatorComponent={() => <View style={styles.listSeparator} />}
            refreshing={isRefreshing}
            onRefresh={() => void loadLogs(true)}
            showsVerticalScrollIndicator={false}
            ListFooterComponent={
              logs.length === 0
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
    paddingBottom: Spacing.xxxl,
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
