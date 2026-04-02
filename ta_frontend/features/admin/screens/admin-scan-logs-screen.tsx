import { useCallback, useMemo, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { Href, router } from 'expo-router';

import { ThemedText } from '@/components/themed-text';
import { Colors } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { getAdminScanLogs } from '@/services/api/admin-scan-logs';
import type { AdminScanLogSummary, ApiError } from '@/services/api/types';

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

function formatStatusLabel(status: string) {
  return status.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function AdminScanLogsScreen() {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const [logs, setLogs] = useState<AdminScanLogSummary[]>([]);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  const loadLogs = useCallback(async () => {
    setIsLoading(true);
    setError('');

    try {
      setLogs(await getAdminScanLogs());
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.message ?? 'Unable to load scan logs right now.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void loadLogs();
    }, [loadLogs]),
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

    if (logs.length === 0) {
      return (
        <View style={styles.centerState}>
          <ThemedText style={styles.emptyText}>No scan logs have been recorded yet.</ThemedText>
        </View>
      );
    }

    return (
      <View style={styles.logList}>
        {logs.map((log) => {
          const isLinked = log.linked_claim_ids.length > 0;

          return (
            <Pressable
              key={log.id}
              style={[styles.logCard, { backgroundColor: colors.cardBackground, borderColor: colors.divider }]}
              onPress={() => router.push(`/admin/logs/${log.id}` as Href)}>
              <View style={styles.rowBetween}>
                <View style={styles.titleBlock}>
                  <ThemedText style={styles.logTitle}>Scan #{log.id}</ThemedText>
                  <ThemedText style={styles.metaText}>{log.user_full_name}</ThemedText>
                  <ThemedText style={styles.metaText}>{log.user_email}</ThemedText>
                </View>
                <View
                  style={[
                    styles.statusPill,
                    {
                      backgroundColor: log.status === 'succeeded' ? `${colors.cta}20` : `${colors.warning}20`,
                    },
                  ]}>
                  <ThemedText
                    style={[
                      styles.statusText,
                      { color: log.status === 'succeeded' ? colors.cta : colors.warning },
                    ]}>
                    {formatStatusLabel(log.status)}
                  </ThemedText>
                </View>
              </View>

              <View style={styles.infoGrid}>
                <View style={styles.infoItem}>
                  <ThemedText style={styles.infoLabel}>Created</ThemedText>
                  <ThemedText style={styles.infoValue}>{formatTimestamp(log.created_at)}</ThemedText>
                </View>
                <View style={styles.infoItem}>
                  <ThemedText style={styles.infoLabel}>Files</ThemedText>
                  <ThemedText style={styles.infoValue}>{log.file_count}</ThemedText>
                </View>
                <View style={styles.infoItem}>
                  <ThemedText style={styles.infoLabel}>Claims</ThemedText>
                  <ThemedText style={styles.infoValue}>
                    {isLinked ? log.linked_claim_ids.join(', ') : 'Not linked'}
                  </ThemedText>
                </View>
                <View style={styles.infoItem}>
                  <ThemedText style={styles.infoLabel}>Model</ThemedText>
                  <ThemedText style={styles.infoValue}>
                    {log.ai_model_name ? `${log.ai_model_provider}/${log.ai_model_name}` : 'Unavailable'}
                  </ThemedText>
                </View>
              </View>
            </Pressable>
          );
        })}
      </View>
    );
  }, [colors.cardBackground, colors.cta, colors.divider, colors.error, colors.primary, colors.warning, error, isLoading, logs]);

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ThemedText type="title">Scan Logs</ThemedText>
        <ThemedText style={styles.subtitle}>
          Review every AI document scan attempt, including failures and scans that were never submitted as claims.
        </ThemedText>
        {content}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  centerState: {
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 220,
    paddingHorizontal: 24,
  },
  container: {
    flex: 1,
  },
  content: {
    padding: 20,
    paddingBottom: 32,
  },
  emptyText: {
    textAlign: 'center',
  },
  errorText: {
    textAlign: 'center',
  },
  infoGrid: {
    gap: 12,
  },
  infoItem: {
    gap: 4,
  },
  infoLabel: {
    fontSize: 12,
    fontWeight: '700',
    opacity: 0.75,
    textTransform: 'uppercase',
  },
  infoValue: {
    fontSize: 14,
  },
  logCard: {
    borderRadius: 16,
    borderWidth: 1,
    gap: 16,
    padding: 16,
  },
  logList: {
    gap: 16,
    marginTop: 24,
  },
  logTitle: {
    fontSize: 18,
    fontWeight: '700',
  },
  metaText: {
    fontSize: 14,
    opacity: 0.75,
  },
  rowBetween: {
    flexDirection: 'row',
    gap: 12,
    justifyContent: 'space-between',
  },
  statusPill: {
    alignItems: 'center',
    alignSelf: 'flex-start',
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '700',
  },
  subtitle: {
    marginTop: 8,
    opacity: 0.7,
  },
  titleBlock: {
    flex: 1,
    gap: 4,
  },
});
