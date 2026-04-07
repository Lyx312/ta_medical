import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, Image, LayoutAnimation, Modal, Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { useLocalSearchParams, router, Href } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { Gesture, GestureDetector, GestureHandlerRootView } from 'react-native-gesture-handler';
import Animated, { interpolate, runOnJS, useAnimatedStyle, useSharedValue, withTiming } from 'react-native-reanimated';

import { ThemedText } from '@/components/themed-text';
import { Colors, getStatusColor, getStatusBadgeBg } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { getAdminScanLog } from '@/services/api/admin-scan-logs';
import type { AdminScanLogAIRun, AdminScanLogDetail, AdminScanLogFile, ApiError } from '@/services/api/types';

const MIN_VIEWER_SCALE = 1;
const MAX_VIEWER_SCALE = 4;
const SWIPE_CLOSE_THRESHOLD = 150;

function clampValue(value: number, min: number, max: number) {
  'worklet';

  return Math.min(Math.max(value, min), max);
}

function getMaxOffset(contentSize: number, viewportSize: number, scale: number) {
  'worklet';

  return Math.max(0, ((contentSize * scale) - viewportSize) / 2);
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

function formatStatusLabel(status: string) {
  return status.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatByteSize(byteSize: number | null | undefined) {
  if (byteSize === null || byteSize === undefined || Number.isNaN(byteSize)) {
    return 'Unknown size';
  }

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
  chevronColor?: string;
  contentStyle?: object;
};

function CollapsibleSection({
  title,
  children,
  initiallyExpanded = true,
  rightElement,
  cardBg,
  borderColor,
  chevronColor,
  contentStyle,
}: CollapsibleSectionProps) {
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
          color={chevronColor ?? '#718096'}
          style={{ transform: [{ rotate: isExpanded ? '0deg' : '-90deg' }] }}
        />
      </Pressable>
      {isExpanded && <View style={[styles.sectionContent, contentStyle]}>{children}</View>}
    </View>
  );
}

type PayloadTabKey = 'validation' | 'extraction' | 'runs' | 'fusion' | 'error';

type PayloadTab = {
  key: PayloadTabKey;
  label: string;
  payload: unknown;
};

type AIRunGroup = {
  producer_id: string;
  provider: string;
  model_name: string;
  producer_rank: number;
  runs: AdminScanLogAIRun[];
};

function formatPayload(payload: unknown) {
  try {
    return JSON.stringify(payload, null, 2);
  } catch {
    return String(payload);
  }
}

function renderPayload(payload: unknown, colors: typeof Colors.light) {
  if (payload === null || payload === undefined) {
    return <ThemedText style={[styles.placeholderText, { color: colors.text }]}>No data recorded.</ThemedText>;
  }

  const formattedPayload = formatPayload(payload);

  return (
    <View
      style={[
        styles.jsonBlock,
        {
          backgroundColor: `${colors.divider}22`,
          borderColor: colors.divider,
        },
      ]}>
      <ThemedText style={[styles.jsonText, { color: colors.text }]}>{formattedPayload}</ThemedText>
    </View>
  );
}

function groupRunsByProducer(runs: AdminScanLogAIRun[]): AIRunGroup[] {
  const grouped = new Map<string, AIRunGroup>();

  for (const run of runs) {
    const existing = grouped.get(run.producer_id);
    if (!existing) {
      grouped.set(run.producer_id, {
        producer_id: run.producer_id,
        provider: run.provider,
        model_name: run.model_name,
        producer_rank: run.producer_rank,
        runs: [run],
      });
      continue;
    }
    existing.runs.push(run);
  }

  return Array.from(grouped.values())
    .map((group) => ({
      ...group,
      runs: [...group.runs].sort((left, right) => left.run_index - right.run_index),
    }))
    .sort((left, right) => left.producer_rank - right.producer_rank || left.producer_id.localeCompare(right.producer_id));
}

function RunsPayloadView({
  runs,
  colors,
  expandedRunPayloads,
  onToggleRunPayload,
}: {
  runs: AdminScanLogAIRun[];
  colors: typeof Colors.light;
  expandedRunPayloads: Record<string, boolean>;
  onToggleRunPayload: (runKey: string) => void;
}) {
  if (runs.length === 0) {
    return <ThemedText style={[styles.placeholderText, { color: colors.text }]}>No run data recorded.</ThemedText>;
  }

  const successfulRuns = runs.filter((run) => run.success).length;
  const failedRuns = runs.length - successfulRuns;
  const groupedRuns = groupRunsByProducer(runs);

  return (
    <View style={styles.runsPanel}>
      <View style={styles.runsSummaryRow}>
        <View style={[styles.runsSummaryPill, { borderColor: colors.divider, backgroundColor: `${colors.divider}1A` }]}> 
          <ThemedText style={[styles.runsSummaryLabel, { color: `${colors.text}B3` }]}>Total Runs</ThemedText>
          <ThemedText style={[styles.runsSummaryValue, { color: colors.text }]}>{runs.length}</ThemedText>
        </View>
        <View style={[styles.runsSummaryPill, { borderColor: colors.divider, backgroundColor: `${colors.cta}1A` }]}> 
          <ThemedText style={[styles.runsSummaryLabel, { color: `${colors.text}B3` }]}>Success</ThemedText>
          <ThemedText style={[styles.runsSummaryValue, { color: colors.text }]}>{successfulRuns}</ThemedText>
        </View>
        <View style={[styles.runsSummaryPill, { borderColor: colors.divider, backgroundColor: `${colors.error}14` }]}> 
          <ThemedText style={[styles.runsSummaryLabel, { color: `${colors.text}B3` }]}>Failed</ThemedText>
          <ThemedText style={[styles.runsSummaryValue, { color: colors.text }]}>{failedRuns}</ThemedText>
        </View>
        <View style={[styles.runsSummaryPill, { borderColor: colors.divider, backgroundColor: `${colors.info}1A` }]}> 
          <ThemedText style={[styles.runsSummaryLabel, { color: `${colors.text}B3` }]}>Producers</ThemedText>
          <ThemedText style={[styles.runsSummaryValue, { color: colors.text }]}>{groupedRuns.length}</ThemedText>
        </View>
      </View>

      {groupedRuns.map((group) => (
        <View key={group.producer_id} style={styles.runGroup}>
          <View style={styles.runGroupHeader}>
            <ThemedText style={[styles.runGroupTitle, { color: colors.text }]}>
              {group.provider}/{group.model_name}
            </ThemedText>
            <ThemedText style={[styles.runGroupMeta, { color: `${colors.text}B3` }]}>Rank #{group.producer_rank + 1}</ThemedText>
          </View>

          {group.runs.map((run) => {
            const runKey = `${run.producer_id}:${run.producer_rank}:${run.run_index}`;
            const isExpanded = Boolean(expandedRunPayloads[runKey]);
            return (
              <View key={runKey} style={[styles.runCard, { borderColor: colors.divider, backgroundColor: `${colors.divider}14` }]}>
                <View style={styles.runCardHeader}>
                  <View style={styles.runCardTitleBlock}>
                    <ThemedText style={[styles.runCardTitle, { color: colors.text }]}>Run #{run.run_index + 1}</ThemedText>
                    <ThemedText style={[styles.runCardMetaText, { color: `${colors.text}B3` }]}>Temp {run.temperature}</ThemedText>
                  </View>
                  <View
                    style={[
                      styles.runStatusPill,
                      {
                        backgroundColor: run.success ? `${colors.cta}22` : `${colors.error}22`,
                        borderColor: run.success ? `${colors.cta}66` : `${colors.error}66`,
                      },
                    ]}>
                    <ThemedText style={[styles.runStatusPillText, { color: run.success ? colors.cta : colors.error }]}>
                      {run.success ? 'Success' : 'Failed'}
                    </ThemedText>
                  </View>
                </View>

                {run.error_message ? (
                  <View style={[styles.runErrorBlock, { backgroundColor: `${colors.error}10`, borderColor: `${colors.error}55` }]}>
                    <ThemedText style={[styles.runErrorTitle, { color: colors.error }]}>Error</ThemedText>
                    <ThemedText style={[styles.runErrorMessage, { color: colors.text }]}>{run.error_message}</ThemedText>
                  </View>
                ) : null}

                <Pressable style={styles.runPayloadToggle} onPress={() => onToggleRunPayload(runKey)}>
                  <ThemedText style={[styles.runPayloadToggleText, { color: colors.primary }]}>
                    {isExpanded ? 'Hide payload' : 'Show payload'}
                  </ThemedText>
                </Pressable>

                {isExpanded ? renderPayload(run.payload, colors) : null}
              </View>
            );
          })}
        </View>
      ))}
    </View>
  );
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function formatShortValue(value: unknown): string {
  if (value === null || value === undefined) {
    return 'null';
  }
  if (typeof value === 'string') {
    return value;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  if (Array.isArray(value)) {
    return `${value.length} item(s)`;
  }
  try {
    const text = JSON.stringify(value);
    if (!text) {
      return 'Object';
    }
    return text.length > 180 ? `${text.slice(0, 177)}...` : text;
  } catch {
    return 'Object';
  }
}

function renderSupportSummary(supportValue: unknown): string {
  if (!Array.isArray(supportValue) || supportValue.length === 0) {
    return 'No support traces';
  }

  const summaries = supportValue
    .map((entry) => asRecord(entry))
    .filter((entry): entry is Record<string, unknown> => Boolean(entry))
    .map((entry) => {
      const provider = typeof entry.provider === 'string' ? entry.provider : 'provider';
      const model = typeof entry.model === 'string' ? entry.model : 'model';
      const runIndices = Array.isArray(entry.run_indices)
        ? entry.run_indices.filter((runIndex): runIndex is number => typeof runIndex === 'number')
        : [];
      const runsLabel = runIndices.length > 0 ? runIndices.map((runIndex) => runIndex + 1).join(', ') : '-';
      return `${provider}/${model} (runs: ${runsLabel})`;
    });

  return summaries.join(' | ');
}

type FusionRunValue = {
  runKey: string;
  producerLabel: string;
  producerId: string;
  producerRank: number;
  runIndex: number;
  value: unknown;
  normalizedKey: string;
};

function normalizeWhitespaceText(value: string): string {
  return value.trim().split(/\s+/).join(' ');
}

function normalizeComparisonTextTs(value: string): string {
  return normalizeWhitespaceText(value).toLowerCase().replace(/[\W_]+/g, '');
}

function coerceNumericValueTs(value: unknown): number | null {
  if (value === null || value === undefined || value === '') {
    return null;
  }
  if (typeof value === 'boolean') {
    return Number(value);
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value !== 'string') {
    return null;
  }

  let cleaned = value.trim().replace(/\s+/g, '').replace(/IDR/gi, '');
  if (cleaned.includes(',') && cleaned.includes('.')) {
    cleaned = cleaned.lastIndexOf(',') > cleaned.lastIndexOf('.') ? cleaned.replace(/\./g, '').replace(',', '.') : cleaned.replace(/,/g, '');
  } else if ((cleaned.match(/,/g) ?? []).length > 1) {
    cleaned = cleaned.replace(/,/g, '');
  } else if ((cleaned.match(/\./g) ?? []).length > 1) {
    cleaned = cleaned.replace(/\./g, '');
  } else {
    cleaned = cleaned.replace(/,/g, '');
  }
  cleaned = cleaned.replace(/[^0-9.\-]/g, '');
  if (!cleaned || cleaned === '-' || cleaned === '.' || cleaned === '-.') {
    return null;
  }
  const parsed = Number.parseFloat(cleaned);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return Number.isInteger(parsed) ? parsed : Number(parsed.toFixed(2));
}

function coerceDateValueTs(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null;
  }
  const candidate = value.trim();
  if (!candidate) {
    return null;
  }

  const normalized = candidate.replace(/[./]/g, '-').replace(/\s+/g, '-');
  const dmy = normalized.match(/^(\d{1,2})-(\d{1,2})-(\d{4})$/);
  if (dmy) {
    const [, day, month, year] = dmy;
    return `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
  }
  const ymd = normalized.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
  if (ymd) {
    const [, year, month, day] = ymd;
    return `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
  }
  return candidate;
}

function normalizeCandidateKeyTs(fieldPath: string, value: unknown): string {
  if (value === null || value === undefined) {
    return '__null__';
  }

  const baseFieldPath = fieldPath.replace(/\[[^\]]+\]/g, '');
  const leafName = baseFieldPath.split('.').pop() ?? baseFieldPath;
  const dateFields = new Set(['service_date', 'transaction_date', 'date_of_birth']);
  const numericFields = new Set([
    'summary.total_amount',
    'patient_info.age',
    'financial_recap.total_medicines',
    'financial_recap.total_other_charges',
    'financial_recap.grand_total',
    'medicines.quantity',
    'medicines.unit_price',
    'medicines.total_price',
    'other_charges.amount',
  ]);
  const semanticFields = new Set(['medical_context.anamnesis', 'medical_context.therapy_note']);

  if (dateFields.has(leafName)) {
    return coerceDateValueTs(value) ?? '__null__';
  }
  if (numericFields.has(baseFieldPath)) {
    const numeric = coerceNumericValueTs(value);
    return numeric === null ? '__null__' : String(numeric);
  }
  if (baseFieldPath.endsWith('currency') || baseFieldPath.endsWith('gender')) {
    return normalizeWhitespaceText(String(value)).toUpperCase();
  }
  if (semanticFields.has(baseFieldPath)) {
    return normalizeWhitespaceText(String(value)).toLowerCase();
  }
  return normalizeComparisonTextTs(String(value));
}

function slugifyKeyTs(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'item';
}

function classifyOtherChargeKindTs(category: unknown, description: unknown): string {
  const text = `${normalizeWhitespaceText(String(category ?? '')).toLowerCase()} ${normalizeWhitespaceText(String(description ?? '')).toLowerCase()}`;
  if (['obat', 'medicine', 'medication', 'drug', 'pharmacy', 'apotek'].some((keyword) => text.includes(keyword))) {
    return 'medicine';
  }
  if (['consult', 'doctor', 'dokter', 'clinic', 'klinik', 'visit', 'treatment', 'therapy', 'consultation'].some((keyword) => text.includes(keyword))) {
    return 'consultation';
  }
  return 'other';
}

function buildListItemKeyTs(listName: string, item: Record<string, unknown>): string {
  if (listName === 'diagnoses') {
    return normalizeCandidateKeyTs('diagnoses.description', item.description);
  }
  if (listName === 'medicines') {
    return normalizeCandidateKeyTs('medicines.name', item.name);
  }
  const amount = coerceNumericValueTs(item.amount);
  if (amount === null) {
    return normalizeCandidateKeyTs('other_charges.description', item.description);
  }
  const kind = classifyOtherChargeKindTs(item.category, item.description);
  return `${kind}:${amount}`;
}

function getFieldValueFromPayload(fieldPath: string, payload: Record<string, unknown>): unknown {
  const listMatch = fieldPath.match(/^([a-z_]+)\[([^\]]+)\]\.([a-z_]+)$/i);
  if (listMatch) {
    const [, listName, itemSlug, fieldName] = listMatch;
    const listValue = payload[listName];
    if (!Array.isArray(listValue)) {
      return undefined;
    }
    for (const item of listValue) {
      const itemRecord = asRecord(item);
      if (!itemRecord) {
        continue;
      }
      const key = buildListItemKeyTs(listName, itemRecord);
      if (slugifyKeyTs(key) === itemSlug) {
        return itemRecord[fieldName];
      }
    }
    return undefined;
  }

  const segments = fieldPath.split('.');
  let current: unknown = payload;
  for (const segment of segments) {
    const currentRecord = asRecord(current);
    if (!currentRecord || !(segment in currentRecord)) {
      return undefined;
    }
    current = currentRecord[segment];
  }
  return current;
}

function deriveRunValuesForField(fieldPath: string, runs: AdminScanLogAIRun[]): FusionRunValue[] {
  const runValues: FusionRunValue[] = [];

  for (const run of runs) {
    const payload = asRecord(run.payload);
    if (!payload) {
      continue;
    }
    const value = getFieldValueFromPayload(fieldPath, payload);
    if (value === undefined) {
      continue;
    }
    runValues.push({
      runKey: `${run.producer_id}:${run.producer_rank}:${run.run_index}`,
      producerLabel: `${run.provider}/${run.model_name}`,
      producerId: run.producer_id,
      producerRank: run.producer_rank,
      runIndex: run.run_index,
      value,
      normalizedKey: normalizeCandidateKeyTs(fieldPath, value),
    });
  }

  return runValues.sort((left, right) => left.producerRank - right.producerRank || left.runIndex - right.runIndex);
}

function buildSupportFromRunValues(runValues: FusionRunValue[]): Record<string, unknown>[] {
  const grouped = new Map<string, { producer_id: string; provider: string; model: string; producer_rank: number; run_indices: number[] }>();
  for (const runValue of runValues) {
    const [provider, ...modelParts] = runValue.producerLabel.split('/');
    const model = modelParts.join('/') || 'model';
    const existing = grouped.get(runValue.producerId);
    if (!existing) {
      grouped.set(runValue.producerId, {
        producer_id: runValue.producerId,
        provider,
        model,
        producer_rank: runValue.producerRank,
        run_indices: [runValue.runIndex],
      });
      continue;
    }
    existing.run_indices.push(runValue.runIndex);
  }
  return Array.from(grouped.values()).sort((left, right) => left.producer_rank - right.producer_rank);
}

function buildCandidatesFromRunValues(fieldPath: string, selectedValue: unknown, runValues: FusionRunValue[]): Record<string, unknown>[] {
  if (runValues.length === 0) {
    return [];
  }

  const selectedKey = normalizeCandidateKeyTs(fieldPath, selectedValue);
  const grouped = new Map<string, { normalized_key: string; value: unknown; runValues: FusionRunValue[]; producerIds: Set<string> }>();

  for (const runValue of runValues) {
    const existing = grouped.get(runValue.normalizedKey);
    if (!existing) {
      grouped.set(runValue.normalizedKey, {
        normalized_key: runValue.normalizedKey,
        value: runValue.value,
        runValues: [runValue],
        producerIds: new Set([runValue.producerId]),
      });
      continue;
    }
    existing.runValues.push(runValue);
    existing.producerIds.add(runValue.producerId);
  }

  return Array.from(grouped.values())
    .sort((left, right) => right.runValues.length - left.runValues.length)
    .map((candidate) => ({
      normalized_key: candidate.normalized_key,
      value: candidate.value,
      selected: candidate.normalized_key === selectedKey,
      support_count: candidate.runValues.length,
      producer_count: candidate.producerIds.size,
      support: buildSupportFromRunValues(candidate.runValues),
    }));
}

function FusionPayloadView({
  payload,
  runs,
  colors,
  expandedFusionFields,
  expandedFusionRunValues,
  onToggleFusionField,
  onToggleFusionRunValues,
}: {
  payload: Record<string, unknown>;
  runs: AdminScanLogAIRun[];
  colors: typeof Colors.light;
  expandedFusionFields: Record<string, boolean>;
  expandedFusionRunValues: Record<string, boolean>;
  onToggleFusionField: (fieldPath: string) => void;
  onToggleFusionRunValues: (fieldPath: string) => void;
}) {
  const overallConfidence = typeof payload.overall_confidence === 'number' ? payload.overall_confidence : null;
  const reviewRequired = typeof payload.review_required === 'boolean' ? payload.review_required : null;
  const reviewReasons = Array.isArray(payload.review_reasons)
    ? payload.review_reasons.filter((reason): reason is string => typeof reason === 'string')
    : [];

  const constraintChecksRecord = asRecord(payload.constraint_checks) ?? {};
  const constraintEntries = Object.entries(constraintChecksRecord)
    .map(([checkName, rawCheck]) => {
      const check = asRecord(rawCheck);
      if (!check) {
        return null;
      }
      const passed = typeof check.passed === 'boolean' ? check.passed : false;
      const usedDerivedValue = typeof check.used_derived_value === 'boolean' ? check.used_derived_value : false;
      return {
        checkName,
        passed,
        usedDerivedValue,
        originalValue: check.original_value,
        finalValue: check.final_value,
        derivedValue: check.derived_value,
      };
    })
    .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry));

  const judgeInvocations = Array.isArray(payload.judge_invocations)
    ? payload.judge_invocations
        .map((invocation) => asRecord(invocation))
        .filter((invocation): invocation is Record<string, unknown> => Boolean(invocation))
    : [];

  const fieldConfidenceRecord = asRecord(payload.field_confidence) ?? {};
  const fieldEntries = Object.entries(fieldConfidenceRecord)
    .map(([fieldPath, rawMeta]) => {
      const meta = asRecord(rawMeta);
      if (!meta) {
        return null;
      }
      const confidence = typeof meta.confidence === 'number' ? meta.confidence : null;
      const reviewFlags = Array.isArray(meta.review_flags)
        ? meta.review_flags.filter((flag): flag is string => typeof flag === 'string')
        : [];
      const candidates = Array.isArray(meta.candidates)
        ? meta.candidates
            .map((candidate) => asRecord(candidate))
            .filter((candidate): candidate is Record<string, unknown> => Boolean(candidate))
        : [];
      const runValues = deriveRunValuesForField(fieldPath, runs);
      const derivedCandidates = buildCandidatesFromRunValues(fieldPath, meta.value, runValues);
      const resolvedCandidates = derivedCandidates.length > 0 ? derivedCandidates : candidates;
      return {
        fieldPath,
        confidence,
        value: meta.value,
        supportCount: typeof meta.support_count === 'number' ? meta.support_count : null,
        producerAgreement: typeof meta.producer_agreement === 'number' ? meta.producer_agreement : null,
        judgeUsed: typeof meta.judge_used === 'boolean' ? meta.judge_used : false,
        judgeReason: typeof meta.judge_reason === 'string' ? meta.judge_reason : null,
        reviewFlags,
        ruleHits: Array.isArray(meta.rule_hits)
          ? meta.rule_hits.filter((rule): rule is string => typeof rule === 'string')
          : [],
        supportSummary: renderSupportSummary(meta.support),
        candidates: resolvedCandidates,
        runValues,
      };
    })
    .filter((entry): entry is NonNullable<typeof entry> => Boolean(entry))
    .sort((left, right) => {
      const leftScore = left.confidence ?? 2;
      const rightScore = right.confidence ?? 2;
      if (leftScore !== rightScore) {
        return leftScore - rightScore;
      }
      return left.fieldPath.localeCompare(right.fieldPath);
    });

  const lowConfidenceCount = fieldEntries.filter((entry) => entry.reviewFlags.includes('low_confidence')).length;
  const failedConstraintCount = constraintEntries.filter((entry) => !entry.passed).length;

  return (
    <View style={styles.fusionPanel}>
      <View style={styles.fusionSummaryRow}>
        <View style={[styles.fusionSummaryPill, { borderColor: colors.divider, backgroundColor: `${colors.divider}1A` }]}> 
          <ThemedText style={[styles.fusionSummaryLabel, { color: `${colors.text}B3` }]}>Overall</ThemedText>
          <ThemedText style={[styles.fusionSummaryValue, { color: colors.text }]}>
            {overallConfidence === null ? 'N/A' : `${Math.round(overallConfidence * 100)}%`}
          </ThemedText>
        </View>
        <View style={[styles.fusionSummaryPill, { borderColor: colors.divider, backgroundColor: `${colors.warning}16` }]}> 
          <ThemedText style={[styles.fusionSummaryLabel, { color: `${colors.text}B3` }]}>Review</ThemedText>
          <ThemedText style={[styles.fusionSummaryValue, { color: colors.text }]}> 
            {reviewRequired === null ? 'N/A' : reviewRequired ? 'Yes' : 'No'}
          </ThemedText>
        </View>
        <View style={[styles.fusionSummaryPill, { borderColor: colors.divider, backgroundColor: `${colors.error}14` }]}> 
          <ThemedText style={[styles.fusionSummaryLabel, { color: `${colors.text}B3` }]}>Low Fields</ThemedText>
          <ThemedText style={[styles.fusionSummaryValue, { color: colors.text }]}>{lowConfidenceCount}</ThemedText>
        </View>
        <View style={[styles.fusionSummaryPill, { borderColor: colors.divider, backgroundColor: `${colors.info}18` }]}> 
          <ThemedText style={[styles.fusionSummaryLabel, { color: `${colors.text}B3` }]}>Judge Calls</ThemedText>
          <ThemedText style={[styles.fusionSummaryValue, { color: colors.text }]}>{judgeInvocations.length}</ThemedText>
        </View>
      </View>

      {reviewReasons.length > 0 ? (
        <View style={[styles.fusionSectionCard, { borderColor: colors.divider, backgroundColor: `${colors.divider}12` }]}>
          <ThemedText style={[styles.fusionSectionTitle, { color: colors.text }]}>Review Reasons</ThemedText>
          <View style={styles.fusionBulletList}>
            {reviewReasons.map((reason) => (
              <View key={reason} style={styles.fusionBulletRow}>
                <ThemedText style={[styles.fusionBulletMark, { color: colors.text }]}>•</ThemedText>
                <ThemedText style={[styles.fusionBulletText, { color: colors.text }]}>{reason}</ThemedText>
              </View>
            ))}
          </View>
        </View>
      ) : null}

      <View style={[styles.fusionSectionCard, { borderColor: colors.divider, backgroundColor: `${colors.divider}12` }]}>
        <View style={styles.fusionSectionHeader}>
          <ThemedText style={[styles.fusionSectionTitle, { color: colors.text }]}>Constraint Checks</ThemedText>
          <ThemedText style={[styles.fusionSectionMeta, { color: `${colors.text}B3` }]}>{failedConstraintCount} failed</ThemedText>
        </View>
        {constraintEntries.length === 0 ? (
          <ThemedText style={[styles.placeholderText, { color: colors.text }]}>No constraint checks available.</ThemedText>
        ) : (
          <View style={styles.fusionList}>
            {constraintEntries.map((entry) => (
              <View key={entry.checkName} style={[styles.fusionListItem, { borderColor: colors.divider, backgroundColor: `${colors.divider}14` }]}> 
                <View style={styles.fusionListItemHeader}>
                  <ThemedText style={[styles.fusionListItemTitle, { color: colors.text }]}>{entry.checkName}</ThemedText>
                  <ThemedText style={[styles.fusionListItemBadge, { color: entry.passed ? colors.cta : colors.error }]}> 
                    {entry.passed ? 'Passed' : 'Failed'}
                  </ThemedText>
                </View>
                <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>Original: {formatShortValue(entry.originalValue)}</ThemedText>
                <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>Derived: {formatShortValue(entry.derivedValue)}</ThemedText>
                <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>Final: {formatShortValue(entry.finalValue)}</ThemedText>
                <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>Used derived value: {entry.usedDerivedValue ? 'Yes' : 'No'}</ThemedText>
              </View>
            ))}
          </View>
        )}
      </View>

      {judgeInvocations.length > 0 ? (
        <View style={[styles.fusionSectionCard, { borderColor: colors.divider, backgroundColor: `${colors.divider}12` }]}>
          <ThemedText style={[styles.fusionSectionTitle, { color: colors.text }]}>Judge Decisions</ThemedText>
          <View style={styles.fusionList}>
            {judgeInvocations.map((invocation, index) => {
              const fieldPath = typeof invocation.field_path === 'string' ? invocation.field_path : `field-${index + 1}`;
              const selectedCandidate = invocation.selected_candidate;
              const reason = typeof invocation.reason === 'string' ? invocation.reason : null;
              return (
                <View key={`${fieldPath}-${index}`} style={[styles.fusionListItem, { borderColor: colors.divider, backgroundColor: `${colors.divider}14` }]}> 
                  <ThemedText style={[styles.fusionListItemTitle, { color: colors.text }]}>{fieldPath}</ThemedText>
                  <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>Picked: {formatShortValue(selectedCandidate)}</ThemedText>
                  {reason ? <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>Reason: {reason}</ThemedText> : null}
                </View>
              );
            })}
          </View>
        </View>
      ) : null}

      <View style={[styles.fusionSectionCard, { borderColor: colors.divider, backgroundColor: `${colors.divider}12` }]}>
        <ThemedText style={[styles.fusionSectionTitle, { color: colors.text }]}>Field Decisions (lowest confidence first)</ThemedText>
        {fieldEntries.length === 0 ? (
          <ThemedText style={[styles.placeholderText, { color: colors.text }]}>No field confidence data available.</ThemedText>
        ) : (
          <View style={styles.fusionList}>
            {fieldEntries.map((entry) => {
              const isExpanded = Boolean(expandedFusionFields[entry.fieldPath]);
              const isRunValuesExpanded = Boolean(expandedFusionRunValues[entry.fieldPath]);
              return (
                <View key={entry.fieldPath} style={[styles.fusionFieldCard, { borderColor: colors.divider, backgroundColor: `${colors.divider}14` }]}> 
                  <View style={styles.fusionFieldHeader}>
                    <ThemedText style={[styles.fusionFieldPath, { color: colors.text }]}>{entry.fieldPath}</ThemedText>
                    <ThemedText style={[styles.fusionFieldConfidence, { color: colors.text }]}> 
                      {entry.confidence === null ? 'N/A' : `${Math.round(entry.confidence * 100)}%`}
                    </ThemedText>
                  </View>
                  <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>Picked: {formatShortValue(entry.value)}</ThemedText>
                  <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>Support count: {entry.supportCount ?? '-'}</ThemedText>
                  <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>Producer agreement: {entry.producerAgreement === null ? '-' : `${Math.round(entry.producerAgreement * 100)}%`}</ThemedText>
                  {entry.reviewFlags.length > 0 ? (
                    <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>Flags: {entry.reviewFlags.join(', ')}</ThemedText>
                  ) : null}
                  {entry.ruleHits.length > 0 ? (
                    <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>Rules: {entry.ruleHits.join(' | ')}</ThemedText>
                  ) : null}
                  <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>Winner support: {entry.supportSummary}</ThemedText>
                  <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>Unique values: {entry.candidates.length} from {entry.runValues.length} run outputs</ThemedText>
                  {entry.judgeUsed && entry.judgeReason ? (
                    <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>Judge reason: {entry.judgeReason}</ThemedText>
                  ) : null}

                  <Pressable style={styles.fusionToggle} onPress={() => onToggleFusionField(entry.fieldPath)}>
                    <ThemedText style={[styles.fusionToggleText, { color: colors.primary }]}> 
                      {isExpanded ? 'Hide candidate values' : `Show candidate values (${entry.candidates.length})`}
                    </ThemedText>
                  </Pressable>

                  {isExpanded ? (
                    <View style={styles.fusionCandidatesList}>
                      {entry.candidates.length === 0 ? (
                        <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>No candidate breakdown available.</ThemedText>
                      ) : (
                        entry.candidates.map((candidate, candidateIndex) => {
                          const isSelected = Boolean(candidate.selected);
                          const supportCount = typeof candidate.support_count === 'number' ? candidate.support_count : 0;
                          const producerCount = typeof candidate.producer_count === 'number' ? candidate.producer_count : 0;
                          return (
                            <View
                              key={`${entry.fieldPath}-candidate-${candidateIndex}`}
                              style={[
                                styles.fusionCandidateItem,
                                {
                                  borderColor: isSelected ? `${colors.cta}66` : colors.divider,
                                  backgroundColor: isSelected ? `${colors.cta}14` : `${colors.divider}12`,
                                },
                              ]}>
                              <ThemedText style={[styles.fusionCandidateValue, { color: colors.text }]}> 
                                {formatShortValue(candidate.value)}
                              </ThemedText>
                              <ThemedText style={[styles.fusionCandidateMeta, { color: `${colors.text}B3` }]}> 
                                {isSelected ? 'Selected' : 'Alternate'} | support {supportCount} | producers {producerCount}
                              </ThemedText>
                              <ThemedText style={[styles.fusionCandidateMeta, { color: `${colors.text}B3` }]}> 
                                Support traces: {renderSupportSummary(candidate.support)}
                              </ThemedText>
                            </View>
                          );
                        })
                      )}
                    </View>
                  ) : null}

                  <Pressable style={styles.fusionToggle} onPress={() => onToggleFusionRunValues(entry.fieldPath)}>
                    <ThemedText style={[styles.fusionToggleText, { color: colors.primary }]}> 
                      {isRunValuesExpanded ? 'Hide per-run values' : `Show per-run values (${entry.runValues.length})`}
                    </ThemedText>
                  </Pressable>

                  {isRunValuesExpanded ? (
                    <View style={styles.fusionRunValuesList}>
                      {entry.runValues.length === 0 ? (
                        <ThemedText style={[styles.fusionListItemText, { color: `${colors.text}B3` }]}>No per-run values found in run payloads.</ThemedText>
                      ) : (
                        entry.runValues.map((runValue) => (
                          <View key={`${entry.fieldPath}-${runValue.runKey}`} style={[styles.fusionRunValueItem, { borderColor: colors.divider, backgroundColor: `${colors.divider}12` }]}> 
                            <ThemedText style={[styles.fusionRunValueHeader, { color: colors.text }]}> 
                              {runValue.producerLabel} • run #{runValue.runIndex + 1}
                            </ThemedText>
                            <ThemedText style={[styles.fusionRunValueText, { color: `${colors.text}B3` }]}> 
                              {formatShortValue(runValue.value)}
                            </ThemedText>
                          </View>
                        ))
                      )}
                    </View>
                  ) : null}
                </View>
              );
            })}
          </View>
        )}
      </View>
    </View>
  );
}

function ImagePairRow({
  file,
  onPressOriginal,
  onPressProcessed,
  colors,
}: {
  file: AdminScanLogFile;
  onPressOriginal: () => void;
  onPressProcessed: () => void;
  colors: typeof Colors.light;
}) {
  const [originalSize, setOriginalSize] = useState<{ width: number; height: number } | null>(null);
  const [processedSize, setProcessedSize] = useState<{ width: number; height: number } | null>(null);

  return (
    <View style={styles.imagePairContainer}>
      <View style={styles.imagePairColumn}>
        <ThemedText style={[styles.imagePairLabel, { color: colors.text }]}>Original</ThemedText>
        <Pressable onPress={onPressOriginal}>
          <Image
            source={{ uri: file.original_image_data_url }}
            style={styles.imagePairThumbnail}
            onLoad={(e) => setOriginalSize({ width: e.nativeEvent.source.width, height: e.nativeEvent.source.height })}
          />
        </Pressable>
        <View style={styles.imageMetaBlock}>
          {originalSize ? (
            <ThemedText style={[styles.imageSize, { color: colors.text }]}>
              {originalSize.width} x {originalSize.height}
            </ThemedText>
          ) : null}
          <ThemedText style={[styles.imageSize, { color: colors.text }]}>
            {formatByteSize(file.original_byte_size)}
          </ThemedText>
        </View>
      </View>
      <View style={[styles.imagePairDivider, { backgroundColor: colors.divider }]} />
      <View style={styles.imagePairColumn}>
        <ThemedText style={[styles.imagePairLabel, { color: colors.text }]}>Processed</ThemedText>
        {file.processed_image_data_url ? (
          <>
            <Pressable onPress={onPressProcessed}>
              <Image
                source={{ uri: file.processed_image_data_url }}
                style={styles.imagePairThumbnail}
                onLoad={(e) => setProcessedSize({ width: e.nativeEvent.source.width, height: e.nativeEvent.source.height })}
              />
            </Pressable>
            <View style={styles.imageMetaBlock}>
              {processedSize ? (
                <ThemedText style={[styles.imageSize, { color: colors.text }]}>
                  {processedSize.width} x {processedSize.height}
                </ThemedText>
              ) : null}
              <ThemedText style={[styles.imageSize, { color: colors.text }]}>
                {formatByteSize(file.processed_byte_size)}
              </ThemedText>
            </View>
          </>
        ) : (
          <View style={[styles.imagePairThumbnail, styles.imagePlaceholder, { backgroundColor: colors.divider }]}>
            <ThemedText style={[styles.placeholderText, { color: colors.text }]}>N/A</ThemedText>
          </View>
        )}
      </View>
    </View>
  );
}

export function AdminScanLogDetailScreen() {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const params = useLocalSearchParams<{ scanLogId?: string }>();
  const [log, setLog] = useState<AdminScanLogDetail | null>(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [viewerImageUri, setViewerImageUri] = useState<string | null>(null);
  const [viewerImageTitle, setViewerImageTitle] = useState('');
  const [viewerImageNaturalSize, setViewerImageNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [viewerStageSize, setViewerStageSize] = useState({ width: 0, height: 0 });
  const [selectedPayloadTab, setSelectedPayloadTab] = useState<PayloadTabKey>('validation');
  const [expandedRunPayloads, setExpandedRunPayloads] = useState<Record<string, boolean>>({});
  const [expandedFusionFields, setExpandedFusionFields] = useState<Record<string, boolean>>({});
  const [expandedFusionRunValues, setExpandedFusionRunValues] = useState<Record<string, boolean>>({});

  const viewerScale = useSharedValue(1);
  const viewerScaleOffset = useSharedValue(1);
  const viewerTranslateX = useSharedValue(0);
  const viewerTranslateY = useSharedValue(0);
  const viewerTranslateXOffset = useSharedValue(0);
  const viewerTranslateYOffset = useSharedValue(0);
  const viewerCloseSwipeY = useSharedValue(0);
  const viewerCloseOpacity = useSharedValue(1);
  const viewerSwipeStartScale = useSharedValue(1);

  const numericScanLogId = useMemo(() => Number(params.scanLogId), [params.scanLogId]);
  const viewerImageLayout = useMemo(() => {
    if (!viewerImageNaturalSize || viewerStageSize.width <= 0 || viewerStageSize.height <= 0) {
      return null;
    }

    const fitScale = Math.min(
      viewerStageSize.width / viewerImageNaturalSize.width,
      viewerStageSize.height / viewerImageNaturalSize.height,
    );
    const displayedWidth = viewerImageNaturalSize.width * fitScale;
    const displayedHeight = viewerImageNaturalSize.height * fitScale;

    return {
      baseWidth: displayedWidth,
      baseHeight: displayedHeight,
      displayedWidth,
      displayedHeight,
    };
  }, [viewerImageNaturalSize, viewerStageSize.height, viewerStageSize.width]);

  const loadLog = useCallback(async () => {
    if (!Number.isInteger(numericScanLogId) || numericScanLogId <= 0) {
      setError('Invalid scan log id.');
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      setLog(await getAdminScanLog(numericScanLogId));
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.message ?? 'Unable to load scan log detail.');
    } finally {
      setIsLoading(false);
    }
  }, [numericScanLogId]);

  useFocusEffect(
    useCallback(() => {
      void loadLog();
    }, [loadLog]),
  );

  const payloadTabs = useMemo<PayloadTab[]>(() => {
    if (!log) {
      return [];
    }

    const tabs: PayloadTab[] = [];

    if (log.validation_payload !== null && log.validation_payload !== undefined) {
      tabs.push({ key: 'validation', label: 'Validation', payload: log.validation_payload });
    }

    if (log.extraction_payload !== null && log.extraction_payload !== undefined) {
      tabs.push({ key: 'extraction', label: 'Extraction', payload: log.extraction_payload });
    }

    if (log.ai_model_runs && log.ai_model_runs.length > 0) {
      tabs.push({ key: 'runs', label: 'Runs', payload: log.ai_model_runs });
    }

    if (log.extraction_meta_payload !== null && log.extraction_meta_payload !== undefined) {
      tabs.push({ key: 'fusion', label: 'Fusion', payload: log.extraction_meta_payload });
    }

    if (log.error_payload && Object.keys(log.error_payload).length > 0) {
      tabs.push({ key: 'error', label: 'Error', payload: log.error_payload });
    }

    return tabs;
  }, [log]);

  useEffect(() => {
    if (payloadTabs.length === 0) {
      setSelectedPayloadTab('validation');
      return;
    }

    const preferredTab =
      payloadTabs.find((tab) => tab.key === 'error') ??
      payloadTabs.find((tab) => tab.key === 'extraction') ??
      payloadTabs.find((tab) => tab.key === 'validation') ??
      payloadTabs[0];

    setSelectedPayloadTab(preferredTab.key);
  }, [log?.id, payloadTabs]);

  useEffect(() => {
    setExpandedRunPayloads({});
    setExpandedFusionFields({});
    setExpandedFusionRunValues({});
  }, [log?.id]);

  const toggleRunPayload = useCallback((runKey: string) => {
    setExpandedRunPayloads((current) => ({
      ...current,
      [runKey]: !current[runKey],
    }));
  }, []);

  const toggleFusionField = useCallback((fieldPath: string) => {
    setExpandedFusionFields((current) => ({
      ...current,
      [fieldPath]: !current[fieldPath],
    }));
  }, []);

  const toggleFusionRunValues = useCallback((fieldPath: string) => {
    setExpandedFusionRunValues((current) => ({
      ...current,
      [fieldPath]: !current[fieldPath],
    }));
  }, []);

  const activePayloadTab = useMemo(
    () => payloadTabs.find((tab) => tab.key === selectedPayloadTab) ?? payloadTabs[0] ?? null,
    [payloadTabs, selectedPayloadTab],
  );

  const openImageViewer = useCallback(
    (uri: string, title: string) => {
      setViewerImageUri(uri);
      setViewerImageTitle(title);
      setViewerImageNaturalSize(null);
      viewerScale.value = 1;
      viewerScaleOffset.value = 1;
      viewerTranslateX.value = 0;
      viewerTranslateY.value = 0;
      viewerTranslateXOffset.value = 0;
      viewerTranslateYOffset.value = 0;
      viewerCloseSwipeY.value = 0;
      viewerCloseOpacity.value = 1;
    },
    [viewerScale, viewerScaleOffset, viewerTranslateX, viewerTranslateXOffset, viewerTranslateY, viewerTranslateYOffset, viewerCloseSwipeY, viewerCloseOpacity],
  );

  const closeImageViewer = useCallback(() => {
    setViewerImageUri(null);
    setViewerImageTitle('');
    setViewerImageNaturalSize(null);
    viewerScale.value = 1;
    viewerScaleOffset.value = 1;
    viewerTranslateX.value = 0;
    viewerTranslateY.value = 0;
    viewerTranslateXOffset.value = 0;
    viewerTranslateYOffset.value = 0;
    viewerCloseSwipeY.value = 0;
    viewerCloseOpacity.value = 1;
  }, [viewerScale, viewerScaleOffset, viewerTranslateX, viewerTranslateXOffset, viewerTranslateY, viewerTranslateYOffset, viewerCloseSwipeY, viewerCloseOpacity]);

  const panGesture = useMemo(
    () =>
      Gesture.Pan()
        .onStart(() => {
          viewerTranslateXOffset.value = viewerTranslateX.value;
          viewerTranslateYOffset.value = viewerTranslateY.value;
          viewerSwipeStartScale.value = viewerScale.value;
        })
        .onUpdate((event) => {
          if (viewerSwipeStartScale.value === 1 && event.translationY > 0) {
            viewerCloseSwipeY.value = event.translationY;
            viewerCloseOpacity.value = interpolate(event.translationY, [0, SWIPE_CLOSE_THRESHOLD * 2], [1, 0.3]);
            viewerScale.value = interpolate(event.translationY, [0, SWIPE_CLOSE_THRESHOLD * 2], [1, 0.85]);
          } else {
            const maxOffsetX = getMaxOffset(
              viewerImageLayout?.displayedWidth ?? viewerStageSize.width,
              viewerStageSize.width,
              viewerScale.value,
            );
            const maxOffsetY = getMaxOffset(
              viewerImageLayout?.displayedHeight ?? viewerStageSize.height,
              viewerStageSize.height,
              viewerScale.value,
            );

            viewerTranslateX.value = clampValue(
              viewerTranslateXOffset.value + event.translationX,
              -maxOffsetX,
              maxOffsetX,
            );
            viewerTranslateY.value = clampValue(
              viewerTranslateYOffset.value + event.translationY,
              -maxOffsetY,
              maxOffsetY,
            );
          }
        })
        .onEnd(() => {
          if (viewerCloseSwipeY.value > 0) {
            if (viewerCloseSwipeY.value > SWIPE_CLOSE_THRESHOLD) {
              runOnJS(closeImageViewer)();
            } else {
              viewerCloseSwipeY.value = withTiming(0);
              viewerCloseOpacity.value = withTiming(1);
              viewerScale.value = withTiming(1);
            }
          } else {
            viewerTranslateXOffset.value = viewerTranslateX.value;
            viewerTranslateYOffset.value = viewerTranslateY.value;
          }
        }),
    [closeImageViewer, viewerCloseSwipeY, viewerCloseOpacity, viewerImageLayout?.displayedHeight, viewerImageLayout?.displayedWidth, viewerScale, viewerStageSize.height, viewerStageSize.width, viewerSwipeStartScale, viewerTranslateX, viewerTranslateXOffset, viewerTranslateY, viewerTranslateYOffset],
  );

  const pinchGesture = useMemo(
    () =>
      Gesture.Pinch()
        .onStart(() => {
          viewerScaleOffset.value = viewerScale.value;
        })
        .onUpdate((event) => {
          const nextScale = clampValue(viewerScaleOffset.value * event.scale, MIN_VIEWER_SCALE, MAX_VIEWER_SCALE);
          const maxOffsetX = getMaxOffset(
            viewerImageLayout?.displayedWidth ?? viewerStageSize.width,
            viewerStageSize.width,
            nextScale,
          );
          const maxOffsetY = getMaxOffset(
            viewerImageLayout?.displayedHeight ?? viewerStageSize.height,
            viewerStageSize.height,
            nextScale,
          );

          viewerScale.value = nextScale;
          viewerTranslateX.value = clampValue(viewerTranslateX.value, -maxOffsetX, maxOffsetX);
          viewerTranslateY.value = clampValue(viewerTranslateY.value, -maxOffsetY, maxOffsetY);
        })
        .onEnd(() => {
          viewerScaleOffset.value = viewerScale.value;
          viewerTranslateXOffset.value = viewerTranslateX.value;
          viewerTranslateYOffset.value = viewerTranslateY.value;
        }),
    [
      viewerImageLayout?.displayedHeight,
      viewerImageLayout?.displayedWidth,
      viewerScale,
      viewerScaleOffset,
      viewerStageSize.height,
      viewerStageSize.width,
      viewerTranslateX,
      viewerTranslateXOffset,
      viewerTranslateY,
      viewerTranslateYOffset,
    ],
  );

  const viewerGesture = useMemo(() => Gesture.Simultaneous(panGesture, pinchGesture), [panGesture, pinchGesture]);

  const viewerImageAnimatedStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: viewerTranslateX.value },
      { translateY: viewerTranslateY.value + viewerCloseSwipeY.value },
      { scale: viewerScale.value },
    ],
    opacity: viewerCloseOpacity.value,
  }));

  const viewerOverlayAnimatedStyle = useAnimatedStyle(() => ({
    opacity: viewerCloseOpacity.value,
  }));

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

    if (!log) {
      return (
        <View style={styles.centerState}>
          <ThemedText>No scan log data found.</ThemedText>
        </View>
      );
    }

    const statusColor = getStatusColor(log.status, colors.text);
    const statusBg = getStatusBadgeBg(log.status);

    return (
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <CollapsibleSection
          title={`Images (${log.file_count})`}
          cardBg={colors.cardBackground}
          borderColor={colors.divider}
          rightElement={
            <View style={[styles.statusPill, { backgroundColor: statusBg }]}>
              <ThemedText style={[styles.statusPillText, { color: statusColor }]}>
                {formatStatusLabel(log.status)}
              </ThemedText>
            </View>
          }>
          {log.files.map((file) => (
            <View key={file.id} style={styles.imagePairWrapper}>
              <ThemedText style={[styles.filename, { color: colors.text }]}>{file.original_filename}</ThemedText>
              <ImagePairRow
                file={file}
                colors={colors}
                onPressOriginal={() => openImageViewer(file.original_image_data_url, 'Original')}
                onPressProcessed={() =>
                  file.processed_image_data_url && openImageViewer(file.processed_image_data_url, 'Processed')
                }
              />
            </View>
          ))}
        </CollapsibleSection>

        <CollapsibleSection
          title="Details"
          cardBg={colors.cardBackground}
          borderColor={colors.divider}
          chevronColor={colors.icon}>
          <View style={styles.detailRow}>
            <ThemedText style={[styles.detailLabel, { color: colors.text }]}>Created</ThemedText>
            <ThemedText style={[styles.detailValue, { color: colors.text }]}>{formatTimestamp(log.created_at)}</ThemedText>
          </View>
          <View style={styles.detailRow}>
            <ThemedText style={[styles.detailLabel, { color: colors.text }]}>Processed At</ThemedText>
            <ThemedText style={[styles.detailValue, { color: colors.text }]}>{formatTimestamp(log.updated_at)}</ThemedText>
          </View>
          <View style={styles.detailRow}>
            <ThemedText style={[styles.detailLabel, { color: colors.text }]}>Linked Claims</ThemedText>
            {log.linked_claim_ids.length > 0 ? (
              <View style={styles.linkedClaimsContainer}>
                {log.linked_claim_ids.map((id) => (
                  <Pressable
                    key={id}
                    onPress={() => router.push(`/admin/claims/${id}` as Href)}
                    style={[styles.linkedClaimButton, { borderColor: colors.primary }]}>
                    <ThemedText style={[styles.linkedClaimButtonText, { color: colors.primary }]}>
                      Open Claim #{id}
                    </ThemedText>
                  </Pressable>
                ))}
              </View>
            ) : (
              <ThemedText style={[styles.detailValue, { color: colors.text }]}>Not linked</ThemedText>
            )}
          </View>
          <View style={styles.detailRow}>
            <ThemedText style={[styles.detailLabel, { color: colors.text }]}>Model</ThemedText>
            <ThemedText style={[styles.detailValue, { color: colors.text }]}>
              {log.primary_model_name ? `${log.primary_model_provider}/${log.primary_model_name}` : 'Unavailable'}
            </ThemedText>
          </View>
          <View style={styles.detailRow}>
            <ThemedText style={[styles.detailLabel, { color: colors.text }]}>Execution</ThemedText>
            <ThemedText style={[styles.detailValue, { color: colors.text }]}>
              {log.ai_execution_mode ? formatStatusLabel(log.ai_execution_mode) : 'Unavailable'}
            </ThemedText>
          </View>
          {log.extraction_meta_payload && typeof log.extraction_meta_payload['overall_confidence'] === 'number' ? (
            <View style={styles.detailRow}>
              <ThemedText style={[styles.detailLabel, { color: colors.text }]}>Overall Confidence</ThemedText>
              <ThemedText style={[styles.detailValue, { color: colors.text }]}>
                {`${Math.round(Number(log.extraction_meta_payload['overall_confidence']) * 100)}%`}
              </ThemedText>
            </View>
          ) : null}
          {log.extraction_meta_payload && typeof log.extraction_meta_payload['review_required'] === 'boolean' ? (
            <View style={styles.detailRow}>
              <ThemedText style={[styles.detailLabel, { color: colors.text }]}>Needs Review</ThemedText>
              <ThemedText style={[styles.detailValue, { color: colors.text }]}>
                {Boolean(log.extraction_meta_payload['review_required']) ? 'Yes' : 'No'}
              </ThemedText>
            </View>
          ) : null}
          {Array.isArray(log.extraction_meta_payload?.['review_reasons']) &&
          (log.extraction_meta_payload?.['review_reasons'] as unknown[]).length > 0 ? (
            <View style={[styles.detailRow, styles.detailRowTopAligned]}>
              <ThemedText style={[styles.detailLabel, { color: colors.text }]}>Review Reasons</ThemedText>
              <View style={styles.detailMultilineValue}>
                {(log.extraction_meta_payload['review_reasons'] as string[]).map((reason) => (
                  <View key={reason} style={styles.reviewReasonRow}>
                    <ThemedText style={[styles.reviewReasonBullet, { color: colors.text }]}>
                      {'\u2022'}
                    </ThemedText>
                    <ThemedText style={[styles.reviewReasonText, { color: colors.text }]}>
                      {reason}
                    </ThemedText>
                  </View>
                ))}
              </View>
            </View>
          ) : null}
        </CollapsibleSection>

        <CollapsibleSection
          title="Technical Data"
          cardBg={colors.cardBackground}
          borderColor={colors.divider}
          chevronColor={colors.icon}
          contentStyle={styles.technicalDataSectionContent}>
          {payloadTabs.length > 0 ? (
            <>
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={[
                  styles.payloadTabBar,
                  {
                    backgroundColor: `${colors.divider}18`,
                    borderColor: colors.divider,
                  },
                ]}>
                {payloadTabs.map((tab) => {
                  const isActive = tab.key === activePayloadTab?.key;

                  return (
                    <Pressable
                      key={tab.key}
                      onPress={() => setSelectedPayloadTab(tab.key)}
                      style={[
                        styles.payloadTabButton,
                        isActive && {
                          backgroundColor: colors.cardBackground,
                          borderColor: colors.primary,
                        },
                      ]}>
                      <ThemedText
                        numberOfLines={1}
                        style={[
                          styles.payloadTabText,
                          {
                            color: isActive ? colors.primary : `${colors.text}B3`,
                          },
                        ]}>
                        {tab.label}
                      </ThemedText>
                    </Pressable>
                  );
                })}
              </ScrollView>

              {activePayloadTab && (
                <View style={styles.payloadPanel}>
                  <ThemedText style={[styles.payloadPanelTitle, { color: `${colors.text}B3` }]}> 
                    {activePayloadTab.label} Payload
                  </ThemedText>
                  {activePayloadTab.key === 'runs' && log ? (
                    <RunsPayloadView
                      runs={log.ai_model_runs}
                      colors={colors}
                      expandedRunPayloads={expandedRunPayloads}
                      onToggleRunPayload={toggleRunPayload}
                    />
                  ) : activePayloadTab.key === 'fusion' ? (
                    asRecord(activePayloadTab.payload) ? (
                      <FusionPayloadView
                        payload={asRecord(activePayloadTab.payload) as Record<string, unknown>}
                        runs={log.ai_model_runs}
                        colors={colors}
                        expandedFusionFields={expandedFusionFields}
                        expandedFusionRunValues={expandedFusionRunValues}
                        onToggleFusionField={toggleFusionField}
                        onToggleFusionRunValues={toggleFusionRunValues}
                      />
                    ) : (
                      renderPayload(activePayloadTab.payload, colors)
                    )
                  ) : (
                    renderPayload(activePayloadTab.payload, colors)
                  )}
                </View>
              )}
            </>
          ) : (
            <ThemedText style={[styles.placeholderText, { color: colors.text }]}>No data recorded.</ThemedText>
          )}
        </CollapsibleSection>
      </ScrollView>
    );
  }, [
    activePayloadTab,
    colors,
    error,
    expandedFusionFields,
    expandedFusionRunValues,
    expandedRunPayloads,
    isLoading,
    log,
    openImageViewer,
    payloadTabs,
    toggleFusionField,
    toggleFusionRunValues,
    toggleRunPayload,
  ]);

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={[styles.header, { backgroundColor: colors.cardBackground, borderColor: colors.divider }]}>
        <Pressable style={styles.backButton} onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={18} color={colors.primary} />
          <ThemedText style={[styles.backButtonText, { color: colors.primary }]}>Back</ThemedText>
        </Pressable>
        {log && (
          <ThemedText style={styles.headerTitle}>Scan #{log.id}</ThemedText>
        )}
        {log && (
          <View style={[styles.statusPill, { backgroundColor: getStatusBadgeBg(log.status) }]}>
            <ThemedText style={[styles.statusPillText, { color: getStatusColor(log.status, colors.text) }]}>
              {formatStatusLabel(log.status)}
            </ThemedText>
          </View>
        )}
      </View>

      {log && (
        <View style={[styles.userStrip, { borderBottomColor: colors.divider }]}>
          <ThemedText style={styles.userName}>{log.user_full_name}</ThemedText>
          <ThemedText style={styles.userEmail}>{log.user_email}</ThemedText>
        </View>
      )}

      {content}

      <Modal animationType="fade" transparent visible={viewerImageUri !== null} onRequestClose={closeImageViewer}>
        <Animated.View style={[styles.viewerOverlay, viewerOverlayAnimatedStyle]}>
          <SafeAreaView style={styles.viewerContainer}>
            <Pressable style={styles.viewerCloseButton} onPress={closeImageViewer}>
              <Ionicons name="close" size={22} color="#FFFFFF" />
            </Pressable>

            <ThemedText style={styles.viewerTitle}>{viewerImageTitle}</ThemedText>

            <GestureHandlerRootView style={styles.viewerStageRoot}>
              <View
                style={styles.viewerImageStage}
                onLayout={(event) => {
                  const { width, height } = event.nativeEvent.layout;
                  setViewerStageSize((current) => {
                    if (current.width === width && current.height === height) {
                      return current;
                    }
                    return { width, height };
                  });
                }}>
                <GestureDetector gesture={viewerGesture}>
                  <Animated.View
                    style={[
                      styles.viewerImageShell,
                      viewerImageLayout
                        ? {
                            width: viewerImageLayout.baseWidth,
                            height: viewerImageLayout.baseHeight,
                          }
                        : styles.viewerImageShellFallback,
                      viewerImageAnimatedStyle,
                    ]}>
                    {viewerImageUri ? (
                      <Image
                        source={{ uri: viewerImageUri }}
                        style={styles.viewerImage}
                        onLoad={(e) => {
                          const { width, height } = e.nativeEvent.source;
                          if (width && height) {
                            setViewerImageNaturalSize({ width, height });
                          }
                        }}
                      />
                    ) : null}
                  </Animated.View>
                </GestureDetector>
              </View>
            </GestureHandlerRootView>
          </SafeAreaView>
        </Animated.View>
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
  detailLabel: {
    fontSize: 14,
    opacity: 0.6,
    width: 140,
  },
  detailRow: {
    flexDirection: 'row',
    paddingVertical: 8,
  },
  detailRowTopAligned: {
    alignItems: 'flex-start',
  },
  detailValue: {
    flex: 1,
    fontSize: 14,
    fontWeight: '500',
  },
  detailMultilineValue: {
    flex: 1,
    gap: 8,
    paddingTop: 1,
  },
  errorText: {
    textAlign: 'center',
  },
  filename: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  header: {
    alignItems: 'center',
    borderBottomWidth: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  headerTitle: {
    fontSize: 16,
    fontWeight: '700',
  },
  imagePairColumn: {
    flex: 1,
    alignItems: 'center',
  },
  imagePairContainer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  imagePairDivider: {
    height: '100%',
    marginHorizontal: 12,
    width: 1,
  },
  imagePairLabel: {
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 8,
    opacity: 0.7,
  },
  imageMetaBlock: {
    alignItems: 'center',
    marginTop: 4,
  },
  imagePairThumbnail: {
    borderRadius: 8,
    height: 80,
    resizeMode: 'cover',
    width: 120,
  },
  imagePairWrapper: {
    marginBottom: 16,
  },
  imagePlaceholder: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  imageSize: {
    fontSize: 11,
    opacity: 0.6,
  },
  placeholderText: {
    fontSize: 14,
    opacity: 0.6,
    fontStyle: 'italic',
  },
  section: {
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
  statusPill: {
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  statusPillText: {
    fontSize: 12,
    fontWeight: '600',
  },
  userEmail: {
    fontSize: 14,
    opacity: 0.7,
  },
  userName: {
    fontSize: 15,
    fontWeight: '600',
  },
  userStrip: {
    borderBottomWidth: 1,
    paddingHorizontal: 20,
    paddingVertical: 12,
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
  viewerImageShell: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  viewerImageShellFallback: {
    height: '100%',
    width: '100%',
  },
  viewerImageStage: {
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center',
    overflow: 'hidden',
  },
  viewerOverlay: {
    backgroundColor: 'rgba(10, 16, 24, 0.96)',
    flex: 1,
  },
  viewerStageRoot: {
    flex: 1,
  },
  viewerTitle: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    paddingTop: 24,
    textAlign: 'center',
  },
  jsonBlock: {
    borderRadius: 10,
    borderWidth: 1,
    padding: 12,
  },
  jsonText: {
    fontSize: 12,
    fontFamily: 'monospace',
    lineHeight: 18,
    width: '100%',
  },
  linkedClaimButton: {
    alignItems: 'center',
    borderRadius: 999,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  linkedClaimButtonText: {
    fontSize: 12,
    fontWeight: '700',
  },
  linkedClaimsContainer: {
    flex: 1,
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    justifyContent: 'flex-end',
  },
  payloadPanel: {
    gap: 8,
  },
  payloadPanelTitle: {
    fontSize: 12,
    fontWeight: '600',
    letterSpacing: 0.3,
    textTransform: 'uppercase',
  },
  payloadTabBar: {
    borderRadius: 12,
    borderWidth: 1,
    flexDirection: 'row',
    gap: 8,
    marginBottom: 12,
    padding: 4,
    paddingRight: 12,
  },
  payloadTabButton: {
    alignItems: 'center',
    borderColor: 'transparent',
    borderRadius: 9,
    borderWidth: 1,
    justifyContent: 'center',
    minWidth: 84,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  payloadTabText: {
    fontSize: 12,
    fontWeight: '600',
  },
  reviewReasonBullet: {
    fontSize: 16,
    lineHeight: 20,
    paddingTop: 1,
  },
  reviewReasonRow: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: 8,
  },
  reviewReasonText: {
    flex: 1,
    fontSize: 14,
    fontWeight: '500',
    lineHeight: 20,
  },
  runCard: {
    borderRadius: 10,
    borderWidth: 1,
    gap: 8,
    padding: 10,
  },
  runCardHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  runCardMetaText: {
    fontSize: 12,
  },
  runCardTitle: {
    fontSize: 14,
    fontWeight: '700',
  },
  runCardTitleBlock: {
    gap: 2,
  },
  runErrorBlock: {
    borderRadius: 8,
    borderWidth: 1,
    gap: 4,
    padding: 8,
  },
  runErrorMessage: {
    fontSize: 12,
    lineHeight: 18,
  },
  runErrorTitle: {
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
  },
  runGroup: {
    gap: 8,
  },
  runGroupHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  runGroupMeta: {
    fontSize: 12,
    fontWeight: '600',
  },
  runGroupTitle: {
    flex: 1,
    fontSize: 13,
    fontWeight: '700',
  },
  runPayloadToggle: {
    alignSelf: 'flex-start',
    paddingVertical: 2,
  },
  runPayloadToggleText: {
    fontSize: 12,
    fontWeight: '700',
  },
  runsPanel: {
    gap: 12,
  },
  runsSummaryLabel: {
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 0.3,
    textTransform: 'uppercase',
  },
  runsSummaryPill: {
    borderRadius: 8,
    borderWidth: 1,
    flex: 1,
    gap: 2,
    minWidth: 70,
    paddingHorizontal: 8,
    paddingVertical: 6,
  },
  runsSummaryRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  runsSummaryValue: {
    fontSize: 15,
    fontWeight: '700',
  },
  runStatusPill: {
    borderRadius: 999,
    borderWidth: 1,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  runStatusPillText: {
    fontSize: 11,
    fontWeight: '700',
  },
  fusionBulletList: {
    gap: 6,
  },
  fusionBulletMark: {
    fontSize: 16,
    lineHeight: 20,
  },
  fusionBulletRow: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: 8,
  },
  fusionBulletText: {
    flex: 1,
    fontSize: 13,
    lineHeight: 19,
  },
  fusionCandidateItem: {
    borderRadius: 8,
    borderWidth: 1,
    gap: 4,
    padding: 8,
  },
  fusionCandidateMeta: {
    fontSize: 11,
    lineHeight: 16,
  },
  fusionCandidateValue: {
    fontSize: 12,
    fontWeight: '700',
  },
  fusionCandidatesList: {
    gap: 8,
  },
  fusionFieldCard: {
    borderRadius: 10,
    borderWidth: 1,
    gap: 6,
    padding: 10,
  },
  fusionFieldConfidence: {
    fontSize: 13,
    fontWeight: '700',
  },
  fusionFieldHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  fusionFieldPath: {
    flex: 1,
    fontSize: 13,
    fontWeight: '700',
    paddingRight: 8,
  },
  fusionList: {
    gap: 8,
  },
  fusionListItem: {
    borderRadius: 8,
    borderWidth: 1,
    gap: 4,
    padding: 8,
  },
  fusionListItemBadge: {
    fontSize: 11,
    fontWeight: '700',
  },
  fusionListItemHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  fusionListItemText: {
    fontSize: 12,
    lineHeight: 18,
  },
  fusionListItemTitle: {
    flex: 1,
    fontSize: 12,
    fontWeight: '700',
    paddingRight: 8,
  },
  fusionPanel: {
    gap: 12,
  },
  fusionSectionCard: {
    borderRadius: 10,
    borderWidth: 1,
    gap: 8,
    padding: 10,
  },
  fusionSectionHeader: {
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  fusionSectionMeta: {
    fontSize: 11,
    fontWeight: '600',
  },
  fusionSectionTitle: {
    fontSize: 13,
    fontWeight: '700',
  },
  fusionRunValuesList: {
    gap: 8,
  },
  fusionRunValueHeader: {
    fontSize: 11,
    fontWeight: '700',
  },
  fusionRunValueItem: {
    borderRadius: 8,
    borderWidth: 1,
    gap: 4,
    padding: 8,
  },
  fusionRunValueText: {
    fontSize: 12,
    lineHeight: 17,
  },
  fusionSummaryLabel: {
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 0.3,
    textTransform: 'uppercase',
  },
  fusionSummaryPill: {
    borderRadius: 8,
    borderWidth: 1,
    flex: 1,
    gap: 2,
    minWidth: 70,
    paddingHorizontal: 8,
    paddingVertical: 6,
  },
  fusionSummaryRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  fusionSummaryValue: {
    fontSize: 15,
    fontWeight: '700',
  },
  fusionToggle: {
    alignSelf: 'flex-start',
    paddingVertical: 2,
  },
  fusionToggleText: {
    fontSize: 12,
    fontWeight: '700',
  },
  technicalDataSectionContent: {
    paddingTop: 0,
    paddingHorizontal: 12,
    paddingBottom: 12,
  },
});
