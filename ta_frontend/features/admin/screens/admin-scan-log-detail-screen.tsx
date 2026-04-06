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
import type { AdminScanLogDetail, AdminScanLogFile, ApiError } from '@/services/api/types';

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

type PayloadTabKey = 'validation' | 'extraction' | 'error';

type PayloadTab = {
  key: PayloadTabKey;
  label: string;
  payload: unknown;
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
              {log.ai_model_name ? `${log.ai_model_provider}/${log.ai_model_name}` : 'Unavailable'}
            </ThemedText>
          </View>
        </CollapsibleSection>

        <CollapsibleSection
          title="Technical Data"
          cardBg={colors.cardBackground}
          borderColor={colors.divider}
          chevronColor={colors.icon}
          contentStyle={styles.technicalDataSectionContent}>
          {payloadTabs.length > 0 ? (
            <>
              <View
                style={[
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
              </View>

              {activePayloadTab && (
                <View style={styles.payloadPanel}>
                  <ThemedText style={[styles.payloadPanelTitle, { color: `${colors.text}B3` }]}>
                    {activePayloadTab.label} Payload
                  </ThemedText>
                  {renderPayload(activePayloadTab.payload, colors)}
                </View>
              )}
            </>
          ) : (
            <ThemedText style={[styles.placeholderText, { color: colors.text }]}>No data recorded.</ThemedText>
          )}
        </CollapsibleSection>
      </ScrollView>
    );
  }, [activePayloadTab, colors, error, isLoading, log, openImageViewer, payloadTabs]);

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
  detailValue: {
    flex: 1,
    fontSize: 14,
    fontWeight: '500',
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
  },
  payloadTabButton: {
    alignItems: 'center',
    borderColor: 'transparent',
    borderRadius: 9,
    borderWidth: 1,
    flex: 1,
    paddingHorizontal: 10,
    paddingVertical: 9,
  },
  payloadTabText: {
    fontSize: 13,
    fontWeight: '600',
  },
  technicalDataSectionContent: {
    paddingTop: 0,
    paddingHorizontal: 12,
    paddingBottom: 12,
  },
});
