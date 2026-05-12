import { useCallback, useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Switch, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';

import { CenteredState } from '@/components/ui/centered-state';
import { ScreenHeader } from '@/components/ui/screen-header';
import { SurfaceCard } from '@/components/ui/surface-card';
import { ThemedText } from '@/components/themed-text';
import { Colors, Opacity, Radii, Spacing } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { getAIModels, updateAIModel } from '@/services/api/admin-ai-models';
import type { AIModelConfiguration, ApiError } from '@/services/api/types';

export function AdminSettingsScreen() {
  const colorScheme = useColorScheme() ?? 'light';
  const colors = Colors[colorScheme];
  const [models, setModels] = useState<AIModelConfiguration[]>([]);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [updatingModelId, setUpdatingModelId] = useState<number | null>(null);

  const loadModels = useCallback(async () => {
    setError('');
    setIsLoading(true);

    try {
      setModels(await getAIModels());
    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError.message ?? 'Unable to load AI configuration right now.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void loadModels();
    }, [loadModels]),
  );

  const handleToggleModel = async (model: AIModelConfiguration) => {
    setUpdatingModelId(model.id);
    setError('');

    try {
      const updatedModel = await updateAIModel(model.id, { is_active: !model.is_active });
      setModels((prev) =>
        prev.map((currentModel) => (currentModel.id === updatedModel.id ? updatedModel : currentModel)),
      );
    } catch (err) {
      const apiError = err as ApiError;
      Alert.alert('Update Failed', apiError.message ?? 'Unable to update AI model status.');
    } finally {
      setUpdatingModelId(null);
    }
  };

  const handlePriorityChange = async (model: AIModelConfiguration, nextPriority: number) => {
    if (nextPriority < 1) {
      return;
    }

    setUpdatingModelId(model.id);
    setError('');

    try {
      const updatedModel = await updateAIModel(model.id, { priority: nextPriority });
      setModels((prev) =>
        prev
          .map((currentModel) => (currentModel.id === updatedModel.id ? updatedModel : currentModel))
          .sort((left, right) => left.priority - right.priority || left.id - right.id),
      );
    } catch (err) {
      const apiError = err as ApiError;
      Alert.alert('Update Failed', apiError.message ?? 'Unable to update AI model priority.');
    } finally {
      setUpdatingModelId(null);
    }
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ScreenHeader
          title="AI Configuration"
          subtitle="Choose which seeded AI models are active for backend inference."
          helperText="Models are listed in order of priority for selection in scan pipelines. Incompatible models cannot be activated until they are updated to meet current scan pipeline requirements."
        />

        {isLoading ? (
          <CenteredState loading minHeight={180} />
        ) : error ? (
          <CenteredState message={error} tone="error" minHeight={180} />
        ) : (
          <SurfaceCard style={styles.modelSection}>
            <View style={styles.modelList}>
              {models.map((model, index) => (
                <View
                  key={model.id}
                  style={[
                    styles.modelRow,
                    index < models.length - 1 && { borderBottomColor: colors.divider, borderBottomWidth: 1 },
                  ]}>
                  <View style={styles.modelInfo}>
                    <ThemedText style={styles.modelName}>{model.display_name}</ThemedText>
                    <ThemedText style={styles.modelMeta}>Provider: {model.provider}</ThemedText>
                    <ThemedText style={styles.modelMeta}>
                      Priority: {model.priority} | Image input: {model.supports_image_input ? 'Yes' : 'No'}
                    </ThemedText>
                    <ThemedText
                      style={[
                        styles.modelStatus,
                        { color: model.is_scan_compatible ? colors.cta : colors.warning },
                      ]}>
                      {model.is_scan_compatible ? 'Scan ready' : 'Unsupported for current scan pipeline'}
                    </ThemedText>
                    {model.activation_block_reason ? (
                      <ThemedText style={styles.modelMeta}>{model.activation_block_reason}</ThemedText>
                    ) : null}
                  </View>
                  <View style={styles.modelControls}>
                    <View style={styles.priorityControls}>
                      <Pressable
                        style={[styles.priorityButton, { borderColor: colors.divider }]}
                        onPress={() => void handlePriorityChange(model, model.priority - 1)}
                        disabled={updatingModelId === model.id}>
                        <ThemedText style={styles.priorityButtonText}>-</ThemedText>
                      </Pressable>
                      <Pressable
                        style={[styles.priorityButton, { borderColor: colors.divider }]}
                        onPress={() => void handlePriorityChange(model, model.priority + 1)}
                        disabled={updatingModelId === model.id}>
                        <ThemedText style={styles.priorityButtonText}>+</ThemedText>
                      </Pressable>
                    </View>
                  </View>
                  <Switch
                    value={model.is_active}
                    onValueChange={() => void handleToggleModel(model)}
                    disabled={updatingModelId === model.id || !model.is_scan_compatible}
                    trackColor={{ false: colors.divider, true: colors.secondary }}
                  />
                </View>
              ))}
            </View>
          </SurfaceCard>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    padding: Spacing.xl,
    paddingBottom: Spacing.xxxl,
  },
  modelInfo: {
    flex: 1,
    gap: 4,
  },
  modelControls: {
    alignItems: 'flex-end',
    gap: Spacing.sm,
  },
  modelList: {
    gap: 0,
  },
  modelMeta: {
    fontSize: 14,
    opacity: Opacity.subtle,
  },
  modelName: {
    fontSize: 17,
    fontWeight: '700',
  },
  modelRow: {
    alignItems: 'flex-start',
    flexDirection: 'row',
    gap: Spacing.lg,
    justifyContent: 'space-between',
    paddingVertical: Spacing.lg,
  },
  modelStatus: {
    fontSize: 13,
    fontWeight: '700',
  },
  modelSection: {
    borderRadius: Radii.lg,
    marginTop: Spacing.xxl,
    overflow: 'hidden',
    paddingHorizontal: Spacing.lg,
  },
  priorityButton: {
    alignItems: 'center',
    borderRadius: Radii.pill,
    borderWidth: 1,
    height: 30,
    justifyContent: 'center',
    width: 30,
  },
  priorityButtonText: {
    fontSize: 18,
    fontWeight: '700',
    lineHeight: 20,
  },
  priorityControls: {
    flexDirection: 'row',
    gap: Spacing.sm,
  },
});
