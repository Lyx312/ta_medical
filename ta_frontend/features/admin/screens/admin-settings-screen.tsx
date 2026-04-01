import { useCallback, useState } from 'react';
import { ActivityIndicator, Alert, ScrollView, StyleSheet, Switch, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';

import { ThemedText } from '@/components/themed-text';
import { Colors } from '@/constants/theme';
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

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <ThemedText type="title">AI Configuration</ThemedText>
        <ThemedText style={styles.subtitle}>Choose which seeded AI models are active for backend inference.</ThemedText>
        <ThemedText style={styles.helperText}>
          Multiple models can be active, but the backend currently uses only the first active model in database order.
        </ThemedText>

        {isLoading ? (
          <View style={styles.centerState}>
            <ActivityIndicator color={colors.primary} />
          </View>
        ) : error ? (
          <View style={styles.centerState}>
            <ThemedText style={[styles.errorText, { color: colors.error }]}>{error}</ThemedText>
          </View>
        ) : (
          <View style={[styles.modelSection, { backgroundColor: colors.cardBackground, borderColor: colors.divider }]}>
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
                  </View>
                  <Switch
                    value={model.is_active}
                    onValueChange={() => void handleToggleModel(model)}
                    disabled={updatingModelId === model.id}
                    trackColor={{ false: colors.divider, true: colors.secondary }}
                  />
                </View>
              ))}
            </View>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  centerState: {
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 180,
    paddingHorizontal: 24,
  },
  container: {
    flex: 1,
  },
  content: {
    padding: 20,
    paddingBottom: 32,
  },
  errorText: {
    textAlign: 'center',
  },
  helperText: {
    lineHeight: 20,
    marginTop: 12,
    opacity: 0.75,
  },
  modelInfo: {
    flex: 1,
    gap: 4,
  },
  modelList: {
    gap: 0,
  },
  modelMeta: {
    fontSize: 14,
    opacity: 0.75,
  },
  modelName: {
    fontSize: 17,
    fontWeight: '700',
  },
  modelRow: {
    alignItems: 'center',
    flexDirection: 'row',
    gap: 16,
    justifyContent: 'space-between',
    paddingVertical: 16,
  },
  modelSection: {
    borderRadius: 14,
    borderWidth: 1,
    marginTop: 24,
    overflow: 'hidden',
    paddingHorizontal: 16,
  },
  subtitle: {
    marginTop: 8,
    opacity: 0.7,
  },
});
