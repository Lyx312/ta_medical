import { getJson, putJson } from './client';
import type { AIModelConfiguration, UpdateAIModelPayload } from './types';

export function getAIModels(): Promise<AIModelConfiguration[]> {
  return getJson<AIModelConfiguration[]>('/admin/ai-models', true);
}

export function updateAIModel(
  modelId: number,
  payload: UpdateAIModelPayload,
): Promise<AIModelConfiguration> {
  return putJson<AIModelConfiguration>(`/admin/ai-models/${modelId}`, payload, true);
}
