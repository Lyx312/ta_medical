import EventSource from 'react-native-sse';

import { API_BASE_URL } from './config';
import { getJson, postFormData, postFormDataWithProgress, postJson } from './client';
import { getAuthToken } from './storage';
import type {
  AIMedicalScanResult,
  AIScanJobStartResponse,
  AIScanJobStatusResponse,
  ScanJobEventName,
} from './types';

export async function scanDocuments(imageUris: string[]): Promise<AIMedicalScanResult> {
  const formData = new FormData();

  for (const uri of imageUris) {
    const filename = uri.split('/').pop() ?? `image_${Date.now()}.jpg`;
    const type = filename.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';

    formData.append('files', {
      uri,
      name: filename,
      type,
    } as never);
  }

  return postFormData<AIMedicalScanResult>('/ai/scan-documents', formData, true);
}

export async function startScanDocumentsJob(
  imageUris: string[],
  onUploadProgress: (progressPercent: number) => void,
): Promise<AIScanJobStartResponse> {
  const formData = new FormData();

  for (const uri of imageUris) {
    const filename = uri.split('/').pop() ?? `image_${Date.now()}.jpg`;
    const type = filename.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';

    formData.append('files', {
      uri,
      name: filename,
      type,
    } as never);
  }

  return postFormDataWithProgress<AIScanJobStartResponse>(
    '/ai/scan-documents/jobs',
    formData,
    onUploadProgress,
    true,
  );
}

export function getScanDocumentsJob(scanLogId: number): Promise<AIScanJobStatusResponse> {
  return getJson<AIScanJobStatusResponse>(`/ai/scan-documents/jobs/${scanLogId}`, true);
}

export function cancelScanDocumentsJob(scanLogId: number): Promise<AIScanJobStatusResponse> {
  return postJson<AIScanJobStatusResponse>(`/ai/scan-documents/jobs/${scanLogId}/cancel`, {}, true);
}

export async function openScanDocumentsJobEventStream(
  scanLogId: number,
): Promise<EventSource<ScanJobEventName>> {
  const token = await getAuthToken();
  const eventSource = new EventSource<ScanJobEventName>(
    `${API_BASE_URL}/ai/scan-documents/jobs/${scanLogId}/events`,
    {
      headers: token
        ? {
            Authorization: `Bearer ${token}`,
          }
        : undefined,
      pollingInterval: 3000,
      timeoutBeforeConnection: 0,
    },
  );
  return eventSource;
}
