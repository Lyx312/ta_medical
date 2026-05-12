import { getJson, postFormData, postFormDataWithProgress, postJson } from './client';
import type { AIMedicalScanResult, AIScanJobStartResponse, AIScanJobStatusResponse } from './types';

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
