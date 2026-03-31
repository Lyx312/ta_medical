import { postFormData } from './client';
import type { AIMedicalScanResult } from './types';

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
