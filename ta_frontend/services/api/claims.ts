import * as FileSystem from 'expo-file-system/legacy';
import * as Sharing from 'expo-sharing';
import { Platform } from 'react-native';

import { getJson, postFormData, postFormDataWithProgress } from './client';
import { API_BASE_URL } from './config';
import { getAuthToken } from './storage';
import type { ClaimDetailResponse, ClaimResponse, CreateClaimPayload } from './types';

export type InhealthFormSaveResult =
  | {
      status: 'saved';
      filename: string;
      uri: string;
    }
  | {
      status: 'shared';
      filename: string;
      uri: string;
    }
  | {
      status: 'cancelled';
      filename: string;
    };

export async function createClaim(payload: CreateClaimPayload): Promise<ClaimDetailResponse> {
  const formData = new FormData();

  formData.append('nama_pasien', payload.nama_pasien);
  formData.append('tanggal_pelayanan', payload.tanggal_pelayanan);
  formData.append('fasilitas_kesehatan', payload.fasilitas_kesehatan);
  formData.append('jenis_manfaat', payload.jenis_manfaat);
  formData.append('nominal_klaim', payload.nominal_klaim.toString());
  formData.append('keterangan_penyakit', payload.keterangan_penyakit);
  formData.append('kronologi', payload.kronologi);
  if (payload.medicines && payload.medicines.length > 0) {
    formData.append('medicines_payload', JSON.stringify(payload.medicines));
  }
  if (payload.source_scan_log_id !== undefined) {
    formData.append('source_scan_log_id', payload.source_scan_log_id.toString());
  }
  if (payload.selected_patient_type) {
    formData.append('selected_patient_type', payload.selected_patient_type);
  }
  if (payload.family_member_id !== undefined) {
    formData.append('family_member_id', payload.family_member_id.toString());
  }

  for (const uri of payload.fileUris) {
    const filename = uri.split('/').pop() ?? `file_${Date.now()}.jpg`;
    const type = filename.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';

    formData.append('files', {
      uri,
      name: filename,
      type,
    } as never);
  }

  return postFormData<ClaimDetailResponse>('/claims', formData, true);
}

export async function createClaimWithProgress(
  payload: CreateClaimPayload,
  onUploadProgress: (progressPercent: number) => void,
): Promise<ClaimDetailResponse> {
  const formData = new FormData();

  formData.append('nama_pasien', payload.nama_pasien);
  formData.append('tanggal_pelayanan', payload.tanggal_pelayanan);
  formData.append('fasilitas_kesehatan', payload.fasilitas_kesehatan);
  formData.append('jenis_manfaat', payload.jenis_manfaat);
  formData.append('nominal_klaim', payload.nominal_klaim.toString());
  formData.append('keterangan_penyakit', payload.keterangan_penyakit);
  formData.append('kronologi', payload.kronologi);
  if (payload.medicines && payload.medicines.length > 0) {
    formData.append('medicines_payload', JSON.stringify(payload.medicines));
  }
  if (payload.source_scan_log_id !== undefined) {
    formData.append('source_scan_log_id', payload.source_scan_log_id.toString());
  }
  if (payload.selected_patient_type) {
    formData.append('selected_patient_type', payload.selected_patient_type);
  }
  if (payload.family_member_id !== undefined) {
    formData.append('family_member_id', payload.family_member_id.toString());
  }

  for (const uri of payload.fileUris) {
    const filename = uri.split('/').pop() ?? `file_${Date.now()}.jpg`;
    const type = filename.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';

    formData.append('files', {
      uri,
      name: filename,
      type,
    } as never);
  }

  return postFormDataWithProgress<ClaimDetailResponse>('/claims', formData, onUploadProgress, true);
}

export function getClaims(): Promise<ClaimResponse[]> {
  return getJson<ClaimResponse[]>('/claims', true);
}

export function getClaim(claimId: number): Promise<ClaimDetailResponse> {
  return getJson<ClaimDetailResponse>(`/claims/${claimId}`, true);
}

export async function downloadInhealthClaimForm(claimId: number): Promise<InhealthFormSaveResult> {
  if (!FileSystem.cacheDirectory) {
    throw { message: 'File download is not available on this device.' };
  }

  const token = await getAuthToken();
  const filename = `inhealth-claim-${claimId}.pdf`;
  const destination = `${FileSystem.cacheDirectory}${filename}`;
  const result = await FileSystem.downloadAsync(
    `${API_BASE_URL}/claims/${claimId}/inhealth-form`,
    destination,
    {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    },
  );

  if (result.status < 200 || result.status >= 300) {
    throw { message: 'Unable to download the InHealth claim form.' };
  }

  if (Platform.OS === 'android') {
    const initialDirectoryUri =
      FileSystem.StorageAccessFramework.getUriForDirectoryInRoot('Download') ??
      FileSystem.StorageAccessFramework.getUriForDirectoryInRoot('Downloads');
    const permissions = await FileSystem.StorageAccessFramework.requestDirectoryPermissionsAsync(initialDirectoryUri);
    if (!permissions.granted) {
      return { status: 'cancelled', filename };
    }

    const base64Pdf = await FileSystem.readAsStringAsync(result.uri, {
      encoding: FileSystem.EncodingType.Base64,
    });
    const targetUri = await FileSystem.StorageAccessFramework.createFileAsync(
      permissions.directoryUri,
      filename.replace(/\.pdf$/i, ''),
      'application/pdf',
    );
    await FileSystem.StorageAccessFramework.writeAsStringAsync(targetUri, base64Pdf, {
      encoding: FileSystem.EncodingType.Base64,
    });
    return {
      status: 'saved',
      filename,
      uri: targetUri,
    };
  }

  if (await Sharing.isAvailableAsync()) {
    await Sharing.shareAsync(result.uri, {
      dialogTitle: 'Save InHealth Form',
      mimeType: 'application/pdf',
      UTI: 'com.adobe.pdf',
    });
    return {
      status: 'shared',
      filename,
      uri: result.uri,
    };
  }

  return {
    status: 'saved',
    filename,
    uri: result.uri,
  };
}
