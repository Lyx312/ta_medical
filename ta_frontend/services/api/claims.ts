import { getJson, postFormData, postFormDataWithProgress } from './client';
import type { ClaimDetailResponse, ClaimResponse, CreateClaimPayload } from './types';

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
