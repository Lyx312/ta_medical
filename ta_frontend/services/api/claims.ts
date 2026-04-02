import { getJson, postFormData } from './client';
import type { ClaimResponse, CreateClaimPayload } from './types';

export async function createClaim(payload: CreateClaimPayload): Promise<ClaimResponse> {
  const formData = new FormData();

  formData.append('nama_pasien', payload.nama_pasien);
  formData.append('tanggal_pelayanan', payload.tanggal_pelayanan);
  formData.append('fasilitas_kesehatan', payload.fasilitas_kesehatan);
  formData.append('jenis_manfaat', payload.jenis_manfaat);
  formData.append('nominal_klaim', payload.nominal_klaim.toString());
  formData.append('keterangan_penyakit', payload.keterangan_penyakit);
  formData.append('kronologi', payload.kronologi);
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

  return postFormData<ClaimResponse>('/claims', formData, true);
}

export function getClaims(): Promise<ClaimResponse[]> {
  return getJson<ClaimResponse[]>('/claims', true);
}

export function getClaim(claimId: number): Promise<ClaimResponse> {
  return getJson<ClaimResponse>(`/claims/${claimId}`, true);
}
