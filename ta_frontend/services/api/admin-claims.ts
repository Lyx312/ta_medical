import { getJson } from './client';
import type { AdminClaimDetail, ClaimResponse } from './types';

export function getAdminClaims(): Promise<ClaimResponse[]> {
  return getJson<ClaimResponse[]>('/admin/claims', true);
}

export function getAdminClaim(claimId: number): Promise<AdminClaimDetail> {
  return getJson<AdminClaimDetail>(`/admin/claims/${claimId}`, true);
}
