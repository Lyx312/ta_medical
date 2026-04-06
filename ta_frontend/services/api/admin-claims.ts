import { getJson } from './client';
import type { AdminClaimDetail, AdminClaimListParams, AdminClaimSummary, ClaimResponse } from './types';

function buildQuery(params: AdminClaimListParams = {}) {
  const query = new URLSearchParams();

  if (params.q?.trim()) {
    query.set('q', params.q.trim());
  }
  if (typeof params.linked === 'boolean') {
    query.set('linked', String(params.linked));
  }
  if (typeof params.limit === 'number') {
    query.set('limit', String(params.limit));
  }
  if (params.sort) {
    query.set('sort', params.sort);
  }

  const queryString = query.toString();
  return queryString ? `?${queryString}` : '';
}

export function getAdminClaims(params: AdminClaimListParams = {}): Promise<ClaimResponse[]> {
  return getJson<ClaimResponse[]>(`/admin/claims${buildQuery(params)}`, true);
}

export function getAdminClaimsSummary(params: Pick<AdminClaimListParams, 'q' | 'linked'> = {}): Promise<AdminClaimSummary> {
  return getJson<AdminClaimSummary>(`/admin/claims/summary${buildQuery(params)}`, true);
}

export function getAdminClaim(claimId: number): Promise<AdminClaimDetail> {
  return getJson<AdminClaimDetail>(`/admin/claims/${claimId}`, true);
}
