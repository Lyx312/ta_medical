import { getJson } from './client';
import type {
  AdminScanLogDetail,
  AdminScanLogListParams,
  AdminScanLogQueueSummary,
  AdminScanLogSummary,
} from './types';

function buildQuery(params: AdminScanLogListParams = {}) {
  const query = new URLSearchParams();

  if (params.status?.trim()) {
    query.set('status', params.status.trim());
  }
  if (params.q?.trim()) {
    query.set('q', params.q.trim());
  }
  if (typeof params.linked === 'boolean') {
    query.set('linked', String(params.linked));
  }
  if (typeof params.limit === 'number') {
    query.set('limit', String(params.limit));
  }

  const queryString = query.toString();
  return queryString ? `?${queryString}` : '';
}

export function getAdminScanLogs(params: AdminScanLogListParams = {}): Promise<AdminScanLogSummary[]> {
  return getJson<AdminScanLogSummary[]>(`/admin/scan-logs${buildQuery(params)}`, true);
}

export function getAdminScanLogsSummary(
  params: Pick<AdminScanLogListParams, 'status' | 'q' | 'linked'> = {},
): Promise<AdminScanLogQueueSummary> {
  return getJson<AdminScanLogQueueSummary>(`/admin/scan-logs/summary${buildQuery(params)}`, true);
}

export function getAdminScanLog(scanLogId: number): Promise<AdminScanLogDetail> {
  return getJson<AdminScanLogDetail>(`/admin/scan-logs/${scanLogId}`, true);
}
