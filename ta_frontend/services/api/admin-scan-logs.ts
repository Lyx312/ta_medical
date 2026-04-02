import { getJson } from './client';
import type { AdminScanLogDetail, AdminScanLogSummary } from './types';

export function getAdminScanLogs(status?: string): Promise<AdminScanLogSummary[]> {
  const query = status ? `?status=${encodeURIComponent(status)}` : '';
  return getJson<AdminScanLogSummary[]>(`/admin/scan-logs${query}`, true);
}

export function getAdminScanLog(scanLogId: number): Promise<AdminScanLogDetail> {
  return getJson<AdminScanLogDetail>(`/admin/scan-logs/${scanLogId}`, true);
}
