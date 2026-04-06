import type {
  AdminClaimSummary,
  AdminScanLogQueueSummary,
  AdminScanLogSummary,
  ClaimResponse,
} from '@/services/api/types';
import type { Colors } from '@/constants/theme';

export const RECENT_DAYS_WINDOW = 7;
export const HIGH_VALUE_CLAIM_THRESHOLD = 500000;

export function formatAdminTimestamp(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

export function formatScanStatusLabel(status: string) {
  return status.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function formatBenefitTypeLabel(value: string) {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function isFailedScanStatus(status: string) {
  return status !== 'pending' && status !== 'succeeded';
}

export function isRecentTimestamp(value: string, days: number = RECENT_DAYS_WINDOW) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return false;
  }

  const now = Date.now();
  return now - date.getTime() <= days * 24 * 60 * 60 * 1000;
}

export function isHighValueClaim(amount: number) {
  return amount >= HIGH_VALUE_CLAIM_THRESHOLD;
}

export function summarizeClaims(claims: ClaimResponse[]): AdminClaimSummary {
  return {
    total_claims: claims.length,
    total_claim_amount: claims.reduce((total, claim) => total + claim.nominal_klaim, 0),
    linked_scan_count: claims.filter((claim) => claim.source_scan_log_id !== null).length,
    manual_submission_count: claims.filter((claim) => claim.source_scan_log_id === null).length,
  };
}

export function summarizeScanLogs(logs: AdminScanLogSummary[]): AdminScanLogQueueSummary {
  return {
    total_logs: logs.length,
    succeeded_count: logs.filter((log) => log.status === 'succeeded').length,
    failed_count: logs.filter((log) => isFailedScanStatus(log.status)).length,
    unlinked_count: logs.filter((log) => log.linked_claim_ids.length === 0).length,
  };
}

export function getScanStatusPresentation(status: string, colors: typeof Colors.light) {
  if (status === 'succeeded') {
    return {
      textColor: colors.cta,
      backgroundColor: `${colors.cta}20`,
    };
  }

  if (status === 'pending') {
    return {
      textColor: colors.warning,
      backgroundColor: `${colors.warning}20`,
    };
  }

  return {
    textColor: colors.error,
    backgroundColor: `${colors.error}20`,
  };
}

export function getScanLinkPresentation(log: AdminScanLogSummary, colors: typeof Colors.light) {
  if (log.linked_claim_ids.length > 0) {
    return {
      label: log.linked_claim_ids.length === 1 ? 'Linked Claim' : `${log.linked_claim_ids.length} Linked Claims`,
      textColor: colors.secondary,
      backgroundColor: `${colors.secondary}20`,
    };
  }

  return {
    label: 'Unlinked',
    textColor: colors.warning,
    backgroundColor: `${colors.warning}20`,
  };
}

export function getClaimSourcePresentation(claim: ClaimResponse, colors: typeof Colors.light) {
  if (claim.source_scan_log_id !== null) {
    return {
      label: 'AI Linked',
      textColor: colors.secondary,
      backgroundColor: `${colors.secondary}20`,
    };
  }

  return {
    label: 'Manual Entry',
    textColor: colors.warning,
    backgroundColor: `${colors.warning}20`,
  };
}
