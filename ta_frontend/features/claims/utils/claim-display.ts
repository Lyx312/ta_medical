import { getStatusBadgeBg, getStatusColor } from '@/constants/theme';

export function formatClaimCurrency(amount: number) {
  return new Intl.NumberFormat('id-ID', {
    style: 'currency',
    currency: 'IDR',
    maximumFractionDigits: 0,
  }).format(amount);
}

export function formatClaimServiceDate(dateString: string) {
  const dmyMatch = dateString.match(/^(\d{2})-(\d{2})-(\d{4})$/);
  if (dmyMatch) {
    const [, day, month, year] = dmyMatch;
    const date = new Date(Number(year), Number(month) - 1, Number(day));
    return Number.isNaN(date.getTime()) ? dateString : date.toLocaleDateString('id-ID');
  }

  const date = new Date(dateString);
  return Number.isNaN(date.getTime()) ? dateString : date.toLocaleDateString('id-ID');
}

export function formatClaimTimestamp(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat('id-ID', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

export function formatBenefitTypeLabel(value: string) {
  const labels: Record<string, string> = {
    rawat_jalan: 'Rawat Jalan',
    rawat_inap: 'Rawat Inap',
    emergency: 'Emergency',
    maternity: 'Maternity',
  };

  return labels[value] ?? value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function formatClaimSourceLabel(sourceScanLogId: number | null) {
  if (sourceScanLogId) {
    return `Linked scan #${sourceScanLogId}`;
  }

  return 'Manual submission';
}

export function getClaimStatusPresentation(status: string, fallbackTextColor: string) {
  return {
    textColor: getStatusColor(status, fallbackTextColor),
    backgroundColor: getStatusBadgeBg(status),
  };
}
