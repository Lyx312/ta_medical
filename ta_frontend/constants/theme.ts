// --- Core Brand Palette ---
const brandNavy = '#1A365D';
const brandSkyBlue = '#4299E1';
const brandMint = '#48BB78';
const brandOffWhite = '#F7FAFC';
const brandLightGray = '#E2E8F0';

// --- System States ---
const sysError = '#E53E3E';
const sysWarning = '#DD6B20';
const sysInfo = '#3182CE';

// --- Status Colors ---
const statusApproved = '#22c55e';
const statusPending = '#eab308';
const statusRejected = sysError;

export function getStatusColor(status: string, textColor: string) {
  const normalizedStatus = status.toLowerCase();

  if (normalizedStatus.includes('approve')) {
    return statusApproved;
  }

  if (normalizedStatus.includes('pending') || normalizedStatus.includes('review')) {
    return statusPending;
  }

  if (normalizedStatus.includes('reject')) {
    return statusRejected;
  }

  return textColor;
}

export function getStatusBadgeBg(status: string) {
  const normalizedStatus = status.toLowerCase();

  if (normalizedStatus.includes('approve')) {
    return `${statusApproved}20`;
  }

  if (normalizedStatus.includes('pending') || normalizedStatus.includes('review')) {
    return `${statusPending}20`;
  }

  if (normalizedStatus.includes('reject')) {
    return `${statusRejected}20`;
  }

  return '#71809620';
}

export const Colors = {
  light: {
    // Standard Navigation & Layout
    text: brandNavy, // Using Navy for standard text maximizes trust and readability
    background: brandOffWhite,
    tint: brandSkyBlue,
    icon: '#718096', // Mid-gray for unselected icons (accessible contrast)
    tabIconDefault: '#718096',
    tabIconSelected: brandSkyBlue,
    
    // Semantic App Elements
    primary: brandNavy,
    secondary: brandSkyBlue,
    cta: brandMint,
    cardBackground: '#FFFFFF', // Pure white cards pop perfectly against the off-white background
    divider: brandLightGray,
    
    // System States
    error: sysError,
    warning: sysWarning,
    info: sysInfo,
  },
  dark: {
    // Standard Navigation & Layout
    text: brandOffWhite,
    background: '#1A202C', // Deep slate replaces pure black for a softer dark mode
    tint: '#63B3ED', // Lightened sky blue for better dark mode contrast
    icon: '#A0AEC0',
    tabIconDefault: '#A0AEC0',
    tabIconSelected: '#63B3ED',
    
    // Semantic App Elements
    primary: '#90CDF4', // Shifted to a lighter blue so headers are readable in dark mode
    secondary: '#63B3ED',
    cta: brandMint, // Mint green retains its success/action meaning in dark mode
    cardBackground: '#2D3748', // Elevated dark slate to distinguish cards from the background
    divider: '#4A5568',
    
    // System States (Lightened for dark mode accessibility)
    error: '#FC8181', 
    warning: '#F6AD55',
    info: '#63B3ED',
  },
};
