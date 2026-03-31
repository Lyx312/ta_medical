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
