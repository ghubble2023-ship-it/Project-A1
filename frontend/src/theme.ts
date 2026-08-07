// Brutalist theme tokens for Project AI - Core Scanner
export const theme = {
  color: {
    surface: '#FFFFFF',
    onSurface: '#000000',
    surfaceSecondary: '#F4F4F5',
    surfaceTertiary: '#E4E4E7',
    surfaceInverse: '#000000',
    onSurfaceInverse: '#FFFFFF',
    brandPrimary: '#000000',
    onBrandPrimary: '#FFFFFF',
    success: '#00E676',
    onSuccess: '#000000',
    warning: '#FFEA00',
    onWarning: '#000000',
    error: '#FF2E00',
    onError: '#FFFFFF',
    border: '#E4E4E7',
    borderStrong: '#000000',
    divider: '#E4E4E7',
    muted: '#71717A',
  },
  font: {
    display: 'SpaceGrotesk_700Bold',
    displayRegular: 'SpaceGrotesk_500Medium',
    mono: 'IBMPlexMono_500Medium',
    monoBold: 'IBMPlexMono_700Bold',
    monoRegular: 'IBMPlexMono_400Regular',
  },
  size: {
    sm: 12,
    base: 14,
    lg: 16,
    xl: 20,
    xxl: 24,
    xxxl: 32,
  },
  space: {
    xs: 4,
    sm: 8,
    md: 12,
    lg: 16,
    xl: 24,
    xxl: 32,
    xxxl: 48,
  },
  border: {
    thin: 1,
    thick: 2,
    heavy: 3,
  },
} as const;

export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'ELEVATED' | 'N/A';

export function riskColor(risk: string): { bg: string; fg: string } {
  const r = (risk || '').toUpperCase();
  if (r === 'HIGH') return { bg: theme.color.error, fg: theme.color.onError };
  if (r === 'ELEVATED') return { bg: theme.color.error, fg: theme.color.onError };
  if (r === 'MEDIUM') return { bg: theme.color.warning, fg: theme.color.onWarning };
  if (r === 'LOW') return { bg: theme.color.success, fg: theme.color.onSuccess };
  return { bg: theme.color.surfaceTertiary, fg: theme.color.onSurface };
}
