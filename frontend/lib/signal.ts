import { Signal } from '../data/types';

/** Bullish signals share the same directional semantics. */
export function isBullishSignal(signal: Signal): boolean {
  return signal === 'STRONG UP' || signal === 'UP';
}

export function isBearishSignal(signal: Signal): boolean {
  return signal === 'DOWN';
}

/**
 * Maps a domain signal to a CSS-module class key.
 * Consumers pass their own module classes so this stays style-agnostic.
 */
export function signalTone(signal: Signal): 'strongUp' | 'up' | 'down' | 'neutral' {
  switch (signal) {
    case 'STRONG UP':
      return 'strongUp';
    case 'UP':
      return 'up';
    case 'DOWN':
      return 'down';
    default:
      return 'neutral';
  }
}

export function formatSigned(value: number, digits = 2): string {
  const abs = Math.abs(value).toFixed(digits);
  if (value > 0) return `+${abs}`;
  if (value < 0) return `-${abs}`;
  return abs;
}

export function formatPercent(value: number, digits = 1): string {
  return `${value.toFixed(digits)}%`;
}
