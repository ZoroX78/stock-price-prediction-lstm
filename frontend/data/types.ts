export type ModelArch = 'LSTM' | 'Transformer';

export type Signal = 'STRONG UP' | 'UP' | 'DOWN' | 'NEUTRAL';

export type RsiDirection = 'RISING' | 'FALLING' | 'FLAT';

export interface Candle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  isPredicted: boolean;
}

export interface Feature {
  name: string;
  /** SHAP contribution; positive supports up-move */
  value: number;
  supportsUp: boolean;
}

export interface RsiData {
  val: number;
  status: string;
  direction: RsiDirection;
  sparkline: number[];
}

export interface MacdData {
  val: number;
  isPositive: boolean;
  status: string;
  time: string;
  bars: number[];
}

export interface BbandsData {
  position: string;
  value: number;
  squeeze: string;
  volume: string;
}

export interface Fold {
  fold: string;
  date: string;
  acc: number;
  correct: boolean;
}

export interface ValidationData {
  rocAuc: number;
  aggF1: number;
  dirAccuracy: number;
  folds: Fold[];
  correctRatio: number;
}

export interface ModelData {
  price: number;
  change: string;
  isPositive: boolean;
  signal: Signal;
  horizon: string;
  confidence: number;
  mlflowRun: string;
  mlflowId: string;
  valDate: string;
  inferenceTime: string;
  rsi: RsiData;
  macd: MacdData;
  bbands: BbandsData;
  history: Candle[];
  features: Feature[];
  validation: ValidationData;
}

export interface TickerData {
  name: string;
  type: string;
  LSTM: ModelData;
  Transformer: ModelData;
}

export interface ResolvedTicker {
  name: string;
  type: string;
  data: ModelData;
}

export interface WatchlistItem {
  ticker: string;
  price: number;
  signal: Signal;
  confidence: number;
  mlflowId: string;
}
