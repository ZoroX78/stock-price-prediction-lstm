/**
 * Curated demo fixtures for primary watchlist tickers.
 * Unknown tickers are generated in generateTicker.ts — do not duplicate that path here.
 *
 * Demo only: not connected to live inference or the Python training pipeline.
 */
import { Candle, Feature, ModelData, Signal, TickerData } from './types';

interface SharedMarket {
  price: number;
  change: string;
  isPositive: boolean;
  rsi: ModelData['rsi'];
  macd: ModelData['macd'];
  bbands: ModelData['bbands'];
  historyBase: Candle[];
}

interface ModelOverride {
  signal: Signal;
  confidence: number;
  mlflowRun: string;
  mlflowId: string;
  valDate: string;
  inferenceTime: string;
  horizon?: string;
  predictionCandles: Candle[];
  features: Feature[];
  validation: ModelData['validation'];
  bbandsSqueeze?: string;
}

function withPredictions(base: Candle[], predictions: Candle[]): Candle[] {
  return [...base, ...predictions];
}

function modelFrom(
  shared: SharedMarket,
  override: ModelOverride
): ModelData {
  return {
    price: shared.price,
    change: shared.change,
    isPositive: shared.isPositive,
    signal: override.signal,
    horizon: override.horizon ?? '5-Day Future',
    confidence: override.confidence,
    mlflowRun: override.mlflowRun,
    mlflowId: override.mlflowId,
    valDate: override.valDate,
    inferenceTime: override.inferenceTime,
    rsi: shared.rsi,
    macd: shared.macd,
    bbands: {
      ...shared.bbands,
      squeeze: override.bbandsSqueeze ?? shared.bbands.squeeze,
    },
    history: withPredictions(shared.historyBase, override.predictionCandles),
    features: override.features,
    validation: override.validation,
  };
}

function ticker(
  name: string,
  type: string,
  shared: SharedMarket,
  lstm: ModelOverride,
  transformer: ModelOverride
): TickerData {
  return {
    name,
    type,
    LSTM: modelFrom(shared, lstm),
    Transformer: modelFrom(shared, transformer),
  };
}

const aaplShared: SharedMarket = {
  price: 189.42,
  change: '+1.24 (+0.65%)',
  isPositive: true,
  rsi: {
    val: 64.2,
    status: 'NEUTRAL-BULLISH',
    direction: 'RISING',
    sparkline: [45, 48, 46, 52, 58, 55, 60, 64.2],
  },
  macd: {
    val: 1.24,
    isPositive: true,
    status: 'BULLISH CROSSOVER',
    time: '2D AGO',
    bars: [0.1, 0.2, 0.3, 0.1, 0.4, 0.8, 1.2, 1.24],
  },
  bbands: {
    position: 'Upper',
    value: 88,
    squeeze: 'SQUEEZE TRIGGERED',
    volume: 'VOL: LOW',
  },
  historyBase: [
    { date: 'Mar 15', open: 181.2, high: 183.5, low: 179.8, close: 182.3, isPredicted: false },
    { date: 'Mar 22', open: 182.3, high: 184.2, low: 181.0, close: 183.1, isPredicted: false },
    { date: 'Mar 29', open: 183.1, high: 183.4, low: 178.5, close: 180.2, isPredicted: false },
    { date: 'Apr 05', open: 180.2, high: 182.1, low: 177.0, close: 181.5, isPredicted: false },
    { date: 'Apr 12', open: 181.5, high: 185.0, low: 180.8, close: 184.4, isPredicted: false },
    { date: 'Apr 19', open: 184.4, high: 184.8, low: 176.2, close: 177.5, isPredicted: false },
    { date: 'Apr 26', open: 177.5, high: 180.2, low: 174.5, close: 176.8, isPredicted: false },
    { date: 'May 03', open: 176.8, high: 182.8, low: 176.0, close: 182.1, isPredicted: false },
    { date: 'May 10', open: 182.1, high: 184.5, low: 181.2, close: 183.9, isPredicted: false },
    { date: 'May 17', open: 183.9, high: 188.2, low: 183.5, close: 187.6, isPredicted: false },
    { date: 'May 24', open: 187.6, high: 190.5, low: 186.2, close: 189.42, isPredicted: false },
  ],
};

const msftShared: SharedMarket = {
  price: 420.55,
  change: '+3.85 (+0.92%)',
  isPositive: true,
  rsi: {
    val: 71.4,
    status: 'OVERBOUGHT',
    direction: 'RISING',
    sparkline: [58, 62, 60, 65, 68, 66, 69, 71.4],
  },
  macd: {
    val: 3.45,
    isPositive: true,
    status: 'BULLISH CROSSOVER',
    time: '1D AGO',
    bars: [0.5, 0.9, 1.4, 1.8, 2.2, 2.8, 3.2, 3.45],
  },
  bbands: {
    position: 'Upper',
    value: 94,
    squeeze: 'EXPANDING',
    volume: 'VOL: HIGH',
  },
  historyBase: [
    { date: 'Mar 15', open: 410.0, high: 415.2, low: 408.5, close: 412.3, isPredicted: false },
    { date: 'Mar 22', open: 412.3, high: 418.0, low: 411.0, close: 415.5, isPredicted: false },
    { date: 'Mar 29', open: 415.5, high: 422.0, low: 413.5, close: 420.2, isPredicted: false },
    { date: 'Apr 05', open: 420.2, high: 424.0, low: 418.0, close: 421.9, isPredicted: false },
    { date: 'Apr 12', open: 421.9, high: 426.5, low: 419.0, close: 423.0, isPredicted: false },
    { date: 'Apr 19', open: 423.0, high: 424.8, low: 412.0, close: 415.1, isPredicted: false },
    { date: 'Apr 26', open: 415.1, high: 419.5, low: 413.0, close: 418.4, isPredicted: false },
    { date: 'May 03', open: 418.4, high: 422.0, low: 415.8, close: 420.1, isPredicted: false },
    { date: 'May 10', open: 420.1, high: 423.5, low: 418.0, close: 421.9, isPredicted: false },
    { date: 'May 17', open: 421.9, high: 425.2, low: 419.5, close: 423.1, isPredicted: false },
    { date: 'May 24', open: 423.1, high: 425.8, low: 419.0, close: 420.55, isPredicted: false },
  ],
};

const tslaShared: SharedMarket = {
  price: 175.22,
  change: '-4.12 (-2.29%)',
  isPositive: false,
  rsi: {
    val: 41.5,
    status: 'BEARISH',
    direction: 'FALLING',
    sparkline: [55, 52, 48, 49, 44, 45, 43, 41.5],
  },
  macd: {
    val: -0.85,
    isPositive: false,
    status: 'BEARISH CROSSOVER',
    time: '3D AGO',
    bars: [0.2, 0.0, -0.2, -0.4, -0.6, -0.7, -0.8, -0.85],
  },
  bbands: {
    position: 'Lower',
    value: 15,
    squeeze: 'EXPANDING',
    volume: 'VOL: HIGH',
  },
  historyBase: [
    { date: 'Mar 15', open: 185.5, high: 188.0, low: 182.1, close: 184.2, isPredicted: false },
    { date: 'Mar 22', open: 184.2, high: 185.4, low: 179.0, close: 181.1, isPredicted: false },
    { date: 'Mar 29', open: 181.1, high: 182.5, low: 175.0, close: 176.4, isPredicted: false },
    { date: 'Apr 05', open: 176.4, high: 179.8, low: 172.5, close: 178.2, isPredicted: false },
    { date: 'Apr 12', open: 178.2, high: 183.0, low: 176.5, close: 181.4, isPredicted: false },
    { date: 'Apr 19', open: 181.4, high: 181.8, low: 170.2, close: 172.5, isPredicted: false },
    { date: 'Apr 26', open: 172.5, high: 175.0, low: 168.5, close: 170.2, isPredicted: false },
    { date: 'May 03', open: 170.2, high: 176.8, low: 169.0, close: 175.8, isPredicted: false },
    { date: 'May 10', open: 175.8, high: 178.5, low: 173.2, close: 177.1, isPredicted: false },
    { date: 'May 17', open: 177.1, high: 180.2, low: 175.5, close: 179.3, isPredicted: false },
    { date: 'May 24', open: 179.3, high: 181.5, low: 174.0, close: 175.22, isPredicted: false },
  ],
};

const nvdaShared: SharedMarket = {
  price: 920.1,
  change: '+12.45 (+1.37%)',
  isPositive: true,
  rsi: {
    val: 59.8,
    status: 'NEUTRAL',
    direction: 'FLAT',
    sparkline: [52, 54, 55, 53, 56, 58, 59, 59.8],
  },
  macd: {
    val: 4.92,
    isPositive: true,
    status: 'FLAT',
    time: '3D AGO',
    bars: [1.2, 1.8, 2.5, 3.1, 3.8, 4.2, 4.8, 4.92],
  },
  bbands: {
    position: 'Mid',
    value: 52,
    squeeze: 'NORMAL',
    volume: 'VOL: MED',
  },
  historyBase: [
    { date: 'Mar 15', open: 880.0, high: 895.5, low: 875.0, close: 890.3, isPredicted: false },
    { date: 'Mar 22', open: 890.3, high: 902.0, low: 882.0, close: 895.5, isPredicted: false },
    { date: 'Mar 29', open: 895.5, high: 915.0, low: 890.5, close: 912.2, isPredicted: false },
    { date: 'Apr 05', open: 912.2, high: 924.0, low: 908.0, close: 918.9, isPredicted: false },
    { date: 'Apr 12', open: 918.9, high: 922.5, low: 905.0, close: 910.0, isPredicted: false },
    { date: 'Apr 19', open: 910.0, high: 912.8, low: 885.0, close: 892.1, isPredicted: false },
    { date: 'Apr 26', open: 892.1, high: 905.5, low: 888.0, close: 900.4, isPredicted: false },
    { date: 'May 03', open: 900.4, high: 912.0, low: 895.8, close: 908.1, isPredicted: false },
    { date: 'May 10', open: 908.1, high: 915.5, low: 902.0, close: 912.9, isPredicted: false },
    { date: 'May 17', open: 912.9, high: 922.2, low: 909.5, close: 918.1, isPredicted: false },
    { date: 'May 24', open: 918.1, high: 924.8, low: 915.0, close: 920.1, isPredicted: false },
  ],
};

export const mockData: Record<string, TickerData> = {
  AAPL: ticker(
    'Apple Inc.',
    'EQUITY',
    aaplShared,
    {
      signal: 'STRONG UP',
      confidence: 78.4,
      mlflowRun: '#882',
      mlflowId: 'r_x882y',
      valDate: '2024-05-18T14:30Z',
      inferenceTime: '42ms',
      predictionCandles: [
        { date: 'May 31 (P)', open: 189.42, high: 192.5, low: 189.0, close: 191.8, isPredicted: true },
        { date: 'Jun 07 (P)', open: 191.8, high: 194.0, low: 191.2, close: 193.5, isPredicted: true },
        { date: 'Jun 14 (P)', open: 193.5, high: 196.2, low: 193.0, close: 195.8, isPredicted: true },
      ],
      features: [
        { name: 'Rolling 20d Vol', value: 0.14, supportsUp: true },
        { name: 'Sector Momentum', value: 0.11, supportsUp: true },
        { name: 'MACD Signal', value: 0.09, supportsUp: true },
        { name: 'RSI_14', value: 0.06, supportsUp: true },
        { name: 'Volume Trend', value: 0.05, supportsUp: true },
        { name: 'VIX Spread', value: -0.03, supportsUp: false },
        { name: 'Short Interest', value: -0.04, supportsUp: false },
      ],
      validation: {
        rocAuc: 0.824,
        aggF1: 0.761,
        dirAccuracy: 61.2,
        folds: [
          { fold: 'Fold 1', date: '2024-Q1', acc: 62.4, correct: true },
          { fold: 'Fold 2', date: '2023-Q4', acc: 60.8, correct: true },
          { fold: 'Fold 3', date: '2023-Q3', acc: 61.5, correct: true },
        ],
        correctRatio: 61.2,
      },
    },
    {
      signal: 'UP',
      confidence: 72.1,
      mlflowRun: '#883',
      mlflowId: 'r_x883y',
      valDate: '2024-05-18T14:32Z',
      inferenceTime: '55ms',
      bbandsSqueeze: 'NORMAL',
      predictionCandles: [
        { date: 'May 31 (P)', open: 189.42, high: 191.8, low: 188.5, close: 190.5, isPredicted: true },
        { date: 'Jun 07 (P)', open: 190.5, high: 193.1, low: 190.0, close: 192.1, isPredicted: true },
        { date: 'Jun 14 (P)', open: 192.1, high: 194.5, low: 191.8, close: 193.8, isPredicted: true },
      ],
      features: [
        { name: 'Rolling 20d Vol', value: 0.12, supportsUp: true },
        { name: 'Sector Momentum', value: 0.09, supportsUp: true },
        { name: 'MACD Signal', value: 0.08, supportsUp: true },
        { name: 'RSI_14', value: 0.05, supportsUp: true },
        { name: 'Volume Trend', value: 0.04, supportsUp: true },
        { name: 'VIX Spread', value: -0.05, supportsUp: false },
        { name: 'Short Interest', value: -0.02, supportsUp: false },
      ],
      validation: {
        rocAuc: 0.801,
        aggF1: 0.742,
        dirAccuracy: 59.8,
        folds: [
          { fold: 'Fold 1', date: '2024-Q1', acc: 61.2, correct: true },
          { fold: 'Fold 2', date: '2023-Q4', acc: 59.1, correct: true },
          { fold: 'Fold 3', date: '2023-Q3', acc: 59.1, correct: true },
        ],
        correctRatio: 59.8,
      },
    }
  ),

  MSFT: ticker(
    'Microsoft Corp.',
    'EQUITY',
    msftShared,
    {
      signal: 'STRONG UP',
      confidence: 81.2,
      mlflowRun: '#891',
      mlflowId: 'r_x891y',
      valDate: '2024-05-18T14:30Z',
      inferenceTime: '38ms',
      predictionCandles: [
        { date: 'May 31 (P)', open: 420.55, high: 424.8, low: 420.0, close: 423.5, isPredicted: true },
        { date: 'Jun 07 (P)', open: 423.5, high: 427.0, low: 422.8, close: 426.1, isPredicted: true },
        { date: 'Jun 14 (P)', open: 426.1, high: 430.5, low: 425.0, close: 429.2, isPredicted: true },
      ],
      features: [
        { name: 'Rolling 20d Vol', value: 0.18, supportsUp: true },
        { name: 'Sector Momentum', value: 0.15, supportsUp: true },
        { name: 'MACD Signal', value: 0.12, supportsUp: true },
        { name: 'RSI_14', value: 0.08, supportsUp: true },
        { name: 'Volume Trend', value: 0.06, supportsUp: true },
        { name: 'VIX Spread', value: -0.01, supportsUp: false },
        { name: 'Short Interest', value: -0.02, supportsUp: false },
      ],
      validation: {
        rocAuc: 0.852,
        aggF1: 0.795,
        dirAccuracy: 64.8,
        folds: [
          { fold: 'Fold 1', date: '2024-Q1', acc: 66.1, correct: true },
          { fold: 'Fold 2', date: '2023-Q4', acc: 63.2, correct: true },
          { fold: 'Fold 3', date: '2023-Q3', acc: 65.1, correct: true },
        ],
        correctRatio: 64.8,
      },
    },
    {
      signal: 'STRONG UP',
      confidence: 79.5,
      mlflowRun: '#892',
      mlflowId: 'r_x892y',
      valDate: '2024-05-18T14:32Z',
      inferenceTime: '48ms',
      predictionCandles: [
        { date: 'May 31 (P)', open: 420.55, high: 423.9, low: 419.5, close: 421.8, isPredicted: true },
        { date: 'Jun 07 (P)', open: 421.8, high: 425.0, low: 421.0, close: 423.5, isPredicted: true },
        { date: 'Jun 14 (P)', open: 423.5, high: 428.2, low: 423.0, close: 426.8, isPredicted: true },
      ],
      features: [
        { name: 'Rolling 20d Vol', value: 0.16, supportsUp: true },
        { name: 'Sector Momentum', value: 0.13, supportsUp: true },
        { name: 'MACD Signal', value: 0.1, supportsUp: true },
        { name: 'RSI_14', value: 0.07, supportsUp: true },
        { name: 'Volume Trend', value: 0.05, supportsUp: true },
        { name: 'VIX Spread', value: -0.02, supportsUp: false },
        { name: 'Short Interest', value: -0.01, supportsUp: false },
      ],
      validation: {
        rocAuc: 0.835,
        aggF1: 0.772,
        dirAccuracy: 62.4,
        folds: [
          { fold: 'Fold 1', date: '2024-Q1', acc: 63.5, correct: true },
          { fold: 'Fold 2', date: '2023-Q4', acc: 61.2, correct: true },
          { fold: 'Fold 3', date: '2023-Q3', acc: 62.5, correct: true },
        ],
        correctRatio: 62.4,
      },
    }
  ),

  TSLA: ticker(
    'Tesla Inc.',
    'EQUITY',
    tslaShared,
    {
      signal: 'DOWN',
      confidence: 65.8,
      mlflowRun: '#756',
      mlflowId: 'r_x756y',
      valDate: '2024-05-18T14:30Z',
      inferenceTime: '55ms',
      predictionCandles: [
        { date: 'May 31 (P)', open: 175.22, high: 175.5, low: 170.0, close: 171.8, isPredicted: true },
        { date: 'Jun 07 (P)', open: 171.8, high: 172.2, low: 166.5, close: 168.1, isPredicted: true },
        { date: 'Jun 14 (P)', open: 168.1, high: 169.0, low: 162.0, close: 164.5, isPredicted: true },
      ],
      features: [
        { name: 'Rolling 20d Vol', value: -0.08, supportsUp: false },
        { name: 'Sector Momentum', value: -0.09, supportsUp: false },
        { name: 'MACD Signal', value: -0.07, supportsUp: false },
        { name: 'RSI_14', value: -0.05, supportsUp: false },
        { name: 'Volume Trend', value: -0.02, supportsUp: false },
        { name: 'VIX Spread', value: 0.1, supportsUp: true },
        { name: 'Short Interest', value: 0.08, supportsUp: true },
      ],
      validation: {
        rocAuc: 0.782,
        aggF1: 0.725,
        dirAccuracy: 58.1,
        folds: [
          { fold: 'Fold 1', date: '2024-Q1', acc: 59.2, correct: true },
          { fold: 'Fold 2', date: '2023-Q4', acc: 56.4, correct: false },
          { fold: 'Fold 3', date: '2023-Q3', acc: 58.7, correct: true },
        ],
        correctRatio: 58.1,
      },
    },
    {
      signal: 'NEUTRAL',
      confidence: 52.4,
      mlflowRun: '#757',
      mlflowId: 'r_x757y',
      valDate: '2024-05-18T14:32Z',
      inferenceTime: '62ms',
      bbandsSqueeze: 'NORMAL',
      predictionCandles: [
        { date: 'May 31 (P)', open: 175.22, high: 177.0, low: 173.5, close: 174.9, isPredicted: true },
        { date: 'Jun 07 (P)', open: 174.9, high: 176.5, low: 172.0, close: 173.5, isPredicted: true },
        { date: 'Jun 14 (P)', open: 173.5, high: 175.0, low: 171.0, close: 172.8, isPredicted: true },
      ],
      features: [
        { name: 'Rolling 20d Vol', value: -0.04, supportsUp: false },
        { name: 'Sector Momentum', value: -0.05, supportsUp: false },
        { name: 'MACD Signal', value: -0.02, supportsUp: false },
        { name: 'RSI_14', value: 0.01, supportsUp: true },
        { name: 'Volume Trend', value: -0.01, supportsUp: false },
        { name: 'VIX Spread', value: 0.05, supportsUp: true },
        { name: 'Short Interest', value: 0.03, supportsUp: true },
      ],
      validation: {
        rocAuc: 0.751,
        aggF1: 0.702,
        dirAccuracy: 55.4,
        folds: [
          { fold: 'Fold 1', date: '2024-Q1', acc: 56.2, correct: false },
          { fold: 'Fold 2', date: '2023-Q4', acc: 54.1, correct: false },
          { fold: 'Fold 3', date: '2023-Q3', acc: 55.9, correct: true },
        ],
        correctRatio: 55.4,
      },
    }
  ),

  NVDA: ticker(
    'NVIDIA Corp.',
    'EQUITY',
    nvdaShared,
    {
      signal: 'NEUTRAL',
      confidence: 48.1,
      mlflowRun: '#910',
      mlflowId: 'r_x910y',
      valDate: '2024-05-18T14:30Z',
      inferenceTime: '49ms',
      predictionCandles: [
        { date: 'May 31 (P)', open: 920.1, high: 923.0, low: 916.0, close: 918.5, isPredicted: true },
        { date: 'Jun 07 (P)', open: 918.5, high: 921.0, low: 914.0, close: 917.2, isPredicted: true },
        { date: 'Jun 14 (P)', open: 917.2, high: 919.5, low: 912.0, close: 915.4, isPredicted: true },
      ],
      features: [
        { name: 'Rolling 20d Vol', value: 0.06, supportsUp: true },
        { name: 'Sector Momentum', value: 0.05, supportsUp: true },
        { name: 'MACD Signal', value: 0.03, supportsUp: true },
        { name: 'RSI_14', value: -0.01, supportsUp: false },
        { name: 'Volume Trend', value: 0.02, supportsUp: true },
        { name: 'VIX Spread', value: -0.04, supportsUp: false },
        { name: 'Short Interest', value: 0.01, supportsUp: true },
      ],
      validation: {
        rocAuc: 0.812,
        aggF1: 0.755,
        dirAccuracy: 60.1,
        folds: [
          { fold: 'Fold 1', date: '2024-Q1', acc: 59.8, correct: true },
          { fold: 'Fold 2', date: '2023-Q4', acc: 60.1, correct: true },
          { fold: 'Fold 3', date: '2023-Q3', acc: 59.1, correct: true },
        ],
        correctRatio: 60.1,
      },
    },
    {
      signal: 'STRONG UP',
      confidence: 84.6,
      mlflowRun: '#911',
      mlflowId: 'r_x911y',
      valDate: '2024-05-18T14:32Z',
      inferenceTime: '45ms',
      predictionCandles: [
        { date: 'May 31 (P)', open: 920.1, high: 928.0, low: 920.0, close: 925.5, isPredicted: true },
        { date: 'Jun 07 (P)', open: 925.5, high: 934.0, low: 924.8, close: 931.2, isPredicted: true },
        { date: 'Jun 14 (P)', open: 931.2, high: 942.0, low: 930.0, close: 939.8, isPredicted: true },
      ],
      features: [
        { name: 'Rolling 20d Vol', value: 0.12, supportsUp: true },
        { name: 'Sector Momentum', value: 0.14, supportsUp: true },
        { name: 'MACD Signal', value: 0.08, supportsUp: true },
        { name: 'RSI_14', value: 0.04, supportsUp: true },
        { name: 'Volume Trend', value: 0.05, supportsUp: true },
        { name: 'VIX Spread', value: -0.01, supportsUp: false },
        { name: 'Short Interest', value: -0.02, supportsUp: false },
      ],
      validation: {
        rocAuc: 0.849,
        aggF1: 0.788,
        dirAccuracy: 63.2,
        folds: [
          { fold: 'Fold 1', date: '2024-Q1', acc: 64.1, correct: true },
          { fold: 'Fold 2', date: '2023-Q4', acc: 62.4, correct: true },
          { fold: 'Fold 3', date: '2023-Q3', acc: 63.1, correct: true },
        ],
        correctRatio: 63.2,
      },
    }
  ),
};
