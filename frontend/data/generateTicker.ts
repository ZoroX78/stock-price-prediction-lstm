import {
  Candle,
  Feature,
  ModelArch,
  ModelData,
  ResolvedTicker,
  Signal,
  TickerData,
} from './types';
import { formatSigned, isBullishSignal } from '../lib/signal';
import { mockData } from './mockData';

const SIGNALS: Signal[] = ['STRONG UP', 'UP', 'DOWN', 'NEUTRAL'];

function tickerSeed(ticker: string): number {
  return ticker.charCodeAt(0) + (ticker.charCodeAt(1) || 0);
}

function buildHistory(seed: number, startPrice: number, isUp: boolean): Candle[] {
  const history: Candle[] = [];
  let currentPrice = startPrice - (isUp ? 20 : -20);

  for (let i = 0; i < 11; i++) {
    const stepDirection = isUp ? 1.5 : -1.5;
    const noise = ((seed + i) % 7) - 3;
    const open = currentPrice;
    const close = currentPrice + stepDirection + noise;
    const high = Math.max(open, close) + Math.abs(noise / 2);
    const low = Math.min(open, close) - Math.abs(noise / 2);

    history.push({
      date: `May ${10 + i * 2}`,
      open,
      high,
      low,
      close,
      isPredicted: false,
    });
    currentPrice = close;
  }

  for (let i = 1; i <= 3; i++) {
    const stepDirection = isUp ? 2.5 : -2.5;
    const noise = ((seed + i) % 5) - 2;
    const open = currentPrice;
    const close = currentPrice + stepDirection + noise;
    const high = Math.max(open, close) + Math.abs(noise / 2);
    const low = Math.min(open, close) - Math.abs(noise / 2);

    history.push({
      date: `Jun ${i * 7} (P)`,
      open,
      high,
      low,
      close,
      isPredicted: true,
    });
    currentPrice = close;
  }

  return history;
}

function buildFeatures(isUp: boolean): Feature[] {
  const signed = (up: number, down: number) => (isUp ? up : down);
  return [
    { name: 'Rolling 20d Vol', value: signed(0.12, -0.07), supportsUp: isUp },
    { name: 'Sector Momentum', value: signed(0.1, -0.09), supportsUp: isUp },
    { name: 'MACD Signal', value: signed(0.08, -0.05), supportsUp: isUp },
    { name: 'RSI_14', value: signed(0.05, -0.02), supportsUp: isUp },
    { name: 'Volume Trend', value: signed(0.03, 0.01), supportsUp: true },
    { name: 'VIX Spread', value: signed(-0.04, 0.08), supportsUp: !isUp },
    { name: 'Short Interest', value: signed(-0.02, 0.06), supportsUp: !isUp },
  ];
}

function buildModelData(ticker: string, model: ModelArch): ModelData {
  const seed = tickerSeed(ticker) + (model === 'Transformer' ? 17 : 0);
  const priceSeed = 100 + (seed % 400);
  const signal = SIGNALS[seed % 4];
  const isUp = isBullishSignal(signal);
  const changePct = ((seed % 50) / 10 + 0.1) * (isUp ? 1 : -1);
  const changeVal = priceSeed * (changePct / 100);
  const confidence = 50 + (seed % 35);
  const valAcc = 55 + (seed % 12);
  const runId = seed % 1000;

  return {
    price: priceSeed,
    change: `${formatSigned(changeVal)} (${formatSigned(changePct)}%)`,
    isPositive: isUp,
    signal,
    horizon: '5-Day Future',
    confidence: Number(confidence.toFixed(1)),
    mlflowRun: `#${runId}`,
    mlflowId: `r_x${runId}y`,
    valDate: '2024-05-18T14:30Z',
    inferenceTime: `${35 + (seed % 25)}ms`,
    rsi: {
      val: isUp ? 62.4 : 38.5,
      status: isUp ? 'NEUTRAL-BULLISH' : 'BEARISH',
      direction: isUp ? 'RISING' : 'FALLING',
      sparkline: isUp
        ? [42, 45, 48, 51, 53, 58, 60, 62.4]
        : [55, 52, 48, 46, 44, 42, 40, 38.5],
    },
    macd: {
      val: changeVal / 4,
      isPositive: isUp,
      status: isUp ? 'BULLISH CROSSOVER' : 'BEARISH CROSSOVER',
      time: '2D AGO',
      bars: isUp
        ? [0.1, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.2]
        : [0.1, 0.0, -0.2, -0.3, -0.5, -0.6, -0.7, -0.8],
    },
    bbands: {
      position: isUp ? 'Upper' : 'Lower',
      value: isUp ? 85 : 18,
      squeeze: isUp ? 'SQUEEZE TRIGGERED' : 'NORMAL',
      volume: isUp ? 'VOL: HIGH' : 'VOL: LOW',
    },
    history: buildHistory(seed, priceSeed, isUp),
    features: buildFeatures(isUp),
    validation: {
      rocAuc: Number((0.75 + (seed % 15) / 100).toFixed(3)),
      aggF1: Number((0.7 + (seed % 12) / 100).toFixed(3)),
      dirAccuracy: Number(valAcc.toFixed(1)),
      folds: [
        { fold: 'Fold 1', date: '2024-Q1', acc: Number((valAcc + 1.2).toFixed(1)), correct: true },
        { fold: 'Fold 2', date: '2023-Q4', acc: Number((valAcc - 0.8).toFixed(1)), correct: true },
        { fold: 'Fold 3', date: '2023-Q3', acc: Number((valAcc + 0.3).toFixed(1)), correct: true },
      ],
      correctRatio: valAcc,
    },
  };
}

function fromFixture(entry: TickerData, model: ModelArch): ResolvedTicker {
  return {
    name: entry.name,
    type: entry.type,
    data: entry[model],
  };
}

/** Resolve ticker data from curated fixtures, or generate a deterministic mock. */
export function getTickerData(ticker: string, model: ModelArch): ResolvedTicker {
  const symbol = ticker.toUpperCase().replace('$', '').trim();
  const fixture = mockData[symbol];
  if (fixture) return fromFixture(fixture, model);

  return {
    name: `${symbol} Holdings Inc.`,
    type: 'EQUITY',
    data: buildModelData(symbol, model),
  };
}

export const DEFAULT_WATCHLIST: string[] = ['AAPL', 'MSFT', 'TSLA', 'NVDA'];
