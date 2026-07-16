'use client';

import React from 'react';
import { BbandsData, MacdData, RsiData } from '../data/types';
import { formatSigned } from '../lib/signal';
import styles from './MetricsPanel.module.css';

interface MetricsPanelProps {
  rsi: RsiData;
  macd: MacdData;
  bbands: BbandsData;
  inferenceTime: string;
}

function getRsiPath(points: number[]): string {
  if (points.length === 0) return '';
  const w = 200;
  const h = 50;
  const padding = 5;
  const stepX = points.length > 1 ? w / (points.length - 1) : 0;

  return points
    .map((val, idx) => {
      const x = idx * stepX;
      const y = h - padding - (val / 100) * (h - 2 * padding);
      return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
    })
    .join(' ');
}

function renderMacdHistogram(bars: number[], posClass: string, negClass: string) {
  if (bars.length === 0) return null;
  const w = 200;
  const h = 50;
  const barGap = 6;
  const barWidth = (w - (bars.length - 1) * barGap) / bars.length;
  const absMax = Math.max(...bars.map(Math.abs), 0.1);
  const zeroY = h / 2;

  return bars.map((bar, idx) => {
    const x = idx * (barWidth + barGap);
    const scaledVal = (bar / absMax) * (h / 2 - 4);
    const y = bar >= 0 ? zeroY - scaledVal : zeroY;
    const height = Math.abs(scaledVal);

    return (
      <rect
        key={idx}
        x={x}
        y={y}
        width={barWidth}
        height={Math.max(height, 1)}
        rx={1}
        className={bar >= 0 ? posClass : negClass}
      />
    );
  });
}

export default function MetricsPanel({
  rsi,
  macd,
  bbands,
  inferenceTime,
}: MetricsPanelProps) {
  return (
    <div className={styles.stack}>
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <span className={styles.cardTitle}>RSI (14)</span>
          <span className={styles.cardValue}>{rsi.val}</span>
        </div>

        <div className={styles.chartContainer}>
          <svg viewBox="0 0 200 50" className={styles.sparklineSvg}>
            <line x1="0" y1="35" x2="200" y2="35" className={styles.rsiRefLine} />
            <line x1="0" y1="15" x2="200" y2="15" className={styles.rsiRefLine} />
            <path d={getRsiPath(rsi.sparkline)} className={styles.rsiPath} />
          </svg>
        </div>

        <div className={styles.cardFooter}>
          <span className={styles.statusLabel}>{rsi.status}</span>
          <span
            className={`${styles.statusBadge} ${
              rsi.direction === 'RISING'
                ? styles.green
                : rsi.direction === 'FALLING'
                  ? styles.red
                  : styles.muted
            }`}
          >
            {rsi.direction}
          </span>
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <span className={styles.cardTitle}>MACD (12,26,9)</span>
          <span
            className={`${styles.cardValue} ${macd.isPositive ? styles.green : styles.red}`}
          >
            {formatSigned(macd.val)}
          </span>
        </div>

        <div className={styles.chartContainer}>
          <svg viewBox="0 0 200 50" className={styles.histogramSvg}>
            <line x1="0" y1="25" x2="200" y2="25" className={styles.macdZeroLine} />
            {renderMacdHistogram(macd.bars, styles.macdBarPos, styles.macdBarNeg)}
          </svg>
        </div>

        <div className={styles.cardFooter}>
          <span className={styles.statusLabel}>{macd.status}</span>
          <span className={styles.timeLabel}>{macd.time}</span>
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.bbHeader}>
          <div className={styles.apiInfo}>
            <div className={styles.apiLabel}>
              <span className={styles.dotMuted} />
              DATA: LIVE
            </div>
            <div className={styles.apiLabel}>INFERENCE: {inferenceTime}</div>
          </div>
          <div className={styles.cardHeaderBB}>
            <span className={styles.cardTitle}>BOLLINGER BANDS</span>
            <span className={styles.bbPosition}>{bbands.position}</span>
          </div>
        </div>

        <div className={styles.sliderContainer}>
          <div className={styles.sliderLabels}>
            <span>LOWER</span>
            <span>MID</span>
            <span>UPPER</span>
          </div>
          <div className={styles.sliderTrack}>
            <div className={styles.sliderTick} style={{ left: '0%' }} />
            <div className={styles.sliderTick} style={{ left: '50%' }} />
            <div className={styles.sliderTick} style={{ left: '100%' }} />
            <div
              className={styles.sliderCursor}
              style={{ left: `${Math.min(Math.max(bbands.value, 0), 100)}%` }}
            />
          </div>
        </div>

        <div className={styles.cardFooter}>
          <span className={`${styles.statusLabel} ${styles.orangeText}`}>
            {bbands.squeeze}
          </span>
          <span className={styles.volLabel}>{bbands.volume}</span>
        </div>
      </div>
    </div>
  );
}
