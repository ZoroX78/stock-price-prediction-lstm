'use client';

import React, { useState } from 'react';
import { Candle } from '../data/types';
import styles from './PriceChart.module.css';

interface PriceChartProps {
  history: Candle[];
  activePrice: number;
  change: string;
  isPositive: boolean;
}

export default function PriceChart({
  history,
  activePrice,
  change,
  isPositive,
}: PriceChartProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const svgWidth = 800;
  const svgHeight = 400;
  const paddingLeft = 30;
  const paddingRight = 75;
  const paddingTop = 40;
  const paddingBottom = 45;

  const chartWidth = svgWidth - paddingLeft - paddingRight;
  const chartHeight = svgHeight - paddingTop - paddingBottom;

  const allPrices = history.flatMap((c) => [c.open, c.close, c.high, c.low]);
  const minPrice = Math.min(...allPrices) * 0.99;
  const maxPrice = Math.max(...allPrices) * 1.01;
  const priceRange = maxPrice - minPrice || 1;

  const getX = (index: number) =>
    paddingLeft + (index * chartWidth) / Math.max(history.length - 1, 1);
  const getY = (val: number) =>
    paddingTop + chartHeight - ((val - minPrice) / priceRange) * chartHeight;

  const predictionStartIndex = history.findIndex((c) => c.isPredicted);
  const predictionXStart =
    predictionStartIndex !== -1 ? getX(predictionStartIndex) : chartWidth;

  const gridLines: number[] = [];
  const step = priceRange / 4;
  for (let i = 0; i <= 4; i++) {
    gridLines.push(minPrice + step * i);
  }

  const formatPrice = (val: number) => val.toFixed(2);

  return (
    <div className={styles.chartCard}>
      <div className={styles.chartHeader}>
        <div className={styles.titleInfo}>
          <h2 className={styles.title}>Price Action</h2>
          <span className={styles.priceValue}>${formatPrice(activePrice)}</span>
          <span
            className={`${styles.changeValue} ${isPositive ? styles.green : styles.red}`}
          >
            {change}
          </span>
        </div>
        <span className={styles.chartHint}>OHLC + prediction window</span>
      </div>

      <div className={styles.chartCanvasContainer}>
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className={styles.chartSvg}
          onMouseLeave={() => setHoveredIndex(null)}
        >
          {predictionStartIndex !== -1 && (
            <g>
              <rect
                x={predictionXStart}
                y={paddingTop}
                width={svgWidth - paddingRight - predictionXStart}
                height={chartHeight}
                className={styles.predictionShade}
              />
              <line
                x1={predictionXStart}
                y1={paddingTop}
                x2={predictionXStart}
                y2={paddingTop + chartHeight}
                className={styles.predictionDivider}
              />
              <rect
                x={predictionXStart + 15}
                y={paddingTop + 12}
                width={112}
                height={18}
                rx={4}
                className={styles.predictionBadgeBg}
              />
              <text
                x={predictionXStart + 71}
                y={paddingTop + 24}
                textAnchor="middle"
                className={styles.predictionBadgeText}
              >
                PREDICTION WINDOW
              </text>
            </g>
          )}

          {gridLines.map((price, idx) => (
            <g key={idx}>
              <line
                x1={paddingLeft}
                y1={getY(price)}
                x2={svgWidth - paddingRight}
                y2={getY(price)}
                className={styles.gridLine}
              />
              <text
                x={svgWidth - paddingRight + 8}
                y={getY(price) + 4}
                className={styles.axisText}
              >
                {Math.round(price)}
              </text>
            </g>
          ))}

          <path
            d={history
              .map(
                (candle, idx) =>
                  `${idx === 0 ? 'M' : 'L'} ${getX(idx)} ${getY(candle.close)}`
              )
              .join(' ')}
            className={styles.trendLine}
          />

          {history.map((candle, idx) => {
            const x = getX(idx);
            const yOpen = getY(candle.open);
            const yClose = getY(candle.close);
            const yHigh = getY(candle.high);
            const yLow = getY(candle.low);
            const isCandlePositive = candle.close >= candle.open;
            const candleClass = isCandlePositive
              ? candle.isPredicted
                ? styles.candlePredictUp
                : styles.candleUp
              : candle.isPredicted
                ? styles.candlePredictDown
                : styles.candleDown;
            const candleWidth = 8;

            return (
              <g
                key={`${candle.date}-${idx}`}
                onMouseEnter={() => setHoveredIndex(idx)}
                style={{ cursor: 'pointer' }}
              >
                <line
                  x1={x}
                  y1={yHigh}
                  x2={x}
                  y2={yLow}
                  className={`${candleClass} ${styles.wick}`}
                />
                <rect
                  x={x - candleWidth / 2}
                  y={Math.min(yOpen, yClose)}
                  width={candleWidth}
                  height={Math.max(Math.abs(yOpen - yClose), 2)}
                  className={`${candleClass} ${styles.candleBody}`}
                  rx={1}
                />
                <rect
                  x={x - 12}
                  y={paddingTop}
                  width={24}
                  height={chartHeight}
                  fill="transparent"
                />
              </g>
            );
          })}

          {history.map((candle, idx) => {
            if (idx % 2 === 0 && !candle.isPredicted) {
              return (
                <text
                  key={`label-${idx}`}
                  x={getX(idx)}
                  y={svgHeight - paddingBottom + 20}
                  textAnchor="middle"
                  className={styles.xAxisText}
                >
                  {candle.date}
                </text>
              );
            }
            if (candle.isPredicted && idx === predictionStartIndex) {
              return (
                <text
                  key={`pred-start-${idx}`}
                  x={getX(idx)}
                  y={svgHeight - paddingBottom + 20}
                  textAnchor="middle"
                  className={`${styles.xAxisText} ${styles.predictLabelColor}`}
                >
                  {candle.date.split(' ')[0]}
                </text>
              );
            }
            if (candle.isPredicted && idx === history.length - 1) {
              return (
                <text
                  key={`pred-end-${idx}`}
                  x={getX(idx)}
                  y={svgHeight - paddingBottom + 20}
                  textAnchor="middle"
                  className={`${styles.xAxisText} ${styles.predictLabelColor}`}
                >
                  Future
                </text>
              );
            }
            return null;
          })}

          <g>
            <line
              x1={paddingLeft}
              y1={getY(activePrice)}
              x2={svgWidth - paddingRight}
              y2={getY(activePrice)}
              className={styles.activePriceLine}
            />
            <rect
              x={svgWidth - paddingRight + 2}
              y={getY(activePrice) - 9}
              width={48}
              height={18}
              rx={3}
              className={styles.priceBadge}
            />
            <text
              x={svgWidth - paddingRight + 26}
              y={getY(activePrice) + 4}
              textAnchor="middle"
              className={styles.priceBadgeText}
            >
              {formatPrice(activePrice)}
            </text>
          </g>
        </svg>

        {hoveredIndex !== null && history[hoveredIndex] && (
          <div
            className={styles.tooltip}
            style={{
              left: `${(getX(hoveredIndex) / svgWidth) * 100}%`,
              top: `${(getY(history[hoveredIndex].close) / svgHeight) * 100 - 32}%`,
            }}
          >
            <div className={styles.tooltipDate}>
              {history[hoveredIndex].date}{' '}
              {history[hoveredIndex].isPredicted && (
                <span className={styles.tooltipPredictBadge}>PREDICTED</span>
              )}
            </div>
            <div className={styles.tooltipGrid}>
              <span>
                Open: <strong>${formatPrice(history[hoveredIndex].open)}</strong>
              </span>
              <span>
                Close: <strong>${formatPrice(history[hoveredIndex].close)}</strong>
              </span>
              <span>
                High: <strong>${formatPrice(history[hoveredIndex].high)}</strong>
              </span>
              <span>
                Low: <strong>${formatPrice(history[hoveredIndex].low)}</strong>
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
