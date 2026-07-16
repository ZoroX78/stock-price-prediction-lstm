'use client';

import React from 'react';
import { LayoutList } from 'lucide-react';
import { WatchlistItem } from '../data/types';
import { signalTone } from '../lib/signal';
import styles from './Watchlist.module.css';

interface WatchlistProps {
  watchlistItems: WatchlistItem[];
  activeTicker: string;
  onSelectTicker: (ticker: string) => void;
}

const BADGE: Record<ReturnType<typeof signalTone>, string> = {
  strongUp: styles.badgeStrongUp,
  up: styles.badgeUp,
  down: styles.badgeDown,
  neutral: styles.badgeNeutral,
};

const SIDE_BAR: Record<ReturnType<typeof signalTone>, string> = {
  strongUp: styles.sideBarUp,
  up: styles.sideBarUp,
  down: styles.sideBarDown,
  neutral: styles.sideBarNeutral,
};

export default function Watchlist({
  watchlistItems,
  activeTicker,
  onSelectTicker,
}: WatchlistProps) {
  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.titleGroup}>
          <LayoutList size={18} className={styles.headerIcon} />
          <h3 className={styles.title}>Active Watchlist</h3>
        </div>
        <span className={styles.itemCount}>{watchlistItems.length} tickers</span>
      </div>

      <div className={styles.tableResponsive}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>TICKER</th>
              <th>PRICE</th>
              <th>SIGNAL (5D)</th>
              <th>CONFIDENCE</th>
              <th className={styles.rightAlign}>MLFLOW ID</th>
            </tr>
          </thead>
          <tbody>
            {watchlistItems.map((item) => {
              const tone = signalTone(item.signal);
              const isActive = item.ticker === activeTicker;
              return (
                <tr
                  key={item.ticker}
                  onClick={() => onSelectTicker(item.ticker)}
                  className={`${styles.row} ${isActive ? styles.activeRow : ''}`}
                >
                  <td className={styles.tickerCell}>
                    <div className={`${styles.sideBar} ${SIDE_BAR[tone]}`} />
                    <span className={styles.symbol}>{item.ticker}</span>
                  </td>
                  <td className={styles.priceCell}>${item.price.toFixed(2)}</td>
                  <td>
                    <span className={`${styles.badge} ${BADGE[tone]}`}>
                      {item.signal}
                    </span>
                  </td>
                  <td className={styles.confidenceCell}>{item.confidence}%</td>
                  <td className={`${styles.mlflowCell} ${styles.rightAlign}`}>
                    {item.mlflowId}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
