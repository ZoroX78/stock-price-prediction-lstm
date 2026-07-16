'use client';

import React, { useState } from 'react';
import { Search, ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { ModelArch, ModelData } from '../data/types';
import { isBullishSignal, isBearishSignal, signalTone } from '../lib/signal';
import styles from './Header.module.css';

interface HeaderProps {
  ticker: string;
  name: string;
  type: string;
  model: ModelArch;
  setModel: (model: ModelArch) => void;
  onSearch: (symbol: string) => void;
  activeData: ModelData;
}

const SIGNAL_BADGE: Record<ReturnType<typeof signalTone>, string> = {
  strongUp: styles.signalStrongUp,
  up: styles.signalUp,
  down: styles.signalDown,
  neutral: styles.signalNeutral,
};

export default function Header({
  ticker,
  name,
  type,
  model,
  setModel,
  onSearch,
  activeData,
}: HeaderProps) {
  const [searchInput, setSearchInput] = useState('');

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchInput.trim()) {
      onSearch(searchInput.trim().toUpperCase());
      setSearchInput('');
    }
  };

  const tone = signalTone(activeData.signal);

  return (
    <header className={styles.header}>
      <div className={styles.tickerSection}>
        <div className={styles.identity}>
          <h1 className={styles.tickerSymbol}>{ticker}</h1>
          <div className={styles.metaInfo}>
            <span className={styles.companyName}>{name}</span>
            <span className={styles.assetType}>{type}</span>
          </div>
        </div>

        <div className={styles.signalWrapper}>
          <div className={styles.label}>SIGNAL</div>
          <div className={`${styles.signalBadge} ${SIGNAL_BADGE[tone]}`}>
            {isBullishSignal(activeData.signal) && (
              <ArrowUpRight className={styles.signalIcon} size={20} />
            )}
            {isBearishSignal(activeData.signal) && (
              <ArrowDownRight className={styles.signalIcon} size={20} />
            )}
            <span>{activeData.signal}</span>
          </div>
        </div>

        <div className={styles.horizonWrapper}>
          <div className={styles.label}>HORIZON</div>
          <div className={styles.horizonValue}>{activeData.horizon}</div>
        </div>
      </div>

      <div className={styles.modelSection}>
        <div className={styles.labelCenter}>MODEL ARCHITECTURE</div>
        <div className={styles.toggleContainer}>
          <button
            type="button"
            className={`${styles.toggleButton} ${model === 'LSTM' ? styles.toggleActive : ''}`}
            onClick={() => setModel('LSTM')}
          >
            LSTM
          </button>
          <button
            type="button"
            className={`${styles.toggleButton} ${model === 'Transformer' ? styles.toggleActive : ''}`}
            onClick={() => setModel('Transformer')}
          >
            TRANSFORMER
          </button>
        </div>
      </div>

      <div className={styles.confidenceSection}>
        <div className={styles.confidenceHeader}>
          <span className={styles.label}>CONFIDENCE SCORE</span>
          <span className={styles.confidencePercent}>{activeData.confidence}%</span>
        </div>
        <div className={styles.progressBarBg}>
          <div
            className={styles.progressBarFill}
            style={{ width: `${Math.min(Math.max(activeData.confidence, 0), 100)}%` }}
          />
        </div>
        <div className={styles.mlflowMetadata}>
          <span>
            MLflow Run: <strong>{activeData.mlflowRun}</strong>
          </span>
          <span>Val: {activeData.valDate}</span>
        </div>
      </div>

      <div className={styles.rightSection}>
        <div className={styles.apiStats}>
          <div className={styles.statItem}>
            <span className={styles.statusDotActive} />
            <span className={styles.statLabel}>DATA: </span>
            <span className={styles.statValue}>LIVE (API)</span>
          </div>
          <div className={styles.statItem}>
            <span className={styles.statLabel}>INFERENCE: </span>
            <span className={styles.statValue}>{activeData.inferenceTime}</span>
          </div>
        </div>

        <form onSubmit={handleSearchSubmit} className={styles.searchBar}>
          <Search size={16} className={styles.searchIcon} />
          <input
            type="text"
            placeholder="Search Ticker (e.g. AAPL)"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className={styles.searchInput}
          />
        </form>
      </div>
    </header>
  );
}
