'use client';

import React, { useState } from 'react';
import Header from '../components/Header';
import PriceChart from '../components/PriceChart';
import MetricsPanel from '../components/MetricsPanel';
import FeatureImportance from '../components/FeatureImportance';
import ValidationPanel from '../components/ValidationPanel';
import Watchlist from '../components/Watchlist';
import { ModelArch, WatchlistItem, ResolvedTicker } from '../data/types';
import styles from './page.module.css';

export default function Dashboard() {
  const [activeTicker, setActiveTicker] = useState('AAPL');
  const [model, setModel] = useState<ModelArch>('LSTM');

  const [activeStock, setActiveStock] = useState<ResolvedTicker | null>(null);
  const [watchlistItems, setWatchlistItems] = useState<WatchlistItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [watchlistLoading, setWatchlistLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  // Fetch active stock prediction
  React.useEffect(() => {
    let active = true;
    const controller = new AbortController();
    async function fetchStock() {
      setIsLoading(true);
      setError(null);
      try {
        const apiModel = model === 'Transformer' ? 'Transformer' : 'LSTM';
        const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const res = await fetch(`${apiBase}/api/predict/${activeTicker}?model=${apiModel}`, { signal: controller.signal });
        if (!res.ok) {
          const errDetail = await res.json().catch(() => ({}));
          throw new Error(errDetail.detail || `Failed to fetch prediction for ${activeTicker}`);
        }
        const data = await res.json();
        if (active) {
          setActiveStock(data);
        }
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        if (active) {
          setError(err instanceof Error ? err.message : 'An error occurred while fetching prediction');
        }
      } finally {
        if (active) {
          setIsLoading(false);
        }
      }
    }
    fetchStock();
    return () => {
      active = false;
      controller.abort();
    };
  }, [activeTicker, model, retryCount]);

  // Fetch watchlist data
  React.useEffect(() => {
    let active = true;
    const controller = new AbortController();
    async function fetchWatchlist() {
      setWatchlistLoading(true);
      try {
        const defaults = ['AAPL', 'MSFT', 'TSLA', 'NVDA'];
        const symbols = defaults.includes(activeTicker)
          ? defaults
          : [...defaults, activeTicker];
        const apiModel = model === 'Transformer' ? 'Transformer' : 'LSTM';
        const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const res = await fetch(`${apiBase}/api/watchlist?tickers=${symbols.join(',')}&model=${apiModel}`, { signal: controller.signal });
        if (res.ok) {
          const data = await res.json();
          if (active) {
            setWatchlistItems(data.items ?? []);
          }
        } else {
          console.error(`Watchlist API returned ${res.status}`);
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') return;
        console.error('Watchlist fetch error:', err);
      } finally {
        if (active) {
          setWatchlistLoading(false);
        }
      }
    }
    fetchWatchlist();
    return () => {
      active = false;
      controller.abort();
    };
  }, [activeTicker, model, retryCount]);

  const handleRetry = () => {
    setRetryCount(prev => prev + 1);
  };

  return (
    <div className={styles.container}>
      {error && (
        <div className={styles.errorState} role="alert">
          <div className={styles.errorTitle}>Live API Connection Error</div>
          <p className={styles.errorText}>{error}</p>
          <button onClick={handleRetry} className={styles.retryButton}>
            RETRY CONNECTION
          </button>
        </div>
      )}

      {isLoading && !activeStock && (
        <div className={styles.loadingState} role="status" aria-live="polite">
          <div className={styles.spinner} />
          <p className={styles.loadingText}>FETCHING MARKET DATA & EXECUTING MODEL INFERENCE...</p>
        </div>
      )}

      {activeStock && (
        <>
          <Header
            ticker={activeTicker}
            name={activeStock.name}
            type={activeStock.type}
            model={model}
            setModel={setModel}
            onSearch={(symbol) => setActiveTicker(symbol.toUpperCase().trim())}
            activeData={activeStock.data}
          />

          <div 
            className={styles.gridTop}
            style={{ 
              opacity: isLoading ? 0.7 : 1, 
              transition: 'opacity 0.2s',
              pointerEvents: isLoading ? 'none' : 'auto'
            }}
          >
            <PriceChart
              history={activeStock.data.history}
              activePrice={activeStock.data.price}
              change={activeStock.data.change}
              isPositive={activeStock.data.isPositive}
            />
            <MetricsPanel
              rsi={activeStock.data.rsi}
              macd={activeStock.data.macd}
              bbands={activeStock.data.bbands}
              inferenceTime={activeStock.data.inferenceTime}
            />
          </div>

          <div className={styles.gridMiddle}>
            <FeatureImportance features={activeStock.data.features} />
            <ValidationPanel validation={activeStock.data.validation} />
          </div>

          <div className={styles.watchlistSection} style={{ opacity: watchlistLoading ? 0.7 : 1 }}>
            <Watchlist
              watchlistItems={watchlistItems}
              activeTicker={activeTicker}
              onSelectTicker={setActiveTicker}
            />
          </div>
        </>
      )}

      <footer className={styles.footer}>
        <p className={styles.disclaimer}>
          LIVE AI INFERENCE DASHBOARD — CONNECTED TO FASTAPI PREDICTION SERVER.
          PREDICTIONS REPRESENT LSTM & TEMPORAL FUSION TRANSFORMER FORWARD PASSES.
        </p>
      </footer>
    </div>
  );
}
