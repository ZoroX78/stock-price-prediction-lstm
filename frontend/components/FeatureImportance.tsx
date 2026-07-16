'use client';

import React from 'react';
import { ShieldCheck } from 'lucide-react';
import { Feature } from '../data/types';
import { formatSigned } from '../lib/signal';
import styles from './FeatureImportance.module.css';

interface FeatureImportanceProps {
  features: Feature[];
}

export default function FeatureImportance({ features }: FeatureImportanceProps) {
  const maxAbsValue = Math.max(...features.map((f) => Math.abs(f.value)), 0.15);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.titleArea}>
          <h3 className={styles.title}>Feature Importance</h3>
          <span className={styles.subtitle}>LOCAL SHAP VALUES</span>
        </div>
        <div className={styles.legend}>
          <div className={styles.legendItem}>
            <span className={styles.dotGreen} />
            SUPPORTS UP
          </div>
          <div className={styles.legendItem}>
            <span className={styles.dotRed} />
            SUPPORTS DOWN
          </div>
        </div>
        <ShieldCheck size={18} className={styles.headerIcon} />
      </div>

      <div className={styles.list}>
        {features.map((feature) => {
          const percent = Math.min((Math.abs(feature.value) / maxAbsValue) * 100, 100);

          return (
            <div key={feature.name} className={styles.row}>
              <span className={styles.featureName}>{feature.name}</span>
              <div className={styles.progressContainer}>
                <div
                  className={`${styles.progressBar} ${
                    feature.supportsUp ? styles.barGreen : styles.barRed
                  }`}
                  style={{ width: `${percent}%` }}
                />
              </div>
              <span
                className={`${styles.featureValue} ${
                  feature.supportsUp ? styles.greenText : styles.redText
                }`}
              >
                {formatSigned(feature.value)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
