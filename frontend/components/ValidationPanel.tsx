'use client';

import React, { useState } from 'react';
import { ValidationData } from '../data/types';
import { formatPercent } from '../lib/signal';
import styles from './ValidationPanel.module.css';

interface ValidationPanelProps {
  validation: ValidationData;
}

export default function ValidationPanel({ validation }: ValidationPanelProps) {
  const [showDetails, setShowDetails] = useState(true);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.titleArea}>
          <h3 className={styles.title}>Walk-Forward Validation</h3>
          <span className={styles.subtitle}>ROLLING 12-MONTH OUT-OF-SAMPLE</span>
        </div>
        <div className={styles.detailsToggle}>
          <span className={styles.toggleLabel}>DETAILS</span>
          <label className={styles.switch}>
            <input
              type="checkbox"
              checked={showDetails}
              onChange={() => setShowDetails((v) => !v)}
            />
            <span className={styles.slider} />
          </label>
        </div>
      </div>

      <div className={styles.metricsRow}>
        <div className={styles.metricItem}>
          <span className={styles.metricLabel}>ROC-AUC</span>
          <span className={styles.metricValue}>{validation.rocAuc.toFixed(3)}</span>
        </div>
        <div className={styles.metricItem}>
          <span className={styles.metricLabel}>AGG F1</span>
          <span className={styles.metricValue}>{validation.aggF1.toFixed(3)}</span>
        </div>
        <div className={styles.metricItem}>
          <span className={styles.metricLabel}>DIRECTIONAL ACCURACY</span>
          <span className={`${styles.metricValue} ${styles.greenText}`}>
            {formatPercent(validation.dirAccuracy)}
          </span>
        </div>
      </div>

      <div
        className={`${styles.tableWrapper} ${
          showDetails ? styles.expanded : styles.collapsed
        }`}
      >
        <table className={styles.table}>
          <thead>
            <tr>
              <th>FOLD #</th>
              <th>DATE RANGE</th>
              <th className={styles.rightAlign}>DIR. ACCURACY</th>
            </tr>
          </thead>
          <tbody>
            {validation.folds.map((fold) => (
              <tr key={fold.fold}>
                <td>{fold.fold}</td>
                <td className={styles.dateRange}>{fold.date}</td>
                <td
                  className={`${styles.rightAlign} ${
                    fold.correct ? styles.greenText : styles.redText
                  }`}
                >
                  {formatPercent(fold.acc)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className={styles.ratioSection}>
        <div className={styles.ratioBarContainer}>
          <div
            className={styles.ratioCorrect}
            style={{ width: `${validation.correctRatio}%` }}
          />
          <div
            className={styles.ratioIncorrect}
            style={{ width: `${100 - validation.correctRatio}%` }}
          />
        </div>
        <div className={styles.ratioLabels}>
          <span className={styles.labelCorrect}>CORRECT</span>
          <span className={styles.labelIncorrect}>INCORRECT</span>
        </div>
      </div>
    </div>
  );
}
