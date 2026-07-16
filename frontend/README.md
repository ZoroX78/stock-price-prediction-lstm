# AlphaPredict — Stock Prediction Dashboard (Frontend)

Next.js App Router demo UI for visualizing LSTM / Transformer stock-movement predictions, walk-forward validation, and SHAP-style feature importance.

> **Demo data only.** This frontend is not wired to live inference or the Python training pipeline. Labels say **DATA: MOCK** on purpose.

## Stack

| Layer | Choice |
|---|---|
| Framework | Next.js 14 (App Router, React 18) |
| Language | TypeScript (`strict`) |
| Styling | CSS Modules + design tokens in `app/globals.css` |
| Charts | Custom SVG (candlesticks, RSI sparkline, MACD histogram) |
| Icons | `lucide-react` |

## Structure

```
frontend/
├── app/                  # Next.js routes (layout + dashboard page)
├── components/           # Presentational UI
├── data/
│   ├── types.ts          # Domain contracts
│   ├── mockData.ts       # Curated fixtures (AAPL/MSFT/TSLA/NVDA)
│   └── generateTicker.ts # Resolve fixtures or generate deterministic mocks
└── lib/
    └── signal.ts         # Signal tone helpers + formatters
```

## Run

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

```bash
npm run build   # production build
npm start       # serve production build
```

## Features

1. Toggle **LSTM** vs **Transformer** — charts, confidence, SHAP bars, and validation metrics update from the active model slice.
2. Search any ticker — curated fixtures when available; otherwise a deterministic mock dataset is generated.
3. Watchlist navigation for AAPL, MSFT, TSLA, NVDA (plus searched symbols).
4. Walk-forward validation panel with collapsible folds.

## Honesty note

Status chrome shows **DATA: MOCK**, not a live API. Wire real predictions later via Next.js route handlers calling your Python service if you want production inference.
