import React from 'react';
import type { Metadata } from 'next';
import { JetBrains_Mono, Outfit } from 'next/font/google';
import './globals.css';

const outfit = Outfit({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600', '700', '800'],
  variable: '--font-outfit',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  weight: ['400', '500'],
  variable: '--font-jetbrains',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'AlphaPredict | Stock Price Movement Prediction',
  description:
    'Demo dashboard for sequence-to-one deep learning predictions (LSTM / Transformer) with walk-forward validation and feature importance.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${outfit.variable} ${jetbrainsMono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
