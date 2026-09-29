import React from 'react';
import { Binary, Network } from 'lucide-react';

export const ArchitectureSpecs: React.FC = () => {
  return (
    <div>
      <div className="hero-header">
        <h1 className="hero-title">System Architecture & Pipeline Specs</h1>
        <p className="hero-subtitle">
          Technical specifications of the deterministic lung segmentation pipeline and the 3 distinct classification paradigms.
        </p>
      </div>

      <div className="upload-grid">
        <div className="card">
          <div className="card-title">
            <Binary size={20} /> 1. Deterministic Lung Segmentation
          </div>
          <p style={{ fontSize: '0.95rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Applies Otsu adaptive thresholding and morphological operations to extract exact lung parenchymal masks while eliminating background tissue artifacts.
          </p>
          <ul style={{ paddingLeft: '1.2rem', fontSize: '0.9rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <li>Gaussian Blur (5x5 kernel) denoising</li>
            <li>Otsu global binarization & inversion</li>
            <li>Border connected component clearing</li>
            <li>Morphological opening & closing (9x9 ellipse)</li>
            <li>CLAHE intensity normalization (clipLimit=2.0)</li>
          </ul>
        </div>

        <div className="card">
          <div className="card-title">
            <Network size={20} /> 2. Tri-Aspect Classification Framework
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', fontSize: '0.9rem' }}>
            <div style={{ borderLeft: '3px solid var(--primary-blue)', paddingLeft: '0.8rem' }}>
              <strong style={{ color: 'var(--primary-cyan)' }}>Aspect 1 • XGBoost Machine Learning</strong>
              <p style={{ color: 'var(--text-muted)' }}>
                Extracts 9-orientation HOG descriptor vectors (16x16 cells), reduces dimensionality to 128 components via PCA, and classifies via gradient boosted decision trees.
              </p>
            </div>
            <div style={{ borderLeft: '3px solid var(--accent-purple)', paddingLeft: '0.8rem' }}>
              <strong style={{ color: 'var(--accent-purple)' }}>Aspect 2 • Symbolic Genetic Programming</strong>
              <p style={{ color: 'var(--text-muted)' }}>
                Evolves non-linear symbolic mathematical functions across generations combining HOG, Local Binary Patterns (LBP), and intensity distribution statistics.
              </p>
            </div>
            <div style={{ borderLeft: '3px solid var(--accent-pink)', paddingLeft: '0.8rem' }}>
              <strong style={{ color: 'var(--accent-pink)' }}>Aspect 3 • ConvNeXt-Tiny CNN Deep Learning</strong>
              <p style={{ color: 'var(--text-muted)' }}>
                Modern convolutional neural network architecture trained with Focal Loss (gamma=1.5, smoothing=0.02) to mitigate severe class imbalance.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
