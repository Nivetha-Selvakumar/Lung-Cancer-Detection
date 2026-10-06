import React from 'react';
import { Binary, Network, ShieldCheck } from 'lucide-react';

export const ArchitectureSpecs: React.FC = () => {
  return (
    <div>
      <div className="hero-header">
        <h1 className="hero-title">System Specifications & Diagnostic Pipeline</h1>
        <p className="hero-subtitle">
          Medical diagnostic specifications for automated thoracic CT lung field segmentation, feature extraction, and multi-stage decision support.
        </p>
      </div>

      <div className="upload-grid">
        <div className="card">
          <div className="card-title">
            <Binary size={20} /> 1. Lung Field Segmentation Pipeline
          </div>
          <p style={{ fontSize: '0.95rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Applies adaptive thresholding and morphological operations to extract parenchymal lung masks while eliminating non-diagnostic tissue artifacts.
          </p>
          <ul style={{ paddingLeft: '1.2rem', fontSize: '0.9rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <li>Gaussian Blur spatial noise suppression</li>
            <li>Otsu global binarization & boundary isolation</li>
            <li>Thoracic border artifact clearing</li>
            <li>Morphological contour closing & hole filling</li>
            <li>CLAHE contrast enhancement (clipLimit=2.0)</li>
          </ul>
        </div>

        <div className="card">
          <div className="card-title">
            <Network size={20} /> 2. Multi-Stage Clinical Decision Support Framework
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', fontSize: '0.9rem' }}>
            <div style={{ borderLeft: '3px solid var(--primary-pink)', paddingLeft: '0.8rem' }}>
              <strong style={{ color: 'var(--accent-pink)' }}>Stage 1 • Deep AI Diagnostic Model (Primary Model)</strong>
              <p style={{ color: 'var(--text-muted)' }}>
                Advanced convolutional network specialized in thoracic CT pattern evaluation, producing primary probabilities across Normal, Benign, and Malignant categories alongside Grad-CAM visual heatmaps.
              </p>
            </div>
            <div style={{ borderLeft: '3px solid var(--primary-blue)', paddingLeft: '0.8rem' }}>
              <strong style={{ color: 'var(--primary-cyan)' }}>Stage 2 • Auxiliary Gradient Feature Classifier (Secondary Check)</strong>
              <p style={{ color: 'var(--text-muted)' }}>
                Extracts 9-orientation spatial gradient descriptors, reduces feature variance, and computes secondary classification validation probabilities.
              </p>
            </div>
            <div style={{ borderLeft: '3px solid var(--accent-purple)', paddingLeft: '0.8rem' }}>
              <strong style={{ color: 'var(--accent-purple)' }}>Stage 3 • Auxiliary Texture Pattern Analysis (Secondary Check)</strong>
              <p style={{ color: 'var(--text-muted)' }}>
                Evaluates parenchymal texture patterns using Local Binary Patterns (LBP) and statistical intensity distributions.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: '1.5rem', background: 'rgba(15, 23, 42, 0.6)' }}>
        <div className="card-title" style={{ color: 'var(--primary-cyan)' }}>
          <ShieldCheck size={20} /> Doctor & Clinical User Authentication
        </div>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.6, margin: 0 }}>
          User details, physician credentials, and access logs are managed securely through user profile table structures. Medical personnel can sign in with their credentials to access live diagnostic evaluation, review AI visual heatmaps, and export clinical decision reports.
        </p>
      </div>
    </div>
  );
};
