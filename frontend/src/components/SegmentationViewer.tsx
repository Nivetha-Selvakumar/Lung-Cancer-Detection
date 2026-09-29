import React from 'react';
import { Layers, Flame, Eye, Info } from 'lucide-react';

interface SegmentationViewerProps {
  images?: {
    original: string;
    mask: string;
    segmented_roi: string;
    gradcam: string;
  };
  camFocusRatio?: number;
  maskCoverage?: number;
}

export const SegmentationViewer: React.FC<SegmentationViewerProps> = ({ images, camFocusRatio, maskCoverage }) => {
  const coveragePct = maskCoverage !== undefined ? (maskCoverage > 1 ? maskCoverage : maskCoverage * 100) : undefined;
  const focusPct = camFocusRatio !== undefined ? (camFocusRatio > 1 ? camFocusRatio : camFocusRatio * 100) : undefined;

  return (
    <div className="card">
      <div className="card-title">
        <Layers size={20} /> Lung Segmentation & Explainable AI (Grad-CAM)
      </div>

      <div className="segmentation-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem' }}>
        {/* 1. Original CT */}
        <div className="img-box">
          <div className="img-box-title">1. Original CT</div>
          <div className="img-preview-wrapper">
            {images?.original ? (
              <img src={images.original} alt="Original CT Scan" />
            ) : (
              <span className="img-placeholder">Awaiting Image...</span>
            )}
          </div>
        </div>

        {/* 2. Lung Mask */}
        <div className="img-box">
          <div className="img-box-title">2. Lung Mask</div>
          <div className="img-preview-wrapper">
            {images?.mask ? (
              <img src={images.mask} alt="Lung Field Mask" />
            ) : (
              <span className="img-placeholder">Awaiting Image...</span>
            )}
          </div>
          {coveragePct !== undefined && (
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.3rem', textAlign: 'center' }}>
              Coverage: <strong>{coveragePct.toFixed(1)}%</strong>
            </div>
          )}
        </div>

        {/* 3. Segmented Lung ROI */}
        <div className="img-box">
          <div className="img-box-title">3. Segmented Lung ROI</div>
          <div className="img-preview-wrapper">
            {images?.segmented_roi ? (
              <img src={images.segmented_roi} alt="Segmented Lung ROI" />
            ) : (
              <span className="img-placeholder">Awaiting Image...</span>
            )}
          </div>
        </div>

        {/* 4. Grad-CAM Visualization */}
        <div className="img-box">
          <div className="img-box-title" style={{ color: 'var(--accent-pink)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.25rem' }}>
            <Flame size={14} /> 4. Model Attention / Grad-CAM
          </div>
          <div className="img-preview-wrapper">
            {images?.gradcam ? (
              <img src={images.gradcam} alt="Grad-CAM Overlay" />
            ) : (
              <span className="img-placeholder">Awaiting Image...</span>
            )}
          </div>
          {focusPct !== undefined && (
            <div style={{ fontSize: '0.75rem', color: 'var(--primary-cyan)', marginTop: '0.3rem', textAlign: 'center' }}>
              Lung Focus: <strong>{focusPct.toFixed(1)}%</strong>
            </div>
          )}
        </div>
      </div>

      {/* Grad-CAM Disclaimer Box */}
      <div style={{
        marginTop: '1rem',
        padding: '0.75rem 1rem',
        background: 'rgba(15, 23, 42, 0.6)',
        borderRadius: 'var(--radius-sm)',
        border: '1px solid var(--border-color)',
        fontSize: '0.8rem',
        color: 'var(--text-muted)',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.5rem'
      }}>
        <Info size={16} style={{ color: 'var(--primary-cyan)', flexShrink: 0, marginTop: '0.1rem' }} />
        <div>
          <strong style={{ color: '#e2e8f0' }}>Model Attention / Grad-CAM Explanation:</strong> The highlighted regions represent image areas that contributed to the model's prediction. They should not be interpreted as confirmed tumor or cancer locations.
        </div>
      </div>
    </div>
  );
};
