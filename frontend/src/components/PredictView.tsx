import React from 'react';
import { PredictResponse } from '../types/api';
import { UploadDropzone } from './UploadDropzone';
import { AlertTriangle, Sparkles, CheckCircle2, ShieldAlert } from 'lucide-react';

interface PredictViewProps {
  predictData: PredictResponse | null;
  onFileSelect: (file: File) => void;
}

export const PredictView: React.FC<PredictViewProps> = ({ predictData, onFileSelect }) => {
  if (!predictData) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div className="page-header">
          <h1 className="page-title">Prediction & Grad-CAM Visualizer</h1>
          <p className="page-subtitle">
            Upload a thoracic CT scan image to perform AI inference, extract segmented lung field ROIs, generate Grad-CAM heatmaps, and produce clinical explanations.
          </p>
        </div>
        <UploadDropzone onFileSelect={onFileSelect} />
      </div>
    );
  }

  const { predicted_class, confidence, probabilities, images, llm_explanation } = predictData;
  const confPct = (confidence * 100).toFixed(2);
  const normPct = (probabilities.Normal * 100).toFixed(2);
  const benignPct = (probabilities.Benign * 100).toFixed(2);
  const maligPct = (probabilities.Malignant * 100).toFixed(2);

  const isMalignant = predicted_class === 'Malignant';
  const isBenign = predicted_class === 'Benign';

  const badgeColor = isMalignant ? '#dc2626' : isBenign ? '#d97706' : '#16a34a';
  const badgeBg = isMalignant ? '#fef2f2' : isBenign ? '#fffbe6' : '#f0fdf4';
  const badgeBorder = isMalignant ? '#fecaca' : isBenign ? '#fef08a' : '#bbf7d0';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Top Header */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-title">Prediction Results</h1>
          <p className="page-subtitle">
            AI-assisted analysis of the uploaded CT scan
          </p>
        </div>
        <button
          onClick={() => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            input.onchange = (e: any) => {
              if (e.target.files && e.target.files[0]) {
                onFileSelect(e.target.files[0]);
              }
            };
            input.click();
          }}
          style={{
            background: '#4f46e5',
            color: '#ffffff',
            border: 'none',
            padding: '0.55rem 1.25rem',
            borderRadius: '8px',
            fontWeight: 600,
            fontSize: '0.86rem',
            cursor: 'pointer',
            boxShadow: '0 4px 12px rgba(79, 70, 229, 0.25)'
          }}
        >
          Upload Another CT Scan
        </button>
      </div>

      {/* 3 Images Row: Original CT Scan | Grad-CAM Heatmap | Overlay (Grad-CAM) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.25rem' }}>
        {/* Original CT */}
        <div className="card" style={{ padding: '1rem', margin: 0, textAlign: 'center' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.75rem' }}>
            Original CT Scan
          </div>
          <div style={{ background: '#000000', borderRadius: '8px', overflow: 'hidden', height: '240px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <img src={images.original} alt="Original CT" style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain' }} />
          </div>
        </div>

        {/* Grad-CAM Heatmap */}
        <div className="card" style={{ padding: '1rem', margin: 0, textAlign: 'center' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.75rem' }}>
            Grad-CAM Heatmap
          </div>
          <div style={{ background: '#000000', borderRadius: '8px', overflow: 'hidden', height: '240px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <img src={images.segmented_roi} alt="Heatmap" style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain' }} />
          </div>
        </div>

        {/* Overlay (Grad-CAM) */}
        <div className="card" style={{ padding: '1rem', margin: 0, textAlign: 'center' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.75rem' }}>
            Overlay (Grad-CAM)
          </div>
          <div style={{ background: '#000000', borderRadius: '8px', overflow: 'hidden', height: '240px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <img src={images.gradcam} alt="Overlay Grad-CAM" style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain' }} />
          </div>
        </div>
      </div>

      {/* 3 Columns Row: AI Prediction | Class Probabilities | AI Interpretation (Gemini) */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr 1.5fr', gap: '1.25rem' }}>
        {/* Card 1: AI Prediction */}
        <div className="card" style={{ margin: 0, padding: '1.25rem' }}>
          <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.75rem' }}>
            AI Prediction
          </div>
          <div style={{
            fontSize: '1.8rem',
            fontWeight: 800,
            color: badgeColor,
            background: badgeBg,
            border: `1px solid ${badgeBorder}`,
            padding: '0.4rem 1rem',
            borderRadius: '10px',
            display: 'inline-block',
            marginBottom: '1.25rem'
          }}>
            {predicted_class}
          </div>

          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#475569', marginBottom: '0.35rem' }}>
            Model Confidence
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ flex: 1, height: '10px', background: '#e2e8f0', borderRadius: '5px', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${confPct}%`, background: badgeColor, borderRadius: '5px', transition: 'width 0.6s' }} />
            </div>
            <span style={{ fontSize: '0.9rem', fontWeight: 800, color: '#0f172a' }}>{confPct}%</span>
          </div>
        </div>

        {/* Card 2: Class Probabilities */}
        <div className="card" style={{ margin: 0, padding: '1.25rem' }}>
          <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '1rem' }}>
            Class Probabilities
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            {/* Normal */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', fontWeight: 600, color: '#334155', marginBottom: '0.25rem' }}>
                <span>Normal</span>
                <span>{normPct}%</span>
              </div>
              <div style={{ height: '8px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${normPct}%`, background: '#16a34a', borderRadius: '4px' }} />
              </div>
            </div>

            {/* Benign */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', fontWeight: 600, color: '#334155', marginBottom: '0.25rem' }}>
                <span>Benign</span>
                <span>{benignPct}%</span>
              </div>
              <div style={{ height: '8px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${benignPct}%`, background: '#d97706', borderRadius: '4px' }} />
              </div>
            </div>

            {/* Malignant */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', fontWeight: 600, color: '#334155', marginBottom: '0.25rem' }}>
                <span>Malignant</span>
                <span>{maligPct}%</span>
              </div>
              <div style={{ height: '8px', background: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${maligPct}%`, background: '#dc2626', borderRadius: '4px' }} />
              </div>
            </div>
          </div>
        </div>

        {/* Card 3: AI Interpretation (Gemini) */}
        <div className="card" style={{ margin: 0, padding: '1.25rem', background: '#faf5ff', border: '1px solid #e9d5ff' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.88rem', fontWeight: 700, color: '#7e22ce', marginBottom: '0.75rem' }}>
            <Sparkles size={18} />
            <span>AI Interpretation (Gemini)</span>
          </div>
          <div style={{ fontSize: '0.82rem', color: '#4c1d95', lineHeight: 1.6, whiteSpace: 'pre-line', maxHeight: '180px', overflowY: 'auto', paddingRight: '0.5rem' }}>
            {llm_explanation?.text || `The model predicts this CT scan as ${predicted_class} with a probability of ${confPct}%. The highlighted regions in the Grad-CAM image indicate areas that contributed most to this prediction. These regions may correspond to suspicious tissue lesions or anatomical findings.`}
          </div>
        </div>
      </div>

      {/* Bottom Important Note Box */}
      <div className="card" style={{ background: '#fffbe6', border: '1px solid #fef08a', padding: '1rem 1.25rem', margin: 0 }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
          <AlertTriangle size={20} style={{ color: '#d97706', flexShrink: 0, marginTop: '0.1rem' }} />
          <div>
            <strong style={{ color: '#92400e', fontSize: '0.9rem' }}>Important Note:</strong>
            <p style={{ fontSize: '0.82rem', color: '#78350f', marginTop: '0.25rem', lineHeight: 1.5 }}>
              This is an AI-assisted prediction. The model may occasionally predict incorrect results. The output is intended for preliminary decision support only and must be verified by a qualified pulmonologist or radiologist using the complete CT examination and clinical information.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
