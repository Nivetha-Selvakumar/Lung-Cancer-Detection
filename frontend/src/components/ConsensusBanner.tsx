import React from 'react';
import { PredictResponse } from '../types/api';
import { ShieldAlert, Activity, AlertTriangle, FileText } from 'lucide-react';

interface PredictionResultCardProps {
  predictData?: PredictResponse;
}

export const ConsensusBanner: React.FC<PredictionResultCardProps> = ({ predictData }) => {
  if (!predictData) return null;

  const {
    case_id,
    predicted_class,
    confidence,
    probabilities,
    malignant_probability,
    model_estimated_malignant_probability
  } = predictData;

  // Convert float probabilities (0-1) to percentage (0-100) if needed
  const normPct = probabilities.Normal > 1 ? probabilities.Normal : probabilities.Normal * 100;
  const benignPct = probabilities.Benign > 1 ? probabilities.Benign : probabilities.Benign * 100;
  const maligPct = probabilities.Malignant > 1 ? probabilities.Malignant : probabilities.Malignant * 100;
  const confPct = confidence > 1 ? confidence : confidence * 100;
  const modelMalignantPct = model_estimated_malignant_probability > 1
    ? model_estimated_malignant_probability
    : (malignant_probability > 1 ? malignant_probability : malignant_probability * 100);

  let badgeClass = 'badge-normal';
  if (predicted_class === 'Malignant') {
    badgeClass = 'badge-malignant';
  } else if (predicted_class === 'Benign') {
    badgeClass = 'badge-benign';
  }

  return (
    <div className="card" style={{ marginTop: '1.5rem', border: '1px solid var(--border-glow)' }}>
      {/* Header Banner */}
      <div style={{
        textAlign: 'center',
        padding: '1rem',
        background: 'rgba(15, 23, 42, 0.8)',
        borderRadius: 'var(--radius-sm)',
        borderBottom: '1px solid var(--border-color)',
        marginBottom: '1.5rem'
      }}>
        <h2 style={{
          letterSpacing: '0.15em',
          fontSize: '1.1rem',
          color: 'var(--primary-cyan)',
          textTransform: 'uppercase',
          margin: 0
        }}>
          AI PREDICTION RESULT
        </h2>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
          Case ID: <strong style={{ color: '#fff' }}>{case_id}</strong> • Main Model: ConvNeXt-Tiny
        </div>
      </div>

      {/* Primary Metrics Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: '1rem',
        marginBottom: '1.5rem'
      }}>
        {/* Classification */}
        <div style={{
          background: 'rgba(30, 41, 59, 0.6)',
          padding: '1.25rem',
          borderRadius: 'var(--radius-sm)',
          textAlign: 'center',
          border: '1px solid var(--border-color)'
        }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
            Predicted Classification
          </div>
          <div className={`consensus-badge ${badgeClass}`} style={{ fontSize: '1.4rem', padding: '0.4rem 1.2rem', display: 'inline-block' }}>
            {predicted_class.toUpperCase()}
          </div>
        </div>

        {/* Model-Estimated Malignant Probability */}
        <div style={{
          background: 'rgba(30, 41, 59, 0.6)',
          padding: '1.25rem',
          borderRadius: 'var(--radius-sm)',
          textAlign: 'center',
          border: '1px solid var(--border-color)'
        }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
            Model-Estimated Malignant Probability
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--status-malignant)' }}>
            {modelMalignantPct.toFixed(1)}%
          </div>
        </div>

        {/* Confidence */}
        <div style={{
          background: 'rgba(30, 41, 59, 0.6)',
          padding: '1.25rem',
          borderRadius: 'var(--radius-sm)',
          textAlign: 'center',
          border: '1px solid var(--border-color)'
        }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
            Confidence
          </div>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--primary-cyan)' }}>
            {confPct.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Important Medical Disclaimer Box */}
      <div style={{
        background: 'rgba(239, 68, 68, 0.08)',
        borderLeft: '4px solid var(--status-malignant)',
        padding: '1rem 1.25rem',
        borderRadius: 'var(--radius-sm)',
        marginBottom: '1.5rem',
        fontSize: '0.85rem',
        lineHeight: 1.5,
        color: '#f87171'
      }}>
        <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
          <ShieldAlert size={18} /> Medical Terminology Notice
        </div>
        <div>
          This value represents the ConvNeXt-Tiny model's estimated probability for the Malignant class for the uploaded CT image. It is not a clinically validated cancer risk and should not be interpreted as a medical diagnosis.
        </div>
      </div>

      {/* Class Probability Distribution Bars */}
      <div style={{
        background: 'rgba(15, 23, 42, 0.5)',
        padding: '1.25rem',
        borderRadius: 'var(--radius-sm)',
        marginBottom: '1.5rem',
        border: '1px solid var(--border-color)'
      }}>
        <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#e2e8f0', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Activity size={18} style={{ color: 'var(--primary-cyan)' }} /> Class Probability Distribution (ConvNeXt-Tiny)
        </div>

        <div className="prob-bars">
          <div className="prob-item">
            <div className="prob-label-row">
              <span>Normal</span>
              <span style={{ fontWeight: 600 }}>{normPct.toFixed(1)}%</span>
            </div>
            <div className="prob-bar-track">
              <div className="prob-bar-fill fill-normal" style={{ width: `${Math.max(normPct, 2)}%` }} />
            </div>
          </div>

          <div className="prob-item">
            <div className="prob-label-row">
              <span>Benign</span>
              <span style={{ fontWeight: 600 }}>{benignPct.toFixed(1)}%</span>
            </div>
            <div className="prob-bar-track">
              <div className="prob-bar-fill fill-benign" style={{ width: `${Math.max(benignPct, 2)}%` }} />
            </div>
          </div>

          <div className="prob-item">
            <div className="prob-label-row">
              <span>Malignant</span>
              <span style={{ fontWeight: 600 }}>{maligPct.toFixed(1)}%</span>
            </div>
            <div className="prob-bar-track">
              <div className="prob-bar-fill fill-malignant" style={{ width: `${Math.max(maligPct, 2)}%` }} />
            </div>
          </div>
        </div>
      </div>

      {/* Case Interpretation */}
      <div style={{
        background: 'rgba(30, 41, 59, 0.4)',
        padding: '1.25rem',
        borderRadius: 'var(--radius-sm)',
        border: '1px solid var(--border-color)',
        fontSize: '0.88rem',
        lineHeight: 1.6,
        color: '#cbd5e1'
      }}>
        <div style={{ fontWeight: 600, color: 'var(--primary-cyan)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FileText size={18} /> Interpretation
        </div>
        <p style={{ margin: '0 0 0.5rem 0' }}>
          For this uploaded CT image, the ConvNeXt-Tiny model assigned the highest probability to the <strong>{predicted_class}</strong> class.
        </p>
        <p style={{ margin: '0 0 0.5rem 0' }}>
          The model-estimated malignant probability is <strong>{modelMalignantPct.toFixed(1)}%</strong>.
        </p>
        <p style={{ margin: 0, fontStyle: 'italic', color: 'var(--text-muted)' }}>
          This is an AI model output and not a clinically validated cancer probability or diagnosis. The system is an AI-assisted preliminary classification/research prototype and does not replace clinical diagnosis.
        </p>
      </div>
    </div>
  );
};
