import React from 'react';
import { ResearchModelResult } from '../types/api';
import { FlaskConical, Activity, Layers } from 'lucide-react';

interface ResearchComparisonViewProps {
  researchModels?: {
    xgboost?: ResearchModelResult;
    genetic_programming?: ResearchModelResult;
  };
}

export const ResearchComparisonView: React.FC<ResearchComparisonViewProps> = ({ researchModels }) => {
  if (!researchModels || (!researchModels.xgboost && !researchModels.genetic_programming)) return null;

  const renderModelCard = (key: 'xgboost' | 'genetic_programming', modelData?: ResearchModelResult) => {
    if (!modelData || !modelData.probabilities) return null;

    const { Normal, Benign, Malignant } = modelData.probabilities;
    const normPct = Normal > 1 ? Normal : Normal * 100;
    const benignPct = Benign > 1 ? Benign : Benign * 100;
    const maligPct = Malignant > 1 ? Malignant : Malignant * 100;
    const confPct = modelData.confidence > 1 ? modelData.confidence : modelData.confidence * 100;

    const isFirst = key === 'xgboost';
    const title = isFirst ? 'Auxiliary Diagnostic Analysis 1' : 'Auxiliary Diagnostic Analysis 2';
    const Icon = isFirst ? Activity : Layers;

    return (
      <div className="card" style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid var(--border-color)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
          <div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Secondary Check • {isFirst ? 'Feature Analysis' : 'Symbolic Pattern Check'}
            </span>
            <h4 style={{ margin: 0, fontSize: '1rem', color: '#e2e8f0' }}>{title}</h4>
          </div>
          <Icon size={20} style={{ color: isFirst ? 'var(--primary-blue)' : 'var(--accent-purple)' }} />
        </div>

        <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', marginBottom: '0.75rem' }}>
          <span style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--primary-cyan)' }}>
            {modelData.predicted_class}
          </span>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Confidence: {confPct.toFixed(1)}%
          </span>
        </div>

        <div className="prob-bars">
          <div className="prob-item">
            <div className="prob-label-row">
              <span>Normal</span>
              <span>{normPct.toFixed(1)}%</span>
            </div>
            <div className="prob-bar-track">
              <div className="prob-bar-fill fill-normal" style={{ width: `${Math.max(normPct, 2)}%` }} />
            </div>
          </div>

          <div className="prob-item">
            <div className="prob-label-row">
              <span>Benign</span>
              <span>{benignPct.toFixed(1)}%</span>
            </div>
            <div className="prob-bar-track">
              <div className="prob-bar-fill fill-benign" style={{ width: `${Math.max(benignPct, 2)}%` }} />
            </div>
          </div>

          <div className="prob-item">
            <div className="prob-label-row">
              <span>Malignant</span>
              <span>{maligPct.toFixed(1)}%</span>
            </div>
            <div className="prob-bar-track">
              <div className="prob-bar-fill fill-malignant" style={{ width: `${Math.max(maligPct, 2)}%` }} />
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ marginTop: '2rem' }}>
      <div style={{
        background: 'rgba(30, 41, 59, 0.4)',
        padding: '0.75rem 1rem',
        borderRadius: 'var(--radius-sm)',
        border: '1px solid var(--border-color)',
        marginBottom: '1rem',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        fontSize: '0.85rem',
        color: 'var(--text-muted)'
      }}>
        <FlaskConical size={18} style={{ color: 'var(--accent-purple)' }} />
        <div>
          <strong style={{ color: '#e2e8f0' }}>Secondary Diagnostic Verification:</strong> Secondary analytical checks provide cross-reference probability validation for attending physicians.
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
        {renderModelCard('xgboost', researchModels.xgboost)}
        {renderModelCard('genetic_programming', researchModels.genetic_programming)}
      </div>
    </div>
  );
};
