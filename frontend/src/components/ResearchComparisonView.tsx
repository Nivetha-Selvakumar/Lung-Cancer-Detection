import React from 'react';
import { ResearchModelResult } from '../types/api';
import { Activity, Layers, Cpu, Award } from 'lucide-react';

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

    const isXGB = key === 'xgboost';
    const title = isXGB ? 'XGBoost Auxiliary Classifier' : 'Genetic Programming Symbolic Model';
    const subtitle = isXGB ? 'Decision Tree Texture & Intensity Features (89.09% Benchmark Acc)' : 'Evolutionary Symbolic Morphological Pattern Rules (62.73% Benchmark Acc)';
    const Icon = isXGB ? Cpu : Layers;

    const isMal = modelData.predicted_class === 'Malignant';
    const isBen = modelData.predicted_class === 'Benign';
    const tagBg = isMal ? '#fef2f2' : isBen ? '#fff7ed' : '#f0fdf4';
    const tagBorder = isMal ? '#fecaca' : isBen ? '#ffedd5' : '#bbf7d0';
    const tagColor = isMal ? '#dc2626' : isBen ? '#ea580c' : '#16a34a';

    return (
      <div
        className="card"
        style={{
          background: '#ffffff',
          borderRadius: '16px',
          padding: '1.25rem 1.5rem',
          border: '1px solid #e2e8f0',
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.03)',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between'
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.6rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem' }}>
              <div style={{
                width: '32px',
                height: '32px',
                borderRadius: '8px',
                background: isXGB ? '#e0e7ff' : '#ede9fe',
                color: isXGB ? '#4f46e5' : '#7c3aed',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0
              }}>
                <Icon size={18} />
              </div>
              <div>
                <h4 style={{ margin: 0, fontSize: '0.96rem', fontWeight: 800, color: '#0f172a' }}>{title}</h4>
                <div style={{ fontSize: '0.74rem', color: '#64748b' }}>{subtitle}</div>
              </div>
            </div>
            <span style={{
              background: tagBg,
              border: `1px solid ${tagBorder}`,
              color: tagColor,
              fontWeight: 800,
              fontSize: '0.82rem',
              padding: '0.2rem 0.65rem',
              borderRadius: '20px'
            }}>
              {modelData.predicted_class}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', margin: '0.75rem 0' }}>
            <span style={{ fontSize: '0.82rem', color: '#64748b', fontWeight: 600 }}>Confidence:</span>
            <span style={{ fontSize: '0.95rem', fontWeight: 800, color: '#0f172a' }}>{confPct.toFixed(1)}%</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', marginTop: '0.75rem' }}>
            {/* Normal */}
            <div style={{ display: 'grid', gridTemplateColumns: '65px 1fr 40px', alignItems: 'center', gap: '0.6rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#475569' }}>Normal</span>
              <div style={{ height: '8px', background: '#f1f5f9', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${normPct}%`, background: '#2dd4bf', borderRadius: '4px' }} />
              </div>
              <span style={{ fontSize: '0.78rem', fontWeight: 700, color: '#0f172a', textAlign: 'right' }}>{normPct.toFixed(1)}%</span>
            </div>

            {/* Benign */}
            <div style={{ display: 'grid', gridTemplateColumns: '65px 1fr 40px', alignItems: 'center', gap: '0.6rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#475569' }}>Benign</span>
              <div style={{ height: '8px', background: '#f1f5f9', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${benignPct}%`, background: '#818cf8', borderRadius: '4px' }} />
              </div>
              <span style={{ fontSize: '0.78rem', fontWeight: 700, color: '#0f172a', textAlign: 'right' }}>{benignPct.toFixed(1)}%</span>
            </div>

            {/* Malignant */}
            <div style={{ display: 'grid', gridTemplateColumns: '65px 1fr 40px', alignItems: 'center', gap: '0.6rem' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#475569' }}>Malignant</span>
              <div style={{ height: '8px', background: '#f1f5f9', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${maligPct}%`, background: '#f43f5e', borderRadius: '4px' }} />
              </div>
              <span style={{ fontSize: '0.78rem', fontWeight: 700, color: '#0f172a', textAlign: 'right' }}>{maligPct.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
      <div style={{
        background: '#f8fafc',
        padding: '0.85rem 1.15rem',
        borderRadius: '12px',
        border: '1px solid #e2e8f0',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <Award size={18} style={{ color: '#4f46e5' }} />
          <span style={{ fontSize: '0.88rem', fontWeight: 700, color: '#0f172a' }}>
            Multi-Model Secondary Verification Checks (Connected Trained Models)
          </span>
        </div>
        <span style={{ fontSize: '0.76rem', color: '#64748b' }}>
          Evaluated in parallel across XGBoost & Genetic Programming models
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>
        {renderModelCard('xgboost', researchModels.xgboost)}
        {renderModelCard('genetic_programming', researchModels.genetic_programming)}
      </div>
    </div>
  );
};
