import React from 'react';
import { ResearchModelResult } from '../types/api';
import { GitFork, Dna, BrainCircuit } from 'lucide-react';

interface AspectCardProps {
  aspectKey: 'xgboost' | 'genetic_programming' | 'convnext';
  aspect?: ResearchModelResult;
}

export const AspectCard: React.FC<AspectCardProps> = ({ aspectKey, aspect }) => {
  if (!aspect || !aspect.probabilities) {
    return (
      <div className={`aspect-card aspect-${aspectKey === 'xgboost' ? '1' : aspectKey === 'genetic_programming' ? '2' : '3'}`}>
        <div className="aspect-header">
          <div>
            <span className="aspect-tag">Aspect • {aspectKey}</span>
            <h3 className="aspect-name">{aspect?.name || aspectKey}</h3>
          </div>
        </div>
        <div style={{ color: 'var(--status-benign)', marginTop: '1rem', fontSize: '0.9rem' }}>
          {aspect?.error || 'Model loading...'}
        </div>
      </div>
    );
  }

  const { Normal = 0, Benign = 0, Malignant = 0 } = aspect.probabilities;

  const renderIcon = () => {
    switch (aspectKey) {
      case 'xgboost':
        return <GitFork size={22} style={{ color: 'var(--primary-blue)' }} />;
      case 'genetic_programming':
        return <Dna size={22} style={{ color: 'var(--accent-purple)' }} />;
      case 'convnext':
        return <BrainCircuit size={22} style={{ color: 'var(--accent-pink)' }} />;
    }
  };

  const getTag = () => {
    switch (aspectKey) {
      case 'xgboost':
        return 'Research • Machine Learning';
      case 'genetic_programming':
        return 'Research • Evolutionary AI';
      case 'convnext':
        return 'Main Model • Deep Learning';
    }
  };

  const cardClass = aspectKey === 'xgboost' ? 'aspect-1' : aspectKey === 'genetic_programming' ? 'aspect-2' : 'aspect-3';

  return (
    <div className={`aspect-card ${cardClass}`}>
      <div className="aspect-header">
        <div>
          <span className="aspect-tag">{getTag()}</span>
          <h3 className="aspect-name">{aspect.name}</h3>
        </div>
        {renderIcon()}
      </div>

      <div className="aspect-result-class">{aspect.predicted_class}</div>
      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
        Confidence: <strong>{(aspect.confidence > 1 ? aspect.confidence : aspect.confidence * 100).toFixed(1)}%</strong>
      </div>

      <div className="prob-bars">
        <div className="prob-item">
          <div className="prob-label-row">
            <span>Normal</span>
            <span>{(Normal > 1 ? Normal : Normal * 100).toFixed(1)}%</span>
          </div>
          <div className="prob-bar-track">
            <div className="prob-bar-fill fill-normal" style={{ width: `${Math.max(Normal > 1 ? Normal : Normal * 100, 2)}%` }} />
          </div>
        </div>

        <div className="prob-item">
          <div className="prob-label-row">
            <span>Benign</span>
            <span>{(Benign > 1 ? Benign : Benign * 100).toFixed(1)}%</span>
          </div>
          <div className="prob-bar-track">
            <div className="prob-bar-fill fill-benign" style={{ width: `${Math.max(Benign > 1 ? Benign : Benign * 100, 2)}%` }} />
          </div>
        </div>

        <div className="prob-item">
          <div className="prob-label-row">
            <span>Malignant</span>
            <span>{(Malignant > 1 ? Malignant : Malignant * 100).toFixed(1)}%</span>
          </div>
          <div className="prob-bar-track">
            <div className="prob-bar-fill fill-malignant" style={{ width: `${Math.max(Malignant > 1 ? Malignant : Malignant * 100, 2)}%` }} />
          </div>
        </div>
      </div>
    </div>
  );
};
