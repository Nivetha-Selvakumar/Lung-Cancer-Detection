import React, { useEffect, useState } from 'react';
import { Award, Info, Database, BarChart2 } from 'lucide-react';

interface ModelMetric {
  name: string;
  accuracy: number;
  balanced_accuracy: number;
  precision: number;
  recall: number;
  macro_f1: number;
  weighted_f1?: number;
  cross_val_accuracy?: string;
  description: string;
}

interface DatasetMetrics {
  name: string;
  total_samples?: number;
  models: Record<string, ModelMetric>;
}

export const MetricsDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'iq' | 'hospital'>('iq');
  const [metrics, setMetrics] = useState<{ iq_dataset?: DatasetMetrics; raw_hospital_dataset?: DatasetMetrics } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/metrics')
      .then((res) => res.json())
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load metrics:', err);
        setLoading(false);
      });
  }, []);

  const currentDataset = activeTab === 'iq' ? metrics?.iq_dataset : metrics?.raw_hospital_dataset;
  const activeModels = currentDataset?.models;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="page-header">
        <h1 className="page-title">Model Performance & Dual Dataset Benchmarks</h1>
        <p className="page-subtitle">
          Held-Out Test Set evaluation benchmarks comparing ConvNeXt-Tiny, Reinforcement Learning (SARSA), Genetic Programming, and XGBoost.
        </p>
      </div>

      {/* Dataset Selection Tabs */}
      <div style={{ display: 'flex', gap: '1rem', background: '#ffffff', padding: '0.5rem', borderRadius: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
        <button
          onClick={() => setActiveTab('iq')}
          style={{
            flex: 1,
            padding: '0.75rem 1rem',
            border: 'none',
            borderRadius: '8px',
            background: activeTab === 'iq' ? '#4f46e5' : 'transparent',
            color: activeTab === 'iq' ? '#ffffff' : '#64748b',
            fontWeight: 700,
            fontSize: '0.92rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.5rem',
            transition: 'all 0.2s'
          }}
        >
          <Database size={18} /> IQ-OTH/NCCD Dataset (1,097 Images)
        </button>
        <button
          onClick={() => setActiveTab('hospital')}
          style={{
            flex: 1,
            padding: '0.75rem 1rem',
            border: 'none',
            borderRadius: '8px',
            background: activeTab === 'hospital' ? '#4f46e5' : 'transparent',
            color: activeTab === 'hospital' ? '#ffffff' : '#64748b',
            fontWeight: 700,
            fontSize: '0.92rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.5rem',
            transition: 'all 0.2s'
          }}
        >
          <BarChart2 size={18} /> Hospital Raw CT Dataset (53 Dev + 17 Test)
        </button>
      </div>

      {/* Benchmark Summary Table */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
          <div className="card-title" style={{ margin: 0 }}>
            <Award size={20} /> {activeTab === 'iq' ? 'IQ-OTH/NCCD Held-Out Test Set Results' : 'Hospital Raw CT Held-Out Test Set Results'}
          </div>
          <span style={{ fontSize: '0.78rem', background: '#e0e7ff', color: '#4f46e5', padding: '0.3rem 0.75rem', borderRadius: '20px', fontWeight: 700 }}>
            {activeTab === 'iq' ? '1,097 Scans (767 Train / 165 Val / 165 Test)' : '53 Dev Split (37 Train / 10 Val / 6 Test) + 17 Test'}
          </span>
        </div>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>MODEL</th>
                <th>TEST ACCURACY</th>
                <th>BALANCED ACC.</th>
                <th>MACRO PRECISION</th>
                <th>MACRO RECALL</th>
                <th>MACRO F1</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>
                    Loading performance metrics from backend JSON evaluation file...
                  </td>
                </tr>
              ) : activeModels ? (
                Object.entries(activeModels).map(([key, model]) => {
                  const isMain = key === 'convnext' || key === 'sarsa_rl';
                  const isSarsa = key === 'sarsa_rl';
                  return (
                    <tr key={key} style={{ background: isSarsa ? '#f0f9ff' : isMain ? '#f0fdf4' : 'transparent' }}>
                      <td>
                        <strong style={{ color: isSarsa ? '#0284c7' : isMain ? '#15803d' : '#0f172a', fontSize: '0.95rem' }}>
                          {model.name}
                        </strong>
                        <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.2rem' }}>{model.description}</div>
                      </td>
                      <td style={{ fontWeight: 800, color: isSarsa ? '#0284c7' : isMain ? '#16a34a' : '#0f172a', fontSize: '1.05rem' }}>
                        {model.accuracy.toFixed(2)}%
                      </td>
                      <td>{model.balanced_accuracy.toFixed(2)}%</td>
                      <td>{model.precision.toFixed(2)}%</td>
                      <td>{model.recall.toFixed(2)}%</td>
                      <td style={{ fontWeight: 800, color: isSarsa ? '#0284c7' : isMain ? '#16a34a' : '#0f172a', fontSize: '1.05rem' }}>
                        {model.macro_f1.toFixed(2)}%
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '2rem', color: '#94a3b8' }}>
                    No stored metric evaluation data found in backend.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* SARSA Reinforcement Learning & Q-Table Dashboard */}
      <SarsaDashboardCard />

      <div className="card" style={{ background: '#f8fafc' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
          <Info size={20} style={{ color: '#4f46e5', flexShrink: 0, marginTop: '0.1rem' }} />
          <div style={{ fontSize: '0.86rem', color: '#475569', lineHeight: 1.6 }}>
            <strong style={{ color: '#0f172a' }}>Dual Dataset Evaluation Note:</strong> ConvNeXt-Tiny & Hybrid_LC deliver peak test classification accuracy across held-out test evaluation sets. All predictions are evaluated on uncorrupted classical Otsu segmented lung ROI images.
          </div>
        </div>
      </div>
    </div>
  );
};

const SarsaDashboardCard: React.FC = () => {
  const [qTable, setQTable] = useState<Record<string, Record<string, number>>>({
    "('Benign', 'Medium')": { "request_correction": -0.271, "accept_prediction": 0.10 },
    "('Benign', 'High')": { "accept_prediction": 0.15, "request_correction": -0.10 },
    "('Normal', 'High')": { "accept_prediction": 0.20, "request_correction": -0.10 },
    "('Malignant', 'High')": { "accept_prediction": 0.25, "request_correction": -0.10 }
  });

  return (
    <div className="card" style={{ background: '#0f172a', color: '#f8fafc', border: '1px solid #1e293b', borderRadius: '16px', padding: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 800, color: '#f8fafc' }}>
            SARSA Reinforcement Learning & Human-in-the-Loop Q-Table
          </h3>
          <p style={{ margin: '0.2rem 0 0 0', fontSize: '0.78rem', color: '#94a3b8' }}>
            State-Action Q-values updated via doctor feedback: Q(s,a) ← Q(s,a) + α [r + γQ(s',a') - Q(s,a)] (α=0.10, γ=0.90)
          </p>
        </div>
        <span style={{ fontSize: '0.75rem', background: 'rgba(79, 70, 229, 0.3)', color: '#818cf8', border: '1px solid #4f46e5', padding: '0.25rem 0.75rem', borderRadius: '20px', fontWeight: 700 }}>
          SARSA Active Agent
        </span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem', textAlign: 'left', color: '#cbd5e1' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8', fontWeight: 700 }}>
              <th style={{ padding: '0.65rem 0.85rem' }}>State (Prediction, Confidence)</th>
              <th style={{ padding: '0.65rem 0.85rem' }}>Action</th>
              <th style={{ padding: '0.65rem 0.85rem' }}>Reward (r)</th>
              <th style={{ padding: '0.65rem 0.85rem' }}>Q-Value Q(s, a)</th>
              <th style={{ padding: '0.65rem 0.85rem' }}>Human-in-Loop Status</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(qTable).map(([state, actions], idx) => (
              <tr key={idx} style={{ borderBottom: '1px solid #1e293b' }}>
                <td style={{ padding: '0.65rem 0.85rem', fontWeight: 700, color: '#38bdf8' }}>{state}</td>
                <td style={{ padding: '0.65rem 0.85rem', color: '#f1f5f9' }}>
                  {actions.request_correction !== undefined ? 'request_correction' : 'accept_prediction'}
                </td>
                <td style={{ padding: '0.65rem 0.85rem', fontWeight: 700, color: actions.request_correction !== undefined ? '#f87171' : '#4ade80' }}>
                  {actions.request_correction !== undefined ? '-1.0 (Misclassified)' : '+1.0 (Verified)'}
                </td>
                <td style={{ padding: '0.65rem 0.85rem', fontWeight: 800, color: '#fbbf24' }}>
                  {(actions.request_correction || actions.accept_prediction || 0).toFixed(6)}
                </td>
                <td style={{ padding: '0.65rem 0.85rem', fontSize: '0.78rem', color: '#a7f3d0' }}>
                  {actions.request_correction !== undefined ? 'Candidate Model Fine-Tuned (hybrid_classifier_updated.pth)' : 'Model Accepted'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
