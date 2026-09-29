import React, { useEffect, useState } from 'react';
import { Table } from 'lucide-react';
import { MetricsResponse } from '../types/api';
import { fetchMetrics } from '../services/api';

export const MetricsDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchMetrics()
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div>
      <div className="hero-header">
        <h1 className="hero-title">Multi-Aspect Quantitative Evaluation</h1>
        <p className="hero-subtitle">
          Comparative benchmarking across Machine Learning (XGBoost), Genetic Programming (Evolutionary AI), and ConvNeXt-Tiny Deep Learning on the IQ-OTHNCCD Lung Dataset.
        </p>
      </div>

      <div className="metrics-table-card">
        <div className="card-title">
          <Table size={20} /> Tri-Aspect Model Performance Comparison
        </div>

        <div className="table-responsive">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Aspect & Model Name</th>
                <th>Accuracy</th>
                <th>Balanced Acc</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Macro F1</th>
              </tr>
            </thead>
            <tbody>
              {loading && (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                    Loading evaluation metrics...
                  </td>
                </tr>
              )}

              {error && (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', color: 'var(--status-benign)' }}>
                    Metrics json not generated yet. Model training running in background.
                  </td>
                </tr>
              )}

              {metrics &&
                Object.entries(metrics.aspects).map(([key, aspect]) => (
                  <tr key={key}>
                    <td>
                      <strong>{aspect.name}</strong>
                    </td>
                    <td className="metric-pill">{aspect.accuracy}%</td>
                    <td>{aspect.balanced_accuracy}%</td>
                    <td>{aspect.precision}%</td>
                    <td>{aspect.recall}%</td>
                    <td className="metric-pill">{aspect.macro_f1}%</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
