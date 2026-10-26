import React, { useEffect, useState } from 'react';
import { Award, Info, Database, RefreshCw, CheckCircle2, Layers, Cpu, ShieldCheck } from 'lucide-react';
import { ComparisonModelRow } from '../types/api';

interface CbrCaseSummary {
  case_id: string;
  timestamp: string;
  image_filename: string;
  ai_prediction: string;
  ai_confidence: number;
  doctor_verified_class: string;
  prediction_correct: boolean;
  notes: string;
  has_82d_features: boolean;
}

interface CbrCasesResponse {
  success: boolean;
  total: number;
  counts: {
    Normal: number;
    Benign: number;
    Malignant: number;
  };
  cases: CbrCaseSummary[];
}

export const MetricsDashboard: React.FC = () => {
  const [comparisonTable, setComparisonTable] = useState<ComparisonModelRow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/metrics')
      .then((res) => res.json())
      .then((data) => {
        if (data.comparison_table && Array.isArray(data.comparison_table)) {
          setComparisonTable(data.comparison_table);
        } else {
          // Fallback to official benchmark rows
          setComparisonTable([
            { rank: 1, model: "Hybrid Classifier — First-Tuned", accuracy: 95.45, balanced_accuracy: 92.06, macro_precision: 96.94, macro_recall: 92.06, macro_f1: 94.19, weighted_f1: 95.39, status: "Selected Champion Model" },
            { rank: 2, model: "Hybrid Classifier — Second-Tuned", accuracy: 93.64, balanced_accuracy: 92.66, macro_precision: 93.57, macro_recall: 92.66, macro_f1: 93.00, weighted_f1: 93.59 },
            { rank: 3, model: "Hybrid Classifier — Baseline", accuracy: 87.27, balanced_accuracy: 83.53, macro_precision: 81.48, macro_recall: 83.53, macro_f1: 82.33, weighted_f1: 87.36 },
            { rank: 4, model: "XGBoost — Fine-Tuned", accuracy: 85.45, balanced_accuracy: 73.81, macro_precision: 84.42, macro_recall: 73.81, macro_f1: 76.74, weighted_f1: 84.50 },
            { rank: 5, model: "XGBoost — Original", accuracy: 83.64, balanced_accuracy: 70.44, macro_precision: 88.34, macro_recall: 70.44, macro_f1: 73.74, weighted_f1: 82.26 },
            { rank: 6, model: "Genetic Programming (GP)", accuracy: 73.64, balanced_accuracy: 54.56, macro_precision: 48.57, macro_recall: 54.56, macro_f1: 51.39, weighted_f1: 69.43 }
          ]);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load metrics:', err);
        setComparisonTable([
          { rank: 1, model: "Hybrid Classifier — First-Tuned", accuracy: 95.45, balanced_accuracy: 92.06, macro_precision: 96.94, macro_recall: 92.06, macro_f1: 94.19, weighted_f1: 95.39, status: "Selected Champion Model" },
          { rank: 2, model: "Hybrid Classifier — Second-Tuned", accuracy: 93.64, balanced_accuracy: 92.66, macro_precision: 93.57, macro_recall: 92.66, macro_f1: 93.00, weighted_f1: 93.59 },
          { rank: 3, model: "Hybrid Classifier — Baseline", accuracy: 87.27, balanced_accuracy: 83.53, macro_precision: 81.48, macro_recall: 83.53, macro_f1: 82.33, weighted_f1: 87.36 },
          { rank: 4, model: "XGBoost — Fine-Tuned", accuracy: 85.45, balanced_accuracy: 73.81, macro_precision: 84.42, macro_recall: 73.81, macro_f1: 76.74, weighted_f1: 84.50 },
          { rank: 5, model: "XGBoost — Original", accuracy: 83.64, balanced_accuracy: 70.44, macro_precision: 88.34, macro_recall: 70.44, macro_f1: 73.74, weighted_f1: 82.26 },
          { rank: 6, model: "Genetic Programming (GP)", accuracy: 73.64, balanced_accuracy: 54.56, macro_precision: 48.57, macro_recall: 54.56, macro_f1: 51.39, weighted_f1: 69.43 }
        ]);
        setLoading(false);
      });
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="page-header">
        <h1 className="page-title">Model Performance & Dataset Benchmarks</h1>
        <p className="page-subtitle">
          Evaluation metrics for models trained and benchmarked on the unified <strong>PSG-IQ Lung CT Dataset</strong> (1,097 IQ-OTHNCCD + 53 PSG IMSR development cohort; 165 internal test scans). Human-in-the-Loop Case-Based Reasoning (CBR) system active.
        </p>
      </div>

      {/* Official Benchmark Comparison Table Matching Reference Image */}
      <div className="card" style={{ background: '#ffffff', borderRadius: '16px', border: '1px solid #e2e8f0', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.03)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: '#fef3c7', color: '#d97706', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Award size={20} />
              </div>
              <h2 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 800, color: '#0f172a', letterSpacing: '-0.01em' }}>
                LUNG CT CLASSIFICATION — FINAL MODEL COMPARISON
              </h2>
            </div>
            <p style={{ margin: '0.35rem 0 0 0', fontSize: '0.8rem', color: '#64748b' }}>
              Comparative benchmark across all 6 diagnostic models on held-out test cohort (165 images). Rank 1 First-Tuned Champion model selected for clinical deployment.
            </p>
          </div>
          <span style={{ fontSize: '0.78rem', background: '#fef9c3', color: '#854d0e', border: '1px solid #fde047', padding: '0.35rem 0.85rem', borderRadius: '20px', fontWeight: 700 }}>
            ★ Champion: 95.45% Accuracy
          </span>
        </div>

        <div className="data-table-container" style={{ overflowX: 'auto' }}>
          <table className="data-table" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: '#f1f5f9', borderBottom: '2px solid #cbd5e1', fontSize: '0.78rem', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                <th style={{ padding: '0.85rem 1rem', textAlign: 'center' }}>Rank</th>
                <th style={{ padding: '0.85rem 1rem' }}>Model</th>
                <th style={{ padding: '0.85rem 1rem', textAlign: 'right' }}>Accuracy (%)</th>
                <th style={{ padding: '0.85rem 1rem', textAlign: 'right' }}>Balanced Accuracy (%)</th>
                <th style={{ padding: '0.85rem 1rem', textAlign: 'right' }}>Macro Precision (%)</th>
                <th style={{ padding: '0.85rem 1rem', textAlign: 'right' }}>Macro Recall (%)</th>
                <th style={{ padding: '0.85rem 1rem', textAlign: 'right' }}>Macro F1 (%)</th>
                <th style={{ padding: '0.85rem 1rem', textAlign: 'right' }}>Weighted F1 (%)</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={8} style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>
                    Loading performance metrics from benchmark configuration...
                  </td>
                </tr>
              ) : comparisonTable.map((row) => {
                const isChampion = row.rank === 1;
                return (
                  <tr
                    key={row.rank}
                    style={{
                      background: isChampion ? '#fef9c3' : 'transparent',
                      borderBottom: isChampion ? '2px solid #fde047' : '1px solid #e2e8f0',
                      transition: 'background 0.2s'
                    }}
                  >
                    <td style={{ padding: '0.95rem 1rem', textAlign: 'center', fontWeight: 800, color: isChampion ? '#854d0e' : '#334155' }}>
                      {row.rank}
                    </td>
                    <td style={{ padding: '0.95rem 1rem' }}>
                      <strong style={{ color: isChampion ? '#854d0e' : '#0f172a', fontSize: '0.94rem' }}>
                        {row.model}
                      </strong>
                      {isChampion && (
                        <span style={{ marginLeft: '0.6rem', fontSize: '0.7rem', background: '#ca8a04', color: '#ffffff', padding: '0.15rem 0.55rem', borderRadius: '10px', fontWeight: 700 }}>
                          Champion
                        </span>
                      )}
                    </td>
                    <td style={{ padding: '0.95rem 1rem', textAlign: 'right', fontWeight: 800, color: isChampion ? '#854d0e' : '#0f172a', fontSize: '1rem' }}>
                      {row.accuracy.toFixed(2)}
                    </td>
                    <td style={{ padding: '0.95rem 1rem', textAlign: 'right', fontWeight: isChampion ? 700 : 500, color: isChampion ? '#854d0e' : '#334155' }}>
                      {row.balanced_accuracy.toFixed(2)}
                    </td>
                    <td style={{ padding: '0.95rem 1rem', textAlign: 'right', fontWeight: isChampion ? 700 : 500, color: isChampion ? '#854d0e' : '#334155' }}>
                      {row.macro_precision.toFixed(2)}
                    </td>
                    <td style={{ padding: '0.95rem 1rem', textAlign: 'right', fontWeight: isChampion ? 700 : 500, color: isChampion ? '#854d0e' : '#334155' }}>
                      {row.macro_recall.toFixed(2)}
                    </td>
                    <td style={{ padding: '0.95rem 1rem', textAlign: 'right', fontWeight: 800, color: isChampion ? '#854d0e' : '#0f172a' }}>
                      {row.macro_f1.toFixed(2)}
                    </td>
                    <td style={{ padding: '0.95rem 1rem', textAlign: 'right', fontWeight: 800, color: isChampion ? '#854d0e' : '#0f172a' }}>
                      {row.weighted_f1.toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>


      {/* Case-Based Reasoning (CBR) Diagnostic Knowledge Base Card */}
      <CbrDashboardCard />

      <div className="card" style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '14px', padding: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
          <Info size={20} style={{ color: '#4f46e5', flexShrink: 0, marginTop: '0.1rem' }} />
          <div style={{ fontSize: '0.86rem', color: '#475569', lineHeight: 1.6 }}>
            <strong style={{ color: '#0f172a' }}>Dataset Allocation Note:</strong> The primary IQ-OTHNCCD cohort (1,097 scans) is used for training, validation, and internal testing. The PSG IMSR development cohort (53 scans) is combined for clinical internal test validation (total 165 test scans). The separate PSG IMSR Test folder (17 scans) is strictly preserved for untouched external validation.
          </div>
        </div>
      </div>
    </div>
  );
};

// Case-Based Reasoning (CBR) Dashboard Component
const CbrDashboardCard: React.FC = () => {
  const [cbrData, setCbrData] = useState<CbrCasesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);
  const [retrainMsg, setRetrainMsg] = useState<string | null>(null);

  const fetchCases = () => {
    fetch('/api/cbr/cases')
      .then((res) => res.json())
      .then((data) => {
        setCbrData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load CBR cases:', err);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchCases();
  }, []);

  const handleRetrain = () => {
    setRetraining(true);
    setRetrainMsg(null);
    fetch('/api/cbr/retrain', { method: 'POST' })
      .then((res) => res.json())
      .then((data) => {
        setRetraining(false);
        setRetrainMsg(data.message || 'Candidate model fine-tuned successfully on verified CBR cases.');
        fetchCases();
      })
      .catch(() => {
        setRetraining(false);
        setRetrainMsg('Retraining failed or server unreachable.');
      });
  };

  return (
    <div className="card" style={{ background: '#0f172a', color: '#f8fafc', border: '1px solid #1e293b', borderRadius: '16px', padding: '1.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'rgba(56, 189, 248, 0.2)', color: '#38bdf8', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Database size={18} />
            </div>
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 800, color: '#f8fafc' }}>
              Case-Based Reasoning (CBR) Knowledge Base & Verification
            </h3>
          </div>
          <p style={{ margin: '0.3rem 0 0 0', fontSize: '0.78rem', color: '#94a3b8' }}>
            Diagnostic retrieval system computing cosine similarity over 82-D fused radiological representations. Retrains candidate models on doctor-verified edge cases.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button
            onClick={handleRetrain}
            disabled={retraining}
            style={{
              background: retraining ? '#334155' : 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
              color: '#ffffff',
              border: 'none',
              padding: '0.5rem 1rem',
              borderRadius: '10px',
              fontWeight: 700,
              fontSize: '0.82rem',
              cursor: retraining ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.45rem',
              boxShadow: '0 4px 12px rgba(2, 132, 199, 0.3)'
            }}
          >
            <RefreshCw size={14} className={retraining ? 'spin-icon' : ''} />
            {retraining ? 'Retraining Candidate...' : 'Trigger CBR Retraining'}
          </button>
          <span style={{ fontSize: '0.75rem', background: 'rgba(16, 185, 129, 0.2)', color: '#34d399', border: '1px solid #059669', padding: '0.35rem 0.85rem', borderRadius: '20px', fontWeight: 700 }}>
            {cbrData ? `${cbrData.total} Verified Cases` : 'CBR Active'}
          </span>
        </div>
      </div>

      {retrainMsg && (
        <div style={{ background: 'rgba(16, 185, 129, 0.15)', border: '1px solid #059669', borderRadius: '10px', padding: '0.75rem 1rem', marginBottom: '1.25rem', fontSize: '0.84rem', color: '#6ee7b7', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <CheckCircle2 size={16} /> {retrainMsg}
        </div>
      )}

      {/* Metric Stat Boxes */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.85rem', marginBottom: '1.25rem' }}>
        <div style={{ background: '#1e293b', padding: '0.9rem', borderRadius: '10px', border: '1px solid #334155' }}>
          <div style={{ fontSize: '0.72rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 700 }}>Total Verified Cases</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#38bdf8', marginTop: '0.2rem' }}>{cbrData?.total ?? 0}</div>
          <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Clinical reference case-base</div>
        </div>

        <div style={{ background: '#1e293b', padding: '0.9rem', borderRadius: '10px', border: '1px solid #334155' }}>
          <div style={{ fontSize: '0.72rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 700 }}>Normal Cases</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#4ade80', marginTop: '0.2rem' }}>{cbrData?.counts?.Normal ?? 0}</div>
          <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Doctor-confirmed normal scans</div>
        </div>

        <div style={{ background: '#1e293b', padding: '0.9rem', borderRadius: '10px', border: '1px solid #334155' }}>
          <div style={{ fontSize: '0.72rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 700 }}>Benign Cases</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#fbbf24', marginTop: '0.2rem' }}>{cbrData?.counts?.Benign ?? 0}</div>
          <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Confirmed non-malignant nodules</div>
        </div>

        <div style={{ background: '#1e293b', padding: '0.9rem', borderRadius: '10px', border: '1px solid #334155' }}>
          <div style={{ fontSize: '0.72rem', color: '#94a3b8', textTransform: 'uppercase', fontWeight: 700 }}>Malignant Cases</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#f87171', marginTop: '0.2rem' }}>{cbrData?.counts?.Malignant ?? 0}</div>
          <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Confirmed malignant neoplasms</div>
        </div>
      </div>

      {/* Recent CBR Cases Table */}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left', color: '#cbd5e1' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8', fontWeight: 700, fontSize: '0.75rem' }}>
              <th style={{ padding: '0.65rem 0.85rem' }}>Case ID</th>
              <th style={{ padding: '0.65rem 0.85rem' }}>Scan File</th>
              <th style={{ padding: '0.65rem 0.85rem' }}>AI Prediction</th>
              <th style={{ padding: '0.65rem 0.85rem' }}>Doctor Verification</th>
              <th style={{ padding: '0.65rem 0.85rem' }}>Status</th>
              <th style={{ padding: '0.65rem 0.85rem' }}>Clinical Notes</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={6} style={{ padding: '1.5rem', textAlign: 'center', color: '#64748b' }}>Loading CBR knowledge base records...</td>
              </tr>
            ) : (!cbrData?.cases || cbrData.cases.length === 0) ? (
              <tr>
                <td colSpan={6} style={{ padding: '1.5rem', textAlign: 'center', color: '#94a3b8' }}>
                  No verified test cases recorded yet. New test cases and doctor verifications will appear here.
                </td>
              </tr>
            ) : cbrData?.cases?.slice(0, 10).map((c, idx) => {
              const isCorrect = c.prediction_correct;
              const badgeColor = c.doctor_verified_class === 'Normal' ? '#4ade80' : c.doctor_verified_class === 'Benign' ? '#fbbf24' : '#f87171';
              return (
                <tr key={idx} style={{ borderBottom: '1px solid #1e293b' }}>
                  <td style={{ padding: '0.65rem 0.85rem', fontWeight: 700, color: '#38bdf8' }}>{c.case_id}</td>
                  <td style={{ padding: '0.65rem 0.85rem', color: '#94a3b8' }}>{c.image_filename}</td>
                  <td style={{ padding: '0.65rem 0.85rem' }}>{c.ai_prediction} ({c.ai_confidence}%)</td>
                  <td style={{ padding: '0.65rem 0.85rem', fontWeight: 800, color: badgeColor }}>
                    {c.doctor_verified_class}
                  </td>
                  <td style={{ padding: '0.65rem 0.85rem' }}>
                    <span style={{ fontSize: '0.7rem', padding: '0.15rem 0.5rem', borderRadius: '8px', background: isCorrect ? 'rgba(74, 222, 128, 0.15)' : 'rgba(248, 113, 113, 0.15)', color: isCorrect ? '#4ade80' : '#f87171', fontWeight: 700 }}>
                      {isCorrect ? 'Verified Agreement' : 'Doctor Correction'}
                    </span>
                  </td>
                  <td style={{ padding: '0.65rem 0.85rem', color: '#cbd5e1', fontSize: '0.76rem', maxWidth: '300px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {c.notes || 'Radiological scan verified.'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
