import React, { useEffect, useState } from 'react';
import { UploadCloud, Scan, Eye, Database, Cpu, BarChart3, ChevronRight, Activity, CheckCircle, Clock } from 'lucide-react';
import { UserProfile } from '../types/api';

interface HomeViewProps {
  currentUser?: UserProfile | null;
  onNavigate: (tab: string) => void;
  onFileSelect: (file: File) => void;
}

export const HomeView: React.FC<HomeViewProps> = ({ currentUser, onNavigate }) => {
  // Load dynamic user data from prop or localStorage
  const user: UserProfile = currentUser || (() => {
    try {
      const saved = localStorage.getItem('lung_ai_user');
      return saved ? JSON.parse(saved) : { full_name: 'Nivetha S.', username: 'doctor', role: 'Research Scholar' };
    } catch (e) {
      return { full_name: 'Nivetha S.', username: 'doctor', role: 'Research Scholar' };
    }
  })();

  const displayName = user.full_name || user.username || 'Practitioner';
  const firstName = displayName.split(' ')[0];

  const [historyRecords, setHistoryRecords] = useState<any[]>([]);

  useEffect(() => {
    fetch('/api/history')
      .then(res => res.json())
      .then(data => {
        if (data.success && data.records) {
          setHistoryRecords(data.records.slice(0, 5));
        }
      })
      .catch(() => {
        // Fallback default sample records matching screen 2
        setHistoryRecords([
          { case_id: 'CASE-017', date_time: '2026-10-06 11:20', predicted_class: 'Malignant', confidence: 0.932, verification_status: 'Verified' },
          { case_id: 'CASE-016', date_time: '2026-10-05 14:12', predicted_class: 'Benign', confidence: 0.687, verification_status: 'Pending Review' },
          { case_id: 'CASE-015', date_time: '2026-10-04 10:05', predicted_class: 'Normal', confidence: 0.765, verification_status: 'Verified' },
        ]);
      });
  }, []);

  const quickActions = [
    { title: 'Upload CT Scan', subtitle: 'Analyze a new CT image', icon: UploadCloud, color: '#4f46e5', target: 'predict' },
    { title: 'Preprocess & Predict', subtitle: 'AI model (ConvNeXt-Tiny)', icon: Scan, color: '#0284c7', target: 'predict' },
    { title: 'View Results', subtitle: 'Prediction with Grad-CAM', icon: Eye, color: '#0d9488', target: 'predict' },
    { title: 'Manage Dataset', subtitle: 'Training and raw data', icon: Database, color: '#d97706', target: 'dataset' },
    { title: 'Performance Metrics', subtitle: 'Accuracy, Confusion Matrix', icon: BarChart3, color: '#16a34a', target: 'dashboard' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      
      {/* 1. Dynamic Welcome Banner */}
      <div style={{
        background: 'linear-gradient(135deg, #eff6ff 0%, #e0e7ff 100%)',
        border: '1px solid #c7d2fe',
        borderRadius: '16px',
        padding: '1.75rem 2rem',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        boxShadow: '0 4px 12px rgba(79, 70, 229, 0.05)'
      }}>
        <div style={{ maxWidth: '720px' }}>
          <h1 style={{ fontSize: '1.65rem', fontWeight: 800, color: '#1e1b4b', margin: '0 0 0.4rem 0', letterSpacing: '-0.02em' }}>
            Welcome, {firstName}!
          </h1>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#4338ca', margin: '0 0 0.4rem 0' }}>
            AI-Assisted Lung Cancer Prediction from CT Scans
          </h3>
          <p style={{ fontSize: '0.88rem', color: '#475569', margin: 0, lineHeight: 1.6 }}>
            Upload a CT scan to predict the probability of Normal, Benign or Malignant with explainable results using Grad-CAM.
          </p>
        </div>
        <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'rgba(99, 102, 241, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#4f46e5' }}>
          <Activity size={40} />
        </div>
      </div>

      {/* 2. Quick Action Grid Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
        {quickActions.map((act, idx) => {
          const Icon = act.icon;
          return (
            <div
              key={idx}
              onClick={() => onNavigate(act.target)}
              style={{
                padding: '1.1rem 1.25rem',
                borderRadius: '14px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background: '#ffffff',
                border: '1px solid #e2e8f0',
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.03)',
                transition: 'all 0.2s ease'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
                <div style={{
                  width: '42px',
                  height: '42px',
                  borderRadius: '10px',
                  background: `${act.color}15`,
                  color: act.color,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <Icon size={22} />
                </div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: '0.92rem', color: '#0f172a' }}>{act.title}</div>
                  <div style={{ fontSize: '0.76rem', color: '#64748b', marginTop: '0.15rem' }}>{act.subtitle}</div>
                </div>
              </div>
              <ChevronRight size={18} style={{ color: '#94a3b8' }} />
            </div>
          );
        })}
      </div>

      {/* 3. Bottom Grid: Recent Activity (Left) + Dataset Overview (Right) */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: '1.5rem' }}>
        
        {/* Recent Activity Card */}
        <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.03)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 700, color: '#0f172a' }}>Recent Activity</h3>
            <button
              onClick={() => onNavigate('history')}
              style={{ background: 'none', border: 'none', color: '#4f46e5', fontWeight: 600, fontSize: '0.82rem', cursor: 'pointer' }}
            >
              View All History →
            </button>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.84rem', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0', color: '#64748b', fontWeight: 600 }}>
                  <th style={{ padding: '0.6rem 0.5rem' }}>Case ID</th>
                  <th style={{ padding: '0.6rem 0.5rem' }}>Date/Time</th>
                  <th style={{ padding: '0.6rem 0.5rem' }}>Prediction</th>
                  <th style={{ padding: '0.6rem 0.5rem' }}>Confidence</th>
                  <th style={{ padding: '0.6rem 0.5rem' }}>Verified</th>
                  <th style={{ padding: '0.6rem 0.5rem' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {historyRecords.length === 0 ? (
                  <tr>
                    <td colSpan={6} style={{ padding: '1.5rem', textAlign: 'center', color: '#94a3b8' }}>
                      No recent CT scan inference activity recorded.
                    </td>
                  </tr>
                ) : (
                  historyRecords.map((rec, i) => {
                    const isMalignant = rec.predicted_class === 'Malignant';
                    const isBenign = rec.predicted_class === 'Benign';
                    const isVerified = rec.verification_status === 'Verified';

                    return (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '0.75rem 0.5rem', fontWeight: 700, color: '#0f172a' }}>{rec.case_id}</td>
                        <td style={{ padding: '0.75rem 0.5rem', color: '#64748b', fontSize: '0.78rem' }}>{rec.date_time}</td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>
                          <span style={{
                            padding: '0.25rem 0.6rem',
                            borderRadius: '12px',
                            fontWeight: 700,
                            fontSize: '0.75rem',
                            background: isMalignant ? '#fef2f2' : isBenign ? '#eff6ff' : '#f0fdf4',
                            color: isMalignant ? '#dc2626' : isBenign ? '#2563eb' : '#16a34a',
                            border: `1px solid ${isMalignant ? '#fecaca' : isBenign ? '#bfdbfe' : '#bbf7d0'}`
                          }}>
                            {rec.predicted_class}
                          </span>
                        </td>
                        <td style={{ padding: '0.75rem 0.5rem', fontWeight: 600, color: '#334155' }}>
                          {(rec.confidence * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '0.75rem 0.5rem', color: '#475569' }}>
                          {isVerified ? 'Yes' : 'Pending'}
                        </td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>
                          <span style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.3rem',
                            padding: '0.25rem 0.6rem',
                            borderRadius: '12px',
                            fontWeight: 600,
                            fontSize: '0.75rem',
                            background: isVerified ? '#f0fdf4' : '#fffbe6',
                            color: isVerified ? '#16a34a' : '#d97706'
                          }}>
                            {isVerified ? <CheckCircle size={12} /> : <Clock size={12} />}
                            {isVerified ? 'Verified' : 'Pending Review'}
                          </span>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Dataset Overview Card with Donut Visualization */}
        <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.03)', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 700, color: '#0f172a' }}>Dataset Overview</h3>
              <button
                onClick={() => onNavigate('dataset')}
                style={{ background: 'none', border: 'none', color: '#4f46e5', fontWeight: 600, fontSize: '0.82rem', cursor: 'pointer' }}
              >
                Manage Dataset →
              </button>
            </div>

            {/* Donut Chart Visual & Class Counts */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', marginTop: '0.5rem' }}>
              
              {/* Donut Chart SVG */}
              <div style={{ position: 'relative', width: '110px', height: '110px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <svg width="110" height="110" viewBox="0 0 36 36" style={{ transform: 'rotate(-90deg)' }}>
                  {/* Background Circle */}
                  <path
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke="#f1f5f9"
                    strokeWidth="3.8"
                  />
                  {/* Normal (Green: 120 / 372 = 32.2%) */}
                  <path
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="3.8"
                    strokeDasharray="32.2, 100"
                  />
                  {/* Benign (Blue: 134 / 372 = 36.0%) */}
                  <path
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke="#3b82f6"
                    strokeWidth="3.8"
                    strokeDasharray="36.0, 100"
                    strokeDashoffset="-32.2"
                  />
                  {/* Malignant (Red: 118 / 372 = 31.8%) */}
                  <path
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke="#ef4444"
                    strokeWidth="3.8"
                    strokeDasharray="31.8, 100"
                    strokeDashoffset="-68.2"
                  />
                </svg>
                <div style={{ position: 'absolute', textAlign: 'center' }}>
                  <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#0f172a', lineHeight: 1 }}>372</div>
                  <div style={{ fontSize: '0.65rem', color: '#64748b', fontWeight: 600 }}>CT Scans</div>
                </div>
              </div>

              {/* Class Legend Breakdown */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.82rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981' }} />
                    <span style={{ color: '#475569' }}>Normal</span>
                  </div>
                  <span style={{ fontWeight: 700, color: '#0f172a' }}>120</span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#3b82f6' }} />
                    <span style={{ color: '#475569' }}>Benign</span>
                  </div>
                  <span style={{ fontWeight: 700, color: '#0f172a' }}>134</span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#ef4444' }} />
                    <span style={{ color: '#475569' }}>Malignant</span>
                  </div>
                  <span style={{ fontWeight: 700, color: '#0f172a' }}>118</span>
                </div>
              </div>
            </div>
          </div>

          <button
            onClick={() => onNavigate('dataset')}
            style={{
              marginTop: '1.25rem',
              width: '100%',
              padding: '0.65rem',
              borderRadius: '8px',
              border: '1px solid #c7d2fe',
              background: '#e0e7ff',
              color: '#4338ca',
              fontWeight: 700,
              fontSize: '0.85rem',
              cursor: 'pointer'
            }}
          >
            Manage Dataset
          </button>
        </div>

      </div>
    </div>
  );
};
