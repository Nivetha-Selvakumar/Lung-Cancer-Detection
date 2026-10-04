import React from 'react';
import { UploadCloud, Scan, Eye, Database, Cpu, BarChart3, Info, ChevronRight } from 'lucide-react';
import { UploadDropzone } from './UploadDropzone';

interface HomeViewProps {
  onNavigate: (tab: string) => void;
  onFileSelect: (file: File) => void;
}

export const HomeView: React.FC<HomeViewProps> = ({ onNavigate, onFileSelect }) => {
  const quickActions = [
    { title: 'Upload CT Scan', subtitle: 'Analyze a new CT image', icon: UploadCloud, color: '#4f46e5', target: 'predict' },
    { title: 'Preprocess & Predict', subtitle: 'AI model (ConvNeXt-Tiny)', icon: Scan, color: '#0284c7', target: 'predict' },
    { title: 'View Results', subtitle: 'Prediction with Grad-CAM', icon: Eye, color: '#0d9488', target: 'predict' },
    { title: 'Manage Dataset', subtitle: 'Training and raw data', icon: Database, color: '#d97706', target: 'dataset' },
    { title: 'Train / Update Model', subtitle: 'Train using IQ Dataset', icon: Cpu, color: '#ea580c', target: 'train' },
    { title: 'Performance Metrics', subtitle: 'Accuracy, Confusion Matrix', icon: BarChart3, color: '#16a34a', target: 'dashboard' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Top Welcome Banner Card */}
      <div className="card" style={{
        background: 'linear-gradient(135deg, #eff6ff 0%, #e0e7ff 100%)',
        border: '1px solid #c7d2fe',
        padding: '1.75rem 2rem',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ maxWidth: '680px' }}>
          <h1 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#1e1b4b', marginBottom: '0.4rem', letterSpacing: '-0.02em' }}>
            Welcome, Nivetha!
          </h1>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#4338ca', marginBottom: '0.5rem' }}>
            AI-Assisted Lung Cancer Prediction from CT Scans
          </h3>
          <p style={{ fontSize: '0.88rem', color: '#475569', lineHeight: 1.6 }}>
            Upload a CT scan to predict the probability of Normal, Benign or Malignant with explainable results using Grad-CAM.
          </p>
        </div>
        <div style={{ display: 'none', width: '120px', height: '90px', background: '#c7d2fe', borderRadius: '12px', opacity: 0.8 }} className="desktop-only" />
      </div>

      {/* 6 Quick Action Grid Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
        {quickActions.map((act, idx) => {
          const Icon = act.icon;
          return (
            <div
              key={idx}
              onClick={() => onNavigate(act.target)}
              className="card"
              style={{
                padding: '1.1rem 1.25rem',
                margin: 0,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background: '#ffffff',
                border: '1px solid #e2e8f0',
                transition: 'all 0.2s ease-in-out'
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

      {/* Bottom Layout: Dropzone Left + About System Right */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '1.5rem' }}>
        <UploadDropzone onFileSelect={onFileSelect} />

        <div className="card" style={{ background: '#ffffff', border: '1px solid #e2e8f0', margin: 0, padding: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1rem', color: '#4f46e5', fontWeight: 700, fontSize: '1rem' }}>
            <Info size={20} />
            <span>About This System</span>
          </div>
          <p style={{ fontSize: '0.88rem', color: '#475569', lineHeight: 1.75 }}>
            This system uses a deep learning model (ConvNeXt-Tiny) trained on IQ-OTH/NCCD and raw lung CT datasets to predict the probability of Normal, Benign or Malignant cases. Grad-CAM is used to highlight important regions contributing to the prediction.
          </p>
        </div>
      </div>
    </div>
  );
};
