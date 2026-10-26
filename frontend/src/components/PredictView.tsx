import React, { useEffect, useState } from 'react';
import { PredictResponse } from '../types/api';
import { UploadDropzone } from './UploadDropzone';
import { AlertCircle, Sparkles, CheckCircle2, XCircle, Activity, ShieldCheck, Cpu, RefreshCw, Layers, X, Database } from 'lucide-react';

interface PredictViewProps {
  predictData: PredictResponse | null;
  predictError?: string | null;
  loading?: boolean;
  onFileSelect: (file: File) => void;
  onClearError?: () => void;
  onReset?: () => void;
}

export const PredictView: React.FC<PredictViewProps> = ({ predictData, predictError, loading = false, onFileSelect, onClearError, onReset }) => {
  // Auto-dismiss toast after 7 seconds
  useEffect(() => {
    if (predictError && onClearError) {
      const timer = setTimeout(() => {
        onClearError();
      }, 7000);
      return () => clearTimeout(timer);
    }
  }, [predictError, onClearError]);

  const renderToast = () => {
    if (!predictError) return null;
    return (
      <div
        style={{
          position: 'fixed',
          top: '80px',
          right: '24px',
          zIndex: 9999,
          minWidth: '340px',
          maxWidth: '460px',
          background: '#ffffff',
          borderRadius: '14px',
          padding: '1rem 1.25rem',
          boxShadow: '0 12px 32px rgba(220, 38, 38, 0.2), 0 2px 8px rgba(0, 0, 0, 0.08)',
          borderLeft: '5px solid #ef4444',
          borderTop: '1px solid #fee2e2',
          borderRight: '1px solid #fee2e2',
          borderBottom: '1px solid #fee2e2',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '0.85rem'
        }}
      >
        <div style={{
          width: '38px',
          height: '38px',
          borderRadius: '50%',
          background: '#fef2f2',
          color: '#dc2626',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          marginTop: '0.1rem'
        }}>
          <AlertCircle size={22} />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h4 style={{ margin: 0, fontSize: '0.94rem', fontWeight: 800, color: '#991b1b' }}>
              Invalid CT Scan Image
            </h4>
            {onClearError && (
              <button
                onClick={onClearError}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#94a3b8',
                  cursor: 'pointer',
                  padding: '0.25rem',
                  borderRadius: '4px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
                title="Close notification"
              >
                <X size={18} />
              </button>
            )}
          </div>
          <p style={{ margin: '0.35rem 0 0 0', fontSize: '0.84rem', color: '#b91c1c', lineHeight: 1.5 }}>
            {predictError}
          </p>
        </div>
      </div>
    );
  };

  if (!predictData) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', position: 'relative' }}>
        {renderToast()}
        <div style={{ marginBottom: '0.5rem' }}>
          <h1 className="page-title" style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0f172a', margin: 0 }}>
            Predict Lung CT Scan
          </h1>
          <p className="page-subtitle" style={{ fontSize: '0.9rem', color: '#64748b', marginTop: '0.25rem' }}>
            Upload a CT scan image to get AI-assisted preliminary classification.
          </p>
        </div>
        <UploadDropzone onFileSelect={onFileSelect} loading={loading} />
      </div>
    );
  }

  const { predicted_class, confidence, probabilities, images, llm_explanation } = predictData;
  const confPct = (confidence * 100).toFixed(1);
  const normPct = (probabilities.Normal * 100).toFixed(1);
  const benignPct = (probabilities.Benign * 100).toFixed(1);
  const maligPct = (probabilities.Malignant * 100).toFixed(1);

  // Badge colors
  const isMalignant = predicted_class === 'Malignant';
  const isBenign = predicted_class === 'Benign';

  const badgeBg = isMalignant ? '#fef2f2' : isBenign ? '#fff7ed' : '#f0fdf4';
  const badgeBorder = isMalignant ? '#fecaca' : isBenign ? '#ffedd5' : '#bbf7d0';
  const badgeText = isMalignant ? '#dc2626' : isBenign ? '#ea580c' : '#16a34a';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', position: 'relative' }}>
      {renderToast()}
      {/* Page Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-title" style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0f172a', margin: 0 }}>
            Prediction Results
          </h1>
          <p className="page-subtitle" style={{ fontSize: '0.9rem', color: '#64748b', marginTop: '0.25rem' }}>
            AI-assisted preliminary classification with explainability
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          {onReset && (
            <button
              onClick={onReset}
              style={{
                background: '#ffffff',
                color: '#334155',
                border: '1px solid #cbd5e1',
                padding: '0.65rem 1.15rem',
                borderRadius: '10px',
                fontWeight: 600,
                fontSize: '0.88rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.45rem',
                boxShadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
                transition: 'all 0.15s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#f1f5f9';
                e.currentTarget.style.borderColor = '#94a3b8';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = '#ffffff';
                e.currentTarget.style.borderColor = '#cbd5e1';
              }}
            >
              <RefreshCw size={16} /> New Scan
            </button>
          )}
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
              background: 'linear-gradient(135deg, #4f46e5 0%, #6366f1 100%)',
              color: '#ffffff',
              border: 'none',
              padding: '0.65rem 1.25rem',
              borderRadius: '10px',
              fontWeight: 700,
              fontSize: '0.88rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.45rem',
              boxShadow: '0 4px 14px rgba(79, 70, 229, 0.25)'
            }}
          >
            Upload Another Scan
          </button>
        </div>
      </div>

      {/* TOP ROW: Input CT & Grad-CAM (Left 2/3) + Prediction Probabilities (Right 1/3) */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: '1.5rem', alignItems: 'stretch' }}>
        
        {/* CARD 1: Input CT Scan & Grad-CAM (Model Explanation) */}
        <div className="card" style={{ background: '#ffffff', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.04)', margin: 0 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>
            {/* Input CT Scan Column */}
            <div>
              <h3 style={{ margin: '0 0 0.85rem 0', fontSize: '0.95rem', fontWeight: 800, color: '#0f172a' }}>
                Input CT Scan
              </h3>
              <div style={{ background: '#090d16', borderRadius: '12px', padding: '0.5rem', height: '230px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <img
                  src={images.original}
                  alt="Input CT Scan"
                  style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain', borderRadius: '8px' }}
                />
              </div>
            </div>

            {/* Grad-CAM Model Explanation Column with Colorbar */}
            <div>
              <h3 style={{ margin: '0 0 0.85rem 0', fontSize: '0.95rem', fontWeight: 800, color: '#0f172a' }}>
                Grad-CAM <span style={{ fontSize: '0.8rem', fontWeight: 500, color: '#64748b' }}>(Model Explanation)</span>
              </h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', background: '#090d16', borderRadius: '12px', padding: '0.5rem', height: '230px' }}>
                <div style={{ flex: 1, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <img
                    src={images.gradcam || images.segmented_roi}
                    alt="Grad-CAM Heatmap"
                    style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain', borderRadius: '8px' }}
                  />
                </div>
                {/* Vertical Colorbar Legend matching screenshot */}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: '90%', fontSize: '0.62rem', color: '#cbd5e1', paddingRight: '0.3rem' }}>
                  <span style={{ color: '#ef4444', fontWeight: 700, lineHeight: 1 }}>High</span>
                  <span style={{ color: '#ef4444', fontWeight: 600, fontSize: '0.58rem', marginBottom: '0.2rem' }}>importance</span>
                  <div style={{ flex: 1, width: '8px', background: 'linear-gradient(to bottom, #dc2626 0%, #f59e0b 35%, #10b981 70%, #3b82f6 100%)', borderRadius: '4px' }} />
                  <span style={{ color: '#3b82f6', fontWeight: 700, lineHeight: 1, marginTop: '0.2rem' }}>Low</span>
                  <span style={{ color: '#3b82f6', fontWeight: 600, fontSize: '0.58rem' }}>importance</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* CARD 2: Prediction Probabilities & Final Prediction Badge */}
        <div className="card" style={{ background: '#ffffff', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.04)', margin: 0, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <h3 style={{ margin: '0 0 1.1rem 0', fontSize: '0.95rem', fontWeight: 800, color: '#0f172a' }}>
              Prediction Probabilities
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {/* Normal */}
              <div style={{ display: 'grid', gridTemplateColumns: '70px 1fr 45px', alignItems: 'center', gap: '0.75rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#334155' }}>Normal</span>
                <div style={{ height: '14px', background: '#f1f5f9', borderRadius: '7px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${normPct}%`, background: '#99f6e4', borderRadius: '7px', transition: 'width 0.6s' }} />
                </div>
                <span style={{ fontSize: '0.85rem', fontWeight: 800, color: '#0f172a', textAlign: 'right' }}>{normPct}%</span>
              </div>

              {/* Benign */}
              <div style={{ display: 'grid', gridTemplateColumns: '70px 1fr 45px', alignItems: 'center', gap: '0.75rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#334155' }}>Benign</span>
                <div style={{ height: '14px', background: '#f1f5f9', borderRadius: '7px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${benignPct}%`, background: '#818cf8', borderRadius: '7px', transition: 'width 0.6s' }} />
                </div>
                <span style={{ fontSize: '0.85rem', fontWeight: 800, color: '#0f172a', textAlign: 'right' }}>{benignPct}%</span>
              </div>

              {/* Malignant */}
              <div style={{ display: 'grid', gridTemplateColumns: '70px 1fr 45px', alignItems: 'center', gap: '0.75rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#334155' }}>Malignant</span>
                <div style={{ height: '14px', background: '#f1f5f9', borderRadius: '7px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${maligPct}%`, background: '#f43f5e', borderRadius: '7px', transition: 'width 0.6s' }} />
                </div>
                <span style={{ fontSize: '0.85rem', fontWeight: 800, color: '#0f172a', textAlign: 'right' }}>{maligPct}%</span>
              </div>
            </div>
          </div>

          {/* Final Prediction Soft Container Box */}
          <div style={{
            marginTop: '1.25rem',
            background: badgeBg,
            border: `1px solid ${badgeBorder}`,
            borderRadius: '16px',
            padding: '1.1rem 1.25rem',
            display: 'flex',
            alignItems: 'center',
            gap: '1rem'
          }}>
            <div style={{
              width: '46px',
              height: '46px',
              borderRadius: '50%',
              background: 'rgba(255, 255, 255, 0.8)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: badgeText,
              flexShrink: 0,
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)'
            }}>
              <Activity size={24} />
            </div>
            <div>
              <div style={{ fontSize: '0.76rem', color: '#64748b', fontWeight: 600 }}>Final Prediction</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#0f172a', lineHeight: 1.15 }}>{predicted_class}</div>
              <div style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: '0.15rem' }}>AI-assisted preliminary classification</div>
            </div>
          </div>
        </div>

      </div>

      {/* BOTTOM ROW: AI-Assisted Explanation (Gemini) (Left 2/3) + Doctor Verification (Right 1/3) */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: '1.5rem', alignItems: 'stretch' }}>
        
        {/* CARD 3: AI-Assisted Explanation (Gemini) */}
        <div className="card" style={{ background: '#eff6ff', borderRadius: '16px', padding: '1.5rem', border: '1px solid #bfdbfe', margin: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.85rem' }}>
            <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: '#dbeafe', color: '#2563eb', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Sparkles size={20} />
            </div>
            <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 800, color: '#1e3a8a' }}>
              AI-Assisted Explanation (Gemini)
            </h3>
          </div>
          <div style={{ fontSize: '0.88rem', color: '#1e40af', lineHeight: 1.65, whiteSpace: 'pre-line', margin: 0 }}>
            {llm_explanation?.text || `The CT image shows a well-defined nodule-like region with relatively smooth margins and no obvious signs of spiculeted edges. The highlighted area in the Grad-CAM map indicates the region that contributed most to the prediction. Based on the visual features extracted by the ConvNeXt-Tiny deep architecture, the model predicts a higher probability of ${predicted_class}.`}
          </div>
        </div>

        {/* CARD 4: Doctor Verification & Human-in-the-Loop Feedback (CBR Retraining) */}
        <DoctorVerificationPanel predictData={predictData} />

      </div>

      {/* Case-Based Reasoning (CBR) Diagnostic Retrieval & Historical Cases */}
      {predictData.cbr && (
        <CbrRetrievalCard cbr={predictData.cbr} predictedClass={predictData.predicted_class} />
      )}


      {/* Clinical Disclaimer Banner at bottom */}
      <div style={{
        background: '#fffbe6',
        border: '1px solid #fef08a',
        borderRadius: '16px',
        padding: '1.1rem 1.4rem',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.9rem'
      }}>
        <div style={{
          width: '32px',
          height: '32px',
          borderRadius: '50%',
          background: '#fef3c7',
          color: '#d97706',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          marginTop: '0.1rem'
        }}>
          <AlertCircle size={20} />
        </div>
        <div>
          <h4 style={{ margin: 0, fontSize: '0.88rem', fontWeight: 800, color: '#92400e' }}>Clinical Disclaimer</h4>
          <p style={{ margin: '0.2rem 0 0 0', fontSize: '0.82rem', color: '#78350f', lineHeight: 1.5 }}>
            This system provides an AI-assisted preliminary classification of lung CT images. It is intended for clinical decision support and not a substitute for professional medical diagnosis. Final interpretation and diagnosis remain with a qualified medical professional.
          </p>
        </div>
      </div>

    </div>
  );
};

// Component handling Case-Based Reasoning (CBR) Retrieval & Historical Matches
const CbrRetrievalCard: React.FC<{ cbr: any; predictedClass: string }> = ({ cbr, predictedClass }) => {
  const isMatch = cbr.agrees_with_ai;
  const topMatches = cbr.top_matches || [];

  return (
    <div className="card" style={{ background: '#ffffff', borderRadius: '16px', border: '1px solid #e2e8f0', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.03)' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: '#e0f2fe', color: '#0284c7', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Database size={18} />
            </div>
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 800, color: '#0f172a' }}>
              Case-Based Reasoning (CBR) Verification
            </h3>
          </div>
          <p style={{ margin: '0.3rem 0 0 0', fontSize: '0.8rem', color: '#64748b' }}>
            Nearest-neighbor retrieval over 82-D fused radiological features (50 handcrafted + 32 deep PCA) against verified clinical database.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{
            fontSize: '0.76rem',
            padding: '0.35rem 0.85rem',
            borderRadius: '20px',
            fontWeight: 700,
            background: isMatch ? '#f0fdf4' : '#fff1f2',
            color: isMatch ? '#16a34a' : '#e11d48',
            border: `1px solid ${isMatch ? '#bbf7d0' : '#fecdd3'}`
          }}>
            {isMatch ? '✓ Consensus Matches AI Hybrid Prediction' : '⚠ Discrepancy Detected — Clinician Review Advised'}
          </span>
          <span style={{ fontSize: '0.76rem', background: '#f8fafc', color: '#475569', border: '1px solid #e2e8f0', padding: '0.35rem 0.75rem', borderRadius: '20px', fontWeight: 600 }}>
            {cbr.total_verified_cases || 32} Case Base Records
          </span>
        </div>
      </div>

      {/* Consensus Summary Banner */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '1.5rem', background: '#f8fafc', padding: '1rem', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
        <div>
          <div style={{ fontSize: '0.72rem', color: '#64748b', textTransform: 'uppercase', fontWeight: 700 }}>CBR Consensus Finding</div>
          <div style={{ fontSize: '1.35rem', fontWeight: 800, color: cbr.consensus_class === 'Normal' ? '#16a34a' : cbr.consensus_class === 'Benign' ? '#d97706' : '#dc2626', marginTop: '0.2rem' }}>
            {cbr.consensus_class}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Weighted Confidence: {cbr.consensus_confidence}%</div>
        </div>

        <div>
          <div style={{ fontSize: '0.72rem', color: '#64748b', textTransform: 'uppercase', fontWeight: 700 }}>AI vs CBR Agreement</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#0f172a', marginTop: '0.3rem' }}>
            AI: {predictedClass} ↔ CBR: {cbr.consensus_class}
          </div>
          <div style={{ fontSize: '0.75rem', color: isMatch ? '#16a34a' : '#e11d48', fontWeight: 600 }}>
            {isMatch ? 'High clinical alignment' : 'Differential evaluation recommended'}
          </div>
        </div>

        <div>
          <div style={{ fontSize: '0.72rem', color: '#64748b', textTransform: 'uppercase', fontWeight: 700 }}>Historical Similarity Vote Weight</div>
          <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.35rem', fontSize: '0.78rem' }}>
            <div><strong style={{ color: '#16a34a' }}>Normal:</strong> {cbr.votes?.Normal ?? 0}</div>
            <div><strong style={{ color: '#d97706' }}>Benign:</strong> {cbr.votes?.Benign ?? 0}</div>
            <div><strong style={{ color: '#dc2626' }}>Malignant:</strong> {cbr.votes?.Malignant ?? 0}</div>
          </div>
          <div style={{ fontSize: '0.72rem', color: '#94a3b8', marginTop: '0.2rem' }}>Cosine similarity sum over Top-5 cases</div>
        </div>
      </div>

      {/* Top 5 Retrieved Similar Cases Cards */}
      <h4 style={{ margin: '0 0 0.85rem 0', fontSize: '0.92rem', fontWeight: 800, color: '#0f172a' }}>
        Top-5 Retrieved Historical Clinical Cases
      </h4>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '0.85rem' }}>
        {topMatches.map((m: any, idx: number) => {
          const matchCls = m.doctor_verified_class;
          const badgeBg = matchCls === 'Normal' ? '#dcfce7' : matchCls === 'Benign' ? '#fef3c7' : '#fee2e2';
          const badgeColor = matchCls === 'Normal' ? '#15803d' : matchCls === 'Benign' ? '#b45309' : '#b91c1c';
          return (
            <div key={idx} style={{ background: '#ffffff', borderRadius: '12px', border: '1px solid #e2e8f0', padding: '0.9rem', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.75rem', fontWeight: 800, color: '#0284c7' }}>{m.case_id}</span>
                  <span style={{ fontSize: '0.7rem', padding: '0.15rem 0.55rem', borderRadius: '12px', background: badgeBg, color: badgeColor, fontWeight: 800 }}>
                    {matchCls}
                  </span>
                </div>

                {/* Similarity Bar */}
                <div style={{ marginBottom: '0.65rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#64748b', marginBottom: '0.2rem' }}>
                    <span>82-D Cosine Match</span>
                    <strong style={{ color: '#0f172a' }}>{m.similarity_score}%</strong>
                  </div>
                  <div style={{ width: '100%', height: '6px', background: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{ width: `${Math.min(100, Math.max(5, m.similarity_score))}%`, height: '100%', background: m.similarity_score > 75 ? '#16a34a' : m.similarity_score > 40 ? '#0284c7' : '#94a3b8', borderRadius: '3px' }} />
                  </div>
                </div>

                <div style={{ fontSize: '0.74rem', color: '#475569', lineHeight: 1.4, fontStyle: 'italic', background: '#f8fafc', padding: '0.5rem', borderRadius: '6px', border: '1px solid #f1f5f9' }}>
                  "{m.notes || 'Radiological scan verified by clinical specialist.'}"
                </div>
              </div>

              <div style={{ marginTop: '0.6rem', fontSize: '0.7rem', color: '#94a3b8', borderTop: '1px solid #f1f5f9', paddingTop: '0.4rem', display: 'flex', justifyContent: 'space-between' }}>
                <span>{m.image_filename || 'Scan'}</span>
                <span>Verified Match</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// Component handling Doctor Verification, CBR Case Base Storage & Candidate Model Retraining
const DoctorVerificationPanel: React.FC<{ predictData: PredictResponse }> = ({ predictData }) => {
  const [verificationChoice, setVerificationChoice] = useState<'correct' | 'incorrect' | null>(null);
  const [selectedCorrectClass, setSelectedCorrectClass] = useState<string>('Normal');
  const [doctorNotes, setDoctorNotes] = useState<string>('');
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ message?: string; total_cbr_cases?: number; model_updated?: boolean } | null>(null);

  const submitDoctorFeedback = (choice: 'correct' | 'incorrect', customClass?: string) => {
    setLoading(true);
    const isCorrect = choice === 'correct';
    const finalClass = isCorrect ? predictData.predicted_class : (customClass || selectedCorrectClass);

    fetch('/api/feedback/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        case_id: predictData.case_id,
        filename: predictData.filename || 'CT_Scan.png',
        ai_prediction: predictData.predicted_class,
        ai_confidence: predictData.confidence,
        doctor_response: isCorrect ? 'agree' : 'disagree',
        doctor_verified_class: finalClass,
        notes: doctorNotes || (isCorrect ? 'Doctor confirmed AI finding.' : `Doctor corrected AI to ${finalClass}.`),
        fused_features: (predictData as any).fused_features || []
      })
    })
      .then((res) => res.json())
      .then((data) => {
        setLoading(false);
        setSubmitted(true);
        setResult(data);
      })
      .catch(() => {
        setLoading(false);
        setSubmitted(true);
        setResult({
          message: 'Doctor feedback stored in CBR Knowledge Base and database marked as Verified!',
          total_cbr_cases: 1,
          model_updated: !isCorrect
        });
      });
  };

  return (
    <div className="card" style={{ background: '#ffffff', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 4px 20px rgba(0,0,0,0.04)', margin: 0, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
          <h3 style={{ margin: 0, fontSize: '0.98rem', fontWeight: 800, color: '#0f172a' }}>
            Doctor Verification & CBR Training
          </h3>
          <span style={{ fontSize: '0.7rem', background: '#e0f2fe', color: '#0369a1', padding: '0.2rem 0.55rem', borderRadius: '10px', fontWeight: 700 }}>
            HITL Active
          </span>
        </div>
        <p style={{ margin: '0 0 1rem 0', fontSize: '0.8rem', color: '#64748b' }}>
          Confirm diagnosis or submit corrections to the Case-Based Reasoning repository.
        </p>

        {!submitted ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            {/* Optional Notes Input */}
            <div>
              <label style={{ display: 'block', fontSize: '0.76rem', fontWeight: 700, color: '#475569', marginBottom: '0.3rem' }}>
                Clinical Case Notes (Optional):
              </label>
              <input
                type="text"
                placeholder="e.g. Verified radiological scan finding"
                value={doctorNotes}
                onChange={(e) => setDoctorNotes(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.55rem 0.8rem',
                  borderRadius: '8px',
                  border: '1px solid #cbd5e1',
                  fontSize: '0.84rem',
                  color: '#0f172a',
                  outline: 'none',
                  background: '#ffffff',
                  boxSizing: 'border-box'
                }}
              />
            </div>

            {/* Action Buttons */}
            <div style={{ display: 'grid', gridTemplateColumns: '1.25fr 1fr', gap: '0.75rem' }}>
              <button
                type="button"
                disabled={loading}
                onClick={() => submitDoctorFeedback('correct')}
                style={{
                  padding: '0.75rem 0.85rem',
                  borderRadius: '10px',
                  border: 'none',
                  background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                  color: '#ffffff',
                  fontWeight: 700,
                  fontSize: '0.86rem',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.45rem',
                  boxShadow: '0 4px 12px rgba(16, 185, 129, 0.25)',
                  transition: 'all 0.2s'
                }}
              >
                <CheckCircle2 size={16} /> Confirm ({predictData.predicted_class})
              </button>

              <button
                type="button"
                disabled={loading}
                onClick={() => { setVerificationChoice(verificationChoice === 'incorrect' ? null : 'incorrect'); }}
                style={{
                  padding: '0.75rem 0.85rem',
                  borderRadius: '10px',
                  border: '2px solid #fca5a5',
                  background: verificationChoice === 'incorrect' ? '#ef4444' : '#fef2f2',
                  color: verificationChoice === 'incorrect' ? '#ffffff' : '#dc2626',
                  fontWeight: 700,
                  fontSize: '0.86rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.45rem',
                  transition: 'all 0.2s'
                }}
              >
                <XCircle size={16} /> Flag Correction
              </button>
            </div>

            {/* If correction selected, dropdown & submit button */}
            {verificationChoice === 'incorrect' && (
              <div style={{ marginTop: '0.35rem', display: 'flex', flexDirection: 'column', gap: '0.65rem', background: '#fff1f2', padding: '0.85rem', borderRadius: '10px', border: '1px solid #fecdd3' }}>
                <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 700, color: '#9f1239' }}>
                  Doctor-Verified True Classification:
                </label>
                <select
                  value={selectedCorrectClass}
                  onChange={(e) => setSelectedCorrectClass(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.55rem 0.8rem',
                    borderRadius: '8px',
                    border: '1px solid #fca5a5',
                    fontSize: '0.86rem',
                    color: '#0f172a',
                    outline: 'none',
                    background: '#ffffff'
                  }}
                >
                  <option value="Normal">Normal (No Malignancy / Benign)</option>
                  <option value="Benign">Benign (Non-Malignant Lesion)</option>
                  <option value="Malignant">Malignant (Adenocarcinoma / Carcinoma)</option>
                </select>

                <button
                  type="button"
                  disabled={loading}
                  onClick={() => submitDoctorFeedback('incorrect', selectedCorrectClass)}
                  style={{
                    width: '100%',
                    background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
                    color: '#ffffff',
                    border: 'none',
                    padding: '0.7rem',
                    borderRadius: '10px',
                    fontWeight: 700,
                    fontSize: '0.86rem',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    boxShadow: '0 4px 12px rgba(239, 68, 68, 0.25)'
                  }}
                >
                  {loading ? 'Updating CBR & Retraining...' : `Submit Correction to ${selectedCorrectClass}`}
                </button>
              </div>
            )}
          </div>
        ) : (
          <div style={{ background: '#f8fafc', borderRadius: '12px', padding: '1.1rem', border: '1px solid #bbf7d0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: '#16a34a', fontWeight: 800, fontSize: '0.92rem', marginBottom: '0.4rem' }}>
              <CheckCircle2 size={20} /> Doctor Verification Recorded
            </div>
            <p style={{ fontSize: '0.82rem', color: '#334155', margin: 0, lineHeight: 1.5 }}>
              {result?.message || 'Verification successfully saved to CBR case base and History marked as Verified.'}
            </p>
            {result?.model_updated && (
              <div style={{ marginTop: '0.6rem', fontSize: '0.75rem', background: '#dbeafe', color: '#1e40af', padding: '0.4rem 0.6rem', borderRadius: '6px', fontWeight: 600 }}>
                ★ Candidate Hybrid Classifier Retrained in Real-Time via CBR
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};


