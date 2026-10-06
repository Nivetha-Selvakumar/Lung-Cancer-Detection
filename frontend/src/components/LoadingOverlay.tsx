import React from 'react';
import { Activity } from 'lucide-react';

interface LoadingOverlayProps {
  active: boolean;
  message?: string;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ active, message }) => {
  if (!active) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 99999,
        background: 'rgba(15, 23, 42, 0.65)',
        backdropFilter: 'blur(6px)',
        WebkitBackdropFilter: 'blur(6px)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center'
      }}
    >
      <div
        style={{
          background: '#ffffff',
          borderRadius: '20px',
          padding: '2.5rem 3rem',
          maxWidth: '440px',
          width: '90%',
          textAlign: 'center',
          boxShadow: '0 20px 50px rgba(0, 0, 0, 0.25), 0 0 40px rgba(79, 70, 229, 0.2)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1.25rem',
          border: '1px solid rgba(226, 232, 240, 0.8)'
        }}
      >
        {/* Animated Glowing Ring Spinner */}
        <div style={{ position: 'relative', width: '72px', height: '72px' }}>
          <div
            style={{
              position: 'absolute',
              inset: 0,
              borderRadius: '50%',
              border: '4px solid #e0e7ff',
              borderTopColor: '#4f46e5',
              borderRightColor: '#6366f1',
              animation: 'spinOverlay 1s linear infinite'
            }}
          />
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#4f46e5'
            }}
          >
            <Activity size={30} style={{ animation: 'pulseIcon 1.5s ease-in-out infinite' }} />
          </div>
        </div>

        {/* Loading Text & Status Description */}
        <div>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#0f172a', margin: '0 0 0.35rem 0' }}>
            {message || 'Analyzing CT Scan Image...'}
          </h3>
          <p style={{ fontSize: '0.85rem', color: '#64748b', margin: 0, lineHeight: 1.5 }}>
            Performing lung parenchyma segmentation & AI model inference
          </p>
        </div>

        {/* Pulsing Progress Bar */}
        <div style={{ width: '100%', height: '6px', background: '#f1f5f9', borderRadius: '3px', overflow: 'hidden' }}>
          <div
            style={{
              height: '100%',
              width: '60%',
              background: 'linear-gradient(90deg, #4f46e5, #06b6d4, #4f46e5)',
              borderRadius: '3px',
              animation: 'loadingProgress 1.8s ease-in-out infinite'
            }}
          />
        </div>
      </div>

      <style>{`
        @keyframes spinOverlay {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        @keyframes pulseIcon {
          0%, 100% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.12); opacity: 0.8; }
        }
        @keyframes loadingProgress {
          0% { transform: translateX(-100%); }
          50% { transform: translateX(50%); }
          100% { transform: translateX(200%); }
        }
      `}</style>
    </div>
  );
};
