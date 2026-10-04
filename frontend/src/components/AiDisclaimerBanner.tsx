import React, { useState } from 'react';
import { AlertTriangle, Info, X } from 'lucide-react';

export const AiDisclaimerBanner: React.FC = () => {
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) return null;

  return (
    <div style={{
      position: 'fixed',
      bottom: '1rem',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 9000,
      width: 'calc(100% - 2rem)',
      maxWidth: '850px',
      background: 'rgba(15, 23, 42, 0.94)',
      backdropFilter: 'blur(12px)',
      border: '1px solid rgba(234, 179, 8, 0.4)',
      borderRadius: 'var(--radius-md)',
      padding: '0.75rem 1.25rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: '1rem',
      boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 0 15px rgba(234, 179, 8, 0.15)',
      fontSize: '0.85rem',
      color: '#e2e8f0'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div style={{
          background: 'rgba(234, 179, 8, 0.2)',
          color: '#facc15',
          padding: '0.4rem',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <AlertTriangle size={18} />
        </div>
        <div>
          <strong style={{ color: '#fef08a' }}>AI Clinical Assistant Disclaimer:</strong> AI diagnostic findings can make mistakes. Always verify key clinical information and CT images with a certified radiologist or pulmonologist.
        </div>
      </div>

      <button
        onClick={() => setDismissed(true)}
        style={{
          background: 'transparent',
          border: 'none',
          color: 'var(--text-muted)',
          cursor: 'pointer',
          padding: '0.3rem',
          borderRadius: '4px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
        title="Dismiss notice"
      >
        <X size={18} />
      </button>
    </div>
  );
};
