import React from 'react';
import { Bot, FileText, CheckCircle2 } from 'lucide-react';
import { LLMExplanation } from '../types/api';

interface ClinicalReportViewProps {
  explanation?: LLMExplanation;
}

export const ClinicalReportView: React.FC<ClinicalReportViewProps> = ({ explanation }) => {
  if (!explanation || !explanation.text) return null;

  return (
    <div className="card" style={{ marginTop: '1.5rem', border: '1px solid var(--border-glow)' }}>
      <div className="card-title" style={{ color: 'var(--primary-cyan)', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Bot size={22} /> AI Explanation (LLM Integration)
        </div>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <CheckCircle2 size={15} style={{ color: 'var(--status-normal)' }} /> Academic Model Decision Synthesis
        </div>
      </div>

      <pre
        style={{
          fontFamily: "'Courier New', Courier, monospace",
          fontSize: '0.88rem',
          lineHeight: '1.6',
          background: 'rgba(10, 15, 26, 0.95)',
          color: '#e2e8f0',
          padding: '1.5rem',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-color)',
          whiteSpace: 'pre-wrap',
          overflowX: 'auto',
          maxHeight: '450px'
        }}
      >
        {explanation.text}
      </pre>
    </div>
  );
};
