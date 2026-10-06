import React, { useRef, useState } from 'react';
import { UploadCloud, Loader2 } from 'lucide-react';

interface UploadDropzoneProps {
  onFileSelect: (file: File) => void;
  loading?: boolean;
}

export const UploadDropzone: React.FC<UploadDropzoneProps> = ({ onFileSelect, loading = false }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!loading) setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (!loading && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!loading && e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div style={{ background: '#ffffff', borderRadius: '16px', padding: '1.25rem' }}>
      <div
        className={`dropzone ${isDragOver ? 'dragover' : ''}`}
        onClick={() => { if (!loading) fileInputRef.current?.click(); }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          border: `2px dashed ${loading ? '#818cf8' : '#a5b4fc'}`,
          borderRadius: '16px',
          padding: '2.5rem 1.5rem',
          textAlign: 'center',
          background: loading ? '#f0f3ff' : '#f8fafc',
          cursor: loading ? 'wait' : 'pointer',
          transition: 'all 0.2s'
        }}
      >
        <div className="dropzone-icon" style={{ background: '#e0e7ff', color: '#4f46e5', width: '56px', height: '56px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1rem auto' }}>
          {loading ? (
            <Loader2 size={28} style={{ animation: 'spin 1s linear infinite' }} />
          ) : (
            <UploadCloud size={28} />
          )}
        </div>
        <div style={{ fontSize: '1rem', fontWeight: 600, color: '#0f172a', marginBottom: '0.4rem' }}>
          {loading ? 'Analyzing CT Scan Image...' : 'Drag and drop a CT scan image here'}
        </div>
        <div style={{ fontSize: '0.85rem', color: '#64748b', marginBottom: '0.8rem' }}>
          {loading ? 'Please wait while AI performs feature extraction' : 'or'}
        </div>
        <button
          type="button"
          disabled={loading}
          onClick={(e) => { e.stopPropagation(); if (!loading) fileInputRef.current?.click(); }}
          style={{
            background: loading ? '#6366f1' : '#4f46e5',
            color: '#ffffff',
            border: 'none',
            padding: '0.55rem 1.4rem',
            borderRadius: '8px',
            fontWeight: 600,
            fontSize: '0.88rem',
            cursor: loading ? 'wait' : 'pointer',
            marginBottom: '0.8rem',
            boxShadow: '0 4px 12px rgba(79, 70, 229, 0.25)',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}
        >
          {loading && <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />}
          {loading ? 'Uploading & Analyzing...' : 'Browse File'}
        </button>
        <div style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
          Supported format: PNG, JPG, JPEG (DICOM conversion required)
        </div>
        <input
          type="file"
          ref={fileInputRef}
          className="file-input"
          accept=".jpg,.jpeg,.png,image/jpeg,image/jpg,image/png"
          style={{ display: 'none' }}
          onChange={handleInputChange}
        />
      </div>

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};
