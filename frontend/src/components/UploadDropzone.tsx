import React, { useRef, useState } from 'react';
import { UploadCloud } from 'lucide-react';

interface UploadDropzoneProps {
  onFileSelect: (file: File) => void;
}

export const UploadDropzone: React.FC<UploadDropzoneProps> = ({ onFileSelect }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div style={{ background: '#ffffff', borderRadius: '16px', padding: '1.25rem' }}>
      <div
        className={`dropzone ${isDragOver ? 'dragover' : ''}`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          border: '2px dashed #a5b4fc',
          borderRadius: '16px',
          padding: '2.5rem 1.5rem',
          textAlign: 'center',
          background: '#f8fafc',
          cursor: 'pointer'
        }}
      >
        <div className="dropzone-icon" style={{ background: '#e0e7ff', color: '#4f46e5', width: '56px', height: '56px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1rem auto' }}>
          <UploadCloud size={28} />
        </div>
        <div style={{ fontSize: '1rem', fontWeight: 600, color: '#0f172a', marginBottom: '0.4rem' }}>
          Drag and drop a CT scan image here
        </div>
        <div style={{ fontSize: '0.85rem', color: '#64748b', marginBottom: '0.8rem' }}>or</div>
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
          style={{
            background: '#4f46e5',
            color: '#ffffff',
            border: 'none',
            padding: '0.55rem 1.4rem',
            borderRadius: '8px',
            fontWeight: 600,
            fontSize: '0.88rem',
            cursor: 'pointer',
            marginBottom: '0.8rem',
            boxShadow: '0 4px 12px rgba(79, 70, 229, 0.25)'
          }}
        >
          Browse File
        </button>
        <div style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
          Supported format: PNG, JPG, JPEG (DICOM conversion required)
        </div>
        <input
          type="file"
          ref={fileInputRef}
          className="file-input"
          accept="image/*"
          onChange={handleInputChange}
        />
      </div>
    </div>
  );
};
