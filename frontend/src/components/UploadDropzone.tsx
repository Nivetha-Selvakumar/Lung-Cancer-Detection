import React, { useRef, useState } from 'react';
import { UploadCloud, ImagePlus } from 'lucide-react';
import { generateSampleCT } from '../utils/sampleGenerator';

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

  const handleSampleClick = async (type: 'normal' | 'benign' | 'malignant') => {
    const file = await generateSampleCT(type);
    onFileSelect(file);
  };

  return (
    <div className="card">
      <div className="card-title">
        <UploadCloud size={20} /> Input CT Scan Image
      </div>

      <div
        className={`dropzone ${isDragOver ? 'dragover' : ''}`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="dropzone-icon">
          <ImagePlus size={28} />
        </div>
        <div className="dropzone-text">Drag & drop CT scan image here</div>
        <div className="dropzone-sub">Supports PNG, JPG, JPEG, BMP, TIF</div>
        <input
          type="file"
          ref={fileInputRef}
          className="file-input"
          accept="image/*"
          onChange={handleInputChange}
        />
      </div>

      <div className="samples-container">
        <div className="samples-label">Or test with dataset CT samples:</div>
        <div className="sample-btns">
          <button
            type="button"
            className="sample-btn"
            onClick={() => handleSampleClick('normal')}
          >
            Normal CT
          </button>
          <button
            type="button"
            className="sample-btn"
            onClick={() => handleSampleClick('benign')}
          >
            Benign CT
          </button>
          <button
            type="button"
            className="sample-btn"
            onClick={() => handleSampleClick('malignant')}
          >
            Malignant CT
          </button>
        </div>
      </div>
    </div>
  );
};
