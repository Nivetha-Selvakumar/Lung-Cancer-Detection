import React from 'react';

interface LoadingOverlayProps {
  active: boolean;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({ active }) => {
  if (!active) return null;

  return (
    <div className="loading-overlay active">
      <div className="spinner" />
      <div className="loading-text">
        Executing Deterministic Lung Segmentation & Tri-Aspect Inference...
      </div>
    </div>
  );
};
