import React, { useState, useEffect } from 'react';
import { Database, Eye, Folder, Layers, X, FileText, Activity, Loader2, ZoomIn } from 'lucide-react';
import { fetchDatasetImages, DatasetImageSample } from '../services/api';

interface DatasetDetail {
  name: string;
  total: number;
  classes: {
    normal: { train: number; test: number; total: number; pct: number };
    benign: { train: number; test: number; total: number; pct: number };
    malignant: { train: number; test: number; total: number; pct: number };
  };
}

export const DatasetView: React.FC = () => {
  const [selectedDataset, setSelectedDataset] = useState<'iq' | 'hospital'>('iq');
  const [sampleCategory, setSampleCategory] = useState<'normal' | 'benign' | 'malignant'>('normal');

  // Images state
  const [sampleImages, setSampleImages] = useState<DatasetImageSample[]>([]);
  const [loadingImages, setLoadingImages] = useState<boolean>(false);
  const [selectedPreviewImage, setSelectedPreviewImage] = useState<DatasetImageSample | null>(null);

  // Modal State for View Action
  const [viewingFolder, setViewingFolder] = useState<string | null>(null);
  const [folderImages, setFolderImages] = useState<DatasetImageSample[]>([]);
  const [loadingFolderImages, setLoadingFolderImages] = useState<boolean>(false);

  useEffect(() => {
    let active = true;
    setLoadingImages(true);
    fetchDatasetImages(selectedDataset, sampleCategory, 6)
      .then((res) => {
        if (active && res.success) {
          setSampleImages(res.images);
        }
      })
      .catch((err) => console.error('Failed to load sample images:', err))
      .finally(() => {
        if (active) setLoadingImages(false);
      });
    return () => {
      active = false;
    };
  }, [selectedDataset, sampleCategory]);

  useEffect(() => {
    if (!viewingFolder) {
      setFolderImages([]);
      return;
    }
    let active = true;
    setLoadingFolderImages(true);
    fetchDatasetImages(selectedDataset, viewingFolder.toLowerCase(), 200)
      .then((res) => {
        if (active && res.success) {
          setFolderImages(res.images);
        }
      })
      .catch((err) => console.error('Failed to load folder images:', err))
      .finally(() => {
        if (active) setLoadingFolderImages(false);
      });
    return () => {
      active = false;
    };
  }, [selectedDataset, viewingFolder]);

  const datasets: Record<'iq' | 'hospital', DatasetDetail> = {
    iq: {
      name: 'IQ-OTH/NCCD Lung Cancer Dataset',
      total: 1097,
      classes: {
        normal: { train: 374, test: 42, total: 416, pct: 37.9 },
        benign: { train: 108, test: 12, total: 120, pct: 10.9 },
        malignant: { train: 505, test: 56, total: 561, pct: 51.2 }
      }
    },
    hospital: {
      name: 'Hospital Raw CT Dataset',
      total: 70,
      classes: {
        normal: { train: 15, test: 5, total: 20, pct: 28.6 },
        benign: { train: 20, test: 7, total: 27, pct: 38.6 },
        malignant: { train: 18, test: 5, total: 23, pct: 32.8 }
      }
    }
  };

  const currentDataset = datasets[selectedDataset];
  const { normal, benign, malignant } = currentDataset.classes;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', fontFamily: "'Inter', -apple-system, sans-serif" }}>
      
      {/* 1. Page Title Header (NO Upload Dataset Button as requested) */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0f172a', margin: '0 0 0.3rem 0', letterSpacing: '-0.02em' }}>
            Dataset Management
          </h1>
          <p style={{ fontSize: '0.88rem', color: '#64748b', margin: 0 }}>
            Manage training data and raw CT datasets for model benchmarking
          </p>
        </div>

        {/* Dataset Switcher Tabs */}
        <div style={{ display: 'flex', gap: '0.4rem', background: '#e2e8f0', padding: '0.3rem', borderRadius: '12px' }}>
          <button
            onClick={() => setSelectedDataset('iq')}
            style={{
              padding: '0.55rem 1.1rem',
              borderRadius: '9px',
              border: 'none',
              background: selectedDataset === 'iq' ? '#4f46e5' : 'transparent',
              color: selectedDataset === 'iq' ? '#ffffff' : '#475569',
              fontWeight: selectedDataset === 'iq' ? 700 : 600,
              fontSize: '0.84rem',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            IQ-OTH/NCCD Dataset (1097)
          </button>

          <button
            onClick={() => setSelectedDataset('hospital')}
            style={{
              padding: '0.55rem 1.1rem',
              borderRadius: '9px',
              border: 'none',
              background: selectedDataset === 'hospital' ? '#4f46e5' : 'transparent',
              color: selectedDataset === 'hospital' ? '#ffffff' : '#475569',
              fontWeight: selectedDataset === 'hospital' ? 700 : 600,
              fontSize: '0.84rem',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
          >
            Raw Dataset (Hospital - 70)
          </button>
        </div>
      </div>

      {/* 2. Top 4 Stat Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.25rem' }}>
        
        {/* Card 1: Total Scans */}
        <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '1.25rem 1.5rem', display: 'flex', alignItems: 'center', gap: '1.1rem', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.03)' }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '14px', background: '#e0e7ff', color: '#4f46e5', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Database size={24} />
          </div>
          <div>
            <div style={{ fontSize: '0.78rem', color: '#64748b', fontWeight: 600 }}>Total Scans</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#0f172a', lineHeight: 1.1, marginTop: '0.15rem' }}>
              {currentDataset.total}
            </div>
          </div>
        </div>

        {/* Card 2: Normal Scans */}
        <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '1.25rem 1.5rem', display: 'flex', alignItems: 'center', gap: '1.1rem', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.03)' }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '14px', background: '#dcfce7', color: '#16a34a', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Activity size={24} />
          </div>
          <div>
            <div style={{ fontSize: '0.78rem', color: '#64748b', fontWeight: 600 }}>Normal</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#16a34a', lineHeight: 1.1, marginTop: '0.15rem' }}>
              {normal.total}
            </div>
          </div>
        </div>

        {/* Card 3: Benign Scans */}
        <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '1.25rem 1.5rem', display: 'flex', alignItems: 'center', gap: '1.1rem', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.03)' }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '14px', background: '#dbeafe', color: '#2563eb', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Layers size={24} />
          </div>
          <div>
            <div style={{ fontSize: '0.78rem', color: '#64748b', fontWeight: 600 }}>Benign</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#2563eb', lineHeight: 1.1, marginTop: '0.15rem' }}>
              {benign.total}
            </div>
          </div>
        </div>

        {/* Card 4: Malignant Scans */}
        <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '1.25rem 1.5rem', display: 'flex', alignItems: 'center', gap: '1.1rem', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.03)' }}>
          <div style={{ width: '48px', height: '48px', borderRadius: '14px', background: '#fee2e2', color: '#dc2626', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <FileText size={24} />
          </div>
          <div>
            <div style={{ fontSize: '0.78rem', color: '#64748b', fontWeight: 600 }}>Malignant</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#dc2626', lineHeight: 1.1, marginTop: '0.15rem' }}>
              {malignant.total}
            </div>
          </div>
        </div>

      </div>

      {/* 3. Middle Grid: Class Distribution Donut Chart (Left) + Sample Images (Right) */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: '1.5rem' }}>
        
        {/* Class Distribution Card with Donut Visualization */}
        <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.03)' }}>
          <h3 style={{ margin: '0 0 1.25rem 0', fontSize: '1rem', fontWeight: 700, color: '#0f172a' }}>
            Class Distribution ({selectedDataset === 'iq' ? 'IQ-OTH/NCCD' : 'Hospital Raw'})
          </h3>

          <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
            
            {/* SVG Donut Chart */}
            <div style={{ position: 'relative', width: '140px', height: '140px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <svg width="140" height="140" viewBox="0 0 36 36" style={{ transform: 'rotate(-90deg)' }}>
                {/* Track */}
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#f1f5f9" strokeWidth="4" />
                
                {/* Normal Segment */}
                <path
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="4"
                  strokeDasharray={`${normal.pct}, 100`}
                />
                {/* Benign Segment */}
                <path
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="4"
                  strokeDasharray={`${benign.pct}, 100`}
                  strokeDashoffset={`-${normal.pct}`}
                />
                {/* Malignant Segment */}
                <path
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="4"
                  strokeDasharray={`${malignant.pct}, 100`}
                  strokeDashoffset={`-${normal.pct + benign.pct}`}
                />
              </svg>
              <div style={{ position: 'absolute', textAlign: 'center' }}>
                <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#0f172a', lineHeight: 1 }}>{currentDataset.total}</div>
                <div style={{ fontSize: '0.68rem', color: '#64748b', fontWeight: 600, marginTop: '0.2rem' }}>CT Scans</div>
              </div>
            </div>

            {/* Percentage Breakdown Legend List */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.86rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#10b981' }} />
                  <span style={{ color: '#334155', fontWeight: 600 }}>Normal</span>
                </div>
                <span style={{ fontWeight: 700, color: '#0f172a' }}>{normal.total} ({normal.pct}%)</span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#3b82f6' }} />
                  <span style={{ color: '#334155', fontWeight: 600 }}>Benign</span>
                </div>
                <span style={{ fontWeight: 700, color: '#0f172a' }}>{benign.total} ({benign.pct}%)</span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#ef4444' }} />
                  <span style={{ color: '#334155', fontWeight: 600 }}>Malignant</span>
                </div>
                <span style={{ fontWeight: 700, color: '#0f172a' }}>{malignant.total} ({malignant.pct}%)</span>
              </div>
            </div>

          </div>
        </div>

        {/* Sample Images Preview Card */}
        <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.03)', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 700, color: '#0f172a' }}>Sample Images</h3>
              
              {/* Filter Pills */}
              <div style={{ display: 'flex', gap: '0.25rem', background: '#f1f5f9', padding: '0.25rem', borderRadius: '8px' }}>
                <button
                  onClick={() => setSampleCategory('normal')}
                  style={{
                    padding: '0.3rem 0.65rem',
                    border: 'none',
                    borderRadius: '6px',
                    background: sampleCategory === 'normal' ? '#ffffff' : 'transparent',
                    color: sampleCategory === 'normal' ? '#10b981' : '#64748b',
                    fontWeight: 700,
                    fontSize: '0.76rem',
                    cursor: 'pointer'
                  }}
                >
                  Normal
                </button>
                <button
                  onClick={() => setSampleCategory('benign')}
                  style={{
                    padding: '0.3rem 0.65rem',
                    border: 'none',
                    borderRadius: '6px',
                    background: sampleCategory === 'benign' ? '#ffffff' : 'transparent',
                    color: sampleCategory === 'benign' ? '#3b82f6' : '#64748b',
                    fontWeight: 700,
                    fontSize: '0.76rem',
                    cursor: 'pointer'
                  }}
                >
                  Benign
                </button>
                <button
                  onClick={() => setSampleCategory('malignant')}
                  style={{
                    padding: '0.3rem 0.65rem',
                    border: 'none',
                    borderRadius: '6px',
                    background: sampleCategory === 'malignant' ? '#ffffff' : 'transparent',
                    color: sampleCategory === 'malignant' ? '#ef4444' : '#64748b',
                    fontWeight: 700,
                    fontSize: '0.76rem',
                    cursor: 'pointer'
                  }}
                >
                  Malignant
                </button>
              </div>
            </div>

            {/* Sample CT Thumbnails Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', marginTop: '0.75rem' }}>
              {loadingImages ? (
                <div style={{ gridColumn: 'span 3', height: '110px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#090d16', borderRadius: '12px', color: '#94a3b8', gap: '0.5rem', fontSize: '0.82rem' }}>
                  <Loader2 size={18} className="animate-spin" style={{ color: '#6366f1' }} /> Loading CT Scans...
                </div>
              ) : sampleImages.length > 0 ? (
                sampleImages.slice(0, 3).map((img, idx) => (
                  <div
                    key={img.id || idx}
                    onClick={() => setSelectedPreviewImage(img)}
                    title={`Click to preview ${img.filename}`}
                    style={{
                      position: 'relative',
                      height: '110px',
                      background: '#090d16',
                      borderRadius: '12px',
                      border: '1px solid #1e293b',
                      overflow: 'hidden',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <img
                      src={img.image_url}
                      alt={img.filename}
                      style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                        display: 'block'
                      }}
                    />
                    <div style={{
                      position: 'absolute',
                      inset: 0,
                      background: 'rgba(15, 23, 42, 0.4)',
                      opacity: 0,
                      transition: 'opacity 0.2s ease',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#ffffff'
                    }}
                    className="hover-overlay"
                    onMouseEnter={(e) => (e.currentTarget.style.opacity = '1')}
                    onMouseLeave={(e) => (e.currentTarget.style.opacity = '0')}
                    >
                      <ZoomIn size={20} />
                      <span style={{ fontSize: '0.65rem', fontWeight: 600, marginTop: '0.2rem' }}>View Scan</span>
                    </div>
                    <div style={{
                      position: 'absolute',
                      bottom: 0,
                      left: 0,
                      right: 0,
                      background: 'rgba(15, 23, 42, 0.85)',
                      backdropFilter: 'blur(4px)',
                      padding: '0.2rem 0.4rem',
                      fontSize: '0.64rem',
                      color: '#cbd5e1',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      textAlign: 'center',
                      fontWeight: 600
                    }}>
                      {img.filename}
                    </div>
                  </div>
                ))
              ) : (
                <div style={{ gridColumn: 'span 3', height: '110px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#090d16', borderRadius: '12px', color: '#64748b', fontSize: '0.82rem' }}>
                  No sample images found for this category
                </div>
              )}
            </div>
          </div>

          <div style={{ fontSize: '0.76rem', color: '#64748b', textAlign: 'center', marginTop: '1rem' }}>
            Showing {sampleImages.length} real sample CT scans for <strong style={{ color: '#0f172a' }}>{sampleCategory}</strong> category
          </div>
        </div>

      </div>

      {/* 4. Bottom Table: Dataset Files (ONLY View Action allowed, NO Delete / Edit) */}
      <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '1.5rem', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.03)' }}>
        <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', fontWeight: 700, color: '#0f172a' }}>
          Dataset Files
        </h3>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.86rem', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0', color: '#64748b', fontWeight: 600, background: '#f8fafc' }}>
                <th style={{ padding: '0.75rem 1rem' }}>File/Folder Name</th>
                <th style={{ padding: '0.75rem 1rem' }}>Type</th>
                <th style={{ padding: '0.75rem 1rem' }}>Total Images</th>
                <th style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              
              {/* Row 1: Normal */}
              <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '0.85rem 1rem', display: 'flex', alignItems: 'center', gap: '0.6rem', fontWeight: 700, color: '#0f172a' }}>
                  <Folder size={18} style={{ color: '#10b981' }} />
                  <span>Normal</span>
                </td>
                <td style={{ padding: '0.85rem 1rem', color: '#64748b' }}>Folder</td>
                <td style={{ padding: '0.85rem 1rem', fontWeight: 800, color: '#10b981' }}>{normal.total} images</td>
                <td style={{ padding: '0.85rem 1rem', textAlign: 'center' }}>
                  <button
                    onClick={() => setViewingFolder('Normal')}
                    style={{
                      background: '#eff6ff',
                      border: '1px solid #bfdbfe',
                      color: '#2563eb',
                      padding: '0.35rem 0.85rem',
                      borderRadius: '8px',
                      fontSize: '0.8rem',
                      fontWeight: 700,
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.4rem'
                    }}
                  >
                    <Eye size={14} /> View Images
                  </button>
                </td>
              </tr>

              {/* Row 2: Benign */}
              <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '0.85rem 1rem', display: 'flex', alignItems: 'center', gap: '0.6rem', fontWeight: 700, color: '#0f172a' }}>
                  <Folder size={18} style={{ color: '#3b82f6' }} />
                  <span>Benign</span>
                </td>
                <td style={{ padding: '0.85rem 1rem', color: '#64748b' }}>Folder</td>
                <td style={{ padding: '0.85rem 1rem', fontWeight: 800, color: '#3b82f6' }}>{benign.total} images</td>
                <td style={{ padding: '0.85rem 1rem', textAlign: 'center' }}>
                  <button
                    onClick={() => setViewingFolder('Benign')}
                    style={{
                      background: '#eff6ff',
                      border: '1px solid #bfdbfe',
                      color: '#2563eb',
                      padding: '0.35rem 0.85rem',
                      borderRadius: '8px',
                      fontSize: '0.8rem',
                      fontWeight: 700,
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.4rem'
                    }}
                  >
                    <Eye size={14} /> View Images
                  </button>
                </td>
              </tr>

              {/* Row 3: Malignant */}
              <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '0.85rem 1rem', display: 'flex', alignItems: 'center', gap: '0.6rem', fontWeight: 700, color: '#0f172a' }}>
                  <Folder size={18} style={{ color: '#ef4444' }} />
                  <span>Malignant</span>
                </td>
                <td style={{ padding: '0.85rem 1rem', color: '#64748b' }}>Folder</td>
                <td style={{ padding: '0.85rem 1rem', fontWeight: 800, color: '#ef4444' }}>{malignant.total} images</td>
                <td style={{ padding: '0.85rem 1rem', textAlign: 'center' }}>
                  <button
                    onClick={() => setViewingFolder('Malignant')}
                    style={{
                      background: '#eff6ff',
                      border: '1px solid #bfdbfe',
                      color: '#2563eb',
                      padding: '0.35rem 0.85rem',
                      borderRadius: '8px',
                      fontSize: '0.8rem',
                      fontWeight: 700,
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.4rem'
                    }}
                  >
                    <Eye size={14} /> View Images
                  </button>
                </td>
              </tr>

            </tbody>
          </table>
        </div>
      </div>

      {/* View Folder Modal */}
      {viewingFolder && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(15, 23, 42, 0.8)',
          backdropFilter: 'blur(5px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          padding: '1.5rem'
        }}>
          <div style={{ background: '#ffffff', borderRadius: '24px', maxWidth: '720px', width: '100%', padding: '2rem', position: 'relative', maxHeight: '90vh', overflowY: 'auto', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)' }}>
            <button
              onClick={() => setViewingFolder(null)}
              style={{ position: 'absolute', top: '1.25rem', right: '1.25rem', background: '#f1f5f9', border: 'none', width: '36px', height: '36px', borderRadius: '50%', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}
            >
              <X size={20} />
            </button>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
              <div style={{ background: '#e0e7ff', color: '#4f46e5', width: '44px', height: '44px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Eye size={22} />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '1.2rem', color: '#0f172a', fontWeight: 800 }}>
                  {viewingFolder} Category CT Scans
                </h3>
                <div style={{ fontSize: '0.84rem', color: '#64748b' }}>
                  {currentDataset.name} • Total: <strong style={{ color: '#4f46e5' }}>{currentDataset.classes[viewingFolder.toLowerCase() as 'normal' | 'benign' | 'malignant'].total} images</strong>
                </div>
              </div>
            </div>

            <p style={{ fontSize: '0.86rem', color: '#475569', lineHeight: 1.5, marginBottom: '1.25rem' }}>
              Viewing image files in the <strong style={{ color: '#0f172a' }}>{viewingFolder}</strong> folder from the dataset. Click any image to view a high-resolution preview.
            </p>

            {/* Folder Image Gallery Grid (Scrollable Container) */}
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '16px', padding: '1rem', minHeight: '220px', maxHeight: '420px', overflowY: 'auto' }}>
              {loadingFolderImages ? (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '180px', color: '#64748b', gap: '0.5rem' }}>
                  <Loader2 size={20} className="animate-spin" style={{ color: '#4f46e5' }} /> Loading CT scan gallery...
                </div>
              ) : folderImages.length > 0 ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem' }}>
                  {folderImages.map((img, idx) => (
                    <div
                      key={img.id || idx}
                      onClick={() => setSelectedPreviewImage(img)}
                      title={`Preview ${img.filename}`}
                      style={{
                        position: 'relative',
                        height: '110px',
                        background: '#090d16',
                        borderRadius: '10px',
                        border: '1px solid #cbd5e1',
                        overflow: 'hidden',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
                        transition: 'transform 0.15s ease'
                      }}
                    >
                      <img
                        src={img.image_url}
                        alt={img.filename}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                      <div
                        style={{
                          position: 'absolute',
                          inset: 0,
                          background: 'rgba(15, 23, 42, 0.4)',
                          opacity: 0,
                          transition: 'opacity 0.2s ease',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: '#ffffff'
                        }}
                        onMouseEnter={(e) => (e.currentTarget.style.opacity = '1')}
                        onMouseLeave={(e) => (e.currentTarget.style.opacity = '0')}
                      >
                        <ZoomIn size={20} />
                      </div>
                      <div style={{
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        background: 'rgba(15, 23, 42, 0.85)',
                        padding: '0.2rem',
                        fontSize: '0.6rem',
                        color: '#f8fafc',
                        textAlign: 'center',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}>
                        {img.filename}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '180px', color: '#94a3b8' }}>
                  No CT scan images available in this folder.
                </div>
              )}
            </div>

            <button
              onClick={() => setViewingFolder(null)}
              style={{ marginTop: '1.25rem', width: '100%', background: '#4f46e5', color: '#ffffff', border: 'none', padding: '0.75rem', borderRadius: '12px', fontWeight: 700, fontSize: '0.9rem', cursor: 'pointer' }}
            >
              Close Folder Gallery
            </button>
          </div>
        </div>
      )}

      {/* Selected Image Full Preview Modal */}
      {selectedPreviewImage && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(15, 23, 42, 0.85)',
          backdropFilter: 'blur(6px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 99999,
          padding: '1.5rem'
        }}>
          <div style={{
            background: '#0f172a',
            border: '1px solid #334155',
            borderRadius: '24px',
            maxWidth: '560px',
            width: '100%',
            padding: '1.75rem',
            position: 'relative',
            color: '#f8fafc',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)'
          }}>
            <button
              onClick={() => setSelectedPreviewImage(null)}
              style={{
                position: 'absolute',
                top: '1.25rem',
                right: '1.25rem',
                background: '#1e293b',
                border: '1px solid #475569',
                width: '36px',
                height: '36px',
                borderRadius: '50%',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#94a3b8'
              }}
            >
              <X size={20} />
            </button>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
              <span style={{
                background: selectedPreviewImage.category.toLowerCase() === 'normal' ? 'rgba(16, 185, 129, 0.2)' : selectedPreviewImage.category.toLowerCase() === 'benign' ? 'rgba(59, 130, 246, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                color: selectedPreviewImage.category.toLowerCase() === 'normal' ? '#34d399' : selectedPreviewImage.category.toLowerCase() === 'benign' ? '#60a5fa' : '#f87171',
                border: `1px solid ${selectedPreviewImage.category.toLowerCase() === 'normal' ? '#10b981' : selectedPreviewImage.category.toLowerCase() === 'benign' ? '#3b82f6' : '#ef4444'}`,
                padding: '0.25rem 0.75rem',
                borderRadius: '999px',
                fontSize: '0.75rem',
                fontWeight: 700,
                textTransform: 'uppercase'
              }}>
                {selectedPreviewImage.category}
              </span>
              <h3 style={{ margin: 0, fontSize: '1rem', color: '#f8fafc', fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                {selectedPreviewImage.filename}
              </h3>
            </div>

            <div style={{
              background: '#020617',
              borderRadius: '16px',
              border: '1px solid #1e293b',
              padding: '1rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '280px',
              maxHeight: '400px'
            }}>
              <img
                src={selectedPreviewImage.image_url}
                alt={selectedPreviewImage.filename}
                style={{
                  maxWidth: '100%',
                  maxHeight: '360px',
                  objectFit: 'contain',
                  borderRadius: '8px',
                  boxShadow: '0 8px 24px rgba(0,0,0,0.4)'
                }}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1.25rem', fontSize: '0.8rem', color: '#94a3b8' }}>
              <span>Dataset: <strong style={{ color: '#cbd5e1' }}>{selectedPreviewImage.dataset}</strong></span>
              <button
                onClick={() => setSelectedPreviewImage(null)}
                style={{
                  background: '#3b82f6',
                  color: '#ffffff',
                  border: 'none',
                  padding: '0.55rem 1.25rem',
                  borderRadius: '10px',
                  fontWeight: 700,
                  fontSize: '0.85rem',
                  cursor: 'pointer'
                }}
              >
                Close Scan Preview
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};
