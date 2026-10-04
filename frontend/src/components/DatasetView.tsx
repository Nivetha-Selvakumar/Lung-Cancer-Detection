import React, { useState } from 'react';
import { Database, Plus, Eye, Trash2, Folder, CheckCircle } from 'lucide-react';

export const DatasetView: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'iq' | 'hospital'>('iq');

  const iqDatasetRows = [
    { class_name: 'Normal', count: 416, samples: ['normal_1.png', 'normal_2.png', 'normal_3.png'] },
    { class_name: 'Benign', count: 120, samples: ['benign_1.png', 'benign_2.png', 'benign_3.png'] },
    { class_name: 'Malignant', count: 561, samples: ['malignant_1.png', 'malignant_2.png', 'malignant_3.png'] },
  ];

  const hospitalDatasetRows = [
    { class_name: 'Normal', count: 15, samples: ['PIC 4.png', 'PIC 8.png', 'Normal (83).jpg'] },
    { class_name: 'Benign', count: 20, samples: ['Picture16.png', 'Picture19.png', 'Bengin case (48).jpg'] },
    { class_name: 'Malignant', count: 18, samples: ['Picture1.png', 'Picture3.png', 'Malignant (56).jpg'] },
  ];

  const currentRows = activeTab === 'iq' ? iqDatasetRows : hospitalDatasetRows;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Top Header */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-title">Dataset Management</h1>
          <p className="page-subtitle">
            Upload, view and manage training and raw datasets
          </p>
        </div>
        <button
          onClick={() => alert('Dataset upload dialog initialized. Select DICOM/PNG files to import.')}
          style={{
            background: '#4f46e5',
            color: '#ffffff',
            border: 'none',
            padding: '0.6rem 1.25rem',
            borderRadius: '8px',
            fontWeight: 600,
            fontSize: '0.86rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            boxShadow: '0 4px 12px rgba(79, 70, 229, 0.25)'
          }}
        >
          <Plus size={18} /> Upload Dataset
        </button>
      </div>

      {/* Dataset Selection Tabs */}
      <div style={{ display: 'flex', gap: '0.5rem', background: '#ffffff', padding: '0.4rem', borderRadius: '10px', width: 'fit-content', border: '1px solid #e2e8f0' }}>
        <button
          onClick={() => setActiveTab('iq')}
          style={{
            padding: '0.55rem 1.2rem',
            border: 'none',
            borderRadius: '6px',
            background: activeTab === 'iq' ? '#4f46e5' : 'transparent',
            color: activeTab === 'iq' ? '#ffffff' : '#64748b',
            fontWeight: 700,
            fontSize: '0.84rem',
            cursor: 'pointer'
          }}
        >
          IQ-OTH/NCCD Dataset
        </button>
        <button
          onClick={() => setActiveTab('hospital')}
          style={{
            padding: '0.55rem 1.2rem',
            border: 'none',
            borderRadius: '6px',
            background: activeTab === 'hospital' ? '#4f46e5' : 'transparent',
            color: activeTab === 'hospital' ? '#ffffff' : '#64748b',
            fontWeight: 700,
            fontSize: '0.84rem',
            cursor: 'pointer'
          }}
        >
          Raw Dataset (Hospital)
        </button>
      </div>

      {/* Dataset Summary Table */}
      <div className="card" style={{ margin: 0 }}>
        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Class</th>
                <th>No. of Images</th>
                <th>Sample Images</th>
                <th style={{ textAlign: 'center' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {currentRows.map((row, idx) => (
                <tr key={idx}>
                  <td>
                    <strong style={{ color: '#0f172a', fontSize: '0.95rem' }}>{row.class_name}</strong>
                  </td>
                  <td style={{ fontWeight: 700, color: '#4f46e5', fontSize: '1rem' }}>
                    {row.count}
                  </td>
                  <td>
                    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                      {row.samples.map((s, i) => (
                        <div
                          key={i}
                          style={{
                            width: '44px',
                            height: '44px',
                            background: '#0f172a',
                            borderRadius: '6px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: '#94a3b8',
                            fontSize: '0.65rem',
                            textAlign: 'center',
                            overflow: 'hidden',
                            border: '1px solid #cbd5e1'
                          }}
                        >
                          <Folder size={18} style={{ color: '#818cf8' }} />
                        </div>
                      ))}
                    </div>
                  </td>
                  <td>
                    <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center' }}>
                      <button
                        onClick={() => alert(`Viewing ${row.class_name} dataset images`)}
                        style={{ background: '#eff6ff', border: '1px solid #bfdbfe', color: '#2563eb', padding: '0.35rem 0.7rem', borderRadius: '6px', fontSize: '0.78rem', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.3rem' }}
                      >
                        <Eye size={14} /> View
                      </button>
                      <button
                        onClick={() => alert(`Protected: ${row.class_name} dataset images are reserved for model training`)}
                        style={{ background: '#fef2f2', border: '1px solid #fecaca', color: '#dc2626', padding: '0.35rem 0.5rem', borderRadius: '6px', fontSize: '0.78rem', cursor: 'pointer' }}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
