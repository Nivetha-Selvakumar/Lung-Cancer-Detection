import React, { useEffect, useState } from 'react';
import { Search, ChevronDown, Calendar, Eye, Trash2, RefreshCw, X } from 'lucide-react';

interface HistoryRecord {
  id: number;
  case_id: string;
  date_time: string;
  username: string;
  model_name: string;
  model_version: string;
  predicted_class: string;
  confidence: number;
  probabilities: {
    Normal: number;
    Benign: number;
    Malignant: number;
  };
  verification_status: string;
  verified_by_doctor: string;
  image_b64?: string;
}

export const HistoryView: React.FC = () => {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(true);

  // Filters & Search
  const [searchQuery, setSearchQuery] = useState('');
  const [classFilter, setClassFilter] = useState<string>('All');
  const [statusFilter, setStatusFilter] = useState<string>('All');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 8;

  // Selected Case Modal
  const [selectedCase, setSelectedCase] = useState<HistoryRecord | null>(null);

  const fetchHistory = () => {
    setLoading(true);
    fetch('/api/history')
      .then((res) => res.json())
      .then((data) => {
        if (data.records) {
          setRecords(data.records);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load history:', err);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleStatusToggle = (id: number, currentStatus: string) => {
    const nextStatus = currentStatus === 'Verified' ? 'Pending' : currentStatus === 'Review' ? 'Verified' : 'Review';
    const nextVerified = nextStatus === 'Verified' ? 'Yes' : 'No';

    fetch('/api/history/update-status', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, status: nextStatus, verified_by_doctor: nextVerified })
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.success) {
          setRecords((prev) =>
            prev.map((r) => (r.id === id ? { ...r, verification_status: nextStatus, verified_by_doctor: nextVerified } : r))
          );
        }
      });
  };

  const handleDeleteRecord = (id: number) => {
    if (!window.confirm('Are you sure you want to delete this prediction record from the database?')) {
      return;
    }
    fetch('/api/history/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id })
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.success) {
          setRecords((prev) => prev.filter((r) => r.id !== id));
        }
      });
  };

  // Filtering Logic
  const filteredRecords = records.filter((rec) => {
    const matchesSearch =
      rec.case_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      rec.predicted_class.toLowerCase().includes(searchQuery.toLowerCase()) ||
      rec.date_time.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesClass = classFilter === 'All' || rec.predicted_class === classFilter;
    const matchesStatus = statusFilter === 'All' || (rec.verification_status || 'Pending') === statusFilter;
    return matchesSearch && matchesClass && matchesStatus;
  });

  const totalPages = Math.ceil(filteredRecords.length / itemsPerPage) || 1;
  const currentRecords = filteredRecords.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Header Bar matching UI screenshot */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-title" style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0f172a', margin: 0 }}>
            Prediction History
          </h1>
          <p className="page-subtitle" style={{ fontSize: '0.9rem', color: '#64748b', marginTop: '0.25rem' }}>
            View past cases, predictions and doctor verification status
          </p>
        </div>
        <button
          onClick={fetchHistory}
          title="Refresh prediction history from database"
          style={{
            background: '#ffffff',
            color: '#4f46e5',
            border: '1px solid #c7d2fe',
            padding: '0.55rem 1.1rem',
            borderRadius: '10px',
            fontWeight: 700,
            fontSize: '0.85rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.45rem',
            boxShadow: '0 2px 6px rgba(0,0,0,0.04)',
            transition: 'all 0.2s'
          }}
        >
          <RefreshCw size={16} /> Refresh
        </button>
      </div>

      {/* Search & Filter Bar matching uploaded screenshot */}
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center' }}>
        {/* Search Input */}
        <div style={{ flex: '1 1 300px', position: 'relative' }}>
          <Search size={18} style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }} />
          <input
            type="text"
            placeholder="Search by case ID, filename, or date..."
            value={searchQuery}
            onChange={(e) => { setSearchQuery(e.target.value); setCurrentPage(1); }}
            style={{
              width: '100%',
              padding: '0.65rem 1rem 0.65rem 2.6rem',
              borderRadius: '10px',
              border: '1px solid #cbd5e1',
              fontSize: '0.88rem',
              color: '#0f172a',
              background: '#ffffff',
              outline: 'none'
            }}
          />
        </div>

        {/* Class Filter Dropdown */}
        <div style={{ position: 'relative', minWidth: '150px' }}>
          <select
            value={classFilter}
            onChange={(e) => { setClassFilter(e.target.value); setCurrentPage(1); }}
            style={{
              width: '100%',
              appearance: 'none',
              padding: '0.65rem 2rem 0.65rem 1rem',
              borderRadius: '10px',
              border: '1px solid #cbd5e1',
              fontSize: '0.88rem',
              fontWeight: 600,
              color: '#334155',
              background: '#ffffff',
              cursor: 'pointer',
              outline: 'none'
            }}
          >
            <option value="All">All Classes</option>
            <option value="Normal">Normal</option>
            <option value="Benign">Benign</option>
            <option value="Malignant">Malignant</option>
          </select>
          <ChevronDown size={16} style={{ position: 'absolute', right: '0.8rem', top: '50%', transform: 'translateY(-50%)', color: '#64748b', pointerEvents: 'none' }} />
        </div>

        {/* Status Filter Dropdown */}
        <div style={{ position: 'relative', minWidth: '150px' }}>
          <select
            value={statusFilter}
            onChange={(e) => { setStatusFilter(e.target.value); setCurrentPage(1); }}
            style={{
              width: '100%',
              appearance: 'none',
              padding: '0.65rem 2rem 0.65rem 1rem',
              borderRadius: '10px',
              border: '1px solid #cbd5e1',
              fontSize: '0.88rem',
              fontWeight: 600,
              color: '#334155',
              background: '#ffffff',
              cursor: 'pointer',
              outline: 'none'
            }}
          >
            <option value="All">All Status</option>
            <option value="Verified">Verified</option>
            <option value="Review">Review</option>
            <option value="Pending">Pending</option>
          </select>
          <ChevronDown size={16} style={{ position: 'absolute', right: '0.8rem', top: '50%', transform: 'translateY(-50%)', color: '#64748b', pointerEvents: 'none' }} />
        </div>

        {/* Date Range Picker Placeholder */}
        <button
          style={{
            padding: '0.65rem 1.1rem',
            borderRadius: '10px',
            border: '1px solid #cbd5e1',
            fontSize: '0.88rem',
            fontWeight: 600,
            color: '#64748b',
            background: '#ffffff',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            cursor: 'pointer'
          }}
        >
          <Calendar size={16} /> Select date range <ChevronDown size={14} />
        </button>
      </div>

      {/* Main Table Card matching uploaded screenshot */}
      <div className="card" style={{ background: '#ffffff', borderRadius: '16px', padding: '0', boxShadow: '0 4px 20px rgba(0,0,0,0.04)', overflow: 'hidden' }}>
        <div className="data-table-container">
          <table className="data-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: '#f8fafc', borderBottom: '1px solid #e2e8f0', color: '#475569', fontSize: '0.82rem', fontWeight: 800, textAlign: 'left' }}>
                <th style={{ padding: '1rem 1.25rem' }}>Case ID</th>
                <th style={{ padding: '1rem 1.25rem' }}>Image</th>
                <th style={{ padding: '1rem 1.25rem' }}>Date & Time</th>
                <th style={{ padding: '1rem 1.25rem' }}>Prediction</th>
                <th style={{ padding: '1rem 1.25rem' }}>Confidence</th>
                <th style={{ padding: '1rem 1.25rem' }}>Verified by Doctor</th>
                <th style={{ padding: '1rem 1.25rem' }}>Status</th>
                <th style={{ padding: '1rem 1.25rem', textAlign: 'center' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={8} style={{ padding: '3rem', textAlign: 'center', color: '#64748b' }}>
                    Loading prediction history from database...
                  </td>
                </tr>
              ) : currentRecords.length === 0 ? (
                <tr>
                  <td colSpan={8} style={{ padding: '3rem', textAlign: 'center', color: '#94a3b8' }}>
                    No prediction history found matching the selected filters.
                  </td>
                </tr>
              ) : (
                currentRecords.map((rec) => {
                  const confPct = (rec.confidence * 100).toFixed(1);
                  const isMalignant = rec.predicted_class === 'Malignant';
                  const isBenign = rec.predicted_class === 'Benign';

                  const predBg = isMalignant ? '#fee2e2' : isBenign ? '#dbeafe' : '#dcfce7';
                  const predColor = isMalignant ? '#ef4444' : isBenign ? '#3b82f6' : '#22c55e';

                  const status = rec.verification_status || 'Pending';
                  const statusBg = status === 'Verified' ? '#dcfce7' : status === 'Review' ? '#dbeafe' : '#f1f5f9';
                  const statusColor = status === 'Verified' ? '#15803d' : status === 'Review' ? '#1d4ed8' : '#64748b';

                  return (
                    <tr key={rec.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '0.9rem 1.25rem', fontWeight: 800, color: '#1e293b', fontSize: '0.9rem' }}>
                        {rec.case_id}
                      </td>
                      <td style={{ padding: '0.9rem 1.25rem' }}>
                        {rec.image_b64 ? (
                          <img
                            src={rec.image_b64}
                            alt={rec.case_id}
                            style={{
                              width: '40px',
                              height: '40px',
                              borderRadius: '8px',
                              objectFit: 'cover',
                              border: '1px solid #cbd5e1',
                              background: '#0f172a'
                            }}
                          />
                        ) : (
                          <div style={{ width: '40px', height: '40px', borderRadius: '8px', background: '#e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8', fontSize: '0.75rem', fontWeight: 700 }}>
                            CT
                          </div>
                        )}
                      </td>
                      <td style={{ padding: '0.9rem 1.25rem', color: '#64748b', fontSize: '0.85rem' }}>
                        {rec.date_time}
                      </td>
                      <td style={{ padding: '0.9rem 1.25rem' }}>
                        <span
                          style={{
                            background: predBg,
                            color: predColor,
                            padding: '0.25rem 0.75rem',
                            borderRadius: '16px',
                            fontWeight: 700,
                            fontSize: '0.82rem',
                            display: 'inline-block'
                          }}
                        >
                          {rec.predicted_class}
                        </span>
                      </td>
                      <td style={{ padding: '0.9rem 1.25rem', fontWeight: 700, color: '#334155', fontSize: '0.88rem' }}>
                        {confPct}%
                      </td>
                      <td style={{ padding: '0.9rem 1.25rem', fontSize: '0.88rem', color: rec.verified_by_doctor === 'Yes' ? '#16a34a' : '#64748b', fontWeight: 600 }}>
                        {rec.verified_by_doctor === 'Yes' ? 'Yes' : 'No'}
                      </td>
                      <td style={{ padding: '0.9rem 1.25rem' }}>
                        <button
                          onClick={() => handleStatusToggle(rec.id, status)}
                          title="Click to toggle status (Pending -> Review -> Verified)"
                          style={{
                            background: statusBg,
                            color: statusColor,
                            border: 'none',
                            padding: '0.3rem 0.85rem',
                            borderRadius: '16px',
                            fontWeight: 700,
                            fontSize: '0.8rem',
                            cursor: 'pointer',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.35rem'
                          }}
                        >
                          {status}
                        </button>
                      </td>
                      <td style={{ padding: '0.9rem 1.25rem', textAlign: 'center' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.6rem' }}>
                          <button
                            onClick={() => setSelectedCase(rec)}
                            title="View Case Details"
                            style={{
                              background: '#eff6ff',
                              color: '#3b82f6',
                              border: '1px solid #bfdbfe',
                              width: '32px',
                              height: '32px',
                              borderRadius: '8px',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              cursor: 'pointer'
                            }}
                          >
                            <Eye size={16} />
                          </button>
                          <button
                            onClick={() => handleDeleteRecord(rec.id)}
                            title="Delete Record"
                            style={{
                              background: '#fef2f2',
                              color: '#ef4444',
                              border: '1px solid #fecaca',
                              width: '32px',
                              height: '32px',
                              borderRadius: '8px',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              cursor: 'pointer'
                            }}
                          >
                            <Trash2 size={16} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar matching UI screenshot */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '1rem 1.5rem', background: '#ffffff', borderTop: '1px solid #f1f5f9' }}>
          <div style={{ fontSize: '0.85rem', color: '#64748b' }}>
            Showing {filteredRecords.length > 0 ? (currentPage - 1) * itemsPerPage + 1 : 0} to {Math.min(currentPage * itemsPerPage, filteredRecords.length)} of {filteredRecords.length} entries
          </div>
          <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
            {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
              <button
                key={page}
                onClick={() => setCurrentPage(page)}
                style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '8px',
                  border: 'none',
                  background: currentPage === page ? '#4f46e5' : '#f1f5f9',
                  color: currentPage === page ? '#ffffff' : '#475569',
                  fontWeight: 700,
                  fontSize: '0.85rem',
                  cursor: 'pointer'
                }}
              >
                {page}
              </button>
            ))}
            <button
              onClick={() => setCurrentPage((p) => Math.min(p + 1, totalPages))}
              disabled={currentPage === totalPages}
              style={{
                padding: '0.4rem 0.75rem',
                borderRadius: '8px',
                border: 'none',
                background: '#f1f5f9',
                color: currentPage === totalPages ? '#cbd5e1' : '#475569',
                fontWeight: 700,
                fontSize: '0.85rem',
                cursor: currentPage === totalPages ? 'not-allowed' : 'pointer'
              }}
            >
              &gt;
            </button>
          </div>
        </div>
      </div>

      {/* Case Details Modal */}
      {selectedCase && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(15, 23, 42, 0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '1rem' }}>
          <div style={{ background: '#ffffff', borderRadius: '20px', maxWidth: '560px', width: '100%', padding: '1.75rem', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)', position: 'relative' }}>
            <button
              onClick={() => setSelectedCase(null)}
              style={{ position: 'absolute', top: '1.25rem', right: '1.25rem', background: 'none', border: 'none', cursor: 'pointer', color: '#64748b' }}
            >
              <X size={20} />
            </button>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
              <div style={{ background: '#e0e7ff', color: '#4f46e5', width: '42px', height: '42px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800 }}>
                CT
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '1.15rem', color: '#0f172a', fontWeight: 800 }}>Case Details — {selectedCase.case_id}</h3>
                <div style={{ fontSize: '0.8rem', color: '#64748b' }}>Recorded on {selectedCase.date_time} by {selectedCase.username}</div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1.25rem', marginBottom: '1.25rem' }}>
              {selectedCase.image_b64 ? (
                <img
                  src={selectedCase.image_b64}
                  alt={selectedCase.case_id}
                  style={{ width: '120px', height: '120px', borderRadius: '12px', objectFit: 'cover', border: '1px solid #cbd5e1', background: '#0f172a' }}
                />
              ) : (
                <div style={{ width: '120px', height: '120px', borderRadius: '12px', background: '#0f172a', color: '#94a3b8', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700 }}>
                  No Image
                </div>
              )}

              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <div style={{ fontSize: '0.85rem', color: '#475569' }}>
                  <strong>Prediction:</strong>{' '}
                  <span style={{ fontWeight: 800, color: selectedCase.predicted_class === 'Malignant' ? '#ef4444' : selectedCase.predicted_class === 'Benign' ? '#3b82f6' : '#22c55e' }}>
                    {selectedCase.predicted_class} ({(selectedCase.confidence * 100).toFixed(1)}%)
                  </span>
                </div>
                <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
                  <strong>Model:</strong> {selectedCase.model_name} ({selectedCase.model_version})
                </div>
                <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
                  <strong>Doctor Verification:</strong> {selectedCase.verified_by_doctor}
                </div>
                <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
                  <strong>Status:</strong> {selectedCase.verification_status}
                </div>
              </div>
            </div>

            {/* Probability Breakdown */}
            <div style={{ background: '#f8fafc', borderRadius: '12px', padding: '1rem', marginBottom: '1.25rem' }}>
              <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#334155', marginBottom: '0.5rem' }}>Class Probability Distribution</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.5rem', textAlign: 'center' }}>
                <div style={{ background: '#ffffff', padding: '0.5rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Normal</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 800, color: '#16a34a' }}>{((selectedCase.probabilities?.Normal || 0) * 100).toFixed(1)}%</div>
                </div>
                <div style={{ background: '#ffffff', padding: '0.5rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Benign</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 800, color: '#2563eb' }}>{((selectedCase.probabilities?.Benign || 0) * 100).toFixed(1)}%</div>
                </div>
                <div style={{ background: '#ffffff', padding: '0.5rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Malignant</div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 800, color: '#dc2626' }}>{((selectedCase.probabilities?.Malignant || 0) * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setSelectedCase(null)}
                style={{ background: '#4f46e5', color: '#ffffff', border: 'none', padding: '0.5rem 1.25rem', borderRadius: '8px', fontWeight: 700, cursor: 'pointer', fontSize: '0.85rem' }}
              >
                Close Case Details
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
