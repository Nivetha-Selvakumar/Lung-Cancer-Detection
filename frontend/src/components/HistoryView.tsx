import React, { useEffect, useState } from 'react';
import { History, Eye, CheckCircle2, Clock, Key, Database, RefreshCw } from 'lucide-react';

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
}

export const HistoryView: React.FC = () => {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [dbStatus, setDbStatus] = useState<{ engine?: string; connected?: boolean; message?: string } | null>(null);
  
  // MySQL Password modal state
  const [showDbModal, setShowDbModal] = useState(false);
  const [mysqlPassword, setMysqlPassword] = useState('');
  const [dbConfigMessage, setDbConfigMessage] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);

  const fetchHistoryAndDbStatus = () => {
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

    fetch('/api/auth/db-status')
      .then((res) => res.json())
      .then((data) => setDbStatus(data))
      .catch((err) => console.error('Failed to load DB status:', err));
  };

  useEffect(() => {
    fetchHistoryAndDbStatus();
  }, []);

  const handleStatusChange = (id: number, currentStatus: string) => {
    const nextStatus = currentStatus === 'Verified' ? 'Reviewed' : currentStatus === 'Reviewed' ? 'Pending' : 'Verified';
    fetch('/api/history/update-status', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, status: nextStatus })
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.success) {
          setRecords((prev) =>
            prev.map((r) => (r.id === id ? { ...r, verification_status: nextStatus } : r))
          );
        }
      });
  };

  const handleConnectMysql = (e: React.FormEvent) => {
    e.preventDefault();
    setIsConnecting(true);
    setDbConfigMessage(null);

    fetch('/api/auth/configure-db', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password: mysqlPassword })
    })
      .then((res) => res.json())
      .then((data) => {
        setIsConnecting(false);
        if (data.success) {
          setDbConfigMessage('✅ ' + data.message);
          fetchHistoryAndDbStatus();
          setTimeout(() => setShowDbModal(false), 1500);
        } else {
          setDbConfigMessage('❌ ' + (data.error || 'Failed to connect to MySQL database.'));
        }
      })
      .catch(() => {
        setIsConnecting(false);
        setDbConfigMessage('❌ Connection request failed.');
      });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-title">Prediction History & Database Audit Log</h1>
          <p className="page-subtitle">
            Dynamic database logs of CT scan predictions, probabilities, and clinical verification statuses.
          </p>
        </div>
        <button
          onClick={() => setShowDbModal(true)}
          style={{
            background: '#4f46e5',
            color: '#ffffff',
            border: 'none',
            padding: '0.55rem 1.1rem',
            borderRadius: '8px',
            fontWeight: 600,
            fontSize: '0.85rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            boxShadow: '0 4px 12px rgba(79, 70, 229, 0.25)'
          }}
        >
          <Database size={16} /> Configure MySQL DB
        </button>
      </div>

      {/* Database Connection Status Bar */}
      <div className="card" style={{ background: '#f8fafc', padding: '0.85rem 1.25rem', borderLeft: dbStatus?.engine === 'MySQL' ? '4px solid #16a34a' : '4px solid #d97706' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Database size={20} style={{ color: dbStatus?.engine === 'MySQL' ? '#16a34a' : '#d97706' }} />
            <div>
              <strong style={{ fontSize: '0.9rem', color: '#0f172a' }}>Database Engine: {dbStatus?.engine || 'Checking...'}</strong>
              <div style={{ fontSize: '0.8rem', color: '#64748b' }}>{dbStatus?.message || 'Database status monitoring active.'}</div>
            </div>
          </div>
          <button
            onClick={fetchHistoryAndDbStatus}
            style={{ background: '#e2e8f0', border: 'none', padding: '0.4rem 0.75rem', borderRadius: '6px', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.35rem', color: '#334155' }}
          >
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {/* History Table */}
      <div className="card">
        <div className="card-title">
          <History size={20} /> Dynamic Prediction History Records ({records.length})
        </div>

        {loading ? (
          <div style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>Loading records from database...</div>
        ) : records.length === 0 ? (
          <div style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>
            No prediction records saved yet. Perform a CT scan prediction to add entries to the database.
          </div>
        ) : (
          <div className="data-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Case ID</th>
                  <th>Date & Time</th>
                  <th>User</th>
                  <th>Model / Ver</th>
                  <th>Prediction</th>
                  <th>Probabilities</th>
                  <th>Verification Status</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {records.map((rec) => {
                  const normPct = (rec.probabilities?.Normal * 100 || 0).toFixed(1);
                  const benignPct = (rec.probabilities?.Benign * 100 || 0).toFixed(1);
                  const maligPct = (rec.probabilities?.Malignant * 100 || 0).toFixed(1);
                  const status = rec.verification_status || 'Pending';

                  return (
                    <tr key={rec.id}>
                      <td><strong style={{ color: '#4f46e5' }}>{rec.case_id}</strong></td>
                      <td style={{ color: '#64748b', fontSize: '0.82rem' }}>{rec.date_time}</td>
                      <td>{rec.username || 'Doctor'}</td>
                      <td style={{ fontSize: '0.82rem', color: '#64748b' }}>{rec.model_name} ({rec.model_version})</td>
                      <td>
                        <span style={{
                          fontWeight: 700,
                          color: rec.predicted_class === 'Malignant' ? '#dc2626' : rec.predicted_class === 'Benign' ? '#d97706' : '#16a34a',
                          background: rec.predicted_class === 'Malignant' ? '#fef2f2' : rec.predicted_class === 'Benign' ? '#fffbe6' : '#f0fdf4',
                          padding: '0.2rem 0.6rem',
                          borderRadius: '4px'
                        }}>
                          {rec.predicted_class} ({(rec.confidence * 100).toFixed(1)}%)
                        </span>
                      </td>
                      <td style={{ fontSize: '0.78rem', color: '#475569' }}>
                        N: {normPct}% | B: {benignPct}% | M: {maligPct}%
                      </td>
                      <td>
                        <button
                          onClick={() => handleStatusChange(rec.id, status)}
                          title="Click to toggle status"
                          style={{
                            fontSize: '0.75rem',
                            fontWeight: 600,
                            color: status === 'Verified' ? '#16a34a' : status === 'Reviewed' ? '#2563eb' : '#d97706',
                            background: status === 'Verified' ? '#f0fdf4' : status === 'Reviewed' ? '#eff6ff' : '#fefce8',
                            border: `1px solid ${status === 'Verified' ? '#bbf7d0' : status === 'Reviewed' ? '#bfdbfe' : '#fef08a'}`,
                            padding: '0.25rem 0.6rem',
                            borderRadius: '12px',
                            cursor: 'pointer',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.25rem'
                          }}
                        >
                          {status === 'Verified' ? <CheckCircle2 size={12} /> : status === 'Reviewed' ? <Eye size={12} /> : <Clock size={12} />}
                          {status}
                        </button>
                      </td>
                      <td>
                        <button style={{ background: '#f1f5f9', border: '1px solid #cbd5e1', color: '#475569', padding: '0.3rem 0.6rem', borderRadius: '4px', fontSize: '0.78rem', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                          <Eye size={13} /> View Log
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* MySQL Password Modal */}
      {showDbModal && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(15, 23, 42, 0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '1rem' }}>
          <div style={{ background: '#ffffff', borderRadius: '16px', maxWidth: '440px', width: '100%', padding: '1.75rem', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
              <div style={{ background: '#e0e7ff', color: '#4f46e5', width: '40px', height: '40px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Key size={20} />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '1.1rem', color: '#0f172a' }}>Connect MySQL Database</h3>
                <div style={{ fontSize: '0.8rem', color: '#64748b' }}>Enter your local MySQL root password (MySQL-8 at localhost:3306)</div>
              </div>
            </div>

            <form onSubmit={handleConnectMysql} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, color: '#334155', marginBottom: '0.4rem' }}>
                  MySQL Root Password
                </label>
                <input
                  type="password"
                  value={mysqlPassword}
                  onChange={(e) => setMysqlPassword(e.target.value)}
                  placeholder="Enter MySQL password"
                  style={{ width: '100%', padding: '0.65rem 0.85rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.9rem', outline: 'none' }}
                  required
                />
              </div>

              {dbConfigMessage && (
                <div style={{ fontSize: '0.82rem', padding: '0.6rem 0.75rem', borderRadius: '6px', background: dbConfigMessage.startsWith('✅') ? '#f0fdf4' : '#fef2f2', color: dbConfigMessage.startsWith('✅') ? '#16a34a' : '#dc2626', border: `1px solid ${dbConfigMessage.startsWith('✅') ? '#bbf7d0' : '#fecaca'}` }}>
                  {dbConfigMessage}
                </div>
              )}

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '0.5rem' }}>
                <button
                  type="button"
                  onClick={() => setShowDbModal(false)}
                  style={{ background: '#f1f5f9', border: 'none', padding: '0.55rem 1rem', borderRadius: '8px', fontWeight: 600, color: '#64748b', cursor: 'pointer' }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isConnecting}
                  style={{ background: '#4f46e5', border: 'none', padding: '0.55rem 1.25rem', borderRadius: '8px', fontWeight: 600, color: '#ffffff', cursor: 'pointer', boxShadow: '0 4px 12px rgba(79, 70, 229, 0.25)' }}
                >
                  {isConnecting ? 'Connecting...' : 'Connect DB'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
