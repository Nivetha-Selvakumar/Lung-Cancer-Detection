import React, { useState } from 'react';
import { User, Key, Database, Shield, Hospital, Mail } from 'lucide-react';
import { UserProfile } from '../types/api';

interface ProfileViewProps {
  currentUser: UserProfile | null;
}

export const ProfileView: React.FC<ProfileViewProps> = ({ currentUser }) => {
  const [mysqlPassword, setMysqlPassword] = useState('');
  const [msg, setMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleConnectDb = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMsg(null);

    fetch('/api/auth/configure-db', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password: mysqlPassword })
    })
      .then((res) => res.json())
      .then((data) => {
        setLoading(false);
        if (data.success) {
          setMsg('✅ Successfully connected to local MySQL database (localhost:3306 / lung_cancer_db)!');
        } else {
          setMsg('❌ ' + (data.error || 'Failed to connect to MySQL database.'));
        }
      })
      .catch(() => {
        setLoading(false);
        setMsg('❌ Connection request failed.');
      });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="page-header">
        <h1 className="page-title">Practitioner Profile & Database Configuration</h1>
        <p className="page-subtitle">
          Manage user credentials, medical role credentials, and local MySQL database connection settings.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        {/* User Credentials Card */}
        <div className="card" style={{ margin: 0 }}>
          <div className="card-title">
            <User size={20} /> Medical Practitioner Details
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <User size={18} style={{ color: '#4f46e5' }} />
              <div>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Full Name</div>
                <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#0f172a' }}>{currentUser?.full_name || 'Nivetha S.'}</div>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Shield size={18} style={{ color: '#4f46e5' }} />
              <div>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Role & Title</div>
                <div style={{ fontSize: '0.92rem', fontWeight: 600, color: '#0f172a' }}>{currentUser?.role || 'Research Scholar'}</div>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Mail size={18} style={{ color: '#4f46e5' }} />
              <div>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Email Address</div>
                <div style={{ fontSize: '0.92rem', fontWeight: 600, color: '#0f172a' }}>{currentUser?.email || 'doctor@hospital.org'}</div>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Hospital size={18} style={{ color: '#4f46e5' }} />
              <div>
                <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Hospital Affiliation</div>
                <div style={{ fontSize: '0.92rem', fontWeight: 600, color: '#0f172a' }}>{currentUser?.hospital_name || 'General Hospital'}</div>
              </div>
            </div>
          </div>
        </div>

        {/* MySQL Database Configuration Card */}
        <div className="card" style={{ margin: 0 }}>
          <div className="card-title">
            <Database size={20} /> MySQL Database Credentials
          </div>

          <form onSubmit={handleConnectDb} style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, color: '#334155', marginBottom: '0.35rem' }}>
                MySQL Root Password (localhost:3306)
              </label>
              <div style={{ position: 'relative' }}>
                <Key size={16} style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }} />
                <input
                  type="password"
                  value={mysqlPassword}
                  onChange={(e) => setMysqlPassword(e.target.value)}
                  placeholder="Enter MySQL password"
                  style={{ width: '100%', padding: '0.65rem 0.75rem 0.65rem 2.2rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.88rem' }}
                  required
                />
              </div>
            </div>

            {msg && (
              <div style={{ fontSize: '0.82rem', padding: '0.65rem 0.85rem', borderRadius: '8px', background: msg.startsWith('✅') ? '#f0fdf4' : '#fef2f2', color: msg.startsWith('✅') ? '#16a34a' : '#dc2626', border: `1px solid ${msg.startsWith('✅') ? '#bbf7d0' : '#fecaca'}` }}>
                {msg}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              style={{ background: '#4f46e5', color: '#ffffff', border: 'none', padding: '0.7rem', borderRadius: '8px', fontWeight: 700, fontSize: '0.88rem', cursor: 'pointer', boxShadow: '0 4px 12px rgba(79, 70, 229, 0.25)' }}
            >
              {loading ? 'Connecting to MySQL...' : 'Connect to MySQL-8 DB'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};
