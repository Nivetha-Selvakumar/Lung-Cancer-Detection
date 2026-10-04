import React from 'react';
import { Activity, Scan, BarChart3, Cpu, User, LogIn, LogOut, Database, History } from 'lucide-react';
import { HealthResponse, UserProfile } from '../types/api';

interface NavbarProps {
  activeTab: 'diagnostics' | 'dashboard' | 'history' | 'pipeline';
  setActiveTab: (tab: 'diagnostics' | 'dashboard' | 'history' | 'pipeline') => void;
  health: HealthResponse | null;
  healthError: boolean;
  currentUser: UserProfile | null;
  onOpenAuthModal: () => void;
  onLogout: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  health,
  healthError,
  currentUser,
  onOpenAuthModal,
  onLogout,
}) => {
  let statusText = 'Checking System...';
  let isOnline = false;

  if (healthError) {
    statusText = 'Backend Offline';
    isOnline = false;
  } else if (health) {
    isOnline = true;
    if (health.convnext_loaded) {
      statusText = 'AI Engine Connected';
    } else {
      statusText = 'System Initializing...';
    }
  }

  const dbEngine = health?.db?.engine || 'MySQL DB';

  return (
    <header className="navbar">
      <div className="brand">
        <div className="brand-icon">
          <Activity size={24} />
        </div>
        <div>
          <span className="brand-title">LUNG-AI CLINICAL DIAGNOSTICS</span>
        </div>
        <span className="brand-tag">v2.0 Clinical Suite</span>
      </div>

      <nav className="nav-links">
        <button
          className={`nav-btn ${activeTab === 'diagnostics' ? 'active' : ''}`}
          onClick={() => setActiveTab('diagnostics')}
        >
          <Scan size={18} /> Diagnostics & Analysis
        </button>
        <button
          className={`nav-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          <BarChart3 size={18} /> Performance Metrics
        </button>
        <button
          className={`nav-btn ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          <History size={18} /> Prediction History
        </button>
        <button
          className={`nav-btn ${activeTab === 'pipeline' ? 'active' : ''}`}
          onClick={() => setActiveTab('pipeline')}
        >
          <Cpu size={18} /> System Specifications
        </button>
      </nav>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        {/* DB Engine Status Pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          fontSize: '0.75rem',
          background: 'rgba(30, 41, 59, 0.7)',
          padding: '0.35rem 0.65rem',
          borderRadius: '20px',
          border: '1px solid var(--border-color)',
          color: 'var(--text-muted)'
        }} title="MySQL User Details Database Connection">
          <Database size={13} style={{ color: 'var(--primary-cyan)' }} />
          <span>{dbEngine}</span>
        </div>

        {/* Backend Online Pill */}
        <div className="server-status">
          <div className={`status-dot ${isOnline ? '' : 'offline'}`} />
          <span>{statusText}</span>
        </div>

        {/* Doctor Login / Account Control */}
        {currentUser ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              background: 'rgba(56, 189, 248, 0.12)',
              border: '1px solid rgba(56, 189, 248, 0.3)',
              padding: '0.35rem 0.75rem',
              borderRadius: '20px',
              fontSize: '0.8rem',
              color: '#f8fafc'
            }}>
              <User size={14} style={{ color: 'var(--primary-cyan)' }} />
              <span style={{ fontWeight: 600 }}>{currentUser.full_name || currentUser.username}</span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>({currentUser.role || 'Doctor'})</span>
            </div>
            <button
              onClick={onLogout}
              style={{
                background: 'rgba(239, 68, 68, 0.12)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                color: '#fca5a5',
                padding: '0.4rem 0.6rem',
                borderRadius: 'var(--radius-sm)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.3rem',
                fontSize: '0.8rem'
              }}
              title="Logout doctor session"
            >
              <LogOut size={14} /> Logout
            </button>
          </div>
        ) : (
          <button
            onClick={onOpenAuthModal}
            style={{
              background: 'linear-gradient(135deg, var(--primary-cyan), var(--primary-blue))',
              border: 'none',
              color: '#0f172a',
              padding: '0.45rem 0.9rem',
              borderRadius: 'var(--radius-sm)',
              fontWeight: 700,
              fontSize: '0.82rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              boxShadow: '0 4px 12px rgba(56, 189, 248, 0.25)'
            }}
          >
            <LogIn size={15} /> Doctor Login
          </button>
        )}
      </div>
    </header>
  );
};
