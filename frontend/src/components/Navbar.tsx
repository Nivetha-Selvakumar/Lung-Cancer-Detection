import React from 'react';
import { Activity, Scan, BarChart3, Cpu } from 'lucide-react';
import { HealthResponse } from '../types/api';

interface NavbarProps {
  activeTab: 'diagnostics' | 'dashboard' | 'pipeline';
  setActiveTab: (tab: 'diagnostics' | 'dashboard' | 'pipeline') => void;
  health: HealthResponse | null;
  healthError: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  health,
  healthError,
}) => {
  let statusText = 'Checking Backend...';
  let isOnline = false;

  if (healthError) {
    statusText = 'Backend Offline (Port 5000)';
    isOnline = false;
  } else if (health) {
    isOnline = true;
    if (health.convnext_loaded && health.xgb_loaded) {
      statusText = 'Backend Connected • Models Loaded';
    } else {
      statusText = 'Backend Running • Training Models...';
    }
  }

  return (
    <header className="navbar">
      <div className="brand">
        <div className="brand-icon">
          <Activity size={24} />
        </div>
        <div>
          <span className="brand-title">LUNG-AI DIAGNOSTICS</span>
        </div>
        <span className="brand-tag">v2.0 Tri-Aspect</span>
      </div>

      <nav className="nav-links">
        <button
          className={`nav-btn ${activeTab === 'diagnostics' ? 'active' : ''}`}
          onClick={() => setActiveTab('diagnostics')}
        >
          <Scan size={18} /> Live Diagnostics
        </button>
        <button
          className={`nav-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          <BarChart3 size={18} /> Performance Metrics
        </button>
        <button
          className={`nav-btn ${activeTab === 'pipeline' ? 'active' : ''}`}
          onClick={() => setActiveTab('pipeline')}
        >
          <Cpu size={18} /> Architecture Specs
        </button>
      </nav>

      <div className="server-status">
        <div className={`status-dot ${isOnline ? '' : 'offline'}`} />
        <span>{statusText}</span>
      </div>
    </header>
  );
};
