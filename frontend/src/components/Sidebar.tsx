import React from 'react';
import { Home, Scan, Database, Cpu, BarChart3, History, User, LogOut, Activity } from 'lucide-react';
import { UserProfile } from '../types/api';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  currentUser: UserProfile | null;
  onLogout: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab, currentUser, onLogout }) => {
  const menuItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'predict', label: 'Predict', icon: Scan },
    { id: 'dataset', label: 'Dataset', icon: Database },
    { id: 'dashboard', label: 'Model Performance', icon: BarChart3 },
    { id: 'history', label: 'History', icon: History },
    { id: 'profile', label: 'Profile', icon: User },
  ];

  return (
    <aside style={{
      width: '240px',
      background: '#ffffff',
      borderRight: '1px solid #e2e8f0',
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      position: 'fixed',
      left: 0,
      top: 0,
      zIndex: 90
    }}>
      {/* Brand Header */}
      <div style={{
        padding: '1.25rem 1rem',
        borderBottom: '1px solid #f1f5f9',
        display: 'flex',
        alignItems: 'center',
        gap: '0.6rem'
      }}>
        <div style={{
          width: '34px',
          height: '34px',
          background: 'linear-gradient(135deg, #4f46e5, #6366f1)',
          color: '#ffffff',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <Activity size={20} />
        </div>
        <div>
          <div style={{ fontWeight: 800, fontSize: '0.92rem', color: '#0f172a', lineHeight: 1.2 }}>
            Lung Cancer CT Analysis
          </div>
        </div>
      </div>

      {/* Navigation Items */}
      <nav style={{ padding: '1rem 0.75rem', display: 'flex', flexDirection: 'column', gap: '0.25rem', flex: 1 }}>
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem',
                padding: '0.65rem 0.85rem',
                borderRadius: '8px',
                border: 'none',
                background: isActive ? '#4f46e5' : 'transparent',
                color: isActive ? '#ffffff' : '#475569',
                fontWeight: isActive ? 700 : 500,
                fontSize: '0.86rem',
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'all 0.2s'
              }}
            >
              <Icon size={18} style={{ color: isActive ? '#ffffff' : '#64748b' }} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Logout Button */}
      <div style={{ padding: '1rem 0.75rem', borderTop: '1px solid #f1f5f9' }}>
        <button
          onClick={onLogout}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            padding: '0.6rem 0.85rem',
            width: '100%',
            borderRadius: '8px',
            border: '1px solid #fee2e2',
            background: '#fef2f2',
            color: '#dc2626',
            fontWeight: 600,
            fontSize: '0.84rem',
            cursor: 'pointer'
          }}
        >
          <LogOut size={16} /> Logout
        </button>
      </div>
    </aside>
  );
};
