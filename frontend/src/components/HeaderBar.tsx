import React from 'react';
import { Bell, Database, Search } from 'lucide-react';
import { UserProfile, HealthResponse } from '../types/api';

interface HeaderBarProps {
  currentUser: UserProfile | null;
  health: HealthResponse | null;
  healthError: boolean;
  onOpenAuthModal: () => void;
}

export const HeaderBar: React.FC<HeaderBarProps> = ({ currentUser, health, healthError, onOpenAuthModal }) => {
  const isOnline = !healthError && health?.convnext_loaded;
  const dbEngine = health?.db?.engine || 'MySQL DB';

  const userInitial = currentUser?.full_name ? currentUser.full_name.charAt(0).toUpperCase() : 'N';
  const userName = currentUser?.full_name || 'Nivetha S.';
  const userRole = currentUser?.role || 'Research Scholar';

  return (
    <header style={{
      height: '64px',
      background: '#ffffff',
      borderBottom: '1px solid #e2e8f0',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 1.5rem',
      position: 'fixed',
      top: 0,
      right: 0,
      left: '240px',
      zIndex: 80
    }}>
      {/* Search Input */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: '#f8fafc', padding: '0.45rem 0.85rem', borderRadius: '20px', border: '1px solid #e2e8f0', width: '280px' }}>
        <Search size={16} style={{ color: '#94a3b8' }} />
        <input
          type="text"
          placeholder="Search CT scans, cases, inference..."
          style={{ border: 'none', background: 'transparent', outline: 'none', fontSize: '0.82rem', width: '100%', color: '#0f172a' }}
        />
      </div>

      {/* Right Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {/* DB Engine Status Pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          fontSize: '0.76rem',
          background: '#f1f5f9',
          padding: '0.35rem 0.7rem',
          borderRadius: '20px',
          color: '#475569',
          fontWeight: 600
        }}>
          <Database size={13} style={{ color: '#4f46e5' }} />
          <span>{dbEngine}</span>
        </div>

        {/* Backend Online Status Pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          fontSize: '0.76rem',
          background: isOnline ? '#f0fdf4' : '#fef2f2',
          border: `1px solid ${isOnline ? '#bbf7d0' : '#fecaca'}`,
          padding: '0.35rem 0.75rem',
          borderRadius: '20px',
          color: isOnline ? '#15803d' : '#dc2626',
          fontWeight: 600
        }}>
          <div style={{ width: '7px', height: '7px', borderRadius: '50%', background: isOnline ? '#22c55e' : '#ef4444' }} />
          <span>{isOnline ? 'Backend Online' : 'Backend Offline'}</span>
        </div>

        {/* Notification Bell */}
        <button
          onClick={() => alert('No new notifications')}
          style={{ background: '#f8fafc', border: '1px solid #e2e8f0', width: '36px', height: '36px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', color: '#64748b' }}
        >
          <Bell size={18} />
        </button>

        {/* User Profile Avatar Pill */}
        <div
          onClick={onOpenAuthModal}
          style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', cursor: 'pointer', background: '#f8fafc', padding: '0.25rem 0.6rem 0.25rem 0.25rem', borderRadius: '20px', border: '1px solid #e2e8f0' }}
        >
          <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: '#4f46e5', color: '#ffffff', fontWeight: 700, fontSize: '0.85rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {userInitial}
          </div>
          <div style={{ textAlign: 'left' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#0f172a', lineHeight: 1.2 }}>{userName}</div>
            <div style={{ fontSize: '0.7rem', color: '#64748b' }}>{userRole}</div>
          </div>
        </div>
      </div>
    </header>
  );
};
