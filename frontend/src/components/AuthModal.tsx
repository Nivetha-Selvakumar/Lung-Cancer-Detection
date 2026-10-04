import React, { useState } from 'react';
import { UserProfile } from '../types/api';
import { loginUser, registerUser, configureDbPassword } from '../services/api';
import { User, Key, Lock, Mail, Building, Database, X, LogIn, UserPlus } from 'lucide-react';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLoginSuccess: (user: UserProfile) => void;
}

export const AuthModal: React.FC<AuthModalProps> = ({ isOpen, onClose, onLoginSuccess }) => {
  const [mode, setMode] = useState<'login' | 'register' | 'db_config'>('login');
  
  // Login Form
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  
  // Register Form
  const [regUsername, setRegUsername] = useState('');
  const [regEmail, setRegEmail] = useState('');
  const [regPassword, setRegPassword] = useState('');
  const [regFullName, setRegFullName] = useState('');
  const [regRole, setRegRole] = useState('Senior Pulmonologist');
  const [regHospital, setRegHospital] = useState('General Hospital');

  // DB Password Config
  const [dbPassword, setDbPassword] = useState('');

  // Status/Error
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await loginUser(username, password);
      setLoading(false);
      if (res.success && res.user) {
        onLoginSuccess(res.user);
        onClose();
      } else {
        setErrorMsg(res.error || 'Authentication failed');
      }
    } catch (err: any) {
      setLoading(false);
      setErrorMsg(err.message || 'Server login request failed');
    }
  };

  const handleRegisterSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await registerUser({
        username: regUsername,
        email: regEmail,
        password: regPassword,
        full_name: regFullName,
        role: regRole,
        hospital_name: regHospital,
      });
      setLoading(false);
      if (res.success && res.user) {
        onLoginSuccess(res.user);
        onClose();
      } else {
        setErrorMsg(res.error || 'Registration failed');
      }
    } catch (err: any) {
      setLoading(false);
      setErrorMsg(err.message || 'Registration request failed');
    }
  };

  const handleDbConfigSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg(null);
    setSuccessMsg(null);
    try {
      const res = await configureDbPassword(dbPassword);
      setLoading(false);
      if (res.success) {
        setSuccessMsg(res.message || 'MySQL database connected!');
        setTimeout(() => {
          setMode('login');
          setSuccessMsg(null);
        }, 1500);
      } else {
        setErrorMsg(res.error || 'Failed to connect to MySQL database');
      }
    } catch (err: any) {
      setLoading(false);
      setErrorMsg(err.message || 'MySQL configuration request failed');
    }
  };

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(15, 23, 42, 0.75)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '1rem' }}>
      <div style={{ background: '#ffffff', borderRadius: '16px', maxWidth: '460px', width: '100%', padding: '2rem', position: 'relative', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)' }}>
        
        <button
          onClick={onClose}
          style={{ position: 'absolute', top: '1.25rem', right: '1.25rem', background: '#f1f5f9', border: 'none', width: '32px', height: '32px', borderRadius: '50%', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}
        >
          <X size={18} />
        </button>

        {/* Modal Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
          <div style={{ background: '#e0e7ff', color: '#4f46e5', width: '44px', height: '44px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {mode === 'login' ? <LogIn size={22} /> : mode === 'register' ? <UserPlus size={22} /> : <Database size={22} />}
          </div>
          <div>
            <h2 style={{ margin: 0, fontSize: '1.2rem', color: '#0f172a' }}>
              {mode === 'login' ? 'Doctor Portal Login' : mode === 'register' ? 'Register Practitioner Account' : 'Connect MySQL Database'}
            </h2>
            <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
              {mode === 'login' ? 'Access clinical diagnostic system' : mode === 'register' ? 'Create new practitioner credentials' : 'Configure local MySQL-8 root password'}
            </div>
          </div>
        </div>

        {/* Mode Selector Tabs */}
        <div style={{ display: 'flex', background: '#f1f5f9', padding: '0.25rem', borderRadius: '8px', marginBottom: '1.25rem' }}>
          <button
            onClick={() => { setMode('login'); setErrorMsg(null); }}
            style={{ flex: 1, padding: '0.45rem', border: 'none', borderRadius: '6px', background: mode === 'login' ? '#ffffff' : 'transparent', color: mode === 'login' ? '#4f46e5' : '#64748b', fontWeight: 600, fontSize: '0.82rem', cursor: 'pointer' }}
          >
            Login
          </button>
          <button
            onClick={() => { setMode('register'); setErrorMsg(null); }}
            style={{ flex: 1, padding: '0.45rem', border: 'none', borderRadius: '6px', background: mode === 'register' ? '#ffffff' : 'transparent', color: mode === 'register' ? '#4f46e5' : '#64748b', fontWeight: 600, fontSize: '0.82rem', cursor: 'pointer' }}
          >
            Register
          </button>
          <button
            onClick={() => { setMode('db_config'); setErrorMsg(null); }}
            style={{ flex: 1, padding: '0.45rem', border: 'none', borderRadius: '6px', background: mode === 'db_config' ? '#ffffff' : 'transparent', color: mode === 'db_config' ? '#4f46e5' : '#64748b', fontWeight: 600, fontSize: '0.82rem', cursor: 'pointer' }}
          >
            MySQL DB
          </button>
        </div>

        {errorMsg && (
          <div style={{ background: '#fef2f2', border: '1px solid #fecaca', color: '#dc2626', padding: '0.65rem 0.85rem', borderRadius: '8px', fontSize: '0.82rem', marginBottom: '1rem' }}>
            {errorMsg}
          </div>
        )}

        {successMsg && (
          <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', color: '#16a34a', padding: '0.65rem 0.85rem', borderRadius: '8px', fontSize: '0.82rem', marginBottom: '1rem' }}>
            {successMsg}
          </div>
        )}

        {/* Login Form */}
        {mode === 'login' && (
          <form onSubmit={handleLoginSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: '#334155', marginBottom: '0.3rem' }}>Username or Email</label>
              <div style={{ position: 'relative' }}>
                <User size={16} style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }} />
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="doctor"
                  style={{ width: '100%', padding: '0.6rem 0.75rem 0.6rem 2.2rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.88rem' }}
                  required
                />
              </div>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: '#334155', marginBottom: '0.3rem' }}>Password</label>
              <div style={{ position: 'relative' }}>
                <Lock size={16} style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }} />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  style={{ width: '100%', padding: '0.6rem 0.75rem 0.6rem 2.2rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.88rem' }}
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              style={{ marginTop: '0.5rem', background: '#4f46e5', color: '#ffffff', border: 'none', padding: '0.7rem', borderRadius: '8px', fontWeight: 700, fontSize: '0.9rem', cursor: 'pointer', boxShadow: '0 4px 12px rgba(79, 70, 229, 0.25)' }}
            >
              {loading ? 'Authenticating...' : 'Sign In to Diagnostic Suite'}
            </button>
          </form>
        )}

        {/* Register Form */}
        {mode === 'register' && (
          <form onSubmit={handleRegisterSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: '#334155', marginBottom: '0.2rem' }}>Full Name</label>
              <input
                type="text"
                value={regFullName}
                onChange={(e) => setRegFullName(e.target.value)}
                placeholder="Dr. Selvakumar"
                style={{ width: '100%', padding: '0.55rem 0.75rem', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                required
              />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: '#334155', marginBottom: '0.2rem' }}>Username</label>
                <input
                  type="text"
                  value={regUsername}
                  onChange={(e) => setRegUsername(e.target.value)}
                  placeholder="nivetha"
                  style={{ width: '100%', padding: '0.55rem 0.75rem', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                  required
                />
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: '#334155', marginBottom: '0.2rem' }}>Email</label>
                <input
                  type="email"
                  value={regEmail}
                  onChange={(e) => setRegEmail(e.target.value)}
                  placeholder="doctor@hospital.org"
                  style={{ width: '100%', padding: '0.55rem 0.75rem', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                  required
                />
              </div>
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: '#334155', marginBottom: '0.2rem' }}>Password</label>
              <input
                type="password"
                value={regPassword}
                onChange={(e) => setRegPassword(e.target.value)}
                placeholder="••••••••"
                style={{ width: '100%', padding: '0.55rem 0.75rem', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                required
              />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: '#334155', marginBottom: '0.2rem' }}>Role</label>
                <input
                  type="text"
                  value={regRole}
                  onChange={(e) => setRegRole(e.target.value)}
                  style={{ width: '100%', padding: '0.55rem 0.75rem', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: '#334155', marginBottom: '0.2rem' }}>Hospital</label>
                <input
                  type="text"
                  value={regHospital}
                  onChange={(e) => setRegHospital(e.target.value)}
                  style={{ width: '100%', padding: '0.55rem 0.75rem', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
                />
              </div>
            </div>
            <button
              type="submit"
              disabled={loading}
              style={{ marginTop: '0.4rem', background: '#4f46e5', color: '#ffffff', border: 'none', padding: '0.65rem', borderRadius: '8px', fontWeight: 700, fontSize: '0.88rem', cursor: 'pointer' }}
            >
              {loading ? 'Creating Account...' : 'Register Practitioner Account'}
            </button>
          </form>
        )}

        {/* MySQL DB Password Config */}
        {mode === 'db_config' && (
          <form onSubmit={handleDbConfigSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: '#334155', marginBottom: '0.3rem' }}>
                MySQL Root Password (localhost:3306)
              </label>
              <div style={{ position: 'relative' }}>
                <Key size={16} style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }} />
                <input
                  type="password"
                  value={dbPassword}
                  onChange={(e) => setDbPassword(e.target.value)}
                  placeholder="Enter your MySQL password"
                  style={{ width: '100%', padding: '0.65rem 0.75rem 0.65rem 2.2rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.88rem' }}
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              style={{ marginTop: '0.2rem', background: '#4f46e5', color: '#ffffff', border: 'none', padding: '0.7rem', borderRadius: '8px', fontWeight: 700, fontSize: '0.9rem', cursor: 'pointer' }}
            >
              {loading ? 'Testing MySQL Connection...' : 'Connect to MySQL-8 DB'}
            </button>
          </form>
        )}
      </div>
    </div>
  );
};
