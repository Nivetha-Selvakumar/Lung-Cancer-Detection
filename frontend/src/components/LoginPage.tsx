import React, { useState } from 'react';
import { UserProfile } from '../types/api';
import { loginUser, requestForgotPassword, resetPassword } from '../services/api';
import { Activity, Lock, Mail, Eye, EyeOff, Sparkles, ArrowLeft, CheckCircle2, ShieldCheck, KeyRound } from 'lucide-react';
import loginArt from '../img/login.png';

interface LoginPageProps {
  onLoginSuccess: (user: UserProfile, token: string) => void;
}

export const LoginPage: React.FC<LoginPageProps> = ({ onLoginSuccess }) => {
  const [mode, setMode] = useState<'login' | 'forgot' | 'reset'>('login');

  // Login form state
  const [emailOrUsername, setEmailOrUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);

  // Forgot / Reset password state
  const [forgotIdentity, setForgotIdentity] = useState('');
  const [resetToken, setResetToken] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showNewPassword, setShowNewPassword] = useState(false);

  // UI status
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    try {
      const res = await loginUser(emailOrUsername.trim(), password);
      setLoading(false);

      if (res.success && res.user && res.token) {
        onLoginSuccess(res.user, res.token);
      } else {
        setErrorMsg(res.error || 'Authentication failed. Please check credentials.');
      }
    } catch (err: any) {
      setLoading(false);
      setErrorMsg(err.message || 'Unable to connect to login authentication service.');
    }
  };

  const handleForgotPasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    try {
      const res = await requestForgotPassword(forgotIdentity.trim());
      setLoading(false);

      if (res.success) {
        setSuccessMsg(res.message || 'Password reset request authorized.');
        if (res.reset_token) {
          setResetToken(res.reset_token);
        }
        setTimeout(() => {
          setMode('reset');
        }, 1200);
      } else {
        setErrorMsg(res.error || 'User account not found.');
      }
    } catch (err: any) {
      setLoading(false);
      setErrorMsg(err.message || 'Password reset request failed.');
    }
  };

  const handleResetPasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (newPassword !== confirmPassword) {
      setErrorMsg('Passwords do not match. Please re-enter.');
      return;
    }

    setLoading(true);
    setErrorMsg(null);
    setSuccessMsg(null);

    try {
      const res = await resetPassword(forgotIdentity.trim() || emailOrUsername.trim(), resetToken, newPassword);
      setLoading(false);

      if (res.success) {
        setSuccessMsg(res.message || 'Password updated successfully! Redirecting to login...');
        setTimeout(() => {
          setMode('login');
          setPassword('');
          setNewPassword('');
          setConfirmPassword('');
          setSuccessMsg(null);
        }, 1800);
      } else {
        setErrorMsg(res.error || 'Password reset failed.');
      }
    } catch (err: any) {
      setLoading(false);
      setErrorMsg(err.message || 'Failed to update password.');
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      height: '100vh',
      width: '100vw',
      display: 'grid',
      gridTemplateColumns: 'minmax(420px, 48%) 1fr',
      overflow: 'hidden',
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    }}>
      
      {/* Left Form Panel - Full Height White Section */}
      <div style={{
        background: '#ffffff',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        padding: '3.5rem 4.5rem',
        boxSizing: 'border-box',
        height: '100vh',
        overflowY: 'auto'
      }}>
        <div style={{ maxWidth: '440px', width: '100%', margin: 'auto 0' }}>
          
          {/* Logo and App Title */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2.5rem' }}>
            <div style={{
              width: '52px',
              height: '52px',
              borderRadius: '16px',
              background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#ffffff',
              boxShadow: '0 10px 20px -4px rgba(79, 70, 229, 0.4)',
              flexShrink: 0
            }}>
              <Activity size={30} />
            </div>
            <div>
              <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: 800, color: '#0f172a', letterSpacing: '-0.025em' }}>
                Lung Cancer CT Analysis
              </h1>
              <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.86rem', color: '#64748b', lineHeight: 1.4 }}>
                AI-assisted preliminary classification of lung CT scans for clinical decision support.
              </p>
            </div>
          </div>

          {/* Notification Error/Success Alert Banners */}
          {errorMsg && (
            <div style={{
              background: '#fef2f2',
              border: '1px solid #fecaca',
              color: '#dc2626',
              padding: '0.85rem 1.1rem',
              borderRadius: '12px',
              fontSize: '0.86rem',
              fontWeight: 500,
              marginBottom: '1.5rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              <span>{errorMsg}</span>
            </div>
          )}

          {successMsg && (
            <div style={{
              background: '#f0fdf4',
              border: '1px solid #bbf7d0',
              color: '#16a34a',
              padding: '0.85rem 1.1rem',
              borderRadius: '12px',
              fontSize: '0.86rem',
              fontWeight: 500,
              marginBottom: '1.5rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              <CheckCircle2 size={18} />
              <span>{successMsg}</span>
            </div>
          )}

          {/* MODE 1: LOGIN FORM */}
          {mode === 'login' && (
            <form onSubmit={handleLoginSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.35rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: '0.45rem' }}>
                  Email address
                </label>
                <div style={{ position: 'relative' }}>
                  <Mail size={19} style={{ position: 'absolute', left: '0.9rem', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }} />
                  <input
                    type="text"
                    value={emailOrUsername}
                    onChange={(e) => setEmailOrUsername(e.target.value)}
                    placeholder="name@example.com"
                    style={{
                      width: '100%',
                      padding: '0.82rem 0.9rem 0.82rem 2.6rem',
                      borderRadius: '12px',
                      border: '1px solid #cbd5e1',
                      fontSize: '0.92rem',
                      outline: 'none',
                      transition: 'border 0.2s',
                      boxSizing: 'border-box'
                    }}
                    required
                  />
                </div>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: '0.45rem' }}>
                  Password
                </label>
                <div style={{ position: 'relative' }}>
                  <Lock size={19} style={{ position: 'absolute', left: '0.9rem', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }} />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter your password"
                    style={{
                      width: '100%',
                      padding: '0.82rem 2.6rem 0.82rem 2.6rem',
                      borderRadius: '12px',
                      border: '1px solid #cbd5e1',
                      fontSize: '0.92rem',
                      outline: 'none',
                      transition: 'border 0.2s',
                      boxSizing: 'border-box'
                    }}
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    style={{
                      position: 'absolute',
                      right: '0.9rem',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      background: 'none',
                      border: 'none',
                      color: '#94a3b8',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                  >
                    {showPassword ? <EyeOff size={19} /> : <Eye size={19} />}
                  </button>
                </div>
              </div>

              {/* Checkbox & Forgot Password Link */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.85rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: '#475569', cursor: 'pointer', userSelect: 'none' }}>
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                    style={{ width: '17px', height: '17px', accentColor: '#4f46e5', borderRadius: '4px' }}
                  />
                  <span>Remember me</span>
                </label>
                <button
                  type="button"
                  onClick={() => { setMode('forgot'); setForgotIdentity(emailOrUsername); setErrorMsg(null); setSuccessMsg(null); }}
                  style={{ background: 'none', border: 'none', color: '#4f46e5', fontWeight: 600, cursor: 'pointer', fontSize: '0.85rem' }}
                >
                  Forgot password?
                </button>
              </div>

              {/* Sign In Button */}
              <button
                type="submit"
                disabled={loading}
                style={{
                  width: '100%',
                  padding: '0.9rem',
                  borderRadius: '12px',
                  border: 'none',
                  background: 'linear-gradient(135deg, #4f46e5 0%, #6366f1 100%)',
                  color: '#ffffff',
                  fontSize: '0.98rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                  boxShadow: '0 4px 16px rgba(79, 70, 229, 0.35)',
                  transition: 'all 0.2s ease',
                  marginTop: '0.4rem'
                }}
              >
                {loading ? 'Authenticating Doctor Account...' : 'Sign In'}
              </button>
            </form>
          )}

          {/* MODE 2: FORGOT PASSWORD */}
          {mode === 'forgot' && (
            <form onSubmit={handleForgotPasswordSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#4f46e5', marginBottom: '0.2rem' }}>
                <KeyRound size={22} />
                <span style={{ fontWeight: 700, fontSize: '1.05rem' }}>Reset Password</span>
              </div>
              <p style={{ fontSize: '0.86rem', color: '#64748b', margin: 0, lineHeight: 1.5 }}>
                Enter your registered email address or username. We will verify your account and generate a secure password reset token.
              </p>

              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: '0.45rem' }}>
                  Registered Email or Username
                </label>
                <div style={{ position: 'relative' }}>
                  <Mail size={19} style={{ position: 'absolute', left: '0.9rem', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }} />
                  <input
                    type="text"
                    value={forgotIdentity}
                    onChange={(e) => setForgotIdentity(e.target.value)}
                    placeholder="nivethaselvakumar23@gmail.com"
                    style={{
                      width: '100%',
                      padding: '0.82rem 0.9rem 0.82rem 2.6rem',
                      borderRadius: '12px',
                      border: '1px solid #cbd5e1',
                      fontSize: '0.92rem',
                      outline: 'none',
                      boxSizing: 'border-box'
                    }}
                    required
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                style={{
                  width: '100%',
                  padding: '0.9rem',
                  borderRadius: '12px',
                  border: 'none',
                  background: 'linear-gradient(135deg, #4f46e5 0%, #6366f1 100%)',
                  color: '#ffffff',
                  fontSize: '0.98rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                  boxShadow: '0 4px 16px rgba(79, 70, 229, 0.35)',
                  marginTop: '0.4rem'
                }}
              >
                {loading ? 'Verifying Account...' : 'Generate Password Reset Token'}
              </button>

              <button
                type="button"
                onClick={() => { setMode('login'); setErrorMsg(null); setSuccessMsg(null); }}
                style={{ background: 'none', border: 'none', color: '#64748b', fontSize: '0.86rem', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem', marginTop: '0.5rem' }}
              >
                <ArrowLeft size={17} /> Back to Sign In
              </button>
            </form>
          )}

          {/* MODE 3: RESET PASSWORD FORM */}
          {mode === 'reset' && (
            <form onSubmit={handleResetPasswordSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#4f46e5', marginBottom: '0.2rem' }}>
                <ShieldCheck size={22} />
                <span style={{ fontWeight: 700, fontSize: '1.05rem' }}>Set New Password</span>
              </div>
              <p style={{ fontSize: '0.84rem', color: '#64748b', margin: 0 }}>
                Authorized reset token generated for <strong style={{ color: '#0f172a' }}>{forgotIdentity}</strong>.
              </p>

              <div>
                <label style={{ display: 'block', fontSize: '0.84rem', fontWeight: 600, color: '#334155', marginBottom: '0.35rem' }}>
                  Reset Code / Token
                </label>
                <input
                  type="text"
                  value={resetToken}
                  onChange={(e) => setResetToken(e.target.value)}
                  placeholder="RESET-..."
                  style={{ width: '100%', padding: '0.72rem 0.9rem', borderRadius: '10px', border: '1px solid #cbd5e1', fontSize: '0.88rem', boxSizing: 'border-box' }}
                  required
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.84rem', fontWeight: 600, color: '#334155', marginBottom: '0.35rem' }}>
                  New Password
                </label>
                <div style={{ position: 'relative' }}>
                  <Lock size={18} style={{ position: 'absolute', left: '0.8rem', top: '50%', transform: 'translateY(-50%)', color: '#94a3b8' }} />
                  <input
                    type={showNewPassword ? 'text' : 'password'}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="Enter new password"
                    style={{ width: '100%', padding: '0.72rem 2.4rem 0.72rem 2.4rem', borderRadius: '10px', border: '1px solid #cbd5e1', fontSize: '0.88rem', boxSizing: 'border-box' }}
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowNewPassword(!showNewPassword)}
                    style={{ position: 'absolute', right: '0.8rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
                  >
                    {showNewPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.84rem', fontWeight: 600, color: '#334155', marginBottom: '0.35rem' }}>
                  Confirm New Password
                </label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Re-enter new password"
                  style={{ width: '100%', padding: '0.72rem 0.9rem', borderRadius: '10px', border: '1px solid #cbd5e1', fontSize: '0.88rem', boxSizing: 'border-box' }}
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                style={{
                  width: '100%',
                  padding: '0.85rem',
                  borderRadius: '12px',
                  border: 'none',
                  background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                  color: '#ffffff',
                  fontSize: '0.95rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                  boxShadow: '0 4px 16px rgba(16, 185, 129, 0.3)',
                  marginTop: '0.4rem'
                }}
              >
                {loading ? 'Updating Password in DB...' : 'Reset Password & Update Database'}
              </button>

              <button
                type="button"
                onClick={() => { setMode('login'); setErrorMsg(null); setSuccessMsg(null); }}
                style={{ background: 'none', border: 'none', color: '#64748b', fontSize: '0.84rem', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem' }}
              >
                <ArrowLeft size={17} /> Cancel
              </button>
            </form>
          )}

        </div>

        {/* Footer Notice */}
        <div style={{ marginTop: 'auto', paddingTop: '2rem', textAlign: 'center' }}>
          <p style={{ margin: 0, fontSize: '0.78rem', color: '#94a3b8' }}>
            For authorized clinical and research users only.
          </p>
        </div>
      </div>

      {/* Right Side Visual Panel - Full 100vh Artwork View */}
      <div style={{
        position: 'relative',
        background: 'linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        padding: '3rem 3.5rem',
        boxSizing: 'border-box',
        height: '100vh',
        overflow: 'hidden'
      }}>
        {/* Background image overlay with soft glow */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundImage: `url(${loginArt})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center center',
          opacity: 0.9,
          filter: 'contrast(1.05) brightness(0.95)'
        }} />

        {/* Soft Ambient Lighting Overlay */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'linear-gradient(180deg, rgba(15, 23, 42, 0.15) 0%, rgba(30, 27, 75, 0.65) 100%)'
        }} />

        {/* Top Decorative Pill Badge */}
        <div style={{ position: 'relative', zIndex: 10, display: 'flex', justifyContent: 'flex-end' }}>
          <div style={{
            background: 'rgba(255, 255, 255, 0.18)',
            backdropFilter: 'blur(12px)',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            padding: '0.45rem 1rem',
            borderRadius: '20px',
            color: '#ffffff',
            fontSize: '0.8rem',
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: '0.45rem',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)'
          }}>
            <ShieldCheck size={16} style={{ color: '#a7f3d0' }} />
            <span>HIPAA Compliant System</span>
          </div>
        </div>

        {/* Bottom Floating Glass Badge (Matching User Mockup Screen 1) */}
        <div style={{ position: 'relative', zIndex: 10 }}>
          <div style={{
            background: 'rgba(15, 23, 42, 0.65)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(255, 255, 255, 0.22)',
            borderRadius: '20px',
            padding: '1.4rem 1.75rem',
            color: '#ffffff',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.4)',
            maxWidth: '480px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
              <div style={{
                width: '42px',
                height: '42px',
                borderRadius: '12px',
                background: 'rgba(99, 102, 241, 0.35)',
                border: '1px solid rgba(165, 180, 252, 0.5)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#a5b4fc',
                flexShrink: 0
              }}>
                <Sparkles size={22} />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: '#ffffff', letterSpacing: '-0.01em' }}>
                  AI for Better Respiratory Care
                </h3>
                <div style={{ fontSize: '0.82rem', color: '#cbd5e1', marginTop: '0.2rem' }}>
                  Research • Explainable AI • Clinical Support
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};
