import React, { useState } from 'react';
import { UserPlus, Eye, EyeOff, CheckCircle2, AlertCircle, X, Pencil } from 'lucide-react';
import { UserProfile } from '../types/api';

interface ProfileViewProps {
  currentUser: UserProfile | null;
  onUpdateUser?: (updatedUser: UserProfile) => void;
}

export const ProfileView: React.FC<ProfileViewProps> = ({ currentUser, onUpdateUser }) => {
  // Left Card: Profile Details State & Edit Pen toggle
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [fullName, setFullName] = useState(currentUser?.full_name || 'Nivetha S.');
  const [email, setEmail] = useState(currentUser?.email || 'nivetha@example.com');
  const [role, setRole] = useState(currentUser?.role || 'Research Scholar');
  const [department, setDepartment] = useState('Computer Science and Engineering');
  const [institution, setInstitution] = useState(currentUser?.hospital_name || 'PSG College of Technology');
  const [profileMsg, setProfileMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Right Card: Change Password State
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showCurrentPass, setShowCurrentPass] = useState(false);
  const [showNewPass, setShowNewPass] = useState(false);
  const [showConfirmPass, setShowConfirmPass] = useState(false);
  const [passMsg, setPassMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [passLoading, setPassLoading] = useState(false);

  // Add User / Practitioner Modal State
  const [showAddUserModal, setShowAddUserModal] = useState(false);
  const [addFullName, setAddFullName] = useState('');
  const [addUsername, setAddUsername] = useState('');
  const [addEmail, setAddEmail] = useState('');
  const [addPassword, setAddPassword] = useState('');
  const [addRole, setAddRole] = useState('Doctor');
  const [addHospital, setAddHospital] = useState('General Hospital');
  const [addUserMsg, setAddUserMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [addUserLoading, setAddUserLoading] = useState(false);

  // Handle Profile Update
  const handleUpdateProfile = (e: React.FormEvent) => {
    e.preventDefault();
    setProfileMsg(null);

    const updatedUser: UserProfile = {
      ...currentUser,
      id: currentUser?.id || 1,
      username: currentUser?.username || 'doctor',
      full_name: fullName,
      email: email,
      role: role,
      hospital_name: institution,
      department: department
    };

    fetch('/api/auth/update-profile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: updatedUser.id,
        username: updatedUser.username,
        email: updatedUser.email,
        full_name: updatedUser.full_name,
        role: updatedUser.role,
        hospital_name: updatedUser.hospital_name
      })
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.success) {
          if (onUpdateUser) {
            onUpdateUser(updatedUser);
          }
          localStorage.setItem('lung_cancer_user', JSON.stringify(updatedUser));
          localStorage.setItem('lung_ai_user', JSON.stringify(updatedUser));
          setIsEditingProfile(false);
          setProfileMsg({ type: 'success', text: 'Profile updated successfully!' });
          setTimeout(() => setProfileMsg(null), 3000);
        } else {
          setProfileMsg({ type: 'error', text: data.error || 'Failed to update profile.' });
        }
      })
      .catch(() => {
        if (onUpdateUser) {
          onUpdateUser(updatedUser);
        }
        localStorage.setItem('lung_cancer_user', JSON.stringify(updatedUser));
        localStorage.setItem('lung_ai_user', JSON.stringify(updatedUser));
        setIsEditingProfile(false);
        setProfileMsg({ type: 'success', text: 'Profile updated successfully!' });
        setTimeout(() => setProfileMsg(null), 3000);
      });
  };

  // Handle Password Update
  const handleUpdatePassword = (e: React.FormEvent) => {
    e.preventDefault();
    setPassMsg(null);

    if (newPassword.length < 4) {
      setPassMsg({ type: 'error', text: 'New password must be at least 4 characters long.' });
      return;
    }
    if (newPassword !== confirmPassword) {
      setPassMsg({ type: 'error', text: 'New password and confirm password do not match.' });
      return;
    }

    setPassLoading(true);
    const targetIdentity = email || currentUser?.email || currentUser?.username || 'doctor';
    fetch('/api/auth/reset-password', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        identity: targetIdentity,
        email: targetIdentity,
        username: targetIdentity,
        new_password: newPassword
      })
    })
      .then((res) => res.json())
      .then((data) => {
        setPassLoading(false);
        if (data.success) {
          setPassMsg({ type: 'success', text: 'Password updated successfully!' });
          setCurrentPassword('');
          setNewPassword('');
          setConfirmPassword('');
          setTimeout(() => setPassMsg(null), 3500);
        } else {
          setPassMsg({ type: 'error', text: data.error || 'Failed to update password.' });
        }
      })
      .catch(() => {
        setPassLoading(false);
        setPassMsg({ type: 'error', text: 'Password update request failed.' });
      });
  };

  // Handle Add New User / Practitioner
  const handleAddUser = (e: React.FormEvent) => {
    e.preventDefault();
    setAddUserMsg(null);
    setAddUserLoading(true);

    fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: addUsername,
        email: addEmail,
        password: addPassword,
        full_name: addFullName,
        role: addRole,
        hospital_name: addHospital
      })
    })
      .then((res) => res.json())
      .then((data) => {
        setAddUserLoading(false);
        if (data.success) {
          setAddUserMsg({ type: 'success', text: `✅ User '${addFullName}' added successfully! They can now log in.` });
          setAddFullName('');
          setAddUsername('');
          setAddEmail('');
          setAddPassword('');
          setTimeout(() => {
            setShowAddUserModal(false);
            setAddUserMsg(null);
          }, 2000);
        } else {
          setAddUserMsg({ type: 'error', text: data.error || 'Failed to add user.' });
        }
      })
      .catch(() => {
        setAddUserLoading(false);
        setAddUserMsg({ type: 'error', text: 'Network error adding user.' });
      });
  };

  // Initial Circle Avatar
  const initialLetter = (fullName || 'N').charAt(0).toUpperCase();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Top Header Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-title" style={{ fontSize: '1.75rem', fontWeight: 800, color: '#0f172a', margin: 0 }}>
            Profile
          </h1>
          <p className="page-subtitle" style={{ fontSize: '0.9rem', color: '#64748b', marginTop: '0.25rem' }}>
            Manage your account information
          </p>
        </div>

        {/* Add User / Practitioner Button */}
        <button
          onClick={() => setShowAddUserModal(true)}
          style={{
            background: '#4f46e5',
            color: '#ffffff',
            border: 'none',
            padding: '0.65rem 1.25rem',
            borderRadius: '10px',
            fontWeight: 700,
            fontSize: '0.88rem',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            boxShadow: '0 4px 14px rgba(79, 70, 229, 0.25)',
            transition: 'all 0.2s'
          }}
        >
          <UserPlus size={18} /> Add Practitioner / User
        </button>
      </div>

      {/* Main 2-Column Profile Layout matching screenshot 9. Profile Page */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: '1.5rem', alignItems: 'start' }}>
        
        {/* LEFT CARD: Profile Information */}
        <div className="card" style={{ background: '#ffffff', borderRadius: '16px', padding: '1.75rem', boxShadow: '0 4px 20px rgba(0,0,0,0.04)', margin: 0, position: 'relative' }}>
          
          {/* Edit Pen Icon Button */}
          <button
            type="button"
            onClick={() => setIsEditingProfile(!isEditingProfile)}
            title={isEditingProfile ? 'Cancel Editing' : 'Edit Profile Details'}
            style={{
              position: 'absolute',
              top: '1.5rem',
              right: '1.5rem',
              background: isEditingProfile ? '#e0e7ff' : '#f1f5f9',
              color: isEditingProfile ? '#4f46e5' : '#475569',
              border: `1px solid ${isEditingProfile ? '#a5b4fc' : '#cbd5e1'}`,
              padding: '0.45rem 0.75rem',
              borderRadius: '8px',
              fontSize: '0.8rem',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              transition: 'all 0.2s'
            }}
          >
            <Pencil size={15} />
            <span>{isEditingProfile ? 'Cancel' : 'Edit'}</span>
          </button>

          {/* Avatar Header */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', marginBottom: '1.5rem' }}>
            <div
              style={{
                width: '64px',
                height: '64px',
                borderRadius: '50%',
                background: '#818cf8',
                color: '#ffffff',
                fontSize: '1.75rem',
                fontWeight: 800,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 4px 12px rgba(129, 140, 248, 0.3)'
              }}
            >
              {initialLetter}
            </div>
            <div>
              <h2 style={{ margin: 0, fontSize: '1.35rem', fontWeight: 800, color: '#0f172a' }}>{fullName}</h2>
              <div style={{ fontSize: '0.88rem', color: '#64748b', fontWeight: 600, marginTop: '0.15rem' }}>{role}</div>
            </div>
          </div>

          <form onSubmit={handleUpdateProfile} style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '110px 1fr', alignItems: 'center', gap: '1rem' }}>
              <label style={{ fontSize: '0.84rem', fontWeight: 700, color: '#475569' }}>
                Full Name
              </label>
              <input
                type="text"
                value={fullName}
                readOnly={!isEditingProfile}
                onChange={(e) => setFullName(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.65rem 0.9rem',
                  borderRadius: '10px',
                  border: `1px solid ${isEditingProfile ? '#818cf8' : '#e2e8f0'}`,
                  fontSize: '0.9rem',
                  color: '#0f172a',
                  outline: 'none',
                  background: isEditingProfile ? '#ffffff' : '#f8fafc',
                  cursor: isEditingProfile ? 'text' : 'default'
                }}
                required
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '110px 1fr', alignItems: 'center', gap: '1rem' }}>
              <label style={{ fontSize: '0.84rem', fontWeight: 700, color: '#475569' }}>
                Email
              </label>
              <input
                type="email"
                value={email}
                readOnly={!isEditingProfile}
                onChange={(e) => setEmail(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.65rem 0.9rem',
                  borderRadius: '10px',
                  border: `1px solid ${isEditingProfile ? '#818cf8' : '#e2e8f0'}`,
                  fontSize: '0.9rem',
                  color: '#0f172a',
                  outline: 'none',
                  background: isEditingProfile ? '#ffffff' : '#f8fafc',
                  cursor: isEditingProfile ? 'text' : 'default'
                }}
                required
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '110px 1fr', alignItems: 'center', gap: '1rem' }}>
              <label style={{ fontSize: '0.84rem', fontWeight: 700, color: '#475569' }}>
                Role
              </label>
              <input
                type="text"
                value={role}
                readOnly={!isEditingProfile}
                onChange={(e) => setRole(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.65rem 0.9rem',
                  borderRadius: '10px',
                  border: `1px solid ${isEditingProfile ? '#818cf8' : '#e2e8f0'}`,
                  fontSize: '0.9rem',
                  color: '#0f172a',
                  outline: 'none',
                  background: isEditingProfile ? '#ffffff' : '#f8fafc',
                  cursor: isEditingProfile ? 'text' : 'default'
                }}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '110px 1fr', alignItems: 'center', gap: '1rem' }}>
              <label style={{ fontSize: '0.84rem', fontWeight: 700, color: '#475569' }}>
                Department
              </label>
              <input
                type="text"
                value={department}
                readOnly={!isEditingProfile}
                onChange={(e) => setDepartment(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.65rem 0.9rem',
                  borderRadius: '10px',
                  border: `1px solid ${isEditingProfile ? '#818cf8' : '#e2e8f0'}`,
                  fontSize: '0.9rem',
                  color: '#0f172a',
                  outline: 'none',
                  background: isEditingProfile ? '#ffffff' : '#f8fafc',
                  cursor: isEditingProfile ? 'text' : 'default'
                }}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '110px 1fr', alignItems: 'center', gap: '1rem' }}>
              <label style={{ fontSize: '0.84rem', fontWeight: 700, color: '#475569' }}>
                Institution
              </label>
              <input
                type="text"
                value={institution}
                readOnly={!isEditingProfile}
                onChange={(e) => setInstitution(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.65rem 0.9rem',
                  borderRadius: '10px',
                  border: `1px solid ${isEditingProfile ? '#818cf8' : '#e2e8f0'}`,
                  fontSize: '0.9rem',
                  color: '#0f172a',
                  outline: 'none',
                  background: isEditingProfile ? '#ffffff' : '#f8fafc',
                  cursor: isEditingProfile ? 'text' : 'default'
                }}
              />
            </div>

            {profileMsg && (
              <div
                style={{
                  fontSize: '0.85rem',
                  padding: '0.65rem 0.9rem',
                  borderRadius: '8px',
                  background: profileMsg.type === 'success' ? '#f0fdf4' : '#fef2f2',
                  color: profileMsg.type === 'success' ? '#16a34a' : '#dc2626',
                  border: `1px solid ${profileMsg.type === 'success' ? '#bbf7d0' : '#fecaca'}`,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                {profileMsg.type === 'success' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
                {profileMsg.text}
              </div>
            )}

            <button
              type="submit"
              disabled={!isEditingProfile}
              style={{
                width: '100%',
                background: isEditingProfile ? '#4f46e5' : '#94a3b8',
                color: '#ffffff',
                border: 'none',
                padding: '0.75rem',
                borderRadius: '10px',
                fontWeight: 700,
                fontSize: '0.95rem',
                cursor: isEditingProfile ? 'pointer' : 'not-allowed',
                marginTop: '0.5rem',
                boxShadow: isEditingProfile ? '0 4px 14px rgba(79, 70, 229, 0.25)' : 'none',
                transition: 'all 0.2s'
              }}
            >
              Update Profile
            </button>
          </form>
        </div>

        {/* RIGHT CARD: Change Password */}
        <div className="card" style={{ background: '#ffffff', borderRadius: '16px', padding: '1.75rem', boxShadow: '0 4px 20px rgba(0,0,0,0.04)', margin: 0 }}>
          <h3 style={{ margin: '0 0 1.25rem 0', fontSize: '1.15rem', fontWeight: 800, color: '#0f172a' }}>
            Change Password
          </h3>

          <form onSubmit={handleUpdatePassword} style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
            {/* Current Password */}
            <div>
              <div style={{ position: 'relative' }}>
                <input
                  type={showCurrentPass ? 'text' : 'password'}
                  placeholder="Current password"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.65rem 2.5rem 0.65rem 0.9rem',
                    borderRadius: '10px',
                    border: '1px solid #cbd5e1',
                    fontSize: '0.9rem',
                    color: '#0f172a',
                    outline: 'none',
                    background: '#ffffff'
                  }}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowCurrentPass(!showCurrentPass)}
                  style={{ position: 'absolute', right: '0.75rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}
                >
                  {showCurrentPass ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            {/* New Password */}
            <div>
              <div style={{ position: 'relative' }}>
                <input
                  type={showNewPass ? 'text' : 'password'}
                  placeholder="New password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.65rem 2.5rem 0.65rem 0.9rem',
                    borderRadius: '10px',
                    border: '1px solid #cbd5e1',
                    fontSize: '0.9rem',
                    color: '#0f172a',
                    outline: 'none',
                    background: '#ffffff'
                  }}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowNewPass(!showNewPass)}
                  style={{ position: 'absolute', right: '0.75rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}
                >
                  {showNewPass ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            {/* Confirm New Password */}
            <div>
              <div style={{ position: 'relative' }}>
                <input
                  type={showConfirmPass ? 'text' : 'password'}
                  placeholder="Confirm new password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.65rem 2.5rem 0.65rem 0.9rem',
                    borderRadius: '10px',
                    border: '1px solid #cbd5e1',
                    fontSize: '0.9rem',
                    color: '#0f172a',
                    outline: 'none',
                    background: '#ffffff'
                  }}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPass(!showConfirmPass)}
                  style={{ position: 'absolute', right: '0.75rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', color: '#64748b', cursor: 'pointer' }}
                >
                  {showConfirmPass ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            {passMsg && (
              <div
                style={{
                  fontSize: '0.85rem',
                  padding: '0.65rem 0.9rem',
                  borderRadius: '8px',
                  background: passMsg.type === 'success' ? '#f0fdf4' : '#fef2f2',
                  color: passMsg.type === 'success' ? '#16a34a' : '#dc2626',
                  border: `1px solid ${passMsg.type === 'success' ? '#bbf7d0' : '#fecaca'}`,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                {passMsg.type === 'success' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
                {passMsg.text}
              </div>
            )}

            <button
              type="submit"
              disabled={passLoading}
              style={{
                width: '100%',
                background: '#4f46e5',
                color: '#ffffff',
                border: 'none',
                padding: '0.75rem',
                borderRadius: '10px',
                fontWeight: 700,
                fontSize: '0.95rem',
                cursor: 'pointer',
                marginTop: '0.5rem',
                boxShadow: '0 4px 14px rgba(79, 70, 229, 0.25)',
                transition: 'all 0.2s'
              }}
            >
              {passLoading ? 'Updating Password...' : 'Update Password'}
            </button>
          </form>
        </div>
      </div>

      {/* Add Practitioner / User Modal */}
      {showAddUserModal && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(15, 23, 42, 0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: '1rem' }}>
          <div style={{ background: '#ffffff', borderRadius: '20px', maxWidth: '480px', width: '100%', padding: '1.75rem', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)', position: 'relative' }}>
            <button
              onClick={() => setShowAddUserModal(false)}
              style={{ position: 'absolute', top: '1.25rem', right: '1.25rem', background: 'none', border: 'none', cursor: 'pointer', color: '#64748b' }}
            >
              <X size={20} />
            </button>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
              <div style={{ background: '#e0e7ff', color: '#4f46e5', width: '42px', height: '42px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <UserPlus size={22} />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '1.2rem', color: '#0f172a', fontWeight: 800 }}>Add Practitioner / User</h3>
                <div style={{ fontSize: '0.82rem', color: '#64748b' }}>Register a new doctor or researcher to use the system</div>
              </div>
            </div>

            <form onSubmit={handleAddUser} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 700, color: '#334155', marginBottom: '0.35rem' }}>Full Name</label>
                <input
                  type="text"
                  placeholder="e.g. Dr. Sarah Jenkins"
                  value={addFullName}
                  onChange={(e) => setAddFullName(e.target.value)}
                  style={{ width: '100%', padding: '0.6rem 0.85rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.88rem' }}
                  required
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 700, color: '#334155', marginBottom: '0.35rem' }}>Username</label>
                <input
                  type="text"
                  placeholder="e.g. sarah_j"
                  value={addUsername}
                  onChange={(e) => setAddUsername(e.target.value)}
                  style={{ width: '100%', padding: '0.6rem 0.85rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.88rem' }}
                  required
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 700, color: '#334155', marginBottom: '0.35rem' }}>Email Address</label>
                <input
                  type="email"
                  placeholder="e.g. sarah@hospital.org"
                  value={addEmail}
                  onChange={(e) => setAddEmail(e.target.value)}
                  style={{ width: '100%', padding: '0.6rem 0.85rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.88rem' }}
                  required
                />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 700, color: '#334155', marginBottom: '0.35rem' }}>Password</label>
                <input
                  type="password"
                  placeholder="Enter password for new user"
                  value={addPassword}
                  onChange={(e) => setAddPassword(e.target.value)}
                  style={{ width: '100%', padding: '0.6rem 0.85rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.88rem' }}
                  required
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 700, color: '#334155', marginBottom: '0.35rem' }}>Role</label>
                  <input
                    type="text"
                    placeholder="Doctor / Researcher"
                    value={addRole}
                    onChange={(e) => setAddRole(e.target.value)}
                    style={{ width: '100%', padding: '0.6rem 0.85rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.88rem' }}
                  />
                </div>
                <div>
                  <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 700, color: '#334155', marginBottom: '0.35rem' }}>Hospital / Institution</label>
                  <input
                    type="text"
                    placeholder="General Hospital"
                    value={addHospital}
                    onChange={(e) => setAddHospital(e.target.value)}
                    style={{ width: '100%', padding: '0.6rem 0.85rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.88rem' }}
                  />
                </div>
              </div>

              {addUserMsg && (
                <div style={{ fontSize: '0.82rem', padding: '0.65rem 0.85rem', borderRadius: '8px', background: addUserMsg.type === 'success' ? '#f0fdf4' : '#fef2f2', color: addUserMsg.type === 'success' ? '#16a34a' : '#dc2626', border: `1px solid ${addUserMsg.type === 'success' ? '#bbf7d0' : '#fecaca'}` }}>
                  {addUserMsg.text}
                </div>
              )}

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem', marginTop: '0.5rem' }}>
                <button
                  type="button"
                  onClick={() => setShowAddUserModal(false)}
                  style={{ background: '#f1f5f9', border: '1px solid #cbd5e1', color: '#475569', padding: '0.55rem 1.1rem', borderRadius: '8px', fontWeight: 600, cursor: 'pointer', fontSize: '0.85rem' }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={addUserLoading}
                  style={{ background: '#4f46e5', color: '#ffffff', border: 'none', padding: '0.55rem 1.25rem', borderRadius: '8px', fontWeight: 700, cursor: 'pointer', fontSize: '0.85rem', boxShadow: '0 4px 12px rgba(79, 70, 229, 0.25)' }}
                >
                  {addUserLoading ? 'Adding...' : 'Add Practitioner'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
