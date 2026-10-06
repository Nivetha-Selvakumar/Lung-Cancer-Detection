import React, { useEffect, useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { HeaderBar } from './components/HeaderBar';
import { HomeView } from './components/HomeView';
import { PredictView } from './components/PredictView';
import { DatasetView } from './components/DatasetView';
import { MetricsDashboard } from './components/MetricsDashboard';
import { HistoryView } from './components/HistoryView';
import { ProfileView } from './components/ProfileView';
import { LoadingOverlay } from './components/LoadingOverlay';
import { AuthModal } from './components/AuthModal';
import { LoginPage } from './components/LoginPage';
import { LogoutModal } from './components/LogoutModal';
import { HealthResponse, PredictResponse, UserProfile } from './types/api';
import { fetchHealth, fetchCurrentUser, logoutUser, predictImage } from './services/api';

// Route Helper Mappers
function getTabFromPath(path: string): string {
  const p = path.toLowerCase().replace(/\/$/, '');
  if (p === '/login') return 'login';
  if (p === '/dashboard' || p === '/home') return 'home';
  if (p === '/predict') return 'predict';
  if (p === '/dataset') return 'dataset';
  if (p === '/metrics') return 'dashboard';
  if (p === '/history') return 'history';
  if (p === '/profile') return 'profile';
  return 'home';
}

function getPathFromTab(tab: string): string {
  switch (tab) {
    case 'login': return '/login';
    case 'home': return '/dashboard';
    case 'predict': return '/predict';
    case 'dataset': return '/dataset';
    case 'dashboard': return '/metrics';
    case 'history': return '/history';
    case 'profile': return '/profile';
    default: return '/dashboard';
  }
}

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>(() => {
    return getTabFromPath(window.location.pathname);
  });

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [loading, setLoading] = useState(false);
  const [predictData, setPredictData] = useState<PredictResponse | null>(null);

  // Authentication State
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [isLogoutModalOpen, setIsLogoutModalOpen] = useState(false);
  const [logoutLoading, setLogoutLoading] = useState(false);

  const [authToken, setAuthToken] = useState<string | null>(() => localStorage.getItem('lung_ai_token'));
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(() => {
    const saved = localStorage.getItem('lung_ai_user');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        return null;
      }
    }
    return null;
  });

  // URL Path Routing Navigation Helper
  const navigateToTab = (tab: string) => {
    setActiveTab(tab);
    const targetPath = getPathFromTab(tab);
    if (window.location.pathname !== targetPath) {
      window.history.pushState({}, '', targetPath);
    }
  };

  // Listen for browser Back/Forward buttons (popstate)
  useEffect(() => {
    const handlePopState = () => {
      const tab = getTabFromPath(window.location.pathname);
      setActiveTab(tab);
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // Sync initial URL path on load or after auth change
  useEffect(() => {
    if (!currentUser || !authToken) {
      if (window.location.pathname !== '/login') {
        window.history.replaceState({}, '', '/login');
      }
    } else {
      if (window.location.pathname === '/' || window.location.pathname === '/login') {
        window.history.replaceState({}, '', '/dashboard');
        setActiveTab('home');
      }
    }
  }, [currentUser, authToken]);

  // Verify Auth Token on initial load
  useEffect(() => {
    const token = localStorage.getItem('lung_ai_token');
    if (token) {
      fetchCurrentUser(token)
        .then((res) => {
          if (res.success && res.user) {
            setCurrentUser(res.user);
            localStorage.setItem('lung_ai_user', JSON.stringify(res.user));
          } else {
            // Token is invalid or expired in DB -> perform logout
            performLogout();
          }
        })
        .catch(() => {
          // Keep local session if network unavailable
        });
    }
  }, []);

  // Periodic Health Check every 8 seconds so Backend status auto-recovers
  useEffect(() => {
    const check = () => {
      fetchHealth()
        .then((data) => {
          setHealth(data);
          setHealthError(false);
        })
        .catch(() => {
          setHealthError(true);
        });
    };
    check();
    const interval = setInterval(check, 8000);
    return () => clearInterval(interval);
  }, []);

  const [predictError, setPredictError] = useState<string | null>(null);

  const handleFileSelect = async (file: File) => {
    setLoading(true);
    setPredictError(null);
    setPredictData(null);
    try {
      const data = await predictImage(file, currentUser?.username);
      setPredictData(data);
      setPredictError(null);
      navigateToTab('predict');
    } catch (err: any) {
      setPredictError(err.message || 'Invalid CT image uploaded.');
      setPredictData(null);
      navigateToTab('predict');
    } finally {
      setLoading(false);
    }
  };

  const handleLoginSuccess = (user: UserProfile, token: string) => {
    setCurrentUser(user);
    setAuthToken(token);
    localStorage.setItem('lung_ai_token', token);
    localStorage.setItem('lung_ai_user', JSON.stringify(user));
    navigateToTab('home');
  };

  const performLogout = async () => {
    setLogoutLoading(true);
    const token = localStorage.getItem('lung_ai_token');
    if (token) {
      try {
        await logoutUser(token);
      } catch (e) {
        // ignore logout network errors
      }
    }
    setCurrentUser(null);
    setAuthToken(null);
    localStorage.removeItem('lung_ai_user');
    localStorage.removeItem('lung_ai_token');
    setLogoutLoading(false);
    setIsLogoutModalOpen(false);
    if (window.location.pathname !== '/login') {
      window.history.pushState({}, '', '/login');
    }
  };

  // If user is not authenticated, render LoginPage Screen on /login
  if (!currentUser || !authToken) {
    return <LoginPage onLoginSuccess={handleLoginSuccess} />;
  }

  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', display: 'flex' }}>
      {/* Left Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={navigateToTab}
        currentUser={currentUser}
        onLogout={() => setIsLogoutModalOpen(true)}
      />

      {/* Top Header Bar */}
      <HeaderBar
        currentUser={currentUser}
        health={health}
        healthError={healthError}
        onOpenAuthModal={() => setIsAuthOpen(true)}
      />

      {/* Loading Overlay */}
      <LoadingOverlay active={loading} />

      {/* Logout Confirmation Modal */}
      <LogoutModal
        isOpen={isLogoutModalOpen}
        onClose={() => setIsLogoutModalOpen(false)}
        onConfirm={performLogout}
        loading={logoutLoading}
      />

      {/* Authentication Modal (For register/switch user inside app) */}
      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
        onLoginSuccess={(user) => {
          if (user.auth_token) {
            handleLoginSuccess(user, user.auth_token);
          } else {
            setCurrentUser(user);
            localStorage.setItem('lung_ai_user', JSON.stringify(user));
          }
        }}
      />

      {/* Main App Workspace View Container */}
      <main style={{ marginLeft: '240px', marginTop: '64px', flex: 1, padding: '2rem 2rem 6rem 2rem', width: 'calc(100% - 240px)' }}>
        {activeTab === 'home' && (
          <HomeView
            currentUser={currentUser}
            onNavigate={(target) => navigateToTab(target)}
            onFileSelect={handleFileSelect}
          />
        )}

        {activeTab === 'predict' && (
          <PredictView
            predictData={predictData}
            predictError={predictError}
            loading={loading}
            onFileSelect={handleFileSelect}
            onClearError={() => setPredictError(null)}
          />
        )}

        {activeTab === 'dataset' && <DatasetView />}
        {activeTab === 'dashboard' && <MetricsDashboard />}
        {activeTab === 'history' && <HistoryView />}
        {activeTab === 'profile' && (
          <ProfileView
            currentUser={currentUser}
            onUpdateUser={(updated) => {
              setCurrentUser(updated);
              localStorage.setItem('lung_ai_user', JSON.stringify(updated));
            }}
          />
        )}
      </main>
    </div>
  );
};

export default App;
