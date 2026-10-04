import React, { useEffect, useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { HeaderBar } from './components/HeaderBar';
import { HomeView } from './components/HomeView';
import { PredictView } from './components/PredictView';
import { DatasetView } from './components/DatasetView';
import { TrainModelView } from './components/TrainModelView';
import { MetricsDashboard } from './components/MetricsDashboard';
import { HistoryView } from './components/HistoryView';
import { ProfileView } from './components/ProfileView';
import { LoadingOverlay } from './components/LoadingOverlay';
import { AuthModal } from './components/AuthModal';
// import { AiDisclaimerBanner } from './components/AiDisclaimerBanner';
import { HealthResponse, PredictResponse, UserProfile } from './types/api';
import { fetchHealth, predictImage } from './services/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('home');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [loading, setLoading] = useState(false);
  const [predictData, setPredictData] = useState<PredictResponse | null>(null);

  // Authentication State
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(() => {
    const saved = localStorage.getItem('lung_ai_user');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        return null;
      }
    }
    return {
      username: 'doctor',
      email: 'doctor@hospital.org',
      full_name: 'Nivetha S.',
      role: 'Research Scholar',
      hospital_name: 'General Hospital'
    };
  });

  useEffect(() => {
    const checkStatus = () => {
      fetchHealth()
        .then((data) => {
          setHealth(data);
          setHealthError(false);
        })
        .catch(() => {
          setHealthError(true);
        });
    };

    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleFileSelect = async (file: File) => {
    setLoading(true);
    try {
      const data = await predictImage(file, currentUser?.username);
      setPredictData(data);
      setActiveTab('predict');
    } catch (err: any) {
      alert(`Prediction failed: ${err.message || 'Check server connection'}`);
    } finally {
      setLoading(false);
    }
  };

  const handleLoginSuccess = (user: UserProfile) => {
    setCurrentUser(user);
    localStorage.setItem('lung_ai_user', JSON.stringify(user));
  };

  const handleLogout = () => {
    setCurrentUser(null);
    localStorage.removeItem('lung_ai_user');
    setIsAuthOpen(true);
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f8fafc', display: 'flex' }}>
      {/* Left Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        currentUser={currentUser}
        onLogout={handleLogout}
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

      {/* Authentication Modal */}
      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
        onLoginSuccess={handleLoginSuccess}
      />

      {/* Main App Workspace View Container */}
      <main style={{ marginLeft: '240px', marginTop: '64px', flex: 1, padding: '2rem 2rem 6rem 2rem', width: 'calc(100% - 240px)' }}>
        {activeTab === 'home' && (
          <HomeView
            onNavigate={(target) => setActiveTab(target)}
            onFileSelect={handleFileSelect}
          />
        )}

        {activeTab === 'predict' && (
          <PredictView
            predictData={predictData}
            onFileSelect={handleFileSelect}
          />
        )}

        {activeTab === 'dataset' && <DatasetView />}
        {activeTab === 'train' && <TrainModelView />}
        {activeTab === 'dashboard' && <MetricsDashboard />}
        {activeTab === 'history' && <HistoryView />}
        {activeTab === 'profile' && <ProfileView currentUser={currentUser} />}
      </main>

      {/* Floating Medical Disclaimer Banner */}
      {/* <AiDisclaimerBanner /> */}
    </div>
  );
};

export default App;
