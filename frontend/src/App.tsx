import React, { useEffect, useState } from 'react';
import { Navbar } from './components/Navbar';
import { UploadDropzone } from './components/UploadDropzone';
import { SegmentationViewer } from './components/SegmentationViewer';
import { ConsensusBanner } from './components/ConsensusBanner';
import { ClinicalReportView } from './components/ClinicalReportView';
import { ResearchComparisonView } from './components/ResearchComparisonView';
import { MetricsDashboard } from './components/MetricsDashboard';
import { ArchitectureSpecs } from './components/ArchitectureSpecs';
import { LoadingOverlay } from './components/LoadingOverlay';
import { HealthResponse, PredictResponse } from './types/api';
import { fetchHealth, predictImage } from './services/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'diagnostics' | 'dashboard' | 'pipeline'>('diagnostics');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [loading, setLoading] = useState(false);
  const [predictData, setPredictData] = useState<PredictResponse | null>(null);

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
      const data = await predictImage(file);
      setPredictData(data);
    } catch (err: any) {
      alert(`Prediction failed: ${err.message || 'Check server connection'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        healthError={healthError}
      />

      <LoadingOverlay active={loading} />

      <main className="container">
        {activeTab === 'diagnostics' && (
          <section className="section active">
            <div className="hero-header">
              <h1 className="hero-title">Prediction of Lung Cancer Probability from CT Scan</h1>
              <p className="hero-subtitle">
                AI-assisted preliminary CT image classification prototype powered by ConvNeXt-Tiny Deep Learning, deterministic lung-field segmentation, Grad-CAM Explainable AI (XAI), and structured LLM decision explanations.
              </p>
            </div>

            {/* Upload & Segmentation Viewer */}
            <div className="upload-grid" style={{ gridTemplateColumns: '1fr 2fr' }}>
              <UploadDropzone onFileSelect={handleFileSelect} />
              <SegmentationViewer
                images={predictData?.images}
                camFocusRatio={predictData?.gradcam_focus_in_lung}
                maskCoverage={predictData?.lung_mask_coverage}
              />
            </div>

            {/* AI Prediction Result Card (Case specific, ConvNeXt-Tiny powered) */}
            <ConsensusBanner predictData={predictData || undefined} />

            {/* LLM Explanation View & Research Comparison */}
            {predictData && (
              <>
                <ClinicalReportView explanation={predictData.llm_explanation} />

                {/* Research Comparison Models (XGBoost & GP - Not used in final prediction) */}
                <ResearchComparisonView researchModels={predictData.research_models} />
              </>
            )}
          </section>
        )}

        {activeTab === 'dashboard' && <MetricsDashboard />}
        {activeTab === 'pipeline' && <ArchitectureSpecs />}
      </main>
    </div>
  );
};

export default App;
