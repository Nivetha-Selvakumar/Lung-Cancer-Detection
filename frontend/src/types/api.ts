export interface UserProfile {
  id?: number;
  username: string;
  email: string;
  full_name: string;
  role: string;
  hospital_name?: string;
  department?: string;
  institution?: string;
  auth_token?: string;
}

export interface AuthResponse {
  success: boolean;
  token?: string;
  user?: UserProfile;
  db_engine?: string;
  error?: string;
}

export interface DbStatusResponse {
  engine: string;
  connected: boolean;
  host?: string;
  database?: string;
  port?: number;
  message: string;
}

export interface ClassProbabilities {
  Normal: number;
  Benign: number;
  Malignant: number;
}

export interface ResearchModelResult {
  name: string;
  predicted_class: 'Normal' | 'Benign' | 'Malignant';
  confidence: number;
  probabilities: ClassProbabilities;
  error?: string;
}

export interface LLMExplanation {
  status: string;
  text: string;
}

export interface PredictResponse {
  case_id: string;
  filename?: string;
  fused_features?: number[];
  predicted_class: 'Normal' | 'Benign' | 'Malignant';
  predicted_class_index: number;
  confidence: number;
  probabilities: ClassProbabilities;
  malignant_probability: number;
  model_estimated_malignant_probability: number;
  temperature?: number;
  lung_mask_coverage: number;
  gradcam_focus_in_lung: number;
  images: {
    original: string;
    mask: string;
    segmented_roi: string;
    gradcam: string;
  };
  llm_explanation: LLMExplanation;
  research_models?: {
    xgboost?: ResearchModelResult;
    genetic_programming?: ResearchModelResult;
  };
  error?: string;
}

export interface AspectMetric {
  name: string;
  accuracy: number;
  balanced_accuracy: number;
  precision: number;
  recall: number;
  macro_f1: number;
  confusion_matrix: number[][];
  description: string;
}

export interface MetricsResponse {
  dataset: string;
  train_samples: number;
  val_samples: number;
  test_samples: number;
  aspects: Record<string, AspectMetric>;
}

export interface HealthResponse {
  status: string;
  xgb_loaded: boolean;
  gp_loaded: boolean;
  convnext_loaded: boolean;
  db?: DbStatusResponse;
}
