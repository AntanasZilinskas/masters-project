// Type definitions for Everest Solar Flare Prediction app

// Flare classification probabilities
export interface FlareProbabilities {
  C: number;
  M: number;
  M5: number;
}

// Uncertainty measures
export interface UncertaintyMeasures {
  epistemic: FlareProbabilities;
  aleatoric: FlareProbabilities;
}

// Prediction horizon
export interface PredictionHorizon {
  hours: number;
  softmax_dense: FlareProbabilities;
  uncertainty: UncertaintyMeasures;
}

// Forecast data
export interface ForecastData {
  generated_at: string;
  horizons: PredictionHorizon[];
}

// Temporal series datapoint
export interface TimeSeriesPoint {
  timestamp: string;
  prob_C: number;
  prob_M: number;
  prob_M5: number;
  epi: number;
  alea: number;
}

// Temporal evolution data
export interface TemporalEvolutionData {
  series: TimeSeriesPoint[];
}

// SHARP parameters for active regions
export interface SharpParameters {
  usflux: number;
  r: number;
  [key: string]: number;
}

// Extreme Value Theory parameters
export interface EVTParameters {
  xi: number;
  sigma: number;
}

// Active region data
export interface ActiveRegionData {
  noaa_id: number;
  location: string;
  sharp: SharpParameters;
  saliency_map_url: string;
  evt: EVTParameters;
  regional_softmax: FlareProbabilities;
}

// Performance metrics for each flare class
export interface ClassMetrics {
  C: number;
  M: number;
  M5: number;
}

// Overall performance scores
export interface PerformanceScores {
  TSS: number;
  BSS: number;
  F1: ClassMetrics;
  precision: ClassMetrics;
  recall: ClassMetrics;
}

// Reliability bin
export interface ReliabilityBin {
  bin: string;
  count: number;
  observed: number;
}

// Performance and calibration data
export interface PerformanceData {
  metric_window_days: number;
  scores: PerformanceScores;
  reliability_bins: ReliabilityBin[];
}

// GOES X-ray flux datapoint
export interface XRayFluxPoint {
  timestamp: string;
  flux: number;
}

// Coronal Mass Ejection data
export interface CMEData {
  time: string;
  speed: number;
  width: number;
  source_ar: number;
}

// Solar context data
export interface SolarContextData {
  synoptic_url: string;
  goes_xray: XRayFluxPoint[];
  cme_feed: CMEData[];
}

// Alert rule
export interface AlertRule {
  id: number;
  class: 'C' | 'M' | 'M5';
  prob_threshold: number;
  max_epi: number;
  horizon: number;
}

// Alert event
export interface AlertEvent {
  rule_id: number;
  triggered_at: string;
  state: 'firing' | 'resolved';
}

// Alert center data
export interface AlertCenterData {
  user_rules: AlertRule[];
  active_alerts: AlertEvent[];
  history: AlertEvent[];
}

// Combined app state
export interface AppState {
  forecast: ForecastData;
  temporalEvolution: TemporalEvolutionData;
  activeRegions: Record<number, ActiveRegionData>;
  performance: PerformanceData;
  solarContext: SolarContextData;
  alertCenter: AlertCenterData;
  ui: {
    selectedHorizon: number;
    selectedActiveRegion: number | null;
    showActiveRegionDrawer: boolean;
  };
}
