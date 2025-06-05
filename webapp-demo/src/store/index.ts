import { create } from 'zustand';
import { format, subHours, addHours } from 'date-fns';
import type { 
  AppState, 
  ForecastData, 
  TemporalEvolutionData, 
  ActiveRegionData, 
  PerformanceData, 
  SolarContextData, 
  AlertCenterData 
} from '../types/everest';

// Function to load real forecast data
const loadForecastData = async (): Promise<ForecastData> => {
  try {
    const response = await fetch('/src/data/forecast_data.json?t=' + Date.now());
    if (response.ok) {
      const rawData = await response.json();
      
      // Transform X to M5 in the loaded data
      const data: ForecastData = {
        ...rawData,
        horizons: rawData.horizons.map((horizon: any) => ({
          ...horizon,
          softmax_dense: {
            C: horizon.softmax_dense.C,
            M: horizon.softmax_dense.M,
            M5: horizon.softmax_dense.X // Map X to M5
          },
          uncertainty: {
            epistemic: {
              C: horizon.uncertainty.epistemic.C,
              M: horizon.uncertainty.epistemic.M,
              M5: horizon.uncertainty.epistemic.X // Map X to M5
            },
            aleatoric: {
              C: horizon.uncertainty.aleatoric.C,
              M: horizon.uncertainty.aleatoric.M,
              M5: horizon.uncertainty.aleatoric.X // Map X to M5
            }
          }
        }))
      };
      
      console.log('✅ Loaded real forecast data (X→M5 mapped):', data);
      console.log('Sample horizon M5 value:', data.horizons[0]?.softmax_dense?.M5);
      return data;
    } else {
      console.warn('⚠️ Failed to load forecast data, using fallback');
      return createMockForecast();
    }
  } catch (error) {
    console.warn('⚠️ Error loading forecast data, using fallback:', error);
    return createMockForecast();
  }
};

// Function to load real temporal evolution data
const loadTemporalEvolutionData = async (): Promise<TemporalEvolutionData> => {
  try {
    const response = await fetch('/src/data/temporal_evolution.json?t=' + Date.now());
    if (response.ok) {
      const rawData = await response.json();
      
      // Transform prob_X to prob_M5 in the loaded data
      const data: TemporalEvolutionData = {
        series: rawData.series.map((point: any) => ({
          timestamp: point.timestamp,
          prob_C: point.prob_C,
          prob_M: point.prob_M,
          prob_M5: point.prob_X, // Map prob_X to prob_M5
          epi: point.epi,
          alea: point.alea
        }))
      };
      
      console.log('✅ Loaded real temporal evolution data (prob_X→prob_M5 mapped):', data.series?.length, 'points');
      console.log('Sample temporal M5 value:', data.series[0]?.prob_M5);
      return data;
    } else {
      console.warn('⚠️ Failed to load temporal evolution data, using fallback');
      return createMockTemporalEvolution();
    }
  } catch (error) {
    console.warn('⚠️ Error loading temporal evolution data, using fallback:', error);
    return createMockTemporalEvolution();
  }
};

// Create mock forecast data
const createMockForecast = (): ForecastData => {
  const now = new Date();
  return {
    generated_at: now.toISOString(),
    horizons: [
      {
        hours: 24,
        softmax_dense: { C: 0.35, M: 0.15, M5: 0.02 },
        uncertainty: {
          epistemic: { C: 0.10, M: 0.08, M5: 0.03 },
          aleatoric: { C: 0.05, M: 0.07, M5: 0.02 }
        }
      },
      {
        hours: 48,
        softmax_dense: { C: 0.28, M: 0.09, M5: 0.01 },
        uncertainty: {
          epistemic: { C: 0.12, M: 0.06, M5: 0.02 },
          aleatoric: { C: 0.07, M: 0.04, M5: 0.01 }
        }
      },
      {
        hours: 72,
        softmax_dense: { C: 0.22, M: 0.07, M5: 0.005 },
        uncertainty: {
          epistemic: { C: 0.14, M: 0.05, M5: 0.01 },
          aleatoric: { C: 0.09, M: 0.03, M5: 0.005 }
        }
      }
    ]
  };
};

// Create mock temporal evolution data - generate once to avoid infinite renders
const createMockTemporalEvolution = (): TemporalEvolutionData => {
  const now = new Date();
  const series = [];
  
  // For deterministic behavior, use fixed seed values with more distinct patterns
  // Create more visible patterns with larger amplitude and clearer trends
  const seedValues = Array(145).fill(0).map((_, i) => Math.sin(i * 0.5) * 0.5 + 0.5);
  
  // Generate data for the past 72 hours (3 days)
  for (let i = 72; i >= 0; i--) {
    const timestamp = subHours(now, i).toISOString();
    
    // Create more pronounced patterns with trending values that are easily visible
    const timeFactor = i / 72; // Normalized time factor (0 to 1)
    const sinWave = Math.sin(i / 6) * 0.25; // Shorter period, more oscillations
    const trend = 0.15 + (0.3 * timeFactor); // Rising trend from past to present
    
    // Add some deterministic variation
    const variation = seedValues[i] * 0.15;
    
    // C-class probability: Higher baseline with periodic variations
    const probC = Math.min(0.85, Math.max(0.15, trend + sinWave + variation));
    
    // M-class: Lower baseline but follows similar pattern with 0.5 scaling and slight phase shift
    const probM = Math.min(0.45, Math.max(0.05, (trend * 0.5) + Math.sin((i + 10) / 8) * 0.15 + variation * 0.5));
    
    // M5-class: Much lower baseline with occasional spikes
    const probX = Math.min(0.12, Math.max(0.001, (trend * 0.15) + (Math.sin((i + 20) / 4) > 0.7 ? 0.08 : 0.01) + variation * 0.2));
    
    series.push({
      timestamp,
      prob_C: probC,
      prob_M: probM,
      prob_M5: probX,
      epi: 0.08 + seedValues[i + 72] * 0.04,
      alea: 0.05 + seedValues[i + 72] * 0.03
    });
  }
  
  // Generate data for the next 72 hours (future predictions)
  for (let i = 1; i <= 72; i++) {
    const timestamp = addHours(now, i).toISOString();
    
    // Create future trend with increasing uncertainty
    const timeFactor = i / 72; // Normalized time factor (0 to 1)
    const sinWave = Math.sin(i / 8) * 0.2; // Smoother oscillations for future
    const trend = 0.45 - (0.1 * timeFactor); // Slightly decreasing trend for future
    
    // Add some deterministic variation
    const variation = seedValues[i + 72] * (0.12 + timeFactor * 0.1); // Growing variation (uncertainty) with time
    
    // C-class probability: Continues from current level with some projected pattern
    const probC = Math.min(0.85, Math.max(0.15, trend + sinWave + variation));
    
    // M-class: Similar pattern with different phase
    const probM = Math.min(0.45, Math.max(0.05, (trend * 0.5) + Math.sin((i + 15) / 10) * 0.15 + variation * 0.5));
    
    // M5-class: Much lower with occasional projected spikes
    const probX = Math.min(0.12, Math.max(0.001, (trend * 0.15) + (Math.sin((i + 30) / 5) > 0.8 ? 0.07 : 0.01) + variation * 0.2));
    
    // Future uncertainty grows with time
    const futureUncertaintyFactor = 1 + (timeFactor * 0.5);
    
    series.push({
      timestamp,
      prob_C: probC,
      prob_M: probM,
      prob_M5: probX,
      epi: Math.min(0.2, 0.08 + seedValues[i] * 0.04 * futureUncertaintyFactor),
      alea: Math.min(0.15, 0.05 + seedValues[i] * 0.03 * futureUncertaintyFactor)
    });
  }
  
  return { series };
};

// Create mock active regions data
const createMockActiveRegions = (): Record<number, ActiveRegionData> => {
  return {
    13297: {
      noaa_id: 13297,
      location: "N16W34",
      sharp: {
        usflux: 2.7e22,
        r: 0.51,
        wlsg: 12.4,
        entrop: 4.2,
        savncpp: 217,
        totbsq: 8.31e7,
        absnjzh: 5.43e12
      },
      saliency_map_url: "https://example.com/saliency/13297_20250506.png",
      evt: { xi: 0.12, sigma: 0.48 },
      regional_softmax: { C: 0.48, M: 0.12, M5: 0.01 }
    },
    13298: {
      noaa_id: 13298,
      location: "S23E17",
      sharp: {
        usflux: 1.9e22,
        r: 0.38,
        wlsg: 9.7,
        entrop: 3.8,
        savncpp: 156,
        totbsq: 6.42e7,
        absnjzh: 4.12e12
      },
      saliency_map_url: "https://example.com/saliency/13298_20250506.png",
      evt: { xi: 0.09, sigma: 0.32 },
      regional_softmax: { C: 0.36, M: 0.09, M5: 0.005 }
    },
    13301: {
      noaa_id: 13301,
      location: "N04W78",
      sharp: {
        usflux: 3.1e22,
        r: 0.62,
        wlsg: 15.2,
        entrop: 4.7,
        savncpp: 298,
        totbsq: 9.15e7,
        absnjzh: 6.78e12
      },
      saliency_map_url: "https://example.com/saliency/13301_20250506.png",
      evt: { xi: 0.15, sigma: 0.53 },
      regional_softmax: { C: 0.52, M: 0.18, M5: 0.04 }
    }
  };
};

// Create mock performance data
const createMockPerformance = (): PerformanceData => {
  return {
    metric_window_days: 30,
    scores: {
      TSS: 0.83,
      BSS: 0.21,
      F1: { C: 0.71, M: 0.45, M5: 0.08 },
      precision: { C: 0.75, M: 0.48, M5: 0.12 },
      recall: { C: 0.68, M: 0.42, M5: 0.06 }
    },
    reliability_bins: [
      { bin: "0-0.1", count: 318, observed: 0.06 },
      { bin: "0.1-0.2", count: 256, observed: 0.14 },
      { bin: "0.2-0.3", count: 187, observed: 0.26 },
      { bin: "0.3-0.4", count: 143, observed: 0.35 },
      { bin: "0.4-0.5", count: 98, observed: 0.48 },
      { bin: "0.5-0.6", count: 67, observed: 0.54 },
      { bin: "0.6-0.7", count: 45, observed: 0.65 },
      { bin: "0.7-0.8", count: 32, observed: 0.74 },
      { bin: "0.8-0.9", count: 18, observed: 0.83 },
      { bin: "0.9-1.0", count: 9, observed: 0.92 }
    ]
  };
};

// Create mock solar context data with deterministic values
const createMockSolarContext = (): SolarContextData => {
  const now = new Date();
  const goesXray = [];
  
  // Use deterministic values instead of random
  const seedValues = Array(289).fill(0).map((_, i) => Math.sin(i * 0.7) * 0.5 + 0.5);
  
  // Generate GOES X-ray flux for the past 24 hours
  for (let i = 24 * 60; i >= 0; i -= 5) { // 5-minute intervals
    const timestamp = subHours(now, i / 60).toISOString();
    // Base level flux with some deterministic variation
    let flux = 1.2e-7; // Base B-class flux
    
    // Add some variation based on seedValues
    const seedIndex = Math.floor(i / 5) % seedValues.length;
    flux += seedValues[seedIndex] * 5e-7;
    
    // Add occasional flares based on deterministic pattern
    if (seedValues[seedIndex] > 0.98) {
      // M5-class flare (very rare)
      flux = 1e-4 + seedValues[seedIndex] * 9e-4;
    } else if (seedValues[seedIndex] > 0.8) {
      // M-class flare (uncommon)
      flux = 1e-5 + seedValues[seedIndex] * 9e-5;
    } else if (seedValues[seedIndex] > 0.6) {
      // C-class flare (common)
      flux = 1e-6 + seedValues[seedIndex] * 9e-6;
    }
    
    goesXray.push({ timestamp, flux });
  }
  
  return {
    synoptic_url: "https://example.com/synoptic/synoptic_20250506.png",
    goes_xray: goesXray,
    cme_feed: [
      { time: subHours(now, 8).toISOString(), speed: 850, width: 120, source_ar: 13297 },
      { time: subHours(now, 18).toISOString(), speed: 1200, width: 180, source_ar: 13301 },
      { time: subHours(now, 36).toISOString(), speed: 650, width: 90, source_ar: 13298 }
    ]
  };
};

// Create mock alert center data
const createMockAlertCenter = (): AlertCenterData => {
  const now = new Date();
  
  return {
    user_rules: [
      { id: 9, class: 'M', prob_threshold: 0.25, max_epi: 0.15, horizon: 24 },
      { id: 12, class: 'M5', prob_threshold: 0.05, max_epi: 0.10, horizon: 48 },
      { id: 7, class: 'C', prob_threshold: 0.50, max_epi: 0.20, horizon: 12 }
    ],
    active_alerts: [
      { rule_id: 9, triggered_at: subHours(now, 2).toISOString(), state: 'firing' },
      { rule_id: 7, triggered_at: subHours(now, 5).toISOString(), state: 'firing' }
    ],
    history: [
      { rule_id: 12, triggered_at: subHours(now, 48).toISOString(), state: 'resolved' },
      { rule_id: 9, triggered_at: subHours(now, 72).toISOString(), state: 'resolved' },
      { rule_id: 7, triggered_at: subHours(now, 96).toISOString(), state: 'resolved' }
    ]
  };
};

// Generate the mock data ONCE before creating the store
const mockForecast = createMockForecast();
const mockTemporalEvolution = createMockTemporalEvolution();
const mockActiveRegions = createMockActiveRegions();
const mockPerformance = createMockPerformance();
const mockSolarContext = createMockSolarContext();
const mockAlertCenter = createMockAlertCenter();

// Debug: log the temporal evolution data to verify it's being created correctly
console.log('STORE DEBUG - Temporal evolution data points:', mockTemporalEvolution.series.length);
if (mockTemporalEvolution.series.length > 0) {
  console.log('STORE DEBUG - First data point:', mockTemporalEvolution.series[0]);
  console.log('STORE DEBUG - Last data point:', mockTemporalEvolution.series[mockTemporalEvolution.series.length - 1]);
} else {
  console.error('STORE DEBUG - No temporal evolution data points generated!');
}

// Initialize store with pre-generated mock data
const initialState: AppState = {
  forecast: mockForecast,
  temporalEvolution: mockTemporalEvolution,
  activeRegions: mockActiveRegions,
  performance: mockPerformance,
  solarContext: mockSolarContext,
  alertCenter: mockAlertCenter,
  ui: {
    selectedHorizon: 24,
    selectedActiveRegion: null,
    showActiveRegionDrawer: false
  }
};

// Define actions
type Actions = {
  setSelectedHorizon: (hours: number) => void;
  showActiveRegionDetail: (regionId: number) => void;
  hideActiveRegionDetail: () => void;
  toggleHorizon: () => void;
  updateForecastData: (data: ForecastData) => void;
  loadRealData: () => Promise<void>;
};

// Create and export the store
export const useStore = create<AppState & Actions>((set, get) => ({
  ...initialState,
  
  // UI actions
  setSelectedHorizon: (hours) => set({ ui: { ...get().ui, selectedHorizon: hours } }),
  
  showActiveRegionDetail: (regionId) => set({ 
    ui: { 
      ...get().ui, 
      selectedActiveRegion: regionId,
      showActiveRegionDrawer: true
    } 
  }),
  
  hideActiveRegionDetail: () => set({ 
    ui: { 
      ...get().ui, 
      showActiveRegionDrawer: false 
    } 
  }),
  
  toggleHorizon: () => {
    const currentHorizon = get().ui.selectedHorizon;
    const horizons = [24, 48, 72];
    const currentIndex = horizons.indexOf(currentHorizon);
    const nextIndex = (currentIndex + 1) % horizons.length;
    set({ ui: { ...get().ui, selectedHorizon: horizons[nextIndex] } });
  },
  
  // Data actions
  updateForecastData: (data) => set({ forecast: data }),
  
  // Load real data
  loadRealData: async () => {
    try {
      console.log('🔄 Loading real forecast data...');
      const [forecastData, temporalData] = await Promise.all([
        loadForecastData(),
        loadTemporalEvolutionData()
      ]);
      
      set({ 
        forecast: forecastData,
        temporalEvolution: temporalData
      });
      
      console.log('✅ Real data loaded successfully');
    } catch (error) {
      console.error('❌ Failed to load real data:', error);
    }
  }
}));

// Load real data on store initialization
const store = useStore.getState();
if (typeof window !== 'undefined') {
  // Only load in browser environment
  store.loadRealData();
} 