import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ProbabilityBar from '../common/ProbabilityBar';
import { useStore } from '../../store';
import type { ActiveRegionData } from '../../types/everest';

/**
 * A slidable drawer that shows details about a specific solar active region
 * including magnetogram image, probability bars, and SHARP parameters
 */
const ActiveRegionDetailDrawer: React.FC = () => {
  // Use individual selectors for better performance
  const selectedActiveRegion = useStore(state => state.ui.selectedActiveRegion);
  const showActiveRegionDrawer = useStore(state => state.ui.showActiveRegionDrawer);
  const activeRegions = useStore(state => state.activeRegions);
  const hideActiveRegionDetail = useStore(state => state.hideActiveRegionDetail);
  
  // Get the selected active region data
  const activeRegion = useMemo(() => {
    return selectedActiveRegion !== null 
      ? activeRegions[selectedActiveRegion] 
      : null;
  }, [selectedActiveRegion, activeRegions]);
  
  // Track whether the image has been loaded
  const [imageLoaded, setImageLoaded] = useState(false);
  
  // Reset image loaded state when region changes
  useEffect(() => {
    setImageLoaded(false);
  }, [selectedActiveRegion]);
  
  // Format SHARP parameters for display - memoize to avoid recalculation
  const formatParameter = useCallback((value: number): string => {
    if (value === 0) return '0';
    
    // Handle scientific notation for very small or large numbers
    if (Math.abs(value) < 0.01 || Math.abs(value) > 100000) {
      return value.toExponential(2);
    }
    
    return value.toFixed(2);
  }, []);
  
  // If no active region is selected, don't render anything
  if (!activeRegion || !showActiveRegionDrawer) {
    return null;
  }
  
  // Calculate EVT tail properties (for visualization)
  const getExceedanceProbabilities = useCallback((evt: { xi: number; sigma: number }) => {
    const probabilities = [];
    const thresholds = [1, 5, 10, 50, 100, 200, 500, 1000];
    
    for (const threshold of thresholds) {
      // Generalized Pareto Distribution exceedance calculation (simplified)
      const prob = Math.pow(1 + evt.xi * threshold / evt.sigma, -1 / evt.xi);
      probabilities.push({ threshold, probability: prob });
    }
    
    return probabilities;
  }, []);
  
  // Memoize exceedance probabilities calculation
  const exceedanceProbabilities = useMemo(() => {
    return getExceedanceProbabilities(activeRegion.evt);
  }, [activeRegion.evt, getExceedanceProbabilities]);
  
  // Memoize image on load handler
  const handleImageLoad = useCallback(() => {
    setImageLoaded(true);
  }, []);
  
  // Handle JSON export
  const handleExportJSON = useCallback(() => {
    const blob = new Blob(
      [JSON.stringify(activeRegion, null, 2)], 
      { type: 'application/json' }
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AR${activeRegion.noaa_id}_data.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [activeRegion]);
  
  return (
    <AnimatePresence>
      {showActiveRegionDrawer && (
        <>
          {/* Backdrop overlay */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black z-40"
            onClick={hideActiveRegionDetail}
          />
          
          {/* Drawer panel */}
          <motion.div
            initial={{ y: '100%' }}
            animate={{ y: '0%' }}
            exit={{ y: '100%' }}
            transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
            className="fixed bottom-0 left-0 right-0 bg-gradient-to-b from-everest-800 to-everest-1000 glass-panel-heavy rounded-t-2xl z-50 overflow-hidden"
            style={{ height: '70vh' }}
          >
            {/* Header bar */}
            <div className="flex justify-between items-center p-4 border-b border-white/10">
              <div className="flex items-center">
                <h2 className="text-xl font-medium text-snow">
                  Active Region {activeRegion.noaa_id}
                </h2>
                <span className="ml-3 px-2 py-1 bg-white/10 rounded text-sm text-glacier-300">
                  {activeRegion.location}
                </span>
              </div>
              
              <button
                onClick={hideActiveRegionDetail}
                className="text-glacier-300 hover:text-snow p-2"
              >
                <svg className="w-5 h-5" fill="none" strokeWidth="2" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            {/* Content */}
            <div className="p-6 h-full overflow-y-auto">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Left column - Magnetogram */}
                <div>
                  <h3 className="text-lg font-medium text-snow mb-3">Magnetogram with Saliency</h3>
                  <div className="relative aspect-square bg-white/5 rounded-xl overflow-hidden border border-white/10">
                    {!imageLoaded && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-glacier-300">Loading magnetogram...</div>
                      </div>
                    )}
                    <motion.img
                      src={activeRegion.saliency_map_url}
                      alt={`AR ${activeRegion.noaa_id} magnetogram with integrated gradient overlay`}
                      className="w-full h-full object-cover"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: imageLoaded ? 1 : 0 }}
                      onLoad={handleImageLoad}
                    />
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-everest-1000/80 p-3 text-xs text-glacier-300">
                      Heatmap shows areas with highest prediction influence
                    </div>
                  </div>
                </div>
                
                {/* Right column - Data */}
                <div>
                  <h3 className="text-lg font-medium text-snow mb-3">Region Forecast</h3>
                  
                  {/* Probability bars */}
                  <div className="space-y-3 mb-6">
                    <ProbabilityBar
                      label="C-class"
                      probability={activeRegion.regional_softmax.C}
                      variant="C"
                    />
                    <ProbabilityBar
                      label="M-class"
                      probability={activeRegion.regional_softmax.M}
                      variant="M"
                    />
                                          <ProbabilityBar
                        label="M5-class"
                        probability={activeRegion.regional_softmax.M5}
                        variant="M5"
                    />
                  </div>
                  
                  {/* SHARP Parameters */}
                  <h3 className="text-lg font-medium text-snow mb-3 mt-6">SHARP Parameters</h3>
                  <div className="grid grid-cols-2 gap-3 mb-6">
                    <div className="glass-panel p-3">
                      <div className="text-xs text-glacier-300">Total Unsigned Flux</div>
                      <div className="text-sm text-snow font-mono tabular-nums">
                        {formatParameter(activeRegion.sharp.usflux)}
                      </div>
                    </div>
                    <div className="glass-panel p-3">
                      <div className="text-xs text-glacier-300">R Value</div>
                      <div className="text-sm text-snow font-mono tabular-nums">
                        {formatParameter(activeRegion.sharp.r)}
                      </div>
                    </div>
                    <div className="glass-panel p-3">
                      <div className="text-xs text-glacier-300">WLSG</div>
                      <div className="text-sm text-snow font-mono tabular-nums">
                        {formatParameter(activeRegion.sharp.wlsg)}
                      </div>
                    </div>
                    <div className="glass-panel p-3">
                      <div className="text-xs text-glacier-300">SAVNCPP</div>
                      <div className="text-sm text-snow font-mono tabular-nums">
                        {formatParameter(activeRegion.sharp.savncpp)}
                      </div>
                    </div>
                  </div>
                  
                  {/* EVT Tail Visualization */}
                  <h3 className="text-lg font-medium text-snow mb-3">Extreme Value Analysis</h3>
                  <div className="glass-panel p-4">
                    <div className="flex justify-between mb-2">
                      <span className="text-sm text-glacier-300">ξ (tail shape):</span>
                      <span className="text-sm text-snow font-mono">{activeRegion.evt.xi.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between mb-4">
                      <span className="text-sm text-glacier-300">σ (scale):</span>
                      <span className="text-sm text-snow font-mono">{activeRegion.evt.sigma.toFixed(3)}</span>
                    </div>
                    
                    <div className="h-24 relative">
                      <div className="absolute left-0 bottom-0 h-full border-l border-white/20"></div>
                      <div className="absolute left-0 bottom-0 w-full border-b border-white/20"></div>
                      
                      {/* Exceedance probability curve */}
                      <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
                        <path
                          d={`M 0,${100 - exceedanceProbabilities[0].probability * 100} 
                              ${exceedanceProbabilities.map((p, i) => 
                                `L ${i * 100 / (exceedanceProbabilities.length - 1)},${100 - p.probability * 100}`
                              ).join(' ')}`}
                          fill="none"
                          stroke="#B569FF"
                          strokeWidth="2"
                          className="aurora-purple-glow"
                        />
                      </svg>
                      
                      {/* Axis labels */}
                      <div className="absolute -bottom-6 left-0 text-xs text-glacier-300">1</div>
                      <div className="absolute -bottom-6 right-0 text-xs text-glacier-300">1000</div>
                      <div className="absolute top-0 -left-6 text-xs text-glacier-300">1.0</div>
                      <div className="absolute bottom-0 -left-6 text-xs text-glacier-300">0.0</div>
                      
                      <div className="absolute -bottom-12 w-full text-center text-xs text-glacier-300">
                        Flare Magnitude (× Background)
                      </div>
                    </div>
                    <div className="text-xs text-glacier-300 mt-4">
                      Probability of exceeding flux threshold
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Export button */}
              <div className="mt-6 flex justify-end">
                <button
                  className="px-4 py-2 rounded bg-glacier-600/20 hover:bg-glacier-600/30 text-glacier-300 hover:text-snow transition-colors text-sm"
                  onClick={handleExportJSON}
                >
                  Export JSON
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default ActiveRegionDetailDrawer; 