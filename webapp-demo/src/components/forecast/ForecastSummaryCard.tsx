import React, { useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { format } from 'date-fns';
import GlassPanel from '../common/GlassPanel';
import ProbabilityBar from '../common/ProbabilityBar';
import { useStore } from '../../store';
import type { PredictionHorizon } from '../../types/everest';

/**
 * Component that displays the forecast summary with probability bars
 * for different flare classes over selected time horizons
 */
const ForecastSummaryCard: React.FC = () => {
  // Only select the specific state pieces we need
  const forecast = useStore(state => state.forecast);
  const selectedHorizon = useStore(state => state.ui.selectedHorizon);
  const toggleHorizon = useStore(state => state.toggleHorizon);
  
  // Find the selected horizon data using useMemo to prevent unnecessary recalculations
  const selectedHorizonData = useMemo(() => {
    const data = forecast.horizons.find(h => h.hours === selectedHorizon);
    return data || forecast.horizons[0]; // Fallback to first horizon if not found
  }, [forecast.horizons, selectedHorizon]);
  
  // Get all available horizons for display
  const availableHorizons = useMemo(() => {
    return forecast.horizons.map(h => h.hours).sort((a, b) => a - b);
  }, [forecast.horizons]);
  
  // Format generation time
  const formattedDate = useMemo(() => {
    const generatedAt = new Date(forecast.generated_at);
    return format(generatedAt, 'MMM d, yyyy');
  }, [forecast.generated_at]);
  
  const formattedTime = useMemo(() => {
    const generatedAt = new Date(forecast.generated_at);
    return format(generatedAt, 'HH:mm z');
  }, [forecast.generated_at]);
  
  // Child staggered animation
  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };
  
  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <GlassPanel className="p-5 md:p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-medium text-snow tracking-tight">
          Flare Forecast
        </h2>
        
        {/* Enhanced horizon selector */}
        <div className="flex items-center space-x-2">
          <span className="text-xs text-glacier-300 hidden sm:block">Time Horizon:</span>
          <motion.button 
            onClick={toggleHorizon}
            className="relative group bg-everest-800 hover:bg-everest-700 border border-white/10 hover:border-white/20 rounded-lg px-3 py-2 transition-all duration-200"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="flex items-center space-x-2">
              {/* Current selection */}
              <span className="text-sm font-medium text-snow">
                {selectedHorizon}h
              </span>
              
              {/* Dropdown indicator */}
              <svg 
                className="w-3 h-3 text-glacier-300 group-hover:text-snow transition-colors" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
            
            {/* Available options indicator */}
            <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 flex space-x-1">
              {availableHorizons.map((horizon) => (
                <div
                  key={horizon}
                  className={`w-1.5 h-1.5 rounded-full transition-all duration-200 ${
                    horizon === selectedHorizon 
                      ? 'bg-glacier-300 scale-110' 
                      : 'bg-white/20 group-hover:bg-white/40'
                  }`}
                />
              ))}
            </div>
            
            {/* Tooltip on hover */}
            <div className="absolute top-full mt-2 left-1/2 transform -translate-x-1/2 bg-everest-900 border border-white/10 rounded-lg px-3 py-2 text-xs text-glacier-300 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap z-10">
              <div className="text-center">
                <div className="text-snow font-medium mb-1">Available Horizons</div>
                <div className="flex space-x-2">
                  {availableHorizons.map((horizon, index) => (
                    <span key={horizon} className={horizon === selectedHorizon ? 'text-glacier-300 font-medium' : 'text-glacier-400'}>
                      {horizon}h{index < availableHorizons.length - 1 ? ' •' : ''}
                    </span>
                  ))}
                </div>
                <div className="text-glacier-400 mt-1">Click to cycle</div>
              </div>
              
              {/* Tooltip arrow */}
              <div className="absolute -top-1 left-1/2 transform -translate-x-1/2 w-2 h-2 bg-everest-900 border-l border-t border-white/10 rotate-45"></div>
            </div>
          </motion.button>
        </div>
      </div>
      
      <div className="text-sm text-glacier-300 mb-4">
        Generated {formattedDate} at {formattedTime}
      </div>
      
      <AnimatePresence mode="wait">
        <motion.div
          key={selectedHorizon}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 20 }}
          transition={{ duration: 0.3 }}
          className="space-y-6"
        >
          <motion.div variants={container} initial="hidden" animate="show">
            <motion.div variants={item} className="mb-6">
              <ProbabilityBar
                label="C-class flare"
                probability={selectedHorizonData.softmax_dense.C}
                epistemicUncertainty={selectedHorizonData.uncertainty.epistemic.C}
                aleatoricUncertainty={selectedHorizonData.uncertainty.aleatoric.C}
                variant="C"
              />
            </motion.div>
            
            <motion.div variants={item} className="mb-6">
              <ProbabilityBar
                label="M-class flare"
                probability={selectedHorizonData.softmax_dense.M}
                epistemicUncertainty={selectedHorizonData.uncertainty.epistemic.M}
                aleatoricUncertainty={selectedHorizonData.uncertainty.aleatoric.M}
                variant="M"
              />
            </motion.div>
            
            <motion.div variants={item}>
              <ProbabilityBar
                label="M5-class flare"
                probability={selectedHorizonData.softmax_dense.M5}
                epistemicUncertainty={selectedHorizonData.uncertainty.epistemic.M5}
                aleatoricUncertainty={selectedHorizonData.uncertainty.aleatoric.M5}
                variant="M5"
              />
            </motion.div>
          </motion.div>
        </motion.div>
      </AnimatePresence>
      
      <div className="mt-6 text-xs text-glacier-300">
        <p>
          <span className="inline-block w-3 h-3 bg-white/10 border border-dashed border-white/20 mr-1 rounded-sm"></span>
          Epistemic uncertainty (model confidence)
        </p>
        <p>
          <span className="inline-block w-3 h-3 bg-white/20 mr-1 rounded-sm"></span>
          Aleatoric uncertainty (inherent randomness)
        </p>
      </div>
    </GlassPanel>
  );
};

export default ForecastSummaryCard; 