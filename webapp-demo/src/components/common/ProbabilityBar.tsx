import React from 'react';
import { motion } from 'framer-motion';

interface ProbabilityBarProps {
  label: string;
  probability: number;
  epistemicUncertainty?: number;
  aleatoricUncertainty?: number;
  variant: 'C' | 'M' | 'M5';
  showTooltip?: boolean;
  className?: string;
}

/**
 * A horizontal probability bar with uncertainty visualization
 * - Inner band represents aleatoric uncertainty (inherent randomness)
 * - Outer dashed line represents epistemic uncertainty (model uncertainty)
 */
const ProbabilityBar: React.FC<ProbabilityBarProps> = ({
  label,
  probability,
  epistemicUncertainty = 0,
  aleatoricUncertainty = 0,
  variant,
  showTooltip = true,
  className = ''
}) => {
  // Format probability as percentage
  const probabilityPercent = Math.round(probability * 100);
  
  // Calculate uncertainty bounds
  const aleaLower = Math.max(0, probability - aleatoricUncertainty);
  const aleaUpper = Math.min(1, probability + aleatoricUncertainty);
  const epiLower = Math.max(0, probability - epistemicUncertainty);
  const epiUpper = Math.min(1, probability + epistemicUncertainty);
  
  // Define colors based on flare class
  const getVariantColors = () => {
    switch (variant) {
      case 'C':
        return {
          bar: 'bg-gradient-to-r from-aurora-green to-glacier-300',
          aleatoric: 'bg-aurora-green/40',
          epistemic: 'border-aurora-green/30'
        };
      case 'M':
        return {
          bar: 'bg-gradient-to-r from-glacier-300 to-glacier-600',
          aleatoric: 'bg-glacier-300/40',
          epistemic: 'border-glacier-300/30'
        };
      case 'M5':
        return {
          bar: 'bg-gradient-to-r from-glacier-600 to-aurora-purple',
          aleatoric: 'bg-aurora-purple/40',
          epistemic: 'border-aurora-purple/30'
        };
    }
  };
  
  const colors = getVariantColors();
  
  // Format tooltip content
  const tooltipContent = `
    Probability: ${probabilityPercent}%
    Aleatoric uncertainty: ±${Math.round(aleatoricUncertainty * 100)}%
    Epistemic uncertainty: ±${Math.round(epistemicUncertainty * 100)}%
  `;

  return (
    <div className={`relative w-full ${className}`}>
      <div className="flex justify-between mb-1">
        <span className="text-sm font-medium text-glacier-300">{label}</span>
        <span className="text-sm font-medium tabular-nums text-snow">
          {probabilityPercent}%
        </span>
      </div>
      
      {/* Base bar background */}
      <div className="h-4 bg-everest-1000 rounded-full overflow-hidden relative">
        {/* Main probability bar */}
        <motion.div 
          className={`h-full ${colors.bar} rounded-full`}
          initial={{ width: 0 }}
          animate={{ width: `${probabilityPercent}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
        
        {/* Aleatoric uncertainty band */}
        <div 
          className={`absolute top-0 h-full ${colors.aleatoric} rounded-full`}
          style={{
            left: `${aleaLower * 100}%`,
            width: `${(aleaUpper - aleaLower) * 100}%`
          }}
        />
        
        {/* Epistemic uncertainty band (dashed border) */}
        <div 
          className={`absolute top-0 h-full rounded-full border border-dashed ${colors.epistemic}`}
          style={{
            left: `${epiLower * 100}%`,
            width: `${(epiUpper - epiLower) * 100}%`
          }}
        />
      </div>
      
      {/* Tooltip */}
      {showTooltip && (
        <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 absolute mt-1 p-2 bg-everest-800 rounded text-xs text-snow">
          <pre className="whitespace-pre-wrap">{tooltipContent}</pre>
        </div>
      )}
    </div>
  );
};

export default ProbabilityBar; 