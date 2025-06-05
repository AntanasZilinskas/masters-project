import React, { useState } from 'react';
import { motion } from 'framer-motion';

interface ProbabilityBarProps {
  label: string;
  probability: number;
  aleatoricUncertainty: number;
  epistemicUncertainty: number;
  className?: string;
  colorClass: string;
}

const ProbabilityBar: React.FC<ProbabilityBarProps> = ({
  label,
  probability,
  aleatoricUncertainty,
  epistemicUncertainty,
  className = '',
  colorClass
}) => {
  const [showTooltip, setShowTooltip] = useState(false);
  
  // Calculate the total width percentages for the bar and uncertainty bands
  const mainWidth = Math.min(100, Math.max(0, probability * 100));
  const aleaWidth = Math.min(100, Math.max(0, (probability + aleatoricUncertainty) * 100));
  const epiWidth = Math.min(100, Math.max(0, (probability + aleatoricUncertainty + epistemicUncertainty) * 100));
  
  return (
    <div 
      className={`w-full h-8 relative mb-3 ${className}`}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <div className="flex items-center h-full">
        <div className="w-12 text-glacier-300 text-right pr-2 font-medium tracking-tight">
          {label}
        </div>
        
        <div className="relative flex-1 h-4 bg-everest-1000 rounded-full overflow-hidden">
          {/* Epistemic uncertainty (outer band, dashed border) */}
          <motion.div 
            className={`absolute top-0 h-full border-dashed border-2 border-${colorClass}/30 rounded-full`}
            style={{ width: `${epiWidth}%` }}
            initial={{ width: 0 }}
            animate={{ width: `${epiWidth}%` }}
            transition={{ duration: 0.7, ease: "easeOut" }}
          />
          
          {/* Aleatoric uncertainty (semi-transparent band) */}
          <motion.div 
            className={`absolute top-0 h-full bg-${colorClass}/40 rounded-full`}
            style={{ width: `${aleaWidth}%` }}
            initial={{ width: 0 }}
            animate={{ width: `${aleaWidth}%` }}
            transition={{ duration: 0.6, ease: "easeOut" }}
          />
          
          {/* Main probability bar */}
          <motion.div 
            className={`absolute top-0 h-full rounded-full ${
              colorClass === 'aurora-purple' 
                ? 'bg-gradient-to-r from-aurora-purple to-glacier-300' 
                : 'bg-gradient-to-r from-aurora-green to-glacier-300'
            }`}
            style={{ width: `${mainWidth}%` }}
            initial={{ width: 0 }}
            animate={{ width: `${mainWidth}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          />
        </div>
        
        <div className="w-12 text-snow text-right pl-2 tabular-nums">
          {(probability * 100).toFixed(0)}%
        </div>
      </div>
      
      {/* Tooltip */}
      {showTooltip && (
        <div className="absolute -top-14 right-0 glass-panel p-2 text-xs z-10">
          <div className="flex justify-between gap-3">
            <div>μ: {probability.toFixed(2)}</div>
            <div>α: {aleatoricUncertainty.toFixed(2)}</div>
            <div>ε: {epistemicUncertainty.toFixed(2)}</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProbabilityBar; 