import React from 'react';
import { motion } from 'framer-motion';

interface GlassPanelProps {
  children: React.ReactNode;
  className?: string;
  heavy?: boolean; // For heavier blur effect
  animate?: boolean; // Whether to animate the panel on mount
}

/**
 * A glassmorphic panel component with frosted glass appearance
 */
const GlassPanel: React.FC<GlassPanelProps> = ({ 
  children, 
  className = '', 
  heavy = false,
  animate = true
}) => {
  const baseClass = heavy 
    ? 'glass-panel-heavy' 
    : 'glass-panel';

  const containerProps = animate 
    ? {
        initial: { opacity: 0, y: 10 },
        animate: { opacity: 1, y: 0 },
        transition: { 
          duration: 0.4,
          ease: [0.25, 0.1, 0.25, 1.0] // Cubic bezier easing
        }
      } 
    : {};

  return (
    <motion.div 
      className={`${baseClass} ${className}`}
      {...containerProps}
    >
      {children}
    </motion.div>
  );
};

export default GlassPanel; 