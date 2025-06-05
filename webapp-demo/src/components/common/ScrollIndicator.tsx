import React from 'react';
import { motion } from 'framer-motion';

interface ScrollIndicatorProps {
  currentSection: number;
  totalSections: number;
  onSectionClick: (sectionIndex: number) => void;
}

/**
 * A vertical scroll indicator showing current section and allowing navigation
 */
const ScrollIndicator: React.FC<ScrollIndicatorProps> = ({
  currentSection,
  totalSections,
  onSectionClick
}) => {
  return (
    <div className="fixed right-4 top-1/2 transform -translate-y-1/2 z-30 hidden md:flex flex-col space-y-4">
      {Array.from({ length: totalSections }, (_, index) => (
        <motion.button
          key={index}
          onClick={() => onSectionClick(index)}
          className={`w-3 h-3 rounded-full border-2 transition-all duration-300 ${
            currentSection === index
              ? 'bg-glacier-300 border-glacier-300 shadow-lg shadow-glacier-300/30'
              : 'bg-transparent border-glacier-600 hover:border-glacier-300'
          }`}
          whileHover={{ scale: 1.2 }}
          whileTap={{ scale: 0.9 }}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 + 1.5 }}
        >
          <span className="sr-only">Go to section {index + 1}</span>
        </motion.button>
      ))}
    </div>
  );
};

export default ScrollIndicator; 