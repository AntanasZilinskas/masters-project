import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ForecastSummaryCard from '../forecast/ForecastSummaryCard';
import TemporalEvolutionPanel from '../forecast/TemporalEvolutionPanel';
import ActiveRegionDetailDrawer from '../forecast/ActiveRegionDetailDrawer';
import ModelPerformancePanel from '../forecast/ModelPerformancePanel';
import ScrollIndicator from '../common/ScrollIndicator';
// Import other components when they're created
// import PerformanceWorkspace from '../performance/PerformanceWorkspace';
// import SolarContextBoard from '../context/SolarContextBoard';
// import AlertCenter from '../alerts/AlertCenter';

/**
 * Main dashboard layout component with smooth scrolling sections
 */
const Dashboard: React.FC = () => {
  const [currentSection, setCurrentSection] = useState(0);
  const section2Ref = useRef<HTMLDivElement>(null);
  const section1Ref = useRef<HTMLDivElement>(null);

  // Smooth scroll to section
  const scrollToSection = (sectionIndex: number) => {
    const targetRef = sectionIndex === 0 ? section1Ref : section2Ref;
    
    if (targetRef.current) {
      targetRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
      setCurrentSection(sectionIndex);
    }
  };

  // Handle scroll detection with throttling
  const handleScroll = React.useCallback(() => {
    if (section2Ref.current && section1Ref.current) {
      const section1Top = section1Ref.current.offsetTop;
      const section2Top = section2Ref.current.offsetTop;
      const scrollPosition = window.scrollY + window.innerHeight * 0.3;
      
      if (scrollPosition >= section2Top) {
        setCurrentSection(1);
      } else if (scrollPosition >= section1Top) {
        setCurrentSection(0);
      }
    }
  }, []);

  // Add scroll listener with throttling
  React.useEffect(() => {
    let ticking = false;
    
    const throttledScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          handleScroll();
          ticking = false;
        });
        ticking = true;
      }
    };
    
    window.addEventListener('scroll', throttledScroll, { passive: true });
    return () => window.removeEventListener('scroll', throttledScroll);
  }, [handleScroll]);

  // Add keyboard navigation
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'ArrowDown' && currentSection === 0) {
        event.preventDefault();
        scrollToSection(1);
      } else if (event.key === 'ArrowUp' && currentSection === 1) {
        event.preventDefault();
        scrollToSection(0);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentSection]);

  return (
    <div className="bg-everest-1000">
      {/* Scroll Indicator */}
      <ScrollIndicator
        currentSection={currentSection}
        totalSections={2}
        onSectionClick={scrollToSection}
      />
      
      {/* Section 1: Main Forecast View */}
      <section 
        ref={section1Ref}
        className="min-h-screen flex flex-col justify-center py-8 px-4 md:px-8 relative"
      >
        <motion.header 
          className="mb-8 text-center"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <h1 className="text-3xl md:text-4xl font-medium text-snow tracking-tight">
            Everest Solar Flare Prediction
          </h1>
          <p className="text-glacier-300 mt-1">
            Real-time forecast with uncertainty quantification
          </p>
        </motion.header>
        
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-7xl mx-auto w-full"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
        >
          {/* Left column - Forecast Summary */}
          <div>
            <ForecastSummaryCard />
          </div>
          
          {/* Right column - Temporal Evolution */}
          <div className="md:col-span-2">
            <TemporalEvolutionPanel />
          </div>
        </motion.div>

        {/* Animated Scroll Arrow */}
        <AnimatePresence>
          {currentSection === 0 && (
            <motion.div
              className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5, delay: 1 }}
            >
              <motion.button
                onClick={() => scrollToSection(1)}
                className="group flex flex-col items-center space-y-2 text-glacier-300 hover:text-snow transition-colors duration-300"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <span className="text-sm font-medium">More Details</span>
                <motion.div
                  className="w-6 h-6 border-2 border-current rounded-full flex items-center justify-center"
                  animate={{ 
                    y: [0, 4, 0],
                  }}
                  transition={{ 
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                >
                  <motion.svg
                    className="w-3 h-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    animate={{ 
                      y: [0, 2, 0],
                    }}
                    transition={{ 
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut",
                      delay: 0.1
                    }}
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M19 14l-7 7m0 0l-7-7m7 7V3" 
                    />
                  </motion.svg>
                </motion.div>
                
                {/* Subtle glow effect */}
                <motion.div
                  className="absolute inset-0 rounded-full bg-glacier-300/10 blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                  animate={{ 
                    scale: [1, 1.2, 1],
                  }}
                  transition={{ 
                    duration: 3,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>
      </section>

      {/* Section 2: Detailed Analytics */}
      <section 
        ref={section2Ref}
        className="min-h-screen py-8 px-4 md:px-8 relative"
      >
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          viewport={{ once: true, margin: "-100px" }}
        >
          <header className="mb-8 text-center">
            <h2 className="text-2xl md:text-3xl font-medium text-snow tracking-tight">
              Model Performance & Analytics
            </h2>
            <p className="text-glacier-300 mt-1">
              Detailed insights and performance metrics
            </p>
          </header>
          
          {/* Model Performance Section */}
          <div className="max-w-7xl mx-auto">
            <ModelPerformancePanel />
          </div>
        </motion.div>

        {/* Back to Top Arrow */}
        <AnimatePresence>
          {currentSection === 1 && (
            <motion.div
              className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
            >
              <motion.button
                onClick={() => scrollToSection(0)}
                className="group flex flex-col items-center space-y-2 text-glacier-300 hover:text-snow transition-colors duration-300"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <motion.div
                  className="w-6 h-6 border-2 border-current rounded-full flex items-center justify-center"
                  animate={{ 
                    y: [0, -4, 0],
                  }}
                  transition={{ 
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                >
                  <motion.svg
                    className="w-3 h-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    animate={{ 
                      y: [0, -2, 0],
                    }}
                    transition={{ 
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut",
                      delay: 0.1
                    }}
                  >
                    <path 
                      strokeLinecap="round" 
                      strokeLinejoin="round" 
                      strokeWidth={2} 
                      d="M5 10l7-7m0 0l7 7m-7-7v18" 
                    />
                  </motion.svg>
                </motion.div>
                <span className="text-sm font-medium">Back to Forecast</span>
                
                {/* Subtle glow effect */}
                <motion.div
                  className="absolute inset-0 rounded-full bg-glacier-300/10 blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                  animate={{ 
                    scale: [1, 1.2, 1],
                  }}
                  transition={{ 
                    duration: 3,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />
              </motion.button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer */}
        <motion.footer 
          className="border-t border-white/10 pt-6 pb-10 text-sm text-glacier-300 mt-16"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          viewport={{ once: true }}
        >
          <div className="flex flex-col md:flex-row justify-between items-center max-w-7xl mx-auto">
            <div>
              Developed with the Everest Solar Flare Prediction Model
            </div>
            <div className="mt-3 md:mt-0">
              <span className="font-mono bg-white/5 py-1 px-2 rounded text-xs">
                Version 1.0.0
              </span>
            </div>
          </div>
        </motion.footer>
      </section>
      
      {/* Active Region Detail Drawer - appears from bottom when triggered */}
      <ActiveRegionDetailDrawer />
      
      {/* 
      Commented out for now - will be implemented in further iterations
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <SolarContextBoard />
        <AlertCenter />
      </div>
      
      <div className="mb-8">
        <PerformanceWorkspace />
      </div>
      */}
    </div>
  );
};

export default Dashboard; 