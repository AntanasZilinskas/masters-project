import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useStore } from '../../store';

/**
 * Button component to manually refresh forecast data from the prediction system
 */
const ForecastRefreshButton: React.FC = () => {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const loadRealData = useStore(state => state.loadRealData);
  
  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      await loadRealData();
      console.log('✅ Forecast data refreshed successfully');
    } catch (error) {
      console.error('❌ Failed to refresh forecast data:', error);
    } finally {
      setIsRefreshing(false);
    }
  };
  
  return (
    <motion.button
      onClick={handleRefresh}
      disabled={isRefreshing}
      className={`
        flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
        transition-all duration-200
        ${isRefreshing 
          ? 'bg-white/5 text-glacier-400 cursor-not-allowed' 
          : 'bg-white/10 text-glacier-300 hover:bg-white/20 hover:text-snow'
        }
      `}
      whileHover={!isRefreshing ? { scale: 1.02 } : {}}
      whileTap={!isRefreshing ? { scale: 0.98 } : {}}
    >
      <motion.svg
        className="h-4 w-4"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        animate={isRefreshing ? { rotate: 360 } : { rotate: 0 }}
        transition={isRefreshing ? { duration: 1, repeat: Infinity, ease: "linear" } : {}}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
        />
      </motion.svg>
      {isRefreshing ? 'Refreshing...' : 'Refresh Data'}
    </motion.button>
  );
};

export default ForecastRefreshButton; 