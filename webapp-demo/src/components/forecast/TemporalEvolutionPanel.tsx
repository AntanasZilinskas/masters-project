import React, { useState, useCallback, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Brush
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import GlassPanel from '../common/GlassPanel';
import ForecastRefreshButton from './ForecastRefreshButton';
import { useStore } from '../../store';

// Define the data point interface for the chart
interface ChartDataPoint {
  timestamp: string;
  value: number;
  min: number;
  max: number;
}

// Flare class types
type FlareClass = 'C' | 'M' | 'M5';

// Define class settings interface
interface ClassSetting {
  color: string;
  fillOpacity: number;
  strokeDasharray?: string;
}

/**
 * Component that displays temporal evolution of flare probabilities
 * Connected to store data and responds to horizon changes
 */
const TemporalEvolutionPanel: React.FC = () => {
  const [activeClass, setActiveClass] = useState<FlareClass>('M');
  
  // Get data from store
  const temporalEvolution = useStore(state => state.temporalEvolution);
  const selectedHorizon = useStore(state => state.ui.selectedHorizon);
  
  // Transform temporal evolution data for the chart based on selected class and horizon
  const currentData = useMemo(() => {
    if (!temporalEvolution?.series || temporalEvolution.series.length === 0) {
      console.log('No temporal evolution data available');
      return [];
    }
    
    const now = new Date();
    const horizonHours = selectedHorizon;
    const cutoffTime = new Date(now.getTime() + horizonHours * 60 * 60 * 1000);
    
    // Filter data to only show up to the selected horizon
    const filteredSeries = temporalEvolution.series.filter(point => {
      const pointTime = new Date(point.timestamp);
      return pointTime <= cutoffTime;
    });
    
    // Transform data based on selected flare class
    const chartData: ChartDataPoint[] = filteredSeries.map(point => {
      let probability: number;
      
      // Get probability for selected class
      switch (activeClass) {
        case 'C':
          probability = point.prob_C;
          break;
        case 'M':
          probability = point.prob_M;
          break;
        case 'M5':
          probability = point.prob_M5;
          break;
        default:
          probability = point.prob_M;
      }
      
      // Convert to percentage and calculate uncertainty bounds
      const value = probability * 100;
      const epistemicUncertainty = point.epi * 100;
      const aleatoricUncertainty = point.alea * 100;
      
      // Calculate min/max bounds
      const totalUncertainty = epistemicUncertainty + aleatoricUncertainty;
      const min = Math.max(0, value - totalUncertainty);
      const max = Math.min(100, value + totalUncertainty);
      
      return {
        timestamp: point.timestamp,
        value,
        min,
        max
      };
    });
    
    console.log(`Temporal data for ${activeClass}-class (${horizonHours}h horizon):`, {
      totalPoints: temporalEvolution.series.length,
      filteredPoints: chartData.length,
      sampleData: chartData.slice(0, 3)
    });
    
    return chartData;
  }, [temporalEvolution, activeClass, selectedHorizon]);
  
  // Class settings with design system colors
  const CLASS_SETTINGS: Record<FlareClass, ClassSetting> = {
    'C': {
      color: '#16E0A2', // aurora-green
      fillOpacity: 0.25
    },
    'M': {
      color: '#8FAFE0', // glacier-300
      fillOpacity: 0.25
    },
    'M5': {
      color: '#B569FF', // aurora-purple
      fillOpacity: 0.25,
      strokeDasharray: '5 5'
    }
  };
  
  const currentColor = CLASS_SETTINGS[activeClass].color;
  const currentStrokeDasharray = CLASS_SETTINGS[activeClass].strokeDasharray;

  // Format date for x-axis label
  const formatXAxis = useCallback((dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    
    // Show different formats based on time difference
    const diffHours = (date.getTime() - now.getTime()) / (1000 * 60 * 60);
    
    if (Math.abs(diffHours) < 24) {
      // Within 24 hours: show time
      return date.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: false 
      });
    } else {
      // Beyond 24 hours: show date
      return `${date.getDate()}/${date.getMonth() + 1}`;
    }
  }, []);

  // Format date for tooltip
  const formatDate = useCallback((dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffHours = (date.getTime() - now.getTime()) / (1000 * 60 * 60);
    
    let timeLabel = '';
    if (diffHours < -1) {
      timeLabel = ` (${Math.abs(Math.round(diffHours))}h ago)`;
    } else if (diffHours > 1) {
      timeLabel = ` (+${Math.round(diffHours)}h)`;
    } else {
      timeLabel = ' (now)';
    }
    
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit'
    }) + timeLabel;
  }, []);

  // Custom tooltip content
  const CustomTooltip = useCallback(({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const formattedDate = formatDate(data.timestamp);
      
      return (
        <motion.div 
          className="glass-panel p-3 text-xs"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.2 }}
        >
          <p className="text-snow font-medium mb-1">{formattedDate}</p>
          <p style={{ color: currentColor }} className="flex justify-between gap-3">
            <span>{activeClass}-class:</span>
            <span className="font-mono tabular-nums">
              {data.value.toFixed(1)}%
            </span>
          </p>
          <p className="text-glacier-300 flex justify-between gap-3">
            <span>Range:</span>
            <span className="font-mono tabular-nums">
              {data.min.toFixed(1)}% - {data.max.toFixed(1)}%
            </span>
          </p>
        </motion.div>
      );
    }
    return null;
  }, [activeClass, currentColor, formatDate]);

  // Animation variants for smooth transitions
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        duration: 0.5,
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.3 }
    }
  };

  return (
    <GlassPanel className="w-full p-5 md:p-6" animate={false}>
      <motion.div 
        className="flex justify-between items-center mb-6"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.h2 
          className="text-xl font-medium text-snow tracking-tight"
          variants={itemVariants}
        >
          Probability Evolution
        </motion.h2>
        
        <motion.div 
          className="flex items-center space-x-3"
          variants={itemVariants}
        >
          <motion.div 
            className="text-sm text-glacier-300"
            key={selectedHorizon}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            {selectedHorizon}h horizon
          </motion.div>
          <div className="flex space-x-2">
            {(['C', 'M', 'M5'] as FlareClass[]).map((flareClass) => (
              <motion.button
                key={flareClass}
                onClick={() => setActiveClass(flareClass)}
                className={`px-2 py-1 text-xs rounded-full transition-all duration-300 ${
                  activeClass === flareClass 
                    ? flareClass === 'C' 
                      ? 'bg-aurora-green/20 text-aurora-green shadow-lg shadow-aurora-green/20' 
                      : flareClass === 'M'
                        ? 'bg-glacier-300/20 text-glacier-300 shadow-lg shadow-glacier-300/20'
                        : 'bg-aurora-purple/20 text-aurora-purple shadow-lg shadow-aurora-purple/20'
                    : 'bg-white/5 text-glacier-300 hover:bg-white/10'
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                transition={{ duration: 0.2 }}
              >
                {flareClass}-class
              </motion.button>
            ))}
          </div>
          <ForecastRefreshButton />
        </motion.div>
      </motion.div>
      
      <AnimatePresence mode="wait">
        <motion.div 
          key={`${activeClass}-${selectedHorizon}`}
          className="h-[320px] w-full bg-everest-800/50 rounded-lg border border-white/10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.4, ease: "easeInOut" }}
        >
          {currentData && currentData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={currentData}
                margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
              >
                <CartesianGrid 
                  strokeDasharray="3 3" 
                  stroke="rgba(255,255,255,0.1)" 
                  vertical={false}
                />
                <XAxis 
                  dataKey="timestamp" 
                  tickFormatter={formatXAxis} 
                  tick={{ fill: '#8FAFE0', fontSize: 12 }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.2)' }}
                  tickLine={{ stroke: 'rgba(255,255,255,0.2)' }}
                />
                <YAxis 
                  domain={[0, 100]}
                  tickFormatter={(value) => `${value}%`}
                  tick={{ fill: '#8FAFE0', fontSize: 12 }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.2)' }}
                  tickLine={{ stroke: 'rgba(255,255,255,0.2)' }}
                />
                <Tooltip content={<CustomTooltip />} />
                
                {/* Min range line */}
                <Line
                  type="monotone"
                  dataKey="min"
                  stroke={currentColor}
                  strokeOpacity={0.4}
                  strokeWidth={1}
                  strokeDasharray="2 2"
                  dot={false}
                  activeDot={false}
                  name="Min Range"
                  isAnimationActive={true}
                  animationDuration={800}
                  animationEasing="ease-in-out"
                />
                
                {/* Max range line */}
                <Line
                  type="monotone"
                  dataKey="max"
                  stroke={currentColor}
                  strokeOpacity={0.4}
                  strokeWidth={1}
                  strokeDasharray="2 2"
                  dot={false}
                  activeDot={false}
                  name="Max Range"
                  isAnimationActive={true}
                  animationDuration={800}
                  animationEasing="ease-in-out"
                />
                
                {/* Main probability line */}
                <Line
                  type="monotone"
                  dataKey="value"
                  name={`${activeClass}-class Probability`}
                  stroke={currentColor}
                  strokeWidth={3}
                  strokeDasharray={currentStrokeDasharray}
                  dot={{ r: 3, fill: currentColor }}
                  activeDot={{ 
                    r: 6, 
                    fill: currentColor, 
                    stroke: 'rgba(255,255,255,0.5)', 
                    strokeWidth: 2 
                  }}
                  isAnimationActive={true}
                  animationDuration={1000}
                  animationEasing="ease-in-out"
                />
                
                {/* Time window slider */}
                <Brush 
                  dataKey="timestamp" 
                  height={35} 
                  stroke={currentColor}
                  fill="rgba(17, 32, 51, 0.8)"
                  tickFormatter={formatXAxis}
                  startIndex={Math.max(0, currentData.length - 20)}
                  endIndex={currentData.length - 1}
                  travellerWidth={12}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <motion.div 
              className="flex items-center justify-center h-full text-glacier-300"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              <div className="text-center">
                <motion.div 
                  className="text-lg mb-2"
                  initial={{ y: 10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.2 }}
                >
                  No temporal data available
                </motion.div>
                <motion.div 
                  className="text-sm"
                  initial={{ y: 10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.3 }}
                >
                  Horizon: {selectedHorizon}h
                </motion.div>
                <motion.div 
                  className="text-sm"
                  initial={{ y: 10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.4 }}
                >
                  Active class: {activeClass}
                </motion.div>
              </div>
            </motion.div>
          )}
        </motion.div>
      </AnimatePresence>
      
      <motion.div 
        className="text-xs text-glacier-300 mt-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6, duration: 0.3 }}
      >
        <span style={{ color: currentColor }}>{activeClass}-class Flare:</span> Probability evolution over {selectedHorizon}h horizon with uncertainty bounds
      </motion.div>
    </GlassPanel>
  );
};

export default TemporalEvolutionPanel;