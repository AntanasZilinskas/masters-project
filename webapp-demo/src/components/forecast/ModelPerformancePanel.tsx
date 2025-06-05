import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassPanel from '../common/GlassPanel';

interface ModelInfo {
  flare_class: string;
  time_window: number;
  version: string;
  model_path: string;
  threshold: number;
}

interface Performance {
  confusion_matrix: number[][];
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc: number;
  tss: number;
  hss: number;
  specificity: number;
  true_positives: number;
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
  total_samples: number;
  positive_samples: number;
  negative_samples: number;
}

interface DataInfo {
  test_samples: number;
  positive_samples: number;
  negative_samples: number;
  class_distribution: {
    positive_rate: number;
    negative_rate: number;
  };
  raw_stats?: {
    total_raw_samples: number;
    raw_positive_samples: number;
    raw_negative_samples: number;
    unique_active_regions: number;
    date_range: {
      start: string;
      end: string;
    };
  };
}

interface ModelResult {
  model_info: ModelInfo;
  performance: Performance;
  data_info: DataInfo;
}

interface ModelMetadata {
  generated_at: string;
  system_info: {
    model_base_dir: string;
    threshold: number;
    input_shape: number[];
  };
  models_tested: Array<{
    flare_class: string;
    time_window: string;
    version?: string;
    status: string;
  }>;
  summary: {
    total_models_attempted: number;
    successful_models: number;
    failed_models: number;
    success_rate: number;
    aggregate_stats?: {
      mean_accuracy: number;
      std_accuracy: number;
      mean_tss: number;
      std_tss: number;
      best_accuracy: number;
      best_tss: number;
    };
  };
}

interface PredictionsData {
  metadata: ModelMetadata;
  summary: Record<string, ModelResult>;
}

const ModelPerformancePanel: React.FC = () => {
  const [predictionsData, setPredictionsData] = useState<PredictionsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'detailed'>('grid');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(false);

  const loadPredictions = useCallback(async () => {
    try {
      setError(null);
      const response = await fetch('/src/data/latest_predictions_compact.json?t=' + Date.now());
      if (!response.ok) {
        throw new Error(`Failed to load predictions data: ${response.status}`);
      }
      const data = await response.json();
      setPredictionsData(data);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPredictions();
  }, [loadPredictions]);

  // Auto-refresh functionality
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      loadPredictions();
    }, 60000); // Refresh every minute

    return () => clearInterval(interval);
  }, [autoRefresh, loadPredictions]);

  const handleRefresh = () => {
    setLoading(true);
    loadPredictions();
  };

  const formatMetric = (value: number, decimals: number = 3): string => {
    return value.toFixed(decimals);
  };

  const getPerformanceColor = (tss: number): string => {
    if (tss >= 0.5) return 'text-aurora-green';
    if (tss >= 0.2) return 'text-glacier-300';
    return 'text-aurora-purple';
  };

  const getPerformanceIcon = (tss: number) => {
    if (tss >= 0.5) return (
      <svg className="h-4 w-4 text-aurora-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    );
    if (tss >= 0.2) return (
      <svg className="h-4 w-4 text-glacier-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    );
    return (
      <svg className="h-4 w-4 text-aurora-purple" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
      </svg>
    );
  };

  if (loading) {
    return (
      <GlassPanel className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <svg className="h-5 w-5 text-glacier-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <h2 className="text-xl font-medium text-snow tracking-tight">Model Performance</h2>
        </div>
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-aurora-green"></div>
          <span className="ml-2 text-glacier-300">Loading model performance data...</span>
        </div>
      </GlassPanel>
    );
  }

  if (error || !predictionsData) {
    return (
      <GlassPanel className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <svg className="h-5 w-5 text-aurora-purple" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          <h2 className="text-xl font-medium text-snow tracking-tight">Model Performance</h2>
        </div>
        <div className="text-aurora-purple mb-4">
          Error loading predictions: {error || 'No data available'}
        </div>
        <button
          onClick={handleRefresh}
          className="px-4 py-2 bg-aurora-green/20 text-aurora-green rounded-lg hover:bg-aurora-green/30 transition-colors"
        >
          Try Again
        </button>
      </GlassPanel>
    );
  }

  const { metadata, summary } = predictionsData;
  const modelResults = Object.entries(summary);

  return (
    <div className="space-y-6">
      {/* Summary Statistics */}
      <GlassPanel className="p-6">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <svg className="h-5 w-5 text-glacier-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <h2 className="text-xl font-medium text-snow tracking-tight">Model Performance Overview</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-3 py-1 text-sm rounded-lg transition-all duration-200 ${
                autoRefresh 
                  ? 'bg-aurora-green/20 text-aurora-green' 
                  : 'bg-white/5 text-glacier-300 hover:bg-white/10'
              }`}
            >
              {autoRefresh ? '🔄 Auto' : '⏸️ Manual'}
            </button>
            <button
              onClick={handleRefresh}
              className="px-3 py-1 text-sm rounded-lg bg-white/5 text-glacier-300 hover:bg-white/10 transition-all duration-200"
            >
              🔄 Refresh
            </button>
          </div>
        </div>
        <div className="flex items-center justify-between mb-6">
          <p className="text-sm text-glacier-300">
            Generated: {new Date(metadata.generated_at).toLocaleString()}
          </p>
          {lastUpdated && (
            <p className="text-xs text-glacier-400">
              Last updated: {lastUpdated.toLocaleTimeString()}
            </p>
          )}
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-aurora-green">
              {metadata.summary.successful_models}
            </div>
            <div className="text-sm text-glacier-300">Successful Models</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-aurora-green">
              {formatMetric(metadata.summary.success_rate * 100, 1)}%
            </div>
            <div className="text-sm text-glacier-300">Success Rate</div>
          </div>
          {metadata.summary.aggregate_stats && (
            <>
              <div className="text-center">
                <div className="text-2xl font-bold text-aurora-purple">
                  {formatMetric(metadata.summary.aggregate_stats.mean_accuracy)}
                </div>
                <div className="text-sm text-glacier-300">Mean Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-glacier-300">
                  {formatMetric(metadata.summary.aggregate_stats.mean_tss)}
                </div>
                <div className="text-sm text-glacier-300">Mean TSS</div>
              </div>
            </>
          )}
        </div>

        <div className="mb-4">
          <div className="flex justify-between text-sm mb-2 text-glacier-300">
            <span>Model Success Rate</span>
            <span>{metadata.summary.successful_models}/{metadata.summary.total_models_attempted}</span>
          </div>
          <div className="w-full bg-white/10 rounded-full h-2">
            <div 
              className="bg-aurora-green h-2 rounded-full transition-all duration-300"
              style={{ width: `${metadata.summary.success_rate * 100}%` }}
            />
          </div>
        </div>
      </GlassPanel>

      {/* Individual Model Results */}
      <GlassPanel className="p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-medium text-snow tracking-tight">Individual Model Performance</h2>
          <div className="flex space-x-2">
            <button
              onClick={() => setViewMode('grid')}
              className={`px-3 py-1 text-sm rounded-lg transition-all duration-200 ${
                viewMode === 'grid' 
                  ? 'bg-aurora-green/20 text-aurora-green' 
                  : 'bg-white/5 text-glacier-300 hover:bg-white/10'
              }`}
            >
              Grid View
            </button>
            <button
              onClick={() => setViewMode('detailed')}
              className={`px-3 py-1 text-sm rounded-lg transition-all duration-200 ${
                viewMode === 'detailed' 
                  ? 'bg-aurora-green/20 text-aurora-green' 
                  : 'bg-white/5 text-glacier-300 hover:bg-white/10'
              }`}
            >
              Detailed View
            </button>
          </div>
        </div>
        
        <AnimatePresence mode="wait">
          {viewMode === 'grid' ? (
            <motion.div
              key="grid"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
            >
              {modelResults.map(([key, result]) => (
                <motion.div
                  key={key}
                  className="bg-white/5 border border-white/10 rounded-xl p-4 border-l-4 border-l-aurora-green"
                  whileHover={{ scale: 1.02 }}
                  transition={{ duration: 0.2 }}
                >
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-medium text-snow">
                      {result.model_info.flare_class}-class
                    </h3>
                    <span className="px-2 py-1 bg-white/10 rounded text-xs text-glacier-300">
                      {result.model_info.time_window}h
                    </span>
                  </div>
                  <div className="flex items-center gap-2 mb-3">
                    <span className="px-2 py-1 bg-white/5 rounded text-xs text-glacier-300">
                      v{result.model_info.version}
                    </span>
                    {getPerformanceIcon(result.performance.tss)}
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-glacier-300">Accuracy:</span>
                      <div className="font-semibold text-snow">{formatMetric(result.performance.accuracy)}</div>
                    </div>
                    <div>
                      <span className="text-glacier-300">TSS:</span>
                      <div className={`font-semibold ${getPerformanceColor(result.performance.tss)}`}>
                        {formatMetric(result.performance.tss)}
                      </div>
                    </div>
                    <div>
                      <span className="text-glacier-300">Precision:</span>
                      <div className="font-semibold text-snow">{formatMetric(result.performance.precision)}</div>
                    </div>
                    <div>
                      <span className="text-glacier-300">Recall:</span>
                      <div className="font-semibold text-snow">{formatMetric(result.performance.recall)}</div>
                    </div>
                  </div>
                  <div className="text-xs text-glacier-300 mt-3">
                    {result.data_info.raw_stats ? (
                      <>
                        <div className="font-medium text-snow mb-1">Dataset: {result.data_info.raw_stats.total_raw_samples.toLocaleString()} samples</div>
                        <div>({result.data_info.raw_stats.raw_positive_samples.toLocaleString()} positive, {result.data_info.raw_stats.raw_negative_samples.toLocaleString()} negative)</div>
                        <div className="mt-1">{result.data_info.raw_stats.unique_active_regions} active regions</div>
                      </>
                    ) : (
                      <>
                        {result.data_info.test_samples} test samples
                        ({result.data_info.positive_samples} positive, {result.data_info.negative_samples} negative)
                      </>
                    )}
                  </div>
                </motion.div>
              ))}
            </motion.div>
          ) : (
            <motion.div
              key="detailed"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-4"
            >
              {modelResults.map(([key, result]) => (
                <motion.div
                  key={key}
                  className="bg-white/5 border border-white/10 rounded-xl p-6"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <h3 className="text-lg font-medium text-snow">
                        {result.model_info.flare_class}-class {result.model_info.time_window}h Model
                      </h3>
                      {getPerformanceIcon(result.performance.tss)}
                    </div>
                    <span className="px-2 py-1 bg-white/10 rounded text-sm text-glacier-300">
                      v{result.model_info.version}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="space-y-2">
                      <h4 className="font-semibold text-sm text-glacier-300">Classification Metrics</h4>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span className="text-glacier-300">Accuracy:</span>
                          <span className="font-mono text-snow">{formatMetric(result.performance.accuracy)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-glacier-300">Precision:</span>
                          <span className="font-mono text-snow">{formatMetric(result.performance.precision)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-glacier-300">Recall:</span>
                          <span className="font-mono text-snow">{formatMetric(result.performance.recall)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-glacier-300">F1-Score:</span>
                          <span className="font-mono text-snow">{formatMetric(result.performance.f1_score)}</span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h4 className="font-semibold text-sm text-glacier-300">Skill Scores</h4>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span className="text-glacier-300">TSS:</span>
                          <span className={`font-mono ${getPerformanceColor(result.performance.tss)}`}>
                            {formatMetric(result.performance.tss)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-glacier-300">HSS:</span>
                          <span className="font-mono text-snow">{formatMetric(result.performance.hss)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-glacier-300">AUC-ROC:</span>
                          <span className="font-mono text-snow">{formatMetric(result.performance.auc_roc)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-glacier-300">Specificity:</span>
                          <span className="font-mono text-snow">{formatMetric(result.performance.specificity)}</span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h4 className="font-semibold text-sm text-glacier-300">Confusion Matrix</h4>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span className="text-glacier-300">True Positives:</span>
                          <span className="font-mono text-aurora-green">{result.performance.true_positives}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-glacier-300">True Negatives:</span>
                          <span className="font-mono text-aurora-green">{result.performance.true_negatives}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-glacier-300">False Positives:</span>
                          <span className="font-mono text-aurora-purple">{result.performance.false_positives}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-glacier-300">False Negatives:</span>
                          <span className="font-mono text-aurora-purple">{result.performance.false_negatives}</span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <h4 className="font-semibold text-sm text-glacier-300">Dataset Info</h4>
                      <div className="space-y-1 text-sm">
                        {result.data_info.raw_stats ? (
                          <>
                            <div className="flex justify-between">
                              <span className="text-glacier-300">Raw Dataset:</span>
                              <span className="font-mono text-snow">{result.data_info.raw_stats.total_raw_samples.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-glacier-300">Positive:</span>
                              <span className="font-mono text-aurora-green">{result.data_info.raw_stats.raw_positive_samples.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-glacier-300">Negative:</span>
                              <span className="font-mono text-aurora-purple">{result.data_info.raw_stats.raw_negative_samples.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-glacier-300">Active Regions:</span>
                              <span className="font-mono text-snow">{result.data_info.raw_stats.unique_active_regions}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-glacier-300">Test Sequences:</span>
                              <span className="font-mono text-glacier-300">{result.data_info.test_samples}</span>
                            </div>
                          </>
                        ) : (
                          <>
                            <div className="flex justify-between">
                              <span className="text-glacier-300">Total Samples:</span>
                              <span className="font-mono text-snow">{result.data_info.test_samples}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-glacier-300">Positive:</span>
                              <span className="font-mono text-snow">{result.data_info.positive_samples}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-glacier-300">Negative:</span>
                              <span className="font-mono text-snow">{result.data_info.negative_samples}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-glacier-300">Pos. Rate:</span>
                              <span className="font-mono text-snow">{formatMetric(result.data_info.class_distribution.positive_rate)}</span>
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </GlassPanel>
    </div>
  );
};

export default ModelPerformancePanel; 