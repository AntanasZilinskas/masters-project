# September 6, 2017 X9.3 Solar Flare Case Study - Test Directory

This directory contains all the analysis files, scripts, and results for the September 6, 2017 X9.3 solar flare case study - the largest flare of Solar Cycle 24.

## 📁 Directory Contents

### 🐍 Python Analysis Scripts
- `september_6_2017_proper_inference.py` - **Main analysis script** with proper utils pipeline
- `september_6_2017_figures.py` - **Figure generation** for prospective and multimodal plots
- `september_6_2017_detailed_commentary.py` - **Comprehensive commentary** parallel to July 2012 analysis
- `september_6_2017_comprehensive_analysis.py` - Full analysis with uncertainty quantification
- `september_6_2017_complete_analysis.py` - Complete end-to-end analysis
- `september_6_2017_nature_analysis.py` - Nature data format analysis
- `september_6_2017_proper_analysis.py` - Proper methodology analysis
- `september_6_2017_case_study.py` - Basic case study implementation
- `september_6_2017_simple_utils_analysis.py` - Simplified utils-based analysis

### 📊 Generated Results
- `september_6_2017_x93_comprehensive_results.csv` - **Main results** (480 sequences, probabilities, uncertainties)
- `september_6_2017_analysis_results.csv` - Analysis summary results

### 🖼️ Generated Figures
- `september_6_2017_x93_prospective.png` - **Main prospective figure** with GOES data, thresholds, lead times
- `september_6_2017_x93_prospective_multimodal.png` - **Multimodal analysis** (5-panel: probability, uncertainty, EVT, precursor, ensemble)
- `september_6_2017_x93_comprehensive_analysis.png` - Comprehensive analysis visualization
- `september_6_2017_x93_complete_analysis.png` - Complete analysis figure
- `september_6_2017_x93_analysis.png` - Basic analysis figure
- `september_6_2017_x93_inference_proper utils.png` - Proper utils inference figure
- `september_6_2017_comprehensive_analysis.pdf` - PDF version of comprehensive analysis

### 📖 Documentation
- `case_study_technical_assessment.md` - Technical assessment and methodology
- `comprehensive_case_study_analysis.md` - Comprehensive case study analysis
- `september_2017_analysis/` - Additional analysis directory

## 🎯 Key Results Summary

**Event**: September 6, 2017 X9.3 Solar Flare (largest of Solar Cycle 24)
- **Maximum Probability**: 59.63% at 2017-09-06 11:46:42 UTC
- **Mean Probability**: 22.86% ± 12.22%
- **Operational Alert Lead Time**: 2.7 hours (46% threshold)
- **Analysis Period**: 72h before to 24h after flare
- **Active Region**: HARPNUM 7115
- **Total Sequences**: 480

## 🚀 Usage

### Run Main Analysis
```bash
python september_6_2017_proper_inference.py
```

### Generate Figures
```bash
python september_6_2017_figures.py
```

### View Detailed Commentary
```bash
python september_6_2017_detailed_commentary.py
```

## 📈 Scientific Significance

This case study demonstrates:
1. **Extreme Event Prediction**: Successfully predicted largest Solar Cycle 24 flare
2. **Model Scalability**: 59.63% max probability for X9.3 vs 15.8% for X1.4 (July 2012)
3. **Operational Readiness**: Met all operational alert requirements
4. **Late-Cycle Performance**: Validated model during solar minimum transition
5. **Multimodal Capabilities**: Full uncertainty quantification and EVT analysis

## 🔬 Methodology

- **Model**: EVEREST (evidential + EVT + precursor + attention)
- **Data Pipeline**: utils.get_training_data() with proper preprocessing
- **Prediction Method**: model.predict_proba() with all multimodal outputs
- **Uncertainty**: Evidential (epistemic + aleatoric) with 95% credible intervals
- **Validation**: Comparison with actual GOES observations and flare timing

## 📊 Comparative Performance

| Metric | July 12, 2012 X1.4 | September 6, 2017 X9.3 | Improvement |
|--------|---------------------|-------------------------|-------------|
| Max Probability | 15.8% | 59.63% | 3.78× higher |
| Flare Magnitude | X1.4 | X9.3 | 6.6× stronger |
| Lead Time (46%) | N/A | 2.7h | Operational |
| Solar Cycle Phase | Maximum | Declining | Robust |

This analysis establishes a new benchmark for extreme solar flare prediction capabilities. 