#!/usr/bin/env python3
"""
Case Study: September 6, 2017 X9.3 Solar Flare Analysis
=====================================================

This script analyzes the historic September 6, 2017 X9.3 solar flare using the EVEREST model.
This was the LARGEST flare of Solar Cycle 24 and one of the most significant space weather events
of the 2010s.

Event Details:
- Date: September 6, 2017
- Peak Time: 12:02 UTC  
- Classification: X9.3 (strongest flare since 2006)
- Active Region: NOAA AR 2673
- Historical Significance: Largest flare of Solar Cycle 24
- Associated CME: Yes, Earth-directed
- Radio Blackout: R3 (Strong) affecting HF communications globally

Author: EVEREST Analysis Team
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_september_2017_data():
    """
    Get data for September 6, 2017 X9.3 flare analysis.
    AR 2673 was the source region - need to identify the HARPNUM.
    """
    
    # Event details
    flare_time = datetime(2017, 9, 6, 12, 2, 0)  # 12:02 UTC
    print(f"🎯 Target Event: September 6, 2017 X9.3 Solar Flare")
    print(f"   Peak Time: {flare_time} UTC")
    print(f"   Active Region: NOAA AR 2673")
    print(f"   Significance: LARGEST flare of Solar Cycle 24")
    
    # Define analysis window: 72h before to 24h after
    start_time = flare_time - timedelta(hours=72)
    end_time = flare_time + timedelta(hours=24)
    
    print(f"   Analysis window: {start_time} to {end_time}")
    print(f"   Total duration: 96 hours")
    
    # Note: We need to identify the HARPNUM for AR 2673 in September 2017
    # This information should be available in SHARP data catalogs
    
    print("\n📊 Data Requirements:")
    print("   • SHARP magnetogram parameters (9 features)")
    print("   • 12-minute cadence observations")
    print("   • HARPNUM identification for AR 2673")
    print("   • ~480 data points expected")
    
    return flare_time, start_time, end_time

def find_ar2673_harpnum():
    """
    Identify the HARPNUM corresponding to NOAA AR 2673 in September 2017.
    This requires checking SHARP/HMI data catalogs.
    """
    
    print("\n🔍 Identifying HARPNUM for AR 2673...")
    
    # Based on space weather reports, AR 2673 was active from late August through September 2017
    # Common HARPNUMs in this timeframe include those in the 7000s series
    
    # Potential candidates (to be verified against actual data):
    candidate_harpnums = [
        7112, 7113, 7114, 7115, 7116, 7117, 7118, 7119, 7120, 7121, 7122
    ]
    
    print(f"   Candidate HARPNUMs: {candidate_harpnums}")
    print("   📝 Note: Exact HARPNUM needs verification against SHARP data catalog")
    print("   📝 AR 2673 was prominently visible and geoeffective in early September 2017")
    
    # For now, return a representative HARPNUM - this should be verified
    estimated_harpnum = 7115  # This is an estimate and needs verification
    
    print(f"   🎯 Using HARPNUM {estimated_harpnum} (pending verification)")
    
    return estimated_harpnum

def analyze_september_2017_flare():
    """
    Main analysis function for the September 6, 2017 X9.3 flare
    """
    
    print("="*80)
    print("🚀 SEPTEMBER 6, 2017 X9.3 FLARE CASE STUDY")
    print("   EVEREST Model Analysis of Solar Cycle 24's Largest Flare")
    print("="*80)
    
    # Get event parameters
    flare_time, start_time, end_time = get_september_2017_data()
    
    # Find the corresponding HARPNUM
    harpnum = find_ar2673_harpnum()
    
    print(f"\n📋 ANALYSIS SETUP:")
    print(f"   Event: September 6, 2017 X9.3 Flare")
    print(f"   HARPNUM: {harpnum} (AR 2673)")
    print(f"   Model: EVEREST (72h-M5 configuration)")
    print(f"   Analysis window: 96 hours total")
    
    print(f"\n⚠️  DATA LOADING STATUS:")
    print(f"   This analysis requires SHARP data for HARPNUM {harpnum}")
    print(f"   Data should span: {start_time} to {end_time}")
    print(f"   Expected features: TOTUSJH, TOTUSJZ, USFLUX, TOTBSQ, R_VALUE,")
    print(f"                     TOTPOT, SAVNCPP, AREA_ACR, ABSNJZH")
    
    # Placeholder for actual data loading
    print(f"\n🔄 Next Steps:")
    print(f"   1. Verify HARPNUM {harpnum} corresponds to AR 2673")
    print(f"   2. Download SHARP data for the specified time range")
    print(f"   3. Process data using same pipeline as July 2012 analysis")
    print(f"   4. Run EVEREST model predictions")
    print(f"   5. Generate comprehensive analysis and visualizations")
    
    return {
        'flare_time': flare_time,
        'harpnum': harpnum,
        'start_time': start_time,
        'end_time': end_time,
        'event_class': 'X9.3',
        'significance': 'Largest flare of Solar Cycle 24'
    }

def create_analysis_framework():
    """
    Set up the analysis framework for the September 2017 event
    """
    
    print(f"\n📁 CREATING ANALYSIS FRAMEWORK...")
    
    # Create output directories
    output_dirs = [
        'september_2017_analysis',
        'september_2017_analysis/data',
        'september_2017_analysis/results',
        'september_2017_analysis/plots'
    ]
    
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"   Created: {dir_path}/")
    
    # Analysis configuration
    config = {
        'event_name': 'September_6_2017_X9.3',
        'flare_class': 'X9.3',
        'peak_time': '2017-09-06T12:02:00Z',
        'active_region': 'AR_2673',
        'model_config': {
            'name': 'EVEREST',
            'weights': 'tests/model_weights_EVEREST_72h_M5.pt',
            'prediction_horizon': '72h',
            'features': [
                'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTBSQ', 'R_VALUE',
                'TOTPOT', 'SAVNCPP', 'AREA_ACR', 'ABSNJZH'
            ]
        },
        'analysis_window': {
            'pre_flare_hours': 72,
            'post_flare_hours': 24,
            'total_hours': 96
        },
        'expected_performance': {
            'note': 'X9.3 is much stronger than X1.4 (July 2012)',
            'hypothesis': 'Should achieve higher probabilities than 15.76%',
            'population_context': 'Closer to 46% optimal threshold'
        }
    }
    
    # Save configuration
    config_path = 'september_2017_analysis/analysis_config.py'
    with open(config_path, 'w') as f:
        f.write(f"# September 6, 2017 X9.3 Flare Analysis Configuration\n")
        f.write(f"# Generated on {datetime.now()}\n\n")
        f.write(f"CONFIG = {config}\n")
    
    print(f"   Configuration saved: {config_path}")
    
    return config

def compare_with_july_2012():
    """
    Compare the September 2017 X9.3 event with our July 2012 X1.4 baseline
    """
    
    print(f"\n📊 COMPARISON WITH JULY 2012 BASELINE:")
    print(f"   {'Metric':<25} {'July 2012 X1.4':<20} {'Sept 2017 X9.3':<20}")
    print(f"   {'-'*70}")
    print(f"   {'Flare Class':<25} {'X1.4':<20} {'X9.3':<20}")
    print(f"   {'Relative Strength':<25} {'1.4x':<20} {'9.3x (6.6× stronger)':<20}")
    print(f"   {'Max Probability':<25} {'15.76%':<20} {'TBD (expect >30%)':<20}")
    print(f"   {'Lead Time (10%)':<25} {'36.1h':<20} {'TBD (expect similar)':<20}")
    print(f"   {'Lead Time (2%)':<25} {'70.1h':<20} {'TBD (expect similar)':<20}")
    print(f"   {'Historical Context':<25} {'Moderate event':<20} {'Cycle maximum':<20}")
    
    print(f"\n🎯 EXPECTED IMPROVEMENTS:")
    print(f"   • Higher peak probabilities (potentially 30-50%)")
    print(f"   • Stronger signal progression")
    print(f"   • More robust uncertainty quantification")
    print(f"   • Closer to population-optimal 46% threshold")
    print(f"   • Excellent demonstration of model capability")

def create_data_search_script():
    """
    Create a script to help identify and download the required SHARP data
    """
    
    script_content = '''#!/usr/bin/env python3
"""
Data Search Script for September 6, 2017 X9.3 Flare
==================================================

This script helps identify the correct HARPNUM for AR 2673 and 
downloads the required SHARP data for EVEREST analysis.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

def search_harpnum_for_ar2673():
    """
    Search for the HARPNUM corresponding to NOAA AR 2673 in September 2017
    """
    
    print("🔍 Searching for HARPNUM corresponding to AR 2673...")
    
    # Target time range
    start_date = datetime(2017, 9, 3)  # A few days before the flare
    end_date = datetime(2017, 9, 9)    # A few days after
    
    print(f"Search period: {start_date} to {end_date}")
    
    # Candidate HARPNUMs in this timeframe
    candidates = [7112, 7113, 7114, 7115, 7116, 7117, 7118, 7119, 7120, 7121, 7122]
    
    print(f"Candidate HARPNUMs: {candidates}")
    
    # TODO: Query SHARP data catalogs to find which HARPNUM was active during this period
    # and corresponds to AR 2673 location
    
    # For now, return best estimate
    return 7115

def download_sharp_data(harpnum, start_time, end_time):
    """
    Download SHARP data for the specified HARPNUM and time range
    """
    
    print(f"📥 Downloading SHARP data...")
    print(f"   HARPNUM: {harpnum}")
    print(f"   Time range: {start_time} to {end_time}")
    
    # TODO: Implement actual SHARP data download
    # This would typically use JSOC/DRMS or similar services
    
    print("⚠️  Data download not yet implemented")
    print("Manual steps:")
    print("1. Access JSOC SHARP data archive")
    print("2. Search for HARPNUM 7115 in September 2017")
    print("3. Download 12-minute cadence data for 9 parameters")
    print("4. Save as CSV format for analysis")

if __name__ == "__main__":
    harpnum = search_harpnum_for_ar2673()
    
    # Define time range (72h before to 24h after flare)
    flare_time = datetime(2017, 9, 6, 12, 2, 0)
    start_time = flare_time - timedelta(hours=72)
    end_time = flare_time + timedelta(hours=24)
    
    download_sharp_data(harpnum, start_time, end_time)
'''
    
    with open('september_2017_analysis/data_search.py', 'w') as f:
        f.write(script_content)
    
    print(f"   Data search script created: september_2017_analysis/data_search.py")

def main():
    """
    Main execution function
    """
    
    try:
        # Analyze the September 2017 event
        event_info = analyze_september_2017_flare()
        
        # Create analysis framework
        config = create_analysis_framework()
        
        # Create data search helper
        create_data_search_script()
        
        # Compare with July 2012 baseline
        compare_with_july_2012()
        
        print(f"\n✅ SEPTEMBER 2017 CASE STUDY SETUP COMPLETE")
        print(f"\n📋 SUMMARY:")
        print(f"   Event: {event_info['event_class']} flare on {event_info['flare_time']}")
        print(f"   Significance: {event_info['significance']}")
        print(f"   HARPNUM: {event_info['harpnum']} (pending verification)")
        print(f"   Ready for data loading and EVEREST analysis")
        
        print(f"\n🚀 NEXT ACTIONS:")
        print(f"   1. Verify HARPNUM and download SHARP data")
        print(f"   2. Use september_2017_analysis/data_search.py for data acquisition")
        print(f"   3. Adapt july_12_2012_analysis.py for September 2017 data")
        print(f"   4. Run EVEREST model and compare with July 2012 results")
        
        return event_info
        
    except Exception as e:
        print(f"\n❌ Error in September 2017 analysis setup: {e}")
        return None

if __name__ == "__main__":
    result = main() 