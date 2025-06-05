#!/usr/bin/env python3
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
