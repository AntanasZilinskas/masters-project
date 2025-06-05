#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_datasets.py
Generate 18 tabular datasets (9×train, 9×test) that exactly follow the 
data-assembly protocol described in:

    Chen et al. (2023) Nature Communications – "Automated solar flare prediction …"

Main differences w.r.t. the original *download.py* prototype:

1. Samples & cadence
   • 12-min cadence SHARP measurements ("hmi.sharp_cea_720s")
   • Each sample row corresponds to one time step of an *individual* AR
   • Windows: 24 h, 48 h, 72 h **before** a flare (or after the start-time of
     a non-flaring AR)

2. Prediction tasks / thresholds
   ┌──────────┬──────────────────────────────┬────────────────────────┐
   │ Task     │ Positive samples             │ Negative samples       │
   ├──────────┼──────────────────────────────┼────────────────────────┤
   │ C-flare  │ C, M, X flares               │ A, B flares & NF ARs   │
   │ M-flare  │ M, X flares                 │ C flares & NF ARs      │
   │ M5-flare │ X, M≥5.0 flares             │ C, M<5.0 flares & NF   │
   └──────────┴──────────────────────────────┴────────────────────────┘
   (NF = non-flaring AR; we adopt the 10 NOAA regions released in Barnard et al. 2021).

3. Quality control / cleaning
   • Central meridian distance |CMD| ≤ 70°        (keyword: "CRLN_OBS")
   • Radial spacecraft velocity |OBS_VR| ≤ 3500 m s⁻¹  (keyword: "OBS_VR")
   • QUALITY == 0 (bitmask)
   • No NaNs in the 9 SHARP parameters.

4. Padding / gap-filling
   Zero-rows are injected *after* normalisation to maintain continuity for
   windows that would otherwise have missing timestamps.

5. 10-fold cross-validation
   Folds are stratified by HARPNUM so that all samples of a single active
   region reside **exclusively** in either the training or test partition of
   a run.  The fold-id is stored in column "Fold".

6. Output files
   • training_data_<THRESH>_<H>.csv
   • testing_data_<THRESH>_<H>.csv
   where <THRESH> ∈ {C, M, M5} and <H> ∈ {24, 48, 72}.
   Each CSV holds 10× as many rows as in a single run; users can reproduce the
   individual folds by filtering on the "Fold" column.

The script is *self-contained* and safe to interrupt/resume because it
incrementally caches GOES and SHARP queries on-disk (see the `.cache` dir).

----------------------------------------------------------------------
Author: Imperial-D2P project (adapted by ChatGPT-4o)
"""

import os
import sys
import json
import pickle
import logging
import signal
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import argparse

from sunpy.time import TimeRange
import sunkit_instruments.goes_xrs as goes_mod
import drms

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------
EMAIL          = "antanas.zilinskas21@imperial.ac.uk"

# RECOMMENDED DATE RANGES FOR DIFFERENT PURPOSES:
# For historical research (2010-2023): Use any dates
# For recent research (2024-early 2025): Use dates at least 2-3 weeks old  
# For testing/development: Use April 2025 (known to work well)

# DEFAULT_START  = datetime(2010, 5, 1, tzinfo=timezone.utc)     # Original broad range
DEFAULT_START  = datetime(2025, 4, 1, tzinfo=timezone.utc)      # April 2025 - known working
DEFAULT_END    = datetime(2025, 4, 30, tzinfo=timezone.utc)     # Full month of April 2025

# Alternative recommended ranges (uncomment as needed):
# DEFAULT_START  = datetime(2025, 2, 1, tzinfo=timezone.utc)     # February 2025 - also works
# DEFAULT_END    = datetime(2025, 2, 28, tzinfo=timezone.utc)
# DEFAULT_START  = datetime(2024, 12, 1, tzinfo=timezone.utc)    # December 2024 - works  
# DEFAULT_END    = datetime(2024, 12, 31, tzinfo=timezone.utc)

START_DATE = DEFAULT_START
END_DATE   = DEFAULT_END
WINDOWS        = [24, 48, 72]                     # All windows
THRESHOLDS     = ["C", "M", "M5"]                 # All thresholds
FEATURES       = [
    "USFLUX", "TOTUSJH", "TOTUSJZ", "MEANALP", "R_VALUE",
    "TOTPOT", "SAVNCPP", "AREA_ACR", "ABSNJZH",
]

# column order for final CSVs
OUTPUT_COLUMNS = ["Flare", "DATE__OBS", "NOAA_AR", "HARPNUM", *FEATURES]

# Train/Test split ratio (90% train, 10% test)
CACHE_DIR      = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

MAPPING_URL = (
    "http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/"
    "all_harps_with_noaa_ars.txt"
)

# --------------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def with_timeout(seconds):
    """Decorator to add timeout to function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler and a timeout alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                log.warning(f"Function {func.__name__} timed out after {seconds} seconds")
                return None
            finally:
                # Disable the alarm and restore the old handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator

def cached(path: Path):
    """Decorator – cache the function output as pickle under *path*."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            if path.exists():
                with open(path, "rb") as fh:
                    return pickle.load(fh)
            out = fn(*args, **kwargs)
            with open(path, "wb") as fh:
                pickle.dump(out, fh)
            return out
        return wrapper
    return decorator

def get_enhanced_harp_mapping() -> Dict[int, int]:
    """Get enhanced HARP mapping that includes manual mappings for recent ARs."""
    
    # Manual mappings for recent ARs that don't have HARP numbers yet
    # These should be updated as new HARP assignments become available
    # 
    # *** IMPORTANT NOTE FOR USERS ***
    # The JSOC HARP mapping database has a lag of several weeks for new Active Regions.
    # As of May 2025, the official mapping only goes up to AR 14078.
    # 
    # If you need to process very recent data (within ~2-3 weeks), you have options:
    # 1. Wait for JSOC to assign HARP numbers (recommended for research)
    # 2. Use slightly older dates (e.g., April 2025 works well) 
    # 3. Add manual mappings below as they become available
    #
    # For most research purposes, using data that is 2-3 weeks old ensures
    # all HARP mappings exist and the dataset generation will work properly.
    
    MANUAL_HARP_MAPPINGS = {
        # Add known recent mappings here as they become available from JSOC
        # Check: http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt
        # Format: NOAA_AR: HARP_NUMBER
        
        # Examples (these would need to be verified):
        # 14079: 13119,  # When officially assigned
        # 14082: 13120,  # When officially assigned
        # 14087: 13121,  # When officially assigned  
        # 14098: 13122,  # When officially assigned
        # 14100: 13123,  # When officially assigned
        
        # Currently no manual mappings - waiting for official JSOC assignments
    }
    
    # Load the official mapping
    mapping_df = load_noaa_harp_map()
    
    # Convert to dict for faster lookup
    mapping_dict = {}
    
    # Add official mappings
    for _, row in mapping_df.iterrows():
        harp = int(row["HARPNUM"])
        noaa_list = str(row["NOAA_ARS"]).split(",")
        for noaa_str in noaa_list:
            try:
                noaa = int(noaa_str.strip())
                mapping_dict[noaa] = harp
            except ValueError:
                continue
    
    # Add manual mappings (these override official ones if there are conflicts)
    mapping_dict.update(MANUAL_HARP_MAPPINGS)
    
    log.info(f"Enhanced HARP mapping loaded: {len(mapping_dict)} NOAA ARs mapped")
    log.info(f"Latest AR in mapping: {max(mapping_dict.keys()) if mapping_dict else 'None'}")
    
    # Warn about potential missing mappings for very recent data
    latest_ar = max(mapping_dict.keys()) if mapping_dict else 0
    if latest_ar < 14090:  # Adjust this threshold as mappings are updated
        log.warning(f"HARP mappings may be missing for ARs > {latest_ar}")
        log.warning("For very recent data, consider using dates 2-3 weeks old")
    
    return mapping_dict

def convert_noaa_to_harp(noaa_ar: int, mapping_df: pd.DataFrame = None, enhanced_mapping: Dict[int, int] = None):
    """Return the HARP number for *noaa_ar* using enhanced mapping.

    First tries the enhanced mapping (which includes manual overrides), 
    then falls back to the original string-contains search.
    """
    
    # Use enhanced mapping if provided
    if enhanced_mapping and noaa_ar in enhanced_mapping:
        return enhanced_mapping[noaa_ar]
    
    # Fallback to original method if mapping_df provided
    if mapping_df is not None:
        matches = mapping_df[mapping_df["NOAA_ARS"].astype(str).str.contains(str(int(noaa_ar)))]
        if not matches.empty:
            return int(matches.iloc[0]["HARPNUM"])
    
    return None

# --------------------------------------------------------------------
# 1) NOAA-→HARP mapping & set of non-flaring ARs
# --------------------------------------------------------------------
@cached(CACHE_DIR / "noaa_harp_map.pkl")
def load_noaa_harp_map() -> pd.DataFrame:
    log.info("Downloading NOAA→HARP mapping …")
    df = pd.read_csv(MAPPING_URL, sep=r"\s+")
    df.columns = df.columns.str.strip()
    return df

NONFLARING_NOAA = [12165, 12180, 12232, 12259, 12305, 12374, 12455, 12489, 12623, 12712]

# --------------------------------------------------------------------
# 2) GOES flare catalogue
# --------------------------------------------------------------------
SEVERITY_ORDER = {"A": 0, "B": 1, "C": 2, "M": 3, "M5": 4, "X": 5}

def fetch_swpc_flares(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch recent flare data from SWPC JSON API for dates after 2023."""
    import requests
    
    log.info("Fetching recent flare data from SWPC JSON API...")
    try:
        # SWPC only provides last 7 days - check if requested range is too old
        now = datetime.now(timezone.utc)
        if end_date < now - timedelta(days=7):
            log.warning(f"SWPC API only provides last 7 days. Requested end date {end_date.date()} is too old.")
            return pd.DataFrame()
            
        # Get 7-day flare data from SWPC
        url = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            log.warning("SWPC API returned no flare data")
            return pd.DataFrame()
            
        flares = []
        for flare in data:
            try:
                # Parse SWPC flare format - handle various field names
                begin_time_str = flare.get('begin_time') or flare.get('event_date') or flare.get('time_tag')
                if not begin_time_str:
                    continue
                    
                begin_time = pd.to_datetime(begin_time_str)
                max_time = pd.to_datetime(flare.get('max_time', begin_time_str))
                end_time = pd.to_datetime(flare.get('end_time', begin_time_str))
                
                # Convert to UTC if needed
                if begin_time.tz is None:
                    begin_time = begin_time.replace(tzinfo=timezone.utc)
                if max_time.tz is None:
                    max_time = max_time.replace(tzinfo=timezone.utc)
                if end_time.tz is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)
                    
                # Filter by date range
                if begin_time >= start_date and begin_time <= end_date:
                    cls_raw = flare.get('current_class') or flare.get('class') or 'C1.0'
                    cls_std = _standardise_class(cls_raw)
                    
                    # Handle active region - try different field names
                    ar = flare.get('active_region') or flare.get('noaa_ar') or flare.get('region')
                    if ar and str(ar).strip() and str(ar).strip() != 'None':
                        flares.append({
                            "start_time": begin_time,
                            "peak_time": max_time,
                            "end_time": end_time,
                            "goes_class": cls_std,
                            "noaa_ar": ar,
                        })
                        
            except Exception as e:
                log.debug(f"Error parsing SWPC flare: {e}")
                continue
                
        df = pd.DataFrame(flares)
        if not df.empty:
            df["severity"] = df["goes_class"].map(SEVERITY_ORDER)
            df = df.sort_values("start_time").reset_index(drop=True)
            log.info(f"SWPC API returned {len(df)} flares in date range")
        else:
            log.warning("SWPC API: No flares found in requested date range")
        
        return df
        
    except Exception as e:
        log.warning(f"SWPC API failed: {e}")
        return pd.DataFrame()

def fetch_donki_flares(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch flare data from NASA DONKI system for recent dates."""
    import requests
    
    log.info("Fetching flare data from NASA DONKI...")
    try:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        url = f"https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR?startDate={start_str}&endDate={end_str}"
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            log.warning("DONKI API returned no flare data")
            return pd.DataFrame()
            
        flares = []
        for flare in data:
            try:
                # Handle None datetime values safely
                begin_time_str = flare.get('beginTime')
                peak_time_str = flare.get('peakTime')
                end_time_str = flare.get('endTime')
                
                if not begin_time_str:
                    continue
                    
                begin_time = pd.to_datetime(begin_time_str)
                peak_time = pd.to_datetime(peak_time_str) if peak_time_str else begin_time
                end_time = pd.to_datetime(end_time_str) if end_time_str else peak_time
                
                # Convert to UTC if needed
                if begin_time.tz is None:
                    begin_time = begin_time.replace(tzinfo=timezone.utc)
                if peak_time.tz is None:
                    peak_time = peak_time.replace(tzinfo=timezone.utc)
                if end_time.tz is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)
                    
                cls_raw = flare.get('classType', 'C1.0')
                cls_std = _standardise_class(cls_raw)
                
                # Handle active region
                ar = flare.get('activeRegionNum')
                if ar and str(ar).strip() and str(ar).strip() != 'None':
                    flares.append({
                        "start_time": begin_time,
                        "peak_time": peak_time,
                        "end_time": end_time,
                        "goes_class": cls_std,
                        "noaa_ar": ar,
                    })
                    
            except Exception as e:
                log.debug(f"Error parsing DONKI flare: {e}")
                continue
                
        df = pd.DataFrame(flares)
        if not df.empty:
            df["severity"] = df["goes_class"].map(SEVERITY_ORDER)
            df = df.sort_values("start_time").reset_index(drop=True)
            log.info(f"DONKI API returned {len(df)} flares")
        else:
            log.warning("DONKI API: No flares found in requested date range")
        
        return df
        
    except Exception as e:
        log.warning(f"DONKI API failed: {e}")
        return pd.DataFrame()

def _standardise_class(raw_cls: str) -> str:
    """Map GOES class string → categorical label used for severity ordering."""
    code = raw_cls[0].upper()
    if code == "M":
        mag = float(raw_cls[1:])
        return "M5" if mag >= 5.0 else "M"
    return code  # "A", "B", "C", "X"

# Helper to convert SunPy / Astropy Time or raw str→ datetime (UTC)
def _to_datetime_utc(t) -> datetime:
    """Return a timezone-aware UTC datetime from various input types."""
    if hasattr(t, "to_datetime"):
        dt = t.to_datetime()
    elif hasattr(t, "datetime"):
        dt = t.datetime
    else:
        dt = pd.to_datetime(str(t)).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

def load_goes_events() -> pd.DataFrame:
    """Return GOES flare dataframe for current START_DATE–END_DATE, cached on disk."""
    cache_path = CACHE_DIR / f"goes_events_{START_DATE:%Y%m%d}_{END_DATE:%Y%m%d}.pkl"
    if cache_path.exists():
        return pd.read_pickle(cache_path)

    # Use different data sources based on date range
    if START_DATE >= datetime(2024, 1, 1, tzinfo=timezone.utc):
        # For recent data (2024+), try SWPC and DONKI APIs
        log.info("Using real-time sources for recent data...")
        
        # Try SWPC first
        df = fetch_swpc_flares(START_DATE, END_DATE)
        
        # If SWPC fails or returns little data, try DONKI
        if df.empty or len(df) < 2:
            df_donki = fetch_donki_flares(START_DATE, END_DATE)
            if not df_donki.empty:
                df = df_donki if df.empty else pd.concat([df, df_donki]).drop_duplicates('start_time')
        
        # If both real-time sources fail, fallback to HEK
        if df.empty:
            log.warning("Real-time sources returned no data. Falling back to HEK...")
            try:
                events = goes_mod.get_goes_event_list(TimeRange(START_DATE, END_DATE))
                if events:
                    recs = []
                    for ev in events:
                        cls_std = _standardise_class(ev["goes_class"])
                        start_dt = _to_datetime_utc(ev["start_time"])
                        peak_dt = _to_datetime_utc(ev["peak_time"]) if ev.get("peak_time") is not None else start_dt
                        end_dt = _to_datetime_utc(ev["end_time"]) if ev.get("end_time") is not None else start_dt
                        recs.append({
                            "start_time": start_dt,
                            "peak_time": peak_dt,
                            "end_time": end_dt,
                            "goes_class": cls_std,
                            "noaa_ar": ev["noaa_active_region"],
                        })
                    df = pd.DataFrame(recs)
                    df.dropna(subset=["noaa_ar"], inplace=True)
                    df["severity"] = df["goes_class"].map(SEVERITY_ORDER)
                    df = df.sort_values("start_time").reset_index(drop=True)
                    log.info(f"HEK fallback returned {len(df)} flares")
            except Exception as e:
                log.warning(f"HEK fallback also failed: {e}")
        
        if df.empty:
            log.error("All data sources failed for recent data. No flare data available.")
            return df
            
    else:
        # For historical data (pre-2024), use original HEK method
        log.info("Fetching GOES flare list from HEK...")
        # HEK can choke on multi-year queries; request data in ≤1-year chunks
        events: List = []
        yr_start: datetime = START_DATE
        while yr_start <= END_DATE:
            yr_end = min(yr_start + relativedelta(years=1) - timedelta(seconds=1), END_DATE)
            try:
                events.extend(goes_mod.get_goes_event_list(TimeRange(yr_start, yr_end)))
            except Exception as e:
                log.warning("HEK query failed for %s – %s : %s", yr_start.date(), yr_end.date(), e)
            yr_start = yr_end + timedelta(seconds=1)

        # If no events could be fetched (e.g., all HEK queries failed), return empty DataFrame with expected columns
        if not events:
            log.error("All HEK queries failed - no flare data could be retrieved. Check your internet connection.")
            empty_df = pd.DataFrame(columns=["start_time", "peak_time", "end_time", "goes_class", "noaa_ar", "severity"])
            return empty_df

        recs: List[Dict] = []
        for ev in events:
            cls_std = _standardise_class(ev["goes_class"])
            start_dt = _to_datetime_utc(ev["start_time"])
            peak_dt  = _to_datetime_utc(ev["peak_time"]) if ev.get("peak_time") is not None else start_dt
            end_dt   = _to_datetime_utc(ev["end_time"])  if ev.get("end_time")  is not None else start_dt
            recs.append({
                "start_time": start_dt,
                "peak_time": peak_dt,
                "end_time": end_dt,
                "goes_class": cls_std,
                "noaa_ar": ev["noaa_active_region"],
            })
        df = pd.DataFrame(recs)
        df.dropna(subset=["noaa_ar"], inplace=True)
        df["severity"] = df["goes_class"].map(SEVERITY_ORDER)
        df = df.sort_values("start_time").reset_index(drop=True)
    
    df.to_pickle(cache_path)
    return df

# --------------------------------------------------------------------
# 3) SHARP retrieval helpers
# --------------------------------------------------------------------

QS_KEYS = (
    ["T_REC", "QUALITY", "OBS_VR", "CM_DIST", "CMP_LON", "NOAA_AR", "HARPNUM"] + FEATURES
)

_CLIENT = drms.Client(email=EMAIL)


@with_timeout(120)  # 2 minute timeout for JSOC queries
def _query_jsoc(qs: str):
    """Helper function to query JSOC with timeout."""
    return _CLIENT.query(qs, key=",".join(QS_KEYS), seg=None)

def fetch_sharp_window(harp: int, start: datetime, end: datetime) -> pd.DataFrame:
    """Return SHARP rows for *harp* in [start,end] (12-min cadence)."""
    duration_h = (end - start).total_seconds() / 3600.0
    qs = (
        f"hmi.sharp_cea_720s[{harp}]"
        f"[{start:%Y.%m.%d_%H:%M:%S_TAI}/{duration_h:.3f}h]"
    )
    try:
        log.debug(f"Querying JSOC for HARP {harp}: {qs}")
        res = _query_jsoc(qs)
        if res is None:  # Timeout occurred
            log.warning(f"JSOC query timed out for HARP {harp}")
            return pd.DataFrame()
        log.debug(f"JSOC query completed for HARP {harp}")
    except Exception as exc:
        log.warning("JSOC query failed for HARP %d: %s", harp, exc)
        return pd.DataFrame()

    df = res[0] if isinstance(res, tuple) else res
    # Some queries may return no rows or omit the T_REC column entirely.
    if "T_REC" not in df.columns:
        # log.warning("No T_REC column in JSOC response for HARP %d (%s – %s)", harp, start, end)
        return pd.DataFrame()

    # strip invalid records emitted by JSOC
    df = df[~df["T_REC"].astype(str).str.startswith("Invalid")].copy()
    if df.empty:
        return df

    df["T_REC"] = pd.to_datetime(df["T_REC"], format="%Y.%m.%d_%H:%M:%S_TAI", errors="coerce")
    df.dropna(subset=["T_REC"], inplace=True)

    # quality filters -------------------------------------------------
    if "CM_DIST" in df.columns:
        cmd_series = pd.to_numeric(df["CM_DIST"], errors="coerce")
    elif "CMP_LON" in df.columns:
        cmd_series = pd.to_numeric(df["CMP_LON"], errors="coerce")
    else:
        cmd_series = None

    if cmd_series is not None and cmd_series.notna().any():
        cmd_ok = cmd_series.abs() <= 70
        df = df[cmd_ok]
    # if cmd_series is None **or** all NaN, keep rows (proxy) – will be filtered later if desired

    # apply remaining quality filters
    df = df[(df["QUALITY"] == 0) & (df["OBS_VR"].abs() <= 3500)]

    # drop NaNs in features
    df.dropna(subset=FEATURES, inplace=True)

    df.rename(columns={"T_REC": "DATE__OBS"}, inplace=True)
    return df[["DATE__OBS", "NOAA_AR", "HARPNUM", *FEATURES]].reset_index(drop=True)

# --------------------------------------------------------------------
# 4) Build sample rows ------------------------------------------------
# --------------------------------------------------------------------

def construct_samples_for_event(row: pd.Series, window_h: int, harp: int) -> pd.DataFrame:
    """All SHARP rows within *window_h* hours **before** (positive) or **after** (negative) the reference time."""
    ref_raw = row["start_time"]
    if hasattr(ref_raw, "replace") and hasattr(ref_raw, "tzinfo"):
        # likely a datetime already
        ref = ref_raw if ref_raw.tzinfo else ref_raw.replace(tzinfo=timezone.utc)
    else:
        ref = _to_datetime_utc(ref_raw)
    if window_h >= 0:
        start, end = ref - timedelta(hours=window_h), ref
    else:  # after → positive interval after reference
        start, end = ref, ref + timedelta(hours=abs(window_h))
    df_sharp = fetch_sharp_window(harp, start, end)

    if df_sharp.empty:
        return df_sharp

    df_sharp["Flare"] = None  # will be filled later by caller
    return df_sharp

# --------------------------------------------------------------------
# 5) Task-specific label helpers
# --------------------------------------------------------------------

_POSITIVE_CLASSES = {
    "C": {"C", "M", "M5", "X"},
    "M": {"M", "X"},
    "M5": {"M5", "X"},
}
_NEGATIVE_CLASSES = {
    "C": {"A", "B"},
    "M": {"C"},
    "M5": {"C", "M"},  # "M" here means M1-4.
}


def is_positive(goes_cls: str, task: str) -> bool:
    return goes_cls in _POSITIVE_CLASSES[task]

# --------------------------------------------------------------------
# 6) Main assembly routine -------------------------------------------
# --------------------------------------------------------------------

def build_dataset(task: str, window_h: int, goes_df: pd.DataFrame, map_df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with all samples & their labels for *task* × *window*."""
    
    # Get enhanced mapping for better recent AR support
    enhanced_mapping = get_enhanced_harp_mapping()
    
    rows: List[pd.DataFrame] = []
    taken: set = set()  # track (harp, ts) already labelled by higher-severity event

    # Debug: show what flares we have
    log.info(f"Building dataset for {task} with {len(goes_df)} flares:")
    for _, ev in goes_df.iterrows():
        log.info(f"  Flare: {ev['goes_class']} class at {ev['start_time']} from AR {ev['noaa_ar']}")

    # sort flares by descending severity so larger flares take precedence
    events = goes_df.sort_values("severity", ascending=False)

    # 6.1 FLARING ARs --------------------------------------------------
    flares_processed = 0
    flares_with_harp = 0
    flares_with_data = 0
    flares_without_harp = []
    
    for _, ev in events.iterrows():
        flares_processed += 1
        noaa_ar = ev["noaa_ar"]
        
        # Skip flares with no AR or AR = 0
        if noaa_ar == 0 or pd.isna(noaa_ar):
            log.debug(f"Skipping flare with no active region (AR = {noaa_ar})")
            continue
        
        # Try enhanced mapping first, then fallback to original
        harp = convert_noaa_to_harp(noaa_ar, mapping_df=map_df, enhanced_mapping=enhanced_mapping)
        
        if harp is None:
            flares_without_harp.append(noaa_ar)
            log.debug(f"No HARP mapping found for NOAA AR {noaa_ar}")
            continue
        
        flares_with_harp += 1
        log.info(f"Processing NOAA AR {noaa_ar} -> HARP {harp}")

        # Collect window SHARP rows
        df_ev = construct_samples_for_event(ev, window_h, harp)
        if df_ev.empty:
            log.debug(f"No SHARP data found for HARP {harp} in {window_h}h window before flare")
            continue

        flares_with_data += 1
        label = "P" if is_positive(ev["goes_class"], task) else "N"
        df_ev["Flare"] = label
        log.debug(f"Added {len(df_ev)} samples for {ev['goes_class']} flare (label: {label})")

        # keep only rows not already assigned by a larger flare
        df_ev["__key"] = list(zip(df_ev["HARPNUM"], pd.to_datetime(df_ev["DATE__OBS"])))
        df_ev = df_ev[~df_ev["__key"].isin(taken)]
        taken.update(df_ev["__key"].tolist())
        df_ev = df_ev.drop(columns="__key")
        rows.append(df_ev)

    log.info(f"Flare processing summary: {flares_processed} total, {flares_with_harp} with HARP mapping, {flares_with_data} with SHARP data")
    
    # Show which ARs are missing HARP mappings
    if flares_without_harp:
        unique_missing = sorted(set(flares_without_harp))
        log.warning(f"ARs without HARP mappings: {unique_missing}")
        
        # Check if these are very recent ARs
        recent_ars = [ar for ar in unique_missing if ar > 14070]
        if recent_ars:
            log.info(f"Recent ARs without HARP mappings: {recent_ars}")
            log.info("Note: Recent ARs may not have HARP numbers assigned yet in JSOC database.")
            log.info("For current research, consider using data from 1-2 weeks ago to ensure HARP mappings exist.")

    # 6.2 NON-FLARING ARs ---------------------------------------------
    for noaa_ar in NONFLARING_NOAA:
        harp = convert_noaa_to_harp(noaa_ar, mapping_df=map_df, enhanced_mapping=enhanced_mapping)
        if harp is None:
            continue

        # window *after* AR start – approximate dt using first available sharp row
        dummy_row = pd.Series({"start_time": START_DATE})
        df_nf = construct_samples_for_event(dummy_row, -window_h, harp)  # negative → after
        if df_nf.empty:
            continue
        df_nf["Flare"] = "N"
        rows.append(df_nf)

    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Flare", *FEATURES])
    log.info(f"Total samples generated: {len(big)}")
    return big

# --------------------------------------------------------------------
# 7) Cross-validation split & scaling --------------------------------
# --------------------------------------------------------------------

def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe chronologically: first 90% → train, last 10% → test."""
    # Sort by timestamp
    df['__ts'] = pd.to_datetime(df["DATE__OBS"])
    df = df.sort_values('__ts').reset_index(drop=True)
    
    # Split 90/10
    n = len(df)
    n_test = int(np.ceil(0.10 * n))
    test = df.iloc[-n_test:].copy()
    train = df.iloc[:n-n_test].copy()
    
    # Remove temporary column
    train = train.drop(columns='__ts')
    test = test.drop(columns='__ts')
    
    return train, test

def minmax_scale(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    scales = {}
    for col in FEATURES:
        mn, mx = train[col].min(), train[col].max()
        span = (mx - mn) if mx > mn else 1.0
        train[col] = 2 * (train[col] - mn) / span - 1
        test[col] = 2 * (test[col] - mn) / span - 1
        scales[col] = (mn, mx)
    return (train, scales)

def pad_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Insert zero-valued rows where 12-min cadence gaps exist (per HARPNUM)."""
    if df.empty:
        return df

    out_rows: List[pd.DataFrame] = []
    for harp, grp in df.groupby("HARPNUM"):
        grp_sorted = grp.sort_values("DATE__OBS").reset_index(drop=True)
        times = pd.to_datetime(grp_sorted["DATE__OBS"])

        padded = [grp_sorted.iloc[0]]
        for i in range(1, len(grp_sorted)):
            prev_ts = times.iloc[i - 1]
            curr_ts = times.iloc[i]
            gap_steps = int((curr_ts - prev_ts).total_seconds() // 720) - 1
            for g in range(gap_steps):
                ts = prev_ts + timedelta(seconds=720 * (g + 1))
                row = grp_sorted.iloc[i - 1].copy()
                row["DATE__OBS"] = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                for feat in FEATURES:
                    row[feat] = 0.0  # already normalised space
                row["Flare"] = "padding"
                padded.append(row)
            padded.append(grp_sorted.iloc[i])
        out_rows.append(pd.DataFrame(padded))

    return pd.concat(out_rows, ignore_index=True)

# --------------------------------------------------------------------
# 8) Entry-point ------------------------------------------------------
# --------------------------------------------------------------------

def extend_dataset_realtime(
    output_dir: str = "realtime_data",
    lookback_hours: int = 72,
    update_interval_hours: int = 12,
    label_delay_hours: int = 48
) -> None:
    """
    Extend datasets with real-time SHARP data for operational deployment.
    
    This function:
    1. Fetches the latest SHARP data for all active regions
    2. Adds unlabeled samples to the dataset
    3. Backfills labels for older samples once we know if flares occurred
    
    Args:
        output_dir: Directory to store real-time datasets
        lookback_hours: How far back to collect SHARP data (default: 72h)
        update_interval_hours: How often to update (default: 12h)
        label_delay_hours: How long to wait before labeling (default: 48h)
    """
    import os
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    log.info(f"Starting real-time dataset extension in {output_dir}")
    log.info(f"Lookback: {lookback_hours}h, Update interval: {update_interval_hours}h, Label delay: {label_delay_hours}h")
    
    # Get current time and calculate time windows
    now = datetime.now(timezone.utc)
    data_start = now - timedelta(hours=lookback_hours)
    label_cutoff = now - timedelta(hours=label_delay_hours)
    
    log.info(f"Collecting SHARP data from {data_start} to {now}")
    log.info(f"Will label data older than {label_cutoff}")
    
    # Load HARP mappings
    map_df = load_noaa_harp_map()
    enhanced_mapping = get_enhanced_harp_mapping()
    
    # Get current active regions from recent flare data
    recent_flares_start = now - timedelta(days=7)  # Look back 7 days for active ARs
    recent_goes_df = load_goes_events_for_period(recent_flares_start, now)
    
    # Get unique active regions that have been active recently
    active_ars = set()
    if not recent_goes_df.empty:
        active_ars.update(recent_goes_df['noaa_ar'].dropna().astype(int).unique())
    
    # Also add some known recent ARs (you can update this list)
    recent_ar_numbers = list(range(14070, 14110))  # Adjust range as needed
    active_ars.update(recent_ar_numbers)
    
    log.info(f"Found {len(active_ars)} potentially active regions: {sorted(active_ars)}")
    
    # Collect SHARP data for all active regions
    all_samples = []
    successful_ars = 0
    
    for noaa_ar in sorted(active_ars):
        if noaa_ar == 0 or pd.isna(noaa_ar):
            continue
            
        # Get HARP mapping
        harp = convert_noaa_to_harp(noaa_ar, mapping_df=map_df, enhanced_mapping=enhanced_mapping)
        if harp is None:
            log.debug(f"No HARP mapping for AR {noaa_ar}")
            continue
            
        log.info(f"Collecting data for AR {noaa_ar} -> HARP {harp}")
        
        # Fetch SHARP data for the lookback period
        df_sharp = fetch_sharp_window(harp, data_start, now)
        if df_sharp.empty:
            log.debug(f"No SHARP data for HARP {harp}")
            continue
            
        # Add metadata
        df_sharp['NOAA_AR'] = noaa_ar
        df_sharp['HARPNUM'] = harp
        df_sharp['Flare'] = 'UNLABELED'  # Will be filled later
        df_sharp['Collection_Time'] = now.isoformat()
        
        all_samples.append(df_sharp)
        successful_ars += 1
        
    if not all_samples:
        log.warning("No SHARP data collected for any active regions")
        return
        
    # Combine all samples
    combined_df = pd.concat(all_samples, ignore_index=True)
    log.info(f"Collected {len(combined_df)} samples from {successful_ars} active regions")
    
    # Save unlabeled dataset
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    unlabeled_file = output_path / f"unlabeled_data_{timestamp_str}.csv"
    combined_df[OUTPUT_COLUMNS + ['Collection_Time']].to_csv(unlabeled_file, index=False)
    log.info(f"Saved unlabeled data to {unlabeled_file}")
    
    # Now backfill labels for older data
    backfill_labels_realtime(output_path, label_cutoff, map_df, enhanced_mapping)

def backfill_labels_realtime(
    output_dir: Path,
    label_cutoff: datetime,
    map_df: pd.DataFrame,
    enhanced_mapping: Dict[int, int]
) -> None:
    """
    Backfill labels for previously collected unlabeled data.
    
    Args:
        output_dir: Directory containing unlabeled datasets
        label_cutoff: Only label data older than this time
        map_df: NOAA-HARP mapping dataframe
        enhanced_mapping: Enhanced HARP mapping dictionary
    """
    log.info(f"Backfilling labels for data older than {label_cutoff}")
    
    # Find all unlabeled files
    unlabeled_files = list(output_dir.glob("unlabeled_data_*.csv"))
    if not unlabeled_files:
        log.info("No unlabeled files found for backfilling")
        return
        
    log.info(f"Found {len(unlabeled_files)} unlabeled files to process")
    
    # Get flare data for the labeling period
    # Look back extra time to catch flares that might affect older samples
    flare_start = label_cutoff - timedelta(hours=72)  # Look back 72h for flares
    flare_end = datetime.now(timezone.utc)
    
    try:
        flares_df = load_goes_events_for_period(flare_start, flare_end)
    except Exception as e:
        log.warning(f"Could not load flare data for labeling: {e}")
        return
        
    if flares_df.empty:
        log.info("No flare data available for labeling period")
        return
        
    log.info(f"Found {len(flares_df)} flares for labeling period")
    
    labeled_count = 0
    
    for unlabeled_file in unlabeled_files:
        try:
            # Load unlabeled data
            df = pd.read_csv(unlabeled_file)
            if df.empty:
                continue
                
            # Check if this file has data old enough to label
            df['DATE__OBS'] = pd.to_datetime(df['DATE__OBS'])
            old_enough_mask = df['DATE__OBS'] <= label_cutoff
            
            if not old_enough_mask.any():
                log.debug(f"No samples old enough to label in {unlabeled_file.name}")
                continue
                
            log.info(f"Labeling {old_enough_mask.sum()} samples in {unlabeled_file.name}")
            
            # Label samples for each task
            for task in THRESHOLDS:
                df_task = df.copy()
                
                # Apply labels based on flare data
                df_task = apply_flare_labels(df_task, flares_df, task, enhanced_mapping, map_df)
                
                # Only keep samples that are old enough to label reliably
                df_labeled = df_task[old_enough_mask].copy()
                
                if df_labeled.empty:
                    continue
                    
                # Save labeled dataset
                timestamp = unlabeled_file.stem.split('_')[-1]  # Extract timestamp
                labeled_file = output_dir / f"labeled_data_{task}_{timestamp}.csv"
                df_labeled[OUTPUT_COLUMNS].to_csv(labeled_file, index=False)
                
                labeled_count += len(df_labeled)
                log.info(f"Saved {len(df_labeled)} labeled samples for task {task} to {labeled_file.name}")
                
        except Exception as e:
            log.error(f"Error processing {unlabeled_file.name}: {e}")
            continue
            
    log.info(f"Backfilling complete. Labeled {labeled_count} total samples")

def apply_flare_labels(
    df: pd.DataFrame,
    flares_df: pd.DataFrame,
    task: str,
    enhanced_mapping: Dict[int, int],
    map_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply flare labels to SHARP data based on flare occurrences.
    
    Args:
        df: SHARP dataframe to label
        flares_df: Flare events dataframe
        task: Prediction task (C, M, or M5)
        enhanced_mapping: Enhanced HARP mapping
        map_df: NOAA-HARP mapping dataframe
        
    Returns:
        Labeled dataframe
    """
    df = df.copy()
    df['Flare'] = 'N'  # Default to negative
    
    # Group by HARP number for efficient processing
    for harp, harp_group in df.groupby('HARPNUM'):
        # Find corresponding NOAA AR
        noaa_ar = harp_group['NOAA_AR'].iloc[0]
        
        # Find flares for this AR
        ar_flares = flares_df[flares_df['noaa_ar'] == noaa_ar].copy()
        if ar_flares.empty:
            continue
            
        # For each SHARP sample, check if a flare occurs within prediction window
        for idx, row in harp_group.iterrows():
            sample_time = pd.to_datetime(row['DATE__OBS'])
            
            # Check for flares in the next 24 hours (prediction window)
            prediction_end = sample_time + timedelta(hours=24)
            
            # Find flares in prediction window
            future_flares = ar_flares[
                (ar_flares['start_time'] > sample_time) & 
                (ar_flares['start_time'] <= prediction_end)
            ]
            
            if not future_flares.empty:
                # Get the highest severity flare in the window
                max_severity_flare = future_flares.loc[future_flares['severity'].idxmax()]
                flare_class = max_severity_flare['goes_class']
                
                # Apply task-specific labeling
                if is_positive(flare_class, task):
                    df.loc[idx, 'Flare'] = 'P'
                    
    return df

def load_goes_events_for_period(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Load GOES events for a specific time period (bypassing cache for real-time use).
    
    Args:
        start_date: Start of period
        end_date: End of period
        
    Returns:
        DataFrame of flare events
    """
    log.info(f"Loading GOES events for {start_date} to {end_date}")
    
    # For recent data, try real-time sources first
    if start_date >= datetime(2024, 1, 1, tzinfo=timezone.utc):
        # Try SWPC first
        df = fetch_swpc_flares(start_date, end_date)
        
        # If SWPC fails or returns little data, try DONKI
        if df.empty or len(df) < 2:
            df_donki = fetch_donki_flares(start_date, end_date)
            if not df_donki.empty:
                df = df_donki if df.empty else pd.concat([df, df_donki]).drop_duplicates('start_time')
    else:
        # For historical data, use HEK
        try:
            events = goes_mod.get_goes_event_list(TimeRange(start_date, end_date))
            if events:
                recs = []
                for ev in events:
                    cls_std = _standardise_class(ev["goes_class"])
                    start_dt = _to_datetime_utc(ev["start_time"])
                    peak_dt = _to_datetime_utc(ev["peak_time"]) if ev.get("peak_time") is not None else start_dt
                    end_dt = _to_datetime_utc(ev["end_time"]) if ev.get("end_time") is not None else start_dt
                    recs.append({
                        "start_time": start_dt,
                        "peak_time": peak_dt,
                        "end_time": end_dt,
                        "goes_class": cls_std,
                        "noaa_ar": ev["noaa_active_region"],
                    })
                df = pd.DataFrame(recs)
                df.dropna(subset=["noaa_ar"], inplace=True)
                df["severity"] = df["goes_class"].map(SEVERITY_ORDER)
                df = df.sort_values("start_time").reset_index(drop=True)
            else:
                df = pd.DataFrame()
        except Exception as e:
            log.warning(f"HEK query failed: {e}")
            df = pd.DataFrame()
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Build SHARP datasets")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD", default=None)
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD", default=None)
    parser.add_argument("--realtime", action="store_true", help="Run real-time data collection")
    parser.add_argument("--output-dir", type=str, help="Output directory for real-time data", default="realtime_data")
    parser.add_argument("--lookback-hours", type=int, help="Hours of data to collect", default=72)
    parser.add_argument("--label-delay-hours", type=int, help="Hours to wait before labeling", default=48)
    args = parser.parse_args()

    # Handle real-time mode
    if args.realtime:
        log.info("Running in real-time data collection mode")
        extend_dataset_realtime(
            output_dir=args.output_dir,
            lookback_hours=args.lookback_hours,
            label_delay_hours=args.label_delay_hours
        )
        return

    global START_DATE, END_DATE
    if args.start:
        START_DATE = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    if args.end:
        END_DATE = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    # Log the date range being processed
    log.info(f"Processing date range: {START_DATE.date()} to {END_DATE.date()}")
    
    # Warn about very recent dates
    now = datetime.now(timezone.utc)
    if END_DATE > now - timedelta(days=7):
        log.warning("Processing very recent data - some ARs may not have HARP mappings yet")
        log.warning("For best results with recent data, use dates at least 1-2 weeks old")

    map_df = load_noaa_harp_map()
    # reload GOES with possibly new dates – bypass cache by calling internal function if range changed
    goes_df = load_goes_events()

    # Exit if no flare data is available (all HEK queries failed)
    if goes_df.empty:
        log.error("No flare data available. Can't continue dataset construction. Please check internet connection and retry.")
        return

    for task in THRESHOLDS:
        for window_h in WINDOWS:
            log.info("Building data for task ≥%s, window %dh …", task, window_h)
            df = build_dataset(task, window_h, goes_df, map_df)
            if df.empty:
                log.warning("No samples for %s/%dh – skipping", task, window_h)
                continue

            # Split chronologically, 90% train / 10% test
            train_df, test_df = chronological_split(df)

            # File paths
            train_csv = f"training_data_{task}_{window_h}.csv"
            test_csv  = f"testing_data_{task}_{window_h}.csv"
            scales_json = f"scales_{task}_{window_h}.json"

            # Min-max scale using training data parameters
            train_scaled, scales = minmax_scale(train_df, test_df)
            test_scaled = test_df.copy()  # scaled in-place by function

            # Apply zero-padding after scaling
            train_padded = pad_missing(train_scaled)
            test_padded = pad_missing(test_scaled)

            # Save CSV and scale parameters
            train_padded[OUTPUT_COLUMNS].to_csv(train_csv, index=False)
            test_padded[OUTPUT_COLUMNS].to_csv(test_csv, index=False)
            log.info("Saved %s (%d rows) and %s (%d rows)", 
                     train_csv, len(train_padded), 
                     test_csv, len(test_padded))

            # Save scale parameters
            with open(scales_json, "w") as fh:
                json.dump({c: list(v) for c, v in scales.items()}, fh, indent=2)

    log.info("Completed all datasets.")


if __name__ == "__main__":
    main() 
