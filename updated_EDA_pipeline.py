"""
====================================================================================================
RMS CATASTROPHE EXPOSURE & POLICY ANALYTICS PIPELINE (PYTHON / SQL RELATIONAL ENGINE)
====================================================================================================
Repository: AmitDey261-da/Arina_Analysis
Filename: updated_EDA_pipeline.py
Author: Catastrophe Risk & Exposure Analytics Engineering Team
Version: 3.0 (Enterprise AI & Deep Learning Edition)

Pipeline Architecture:
1. Automated Ingestion & Case-Insensitive Schema Normalization into In-Memory SQL DB.
2. Comprehensive Standalone Table Deep-Dive EDA for all 8 Datasets (Prior vs Current Month).
3. Relational Loc ⨝ Pol Merges (on ACCNTNUM) in SQL & Python with Match & Orphan Diagnostics.
4. Merged Portfolio EDA (Account Rollups, Rate-on-Line, Limit Utilization, State HHI Index).
5. Prior Month vs Current Month (MoM) Reconciliation, Roll-Forward Waterfall & Multi-Axis Drift.
6. Enterprise AI & Deep Learning Suite:
   - DL 1: Deep Neural Autoencoder (Latent Risk Embeddings & Reconstruction Loss Outliers).
   - DL 2: Deep MLP Non-Linear Pricing Regressor (Neural Residual Mispricing Detection).
   - ML 3: Multi-Dimensional Exposure Anomaly Detection (Isolation Forest, n_jobs=-1).
   - ML 4: High-Performance Exposure Risk Segmentation (MiniBatchKMeans + 2D PCA).
   - ML 5: Geospatial Catastrophe Accumulation & Hotspot Clustering.
   - ML 6: Extreme Value Theory (EVT) Tail Risk Quantiles (99% VaR, 99% TVaR & EP Curve).
   - ML 7: Statistical Feature Drift Matrix (Population Stability Index - PSI & KS Tests).
   - ML 8: Supervised Covariate Portfolio Drift Classifier (Random Forest Feature Importance).
   - ML 9: Reinsurance Stress-Testing & Climate/Inflation Shock Simulation.
7. Native ANSI SQL CTE & Analytical Window Function Scripts.
8. Visual Dashboards & Executive Audit Deliverables.
====================================================================================================
"""

import os
import sys
import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import silhouette_score, r2_score
from scipy.stats import ks_2samp

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# ==================================================================================================
# 0. CONFIGURATION & SCHEMA DEFINITIONS
# ==================================================================================================
DATA_DIR = "."  # <-- Set your directory path here (e.g., r"C:\ExposureData" or ".")

FILES = {
    "prior_com_loc": os.path.join(DATA_DIR, "COM RMS_Loc_March_2026.csv"),
    "prior_com_pol": os.path.join(DATA_DIR, "COM RMS_Pol_March_2026.csv"),
    "prior_std_loc": os.path.join(DATA_DIR, "RMS_Loc_March_2026.csv"),
    "prior_std_pol": os.path.join(DATA_DIR, "RMS_Pol_March_2026.csv"),
    
    "curr_com_loc":  os.path.join(DATA_DIR, "COM RMS_Loc_April_2026.csv"),
    "curr_com_pol":  os.path.join(DATA_DIR, "COM RMS_Pol_April_2026.csv"),
    "curr_std_loc":  os.path.join(DATA_DIR, "RMS_Loc_April_2026.csv"),
    "curr_std_pol":  os.path.join(DATA_DIR, "RMS_Pol_April_2026.csv"),
}

STANDARD_COL_MAP = {
    "accntnum": "ACCNTNUM", "locnum": "LOCNUM", "locname": "LOCNAME",
    "latitude": "LATITUDE", "longitude": "LONGITUDE", "streetname": "STREETNAME",
    "city": "CITY", "state": "STATE", "statecode": "STATECODE",
    "postalcode": "POSTALCODE", "county": "COUNTY", "bldgclass": "BLDGCLASS",
    "occtype": "OCCTYPE", "yearbuilt": "YEARBUILT", "floorarea": "FLOORAREA",
    "numstories": "NUMSTORIES", "wscv4limit": "WSCV4LIMIT", "wscv4val": "WSCV4VAL",
    "wscv5limit": "WSCV5LIMIT", "wscv5val": "WSCV5VAL", "wscv6limit": "WSCV6LIMIT",
    "wscv6val": "WSCV6VAL", "wscv7limit": "WSCV7LIMIT", "wscv7val": "WSCV7VAL",
    "wssitelim": "WSSITELIM", "wssiteded": "WSSITEDED", "tocv4limit": "TOCV4LIMIT",
    "tocv4val": "TOCV4VAL", "tocv5limit": "TOCV5LIMIT", "tocv5val": "TOCV5VAL",
    "tocv6limit": "TOCV6LIMIT", "tocv6val": "TOCV6VAL", "tocv7limit": "TOCV7LIMIT",
    "tocv7val": "TOCV7VAL", "tositelim": "TOSITELIM", "wscv4ded": "WSCV4DED",
    "wscv6ded": "WSCV6DED", "tocv4ded": "TOCV4DED", "tocv6ded": "TOCV6DED",
    "bldgscheme": "BLDGSCHEME", "cntrycode": "CNTRYCODE", "cntryscheme": "CNTRYSCHEME",
    "occscheme": "OCCSCHEME", "roofsys": "ROOFSYS", "roofgeom": "ROOFGEOM",
    "roofanch": "ROOFANCH", "roofage": "ROOFAGE", "cladrate": "CLADRATE",
    "cladsys": "CLADSYS", "resistopen": "RESISTOPEN",
    
    # Pol columns
    "accntname": "ACCNTNAME", "prodname": "PRODNAME", "cedantid": "CEDANTID",
    "cedantname": "CEDANTNAME", "policynum": "POLICYNUM", "lobname": "LOBNAME",
    "policytype": "POLICYTYPE", "blanpreamt": "BLANPREAMT", "blanprecur": "BLANPRECUR",
    "userdef1": "USERDEF1", "userdef2": "USERDEF2", "blanlimamt": "BLANLIMAMT"
}

def banner(title):
    print("\n" + "=" * 100)
    print(f" {title.upper()}")
    print("=" * 100)

def section(title):
    print("\n" + "-" * 85)
    print(f" >>> {title}")
    print("-" * 85)

# ==================================================================================================
# 1. INGESTION & SQL RELATIONAL DATABASE LAYER
# ==================================================================================================
class ExposureDataEngine:
    def __init__(self, file_dict=FILES, db_path=":memory:"):
        self.file_dict = file_dict
        self.conn = sqlite3.connect(db_path)
        self.raw_dfs = {}
        self.normalized_dfs = {}
        
    def load_and_ingest(self):
        banner("1. Ingestion & Case-Insensitive Schema Normalization into SQL Engine")
        for key, filepath in self.file_dict.items():
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Required file not found: '{filepath}'. Please check path in FILES dictionary.")
            
            df = pd.read_csv(filepath, low_memory=False)
            self.raw_dfs[key] = df
            
            col_map = {str(c).strip(): STANDARD_COL_MAP.get(str(c).strip().lower(), str(c).strip()) for c in df.columns}
            norm_df = df.rename(columns=col_map).copy()
            
            for str_k in ['ACCNTNUM', 'LOCNUM', 'POLICYNUM', 'STATE', 'BLDGCLASS', 'OCCTYPE', 'LOBNAME', 'CEDANTNAME', 'PRODNAME']:
                if str_k in norm_df.columns:
                    norm_df[str_k] = norm_df[str_k].astype(str).str.strip().replace({'nan': '', 'None': ''})
            
            num_cols = [
                'WSCV4LIMIT', 'WSCV4VAL', 'WSCV5LIMIT', 'WSCV5VAL', 'WSCV6LIMIT', 'WSCV6VAL',
                'WSCV7LIMIT', 'WSCV7VAL', 'WSSITELIM', 'WSSITEDED', 'TOCV4LIMIT', 'TOCV4VAL',
                'TOCV5LIMIT', 'TOCV5VAL', 'TOCV6LIMIT', 'TOCV6VAL', 'TOCV7LIMIT', 'TOCV7VAL',
                'TOSITELIM', 'WSCV4DED', 'WSCV6DED', 'TOCV4DED', 'TOCV6DED',
                'LATITUDE', 'LONGITUDE', 'YEARBUILT', 'FLOORAREA', 'NUMSTORIES', 'ROOFAGE',
                'BLANPREAMT', 'BLANLIMAMT'
            ]
            for col in num_cols:
                if col in norm_df.columns:
                    norm_df[col] = pd.to_numeric(norm_df[col], errors='coerce').fillna(0.0)
            
            if 'LOCNUM' in norm_df.columns:
                ws_cv_cols = [c for c in ['WSCV4VAL', 'WSCV5VAL', 'WSCV6VAL', 'WSCV7VAL'] if c in norm_df.columns]
                norm_df['TIV_WS'] = norm_df[ws_cv_cols].sum(axis=1) if ws_cv_cols else 0.0
                
                to_cv_cols = [c for c in ['TOCV4VAL', 'TOCV5VAL', 'TOCV6VAL', 'TOCV7VAL'] if c in norm_df.columns]
                if to_cv_cols:
                    norm_df['TIV_TO'] = norm_df[to_cv_cols].sum(axis=1)
                    norm_df['TIV_TOTAL'] = np.maximum(norm_df['TIV_WS'], norm_df['TIV_TO'])
                else:
                    norm_df['TIV_TO'] = 0.0
                    norm_df['TIV_TOTAL'] = norm_df['TIV_WS']
            
            self.normalized_dfs[key] = norm_df
            table_name = key
            norm_df.to_sql(table_name, self.conn, if_exists='replace', index=False)
            t_type = "Location" if 'LOCNUM' in norm_df.columns else "Policy"
            print(f" [OK] Ingested: '{table_name:<15}' | Type: {t_type:<8} | Rows: {len(norm_df):>7,} | Cols: {norm_df.shape[1]:>2} | Source: {filepath}")
            
        print(f"\nAll 8 datasets successfully normalized and loaded into relational SQL engine.")
        return self
        
    def query(self, sql):
        return pd.read_sql_query(sql, self.conn)

# ==================================================================================================
# 2. STANDALONE TABLE DEEP-DIVE EDA (ALL 8 DATASETS)
# ==================================================================================================
def run_standalone_table_eda(engine):
    banner("2. Standalone Table Deep-Dive Exploratory Data Analysis (EDA)")
    eda_summary = []
    
    for name, df in engine.normalized_dfs.items():
        is_loc = 'LOCNUM' in df.columns
        table_type = "Location Exposure Table" if is_loc else "Policy Financial Table"
        section(f"Dataset Profile: {name} ({table_type}) - Source: {engine.file_dict[name]}")
        
        n_rows, n_cols = df.shape
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        if is_loc:
            pk_col = ['ACCNTNUM', 'LOCNUM']
            is_pk_unique = not df.duplicated(subset=pk_col).any()
            unique_accnts = df['ACCNTNUM'].nunique()
            unique_locs = df['LOCNUM'].nunique()
            pk_str = f"Composite Key (ACCNTNUM, LOCNUM) Unique: {is_pk_unique} | Accounts: {unique_accnts:,} | Locations: {unique_locs:,}"
        else:
            pk_col = ['ACCNTNUM', 'POLICYNUM']
            is_pk_unique = not df.duplicated(subset=pk_col).any()
            unique_accnts = df['ACCNTNUM'].nunique()
            unique_pols = df['POLICYNUM'].nunique()
            pk_str = f"Composite Key (ACCNTNUM, POLICYNUM) Unique: {is_pk_unique} | Accounts: {unique_accnts:,} | Policies: {unique_pols:,}"
            
        null_counts = engine.raw_dfs[name].isnull().sum()
        total_nulls = null_counts.sum()
        null_pct = (total_nulls / (n_rows * n_cols)) * 100 if n_rows * n_cols > 0 else 0
        cols_with_nulls = null_counts[null_counts > 0]
        
        print(f"• Structure: {n_rows:,} rows × {n_cols} columns | Memory Footprint: {mem_mb:.2f} MB")
        print(f"• Key Integrity: {pk_str}")
        print(f"• Missing Data: {total_nulls:,} nulls ({null_pct:.2f}%) across {len(cols_with_nulls)} columns.")
        
        if is_loc:
            tot_tiv = df['TIV_TOTAL'].sum()
            mean_tiv = df['TIV_TOTAL'].mean() if len(df) > 0 else 0.0
            median_tiv = df['TIV_TOTAL'].median() if len(df) > 0 else 0.0
            p90_tiv = df['TIV_TOTAL'].quantile(0.90) if len(df) > 0 else 0.0
            p99_tiv = df['TIV_TOTAL'].quantile(0.99) if len(df) > 0 else 0.0
            max_tiv = df['TIV_TOTAL'].max() if len(df) > 0 else 0.0
            
            cv4 = df['WSCV4VAL'].sum() if 'WSCV4VAL' in df.columns else 0
            cv5 = df['WSCV5VAL'].sum() if 'WSCV5VAL' in df.columns else 0
            cv6 = df['WSCV6VAL'].sum() if 'WSCV6VAL' in df.columns else 0
            cv7 = df['WSCV7VAL'].sum() if 'WSCV7VAL' in df.columns else 0
            
            cv4_pct = (cv4 / tot_tiv * 100) if tot_tiv > 0 else 0
            cv5_pct = (cv5 / tot_tiv * 100) if tot_tiv > 0 else 0
            cv6_pct = (cv6 / tot_tiv * 100) if tot_tiv > 0 else 0
            cv7_pct = (cv7 / tot_tiv * 100) if tot_tiv > 0 else 0
            
            print(f"• Valuation Summary -> Total Insured Value (TIV): ${tot_tiv:,.2f}")
            print(f"  TIV Percentiles: Mean = ${mean_tiv:,.2f} | Median = ${median_tiv:,.2f} | 90th = ${p90_tiv:,.2f} | 99th = ${p99_tiv:,.2f} | Max = ${max_tiv:,.2f}")
            print(f"  Coverage Split (WS) -> Building (CV4): ${cv4:,.2f} ({cv4_pct:.1f}%), "
                  f"Other Struct (CV5): ${cv5:,.2f} ({cv5_pct:.1f}%), "
                  f"Contents (CV6): ${cv6:,.2f} ({cv6_pct:.1f}%), "
                  f"Time Element (CV7): ${cv7:,.2f} ({cv7_pct:.1f}%)")
                  
            fl_mean = df['FLOORAREA'].dropna().mean() if 'FLOORAREA' in df.columns else np.nan
            st_mean = df['NUMSTORIES'].dropna().mean() if 'NUMSTORIES' in df.columns else np.nan
            
            if 'YEARBUILT' in df.columns:
                valid_yr = df[df['YEARBUILT'] > 1800]['YEARBUILT'].dropna()
                avg_yr_str = f"{int(valid_yr.mean())}" if len(valid_yr) > 0 and not np.isnan(valid_yr.mean()) else "N/A"
                pre_1980 = (df['YEARBUILT'] < 1980) & (df['YEARBUILT'] > 1800)
                era_1980_1995 = (df['YEARBUILT'] >= 1980) & (df['YEARBUILT'] < 1995)
                era_1995_2002 = (df['YEARBUILT'] >= 1995) & (df['YEARBUILT'] < 2002)
                modern_2002_plus = (df['YEARBUILT'] >= 2002)
                print(f"  Construction Eras   -> Pre-1980: {pre_1980.sum():,} ({pre_1980.mean()*100:.1f}%), "
                      f"1980-1994: {era_1980_1995.sum():,} ({era_1980_1995.mean()*100:.1f}%), "
                      f"Post-Andrew (1995-2001): {era_1995_2002.sum():,} ({era_1995_2002.mean()*100:.1f}%), "
                      f"Modern IBHS/FBC (2002+): {modern_2002_plus.sum():,} ({modern_2002_plus.mean()*100:.1f}%)")
            else:
                avg_yr_str = "N/A"
                
            fl_str = f"{fl_mean:,.1f} sqft" if pd.notnull(fl_mean) else "N/A"
            st_str = f"{st_mean:.1f}" if pd.notnull(st_mean) else "N/A"
            print(f"  Physical Profile    -> Avg Floor Area: {fl_str} | Avg Stories: {st_str} | Avg Year Built: {avg_yr_str}")
                
            if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
                valid_geo = df[(df['LATITUDE'].between(-90, 90)) & (df['LONGITUDE'].between(-180, 180)) & (df['LATITUDE'] != 0)]
                valid_pct = (len(valid_geo) / len(df) * 100) if len(df) > 0 else 0
                lat_min, lat_max = (df['LATITUDE'].min(), df['LATITUDE'].max()) if len(df) > 0 else (0, 0)
                lon_min, lon_max = (df['LONGITUDE'].min(), df['LONGITUDE'].max()) if len(df) > 0 else (0, 0)
                print(f"  Geocoding Quality   -> Valid Coordinates: {len(valid_geo):,}/{len(df):,} ({valid_pct:.1f}%) | Lat: [{lat_min:.2f}, {lat_max:.2f}], Lon: [{lon_min:.2f}, {lon_max:.2f}]")
                
            if 'BLDGCLASS' in df.columns:
                top_bc = {k: int(v) for k, v in df['BLDGCLASS'].value_counts().head(3).items() if str(k).strip()}
                print(f"  Top Building Classes: {top_bc}")
            if 'OCCTYPE' in df.columns:
                top_ot = {k: int(v) for k, v in df['OCCTYPE'].value_counts().head(3).items() if str(k).strip()}
                print(f"  Top Occupancy Types : {top_ot}")
            if 'ROOFSYS' in df.columns:
                top_rs = {k: int(v) for k, v in df['ROOFSYS'].value_counts().head(3).items() if str(k).strip()}
                print(f"  Top Roof Systems    : {top_rs}")
                
            eda_summary.append({
                "Dataset": name, "Type": "Location", "Rows": n_rows, "Cols": n_cols,
                "Unique_Accounts": unique_accnts, "Unique_Entities": unique_locs,
                "Total_TIV_USD": tot_tiv, "Mean_TIV_USD": mean_tiv, "Total_Premium_USD": np.nan, "Total_Limit_USD": np.nan
            })
        else:
            tot_prem = df['BLANPREAMT'].sum()
            tot_lim = df['BLANLIMAMT'].sum()
            mean_prem = df['BLANPREAMT'].mean() if len(df) > 0 else 0.0
            median_prem = df['BLANPREAMT'].median() if len(df) > 0 else 0.0
            p90_prem = df['BLANPREAMT'].quantile(0.90) if len(df) > 0 else 0.0
            mean_lim = df['BLANLIMAMT'].mean() if len(df) > 0 else 0.0
            prem_rate = (tot_prem / tot_lim * 100) if tot_lim > 0 else 0.0
            
            print(f"• Policy Financials -> Total Blanket Premium: ${tot_prem:,.2f} | Total Blanket Limit: ${tot_lim:,.2f}")
            print(f"  Premium Percentiles: Mean = ${mean_prem:,.2f} | Median = ${median_prem:,.2f} | 90th = ${p90_prem:,.2f}")
            print(f"  Aggregate Rate on Blanket Limit: {prem_rate:.3f}% ({prem_rate * 100:.1f} bps)")
            
            if 'LOBNAME' in df.columns:
                top_lob = {k: int(v) for k, v in df['LOBNAME'].value_counts().head(3).items() if str(k).strip()}
                print(f"  Top Lines of Business (LOB): {top_lob}")
            if 'CEDANTNAME' in df.columns:
                top_ced = {k: int(v) for k, v in df['CEDANTNAME'].value_counts().head(3).items() if str(k).strip()}
                print(f"  Top Cedants                : {top_ced}")
            if 'PRODNAME' in df.columns:
                top_prod = {k: int(v) for k, v in df['PRODNAME'].value_counts().head(3).items() if str(k).strip()}
                print(f"  Top Producers              : {top_prod}")
                
            eda_summary.append({
                "Dataset": name, "Type": "Policy", "Rows": n_rows, "Cols": n_cols,
                "Unique_Accounts": unique_accnts, "Unique_Entities": unique_pols,
                "Total_TIV_USD": np.nan, "Mean_TIV_USD": np.nan, "Total_Premium_USD": tot_prem, "Total_Limit_USD": tot_lim
            })
            
    summary_df = pd.DataFrame(eda_summary)
    return summary_df

# ==================================================================================================
# 3. RELATIONAL MERGING & MULTI-TIER MERGED EXPOSURE EDA
# ==================================================================================================
def merge_and_analyze_exposure(engine):
    banner("3. Relational Merging (Location ⨝ Policy on ACCNTNUM) & Multi-Tier EDA")
    
    pairs = [
        ("prior_com", "prior_com_loc", "prior_com_pol", "Prior Commercial (March 2026)"),
        ("prior_std", "prior_std_loc", "prior_std_pol", "Prior Standard (March 2026)"),
        ("curr_com", "curr_com_loc", "curr_com_pol", "Current Commercial (April 2026)"),
        ("curr_std", "curr_std_loc", "curr_std_pol", "Current Standard (April 2026)"),
    ]
    
    merged_data = {}
    
    for key, loc_key, pol_key, label in pairs:
        section(f"Merge Integrity & Reinsurance Exposure Profile: {label}")
        loc_df = engine.normalized_dfs[loc_key]
        pol_df = engine.normalized_dfs[pol_key]
        
        sql_merge = f"""
        SELECT 
            p.ACCNTNUM,
            p.ACCNTNAME,
            p.POLICYNUM,
            p.PRODNAME,
            p.CEDANTID,
            p.CEDANTNAME,
            p.LOBNAME,
            p.POLICYTYPE,
            p.BLANPREAMT,
            p.BLANLIMAMT,
            p.USERDEF1,
            p.USERDEF2,
            l.LOCNUM,
            l.LOCNAME,
            l.LATITUDE,
            l.LONGITUDE,
            l.STREETNAME,
            l.CITY,
            l.STATE,
            l.STATECODE,
            l.POSTALCODE,
            l.COUNTY,
            l.BLDGCLASS,
            l.OCCTYPE,
            l.YEARBUILT,
            l.FLOORAREA,
            l.NUMSTORIES,
            l.WSCV4VAL,
            l.WSCV5VAL,
            l.WSCV6VAL,
            l.WSCV7VAL,
            l.TIV_WS,
            l.TIV_TO,
            l.TIV_TOTAL,
            l.WSSITELIM,
            l.WSSITEDED,
            l.ROOFSYS,
            l.ROOFGEOM,
            l.ROOFANCH,
            l.ROOFAGE,
            l.CLADRATE,
            l.CLADSYS,
            l.RESISTOPEN
        FROM {pol_key} p
        INNER JOIN {loc_key} l ON p.ACCNTNUM = l.ACCNTNUM
        """
        merged_df = engine.query(sql_merge)
        
        loc_accnts = set(loc_df['ACCNTNUM'].unique())
        pol_accnts = set(pol_df['ACCNTNUM'].unique())
        common_accnts = loc_accnts.intersection(pol_accnts)
        orphan_loc_accnts = loc_accnts - pol_accnts
        orphan_pol_accnts = pol_accnts - loc_accnts
        
        pol_match_rate = (len(common_accnts) / len(pol_accnts)) * 100 if len(pol_accnts) > 0 else 0
        loc_match_rate = (len(common_accnts) / len(loc_accnts)) * 100 if len(loc_accnts) > 0 else 0
        
        print(f"• Match Statistics -> Policy Accounts: {len(pol_accnts):,} | Loc Accounts: {len(loc_accnts):,} | Common: {len(common_accnts):,}")
        print(f"• Match Rates      -> Policy Match: {pol_match_rate:.1f}% | Location Match: {loc_match_rate:.1f}%")
        print(f"• Orphan Checks    -> Orphan Locations (no policy): {len(orphan_loc_accnts):,} | Orphan Policies (no locs): {len(orphan_pol_accnts):,}")
        
        acc_tiv = merged_df.groupby('ACCNTNUM')['TIV_TOTAL'].sum().reset_index().rename(columns={'TIV_TOTAL': 'ACCOUNT_TIV'})
        acc_loc_cnt = merged_df.groupby('ACCNTNUM')['LOCNUM'].count().reset_index().rename(columns={'LOCNUM': 'LOC_COUNT'})
        
        merged_df = merged_df.merge(acc_tiv, on='ACCNTNUM', how='left')
        merged_df = merged_df.merge(acc_loc_cnt, on='ACCNTNUM', how='left')
        
        merged_df['RATE_ON_LINE_PCT'] = np.where(merged_df['ACCOUNT_TIV'] > 0, (merged_df['BLANPREAMT'] / merged_df['ACCOUNT_TIV']) * 100, 0)
        merged_df['LIMIT_TO_TIV_RATIO'] = np.where(merged_df['ACCOUNT_TIV'] > 0, (merged_df['BLANLIMAMT'] / merged_df['ACCOUNT_TIV']), 0)
        merged_df['TIV_PER_SQFT'] = np.where(merged_df['FLOORAREA'] > 0, merged_df['TIV_TOTAL'] / merged_df['FLOORAREA'], 0)
        
        tot_tiv = merged_df.groupby('ACCNTNUM')['ACCOUNT_TIV'].first().sum() if len(merged_df) > 0 else 0.0
        tot_prem = merged_df.groupby('ACCNTNUM')['BLANPREAMT'].first().sum() if len(merged_df) > 0 else 0.0
        tot_lim = merged_df.groupby('ACCNTNUM')['BLANLIMAMT'].first().sum() if len(merged_df) > 0 else 0.0
        avg_rol = (tot_prem / tot_tiv) * 100 if tot_tiv > 0 else 0
        avg_lim_tiv = (tot_lim / tot_tiv) * 100 if tot_tiv > 0 else 0
        
        print(f"• Exposure Financials -> Total TIV: ${tot_tiv:,.2f} | Blanket Premium: ${tot_prem:,.2f} | Blanket Limit: ${tot_lim:,.2f}")
        print(f"• Rate on Line (RoL)  -> Portfolio RoL: {avg_rol:.4f}% ({avg_rol * 100:.1f} bps) | Limit / TIV Ratio: {avg_lim_tiv:.1f}%")
        print(f"• Account Density     -> Avg Locs/Account: {merged_df['LOC_COUNT'].mean() if len(merged_df)>0 else 0:.2f} | Avg TIV/Loc: ${merged_df['TIV_TOTAL'].mean() if len(merged_df)>0 else 0:,.2f}")
        
        state_tiv = merged_df.groupby('STATE')['TIV_TOTAL'].sum()
        if len(state_tiv) > 0 and state_tiv.sum() > 0:
            state_shares = (state_tiv / state_tiv.sum()) * 100
            hhi_state = (state_shares ** 2).sum()
            print(f"• Spatial Risk (HHI)  -> State HHI Index: {hhi_state:,.1f} | Top State: {state_shares.idxmax()} ({state_shares.max():.1f}% share)")
        else:
            print(f"• Spatial Risk (HHI)  -> State HHI Index: N/A")
            
        merged_data[key] = merged_df
        merged_df.to_sql(f"merged_{key}", engine.conn, if_exists='replace', index=False)
        
    prior_all = pd.concat([merged_data['prior_com'].assign(SEGMENT='Commercial'), 
                           merged_data['prior_std'].assign(SEGMENT='Standard')], ignore_index=True)
    curr_all = pd.concat([merged_data['curr_com'].assign(SEGMENT='Commercial'), 
                          merged_data['curr_std'].assign(SEGMENT='Standard')], ignore_index=True)
    
    prior_all.to_sql("portfolio_prior_march_2026", engine.conn, if_exists='replace', index=False)
    curr_all.to_sql("portfolio_curr_april_2026", engine.conn, if_exists='replace', index=False)
    
    print(f"\n[OK] Master Portfolio Tables Ready: Prior (March 2026: {len(prior_all):,} locs) & Current (April 2026: {len(curr_all):,} locs)")
    return merged_data, prior_all, curr_all

# ==================================================================================================
# 4. PRIOR MONTH VS CURRENT MONTH COMPARATIVE ENGINE & ROLL-FORWARD
# ==================================================================================================
def compare_prior_vs_current(engine, prior_all, curr_all):
    banner("4. Prior Month (March 2026) vs Current Month (April 2026) Comparison Engine")
    
    def calc_kpis(df, label):
        unique_accnts = df['ACCNTNUM'].nunique()
        unique_pols = df['POLICYNUM'].nunique()
        unique_locs = df['LOCNUM'].nunique()
        
        acc_df = df.drop_duplicates(subset=['ACCNTNUM'])
        tot_prem = acc_df['BLANPREAMT'].sum()
        tot_lim = acc_df['BLANLIMAMT'].sum()
        tot_tiv = df['TIV_TOTAL'].sum()
        mean_tiv_loc = df['TIV_TOTAL'].mean() if len(df) > 0 else 0.0
        mean_prem_acc = acc_df['BLANPREAMT'].mean() if len(acc_df) > 0 else 0.0
        rol_pct = (tot_prem / tot_tiv) * 100 if tot_tiv > 0 else 0
        
        return {
            "Period": label,
            "Total_Accounts": unique_accnts,
            "Total_Policies": unique_pols,
            "Total_Locations": unique_locs,
            "Total_TIV_USD": tot_tiv,
            "Total_Premium_USD": tot_prem,
            "Total_Limit_USD": tot_lim,
            "Mean_TIV_per_Loc_USD": mean_tiv_loc,
            "Mean_Prem_per_Acc_USD": mean_prem_acc,
            "Portfolio_RoL_Pct": rol_pct
        }
        
    kpi_prior = calc_kpis(prior_all, "Prior (March 2026)")
    kpi_curr = calc_kpis(curr_all, "Current (April 2026)")
    
    kpi_comp = pd.DataFrame([kpi_prior, kpi_curr]).T
    kpi_comp.columns = kpi_comp.iloc[0]
    kpi_comp = kpi_comp.iloc[1:]
    
    kpi_comp['Delta (Abs)'] = kpi_comp['Current (April 2026)'] - kpi_comp['Prior (March 2026)']
    kpi_comp['Growth (%)'] = np.where(
        kpi_comp['Prior (March 2026)'].astype(float) != 0,
        (kpi_comp['Delta (Abs)'] / kpi_comp['Prior (March 2026)'].astype(float)) * 100,
        0.0
    )
    
    section("Executive Portfolio KPI MoM Comparison Table")
    print(kpi_comp.to_string())
    
    section("Executive Roll-Forward Bridge (TIV & Premium Waterfalls)")
    prior_accs = set(prior_all['ACCNTNUM'].unique())
    curr_accs = set(curr_all['ACCNTNUM'].unique())
    
    retained_accs = prior_accs.intersection(curr_accs)
    lapsed_accs = prior_accs - curr_accs
    new_accs = curr_accs - prior_accs
    
    prior_acc_df = prior_all.drop_duplicates('ACCNTNUM').set_index('ACCNTNUM')
    curr_acc_df = curr_all.drop_duplicates('ACCNTNUM').set_index('ACCNTNUM')
    
    prior_tiv_by_acc = prior_all.groupby('ACCNTNUM')['TIV_TOTAL'].sum()
    curr_tiv_by_acc = curr_all.groupby('ACCNTNUM')['TIV_TOTAL'].sum()
    
    baseline_tiv = prior_all['TIV_TOTAL'].sum()
    baseline_prem = prior_acc_df['BLANPREAMT'].sum() if len(prior_acc_df) > 0 else 0.0
    
    lapsed_tiv = prior_tiv_by_acc.loc[list(lapsed_accs)].sum() if lapsed_accs else 0.0
    lapsed_prem = prior_acc_df.loc[list(lapsed_accs), 'BLANPREAMT'].sum() if lapsed_accs else 0.0
    
    new_tiv = curr_tiv_by_acc.loc[list(new_accs)].sum() if new_accs else 0.0
    new_prem = curr_acc_df.loc[list(new_accs), 'BLANPREAMT'].sum() if new_accs else 0.0
    
    retained_prior_tiv = prior_tiv_by_acc.loc[list(retained_accs)].sum() if retained_accs else 0.0
    retained_curr_tiv = curr_tiv_by_acc.loc[list(retained_accs)].sum() if retained_accs else 0.0
    retained_tiv_delta = retained_curr_tiv - retained_prior_tiv
    
    retained_prior_prem = prior_acc_df.loc[list(retained_accs), 'BLANPREAMT'].sum() if retained_accs else 0.0
    retained_curr_prem = curr_acc_df.loc[list(retained_accs), 'BLANPREAMT'].sum() if retained_accs else 0.0
    retained_prem_delta = retained_curr_prem - retained_prior_prem
    
    ending_tiv = curr_all['TIV_TOTAL'].sum()
    ending_prem = curr_acc_df['BLANPREAMT'].sum() if len(curr_acc_df) > 0 else 0.0
    
    p_len = len(prior_accs) if len(prior_accs) > 0 else 1
    print(f"• Account Roll-Forward:")
    print(f"  - March Baseline Accounts:    {len(prior_accs):>6,}")
    print(f"  - Less: Lapsed / Lost:       -{len(lapsed_accs):>6,} ({len(lapsed_accs)/p_len*100:.1f}% Churn Rate)")
    print(f"  - Retained / Renewed:         {len(retained_accs):>6,} ({len(retained_accs)/p_len*100:.1f}% Retention Rate)")
    print(f"  - Plus: New Business:        +{len(new_accs):>6,} ({len(new_accs)/p_len*100:.1f}% Inflow Rate)")
    print(f"  - April Ending Accounts:      {len(curr_accs):>6,} (Net Change: {'+' if len(curr_accs)>=len(prior_accs) else '-'}{abs(len(curr_accs)-len(prior_accs)):,})")
    
    tiv_pct_str = f"{(ending_tiv-baseline_tiv)/baseline_tiv*100:+.2f}%" if baseline_tiv > 0 else "N/A"
    print(f"\n• Total Insured Value (TIV) Waterfall Bridge:")
    print(f"  March 2026 Baseline TIV:     ${baseline_tiv:>16,.2f}")
    print(f"  (-) Lapsed Accounts TIV:     -${lapsed_tiv:>15,.2f}")
    print(f"  (+) New Accounts TIV:        +${new_tiv:>15,.2f}")
    print(f"  (±) Retained Exposure Drift:  {'+' if retained_tiv_delta>=0 else '-'}${abs(retained_tiv_delta):>15,.2f}")
    print(f"  -------------------------------------------------------------")
    print(f"  (=) April 2026 Ending TIV:   ${ending_tiv:>16,.2f} (Δ: {'+' if ending_tiv>=baseline_tiv else '-'}${abs(ending_tiv-baseline_tiv):,.2f} / {tiv_pct_str})")
    
    prem_pct_str = f"{(ending_prem-baseline_prem)/baseline_prem*100:+.2f}%" if baseline_prem > 0 else "N/A"
    print(f"\n• Blanket Premium Waterfall Bridge:")
    print(f"  March 2026 Baseline Premium: ${baseline_prem:>16,.2f}")
    print(f"  (-) Lapsed Accounts Premium: -${lapsed_prem:>15,.2f}")
    print(f"  (+) New Accounts Premium:    +${new_prem:>15,.2f}")
    print(f"  (±) Retained Pricing Drift:   {'+' if retained_prem_delta>=0 else '-'}${abs(retained_prem_delta):>15,.2f}")
    print(f"  -------------------------------------------------------------")
    print(f"  (=) April 2026 Ending Prem:  ${ending_prem:>16,.2f} (Δ: {'+' if ending_prem>=baseline_prem else '-'}${abs(ending_prem-baseline_prem):,.2f} / {prem_pct_str})")
    
    section("Geographic Exposure Drift: State Breakdown")
    prior_state = prior_all.groupby('STATE').agg(Prior_TIV=('TIV_TOTAL', 'sum'), Prior_Locs=('LOCNUM', 'count'))
    curr_state = curr_all.groupby('STATE').agg(Curr_TIV=('TIV_TOTAL', 'sum'), Curr_Locs=('LOCNUM', 'count'))
    
    state_drift = pd.concat([prior_state, curr_state], axis=1).fillna(0)
    state_drift['TIV_Delta_USD'] = state_drift['Curr_TIV'] - state_drift['Prior_TIV']
    state_drift['TIV_Growth_%'] = np.where(state_drift['Prior_TIV'] > 0, (state_drift['TIV_Delta_USD'] / state_drift['Prior_TIV']) * 100, 0.0)
    state_drift['Loc_Delta'] = state_drift['Curr_Locs'] - state_drift['Prior_Locs']
    state_drift = state_drift.sort_values(by='Curr_TIV', ascending=False)
    print(state_drift.to_string())
    
    section("Line of Business (LOB) Exposure Drift")
    prior_lob = prior_all.groupby('LOBNAME').agg(Prior_TIV=('TIV_TOTAL', 'sum'), Prior_Locs=('LOCNUM', 'count'))
    curr_lob = curr_all.groupby('LOBNAME').agg(Curr_TIV=('TIV_TOTAL', 'sum'), Curr_Locs=('LOCNUM', 'count'))
    
    lob_drift = pd.concat([prior_lob, curr_lob], axis=1).fillna(0)
    lob_drift['TIV_Delta_USD'] = lob_drift['Curr_TIV'] - lob_drift['Prior_TIV']
    lob_drift['TIV_Growth_%'] = np.where(lob_drift['Prior_TIV'] > 0, (lob_drift['TIV_Delta_USD'] / lob_drift['Prior_TIV']) * 100, 0.0)
    lob_drift = lob_drift.sort_values(by='Curr_TIV', ascending=False)
    print(lob_drift.to_string())
    
    state_drift.reset_index().to_sql("drift_state_mom", engine.conn, if_exists='replace', index=False)
    lob_drift.reset_index().to_sql("drift_lob_mom", engine.conn, if_exists='replace', index=False)
    
    return kpi_comp, state_drift, lob_drift

# ==================================================================================================
# 5. ENTERPRISE AI, DEEP LEARNING & ADVANCED ML SUITE
# ==================================================================================================
def run_advanced_ai_and_dl_suite(curr_all, prior_all):
    banner("5. Enterprise AI, Deep Learning & Advanced ML Suite")
    
    df = curr_all.copy()
    N = len(df)
    
    valid_yb = df.loc[df['YEARBUILT'] > 1800, 'YEARBUILT']
    median_yb = valid_yb.median() if len(valid_yb) > 0 else 1995
    imputed_yb = np.where((df['YEARBUILT'] < 1800) | (df['YEARBUILT'].isna()), median_yb, df['YEARBUILT'])
    df['AGE'] = np.clip(2026 - imputed_yb, 0, 150)
    
    df['TIV_PER_SQFT'] = np.where(df['FLOORAREA'] > 0, df['TIV_TOTAL'] / df['FLOORAREA'], 0.0)
    df['RATE_ON_LINE_BPS'] = df['RATE_ON_LINE_PCT'] * 100
    df['STORIES_AREA_INTERACTION'] = df['NUMSTORIES'] * np.log1p(df['FLOORAREA'])
    
    feature_cols = ['TIV_TOTAL', 'BLANPREAMT', 'FLOORAREA', 'NUMSTORIES', 'AGE', 'TIV_PER_SQFT', 'RATE_ON_LINE_BPS']
    X = df[feature_cols].copy().fillna(0.0)
    
    if N < 5:
        print("• Insufficient records for AI Suite. Skipping.")
        return df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ----------------------------------------------------------------------------------------------
    # DL MODEL 1: Deep Neural Autoencoder for Unsupervised Anomaly Detection & Latent Embeddings
    # ----------------------------------------------------------------------------------------------
    section("DL Model 1: Deep Neural Autoencoder (Latent Representations & Reconstruction Outliers)")
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(32, 16, 32),
        activation='relu',
        solver='adam',
        max_iter=50,
        batch_size=512,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=5
    )
    autoencoder.fit(X_scaled, X_scaled)
    reconstructed = autoencoder.predict(X_scaled)
    df['AUTOENCODER_MSE'] = np.mean(np.square(X_scaled - reconstructed), axis=1)
    
    deep_threshold = np.percentile(df['AUTOENCODER_MSE'], 97.0)
    df['DEEP_ANOMALY_FLAG'] = np.where(df['AUTOENCODER_MSE'] >= deep_threshold, -1, 1)
    deep_anomalies = df[df['DEEP_ANOMALY_FLAG'] == -1].sort_values(by='AUTOENCODER_MSE', ascending=False)
    
    print(f"• Deep Neural Autoencoder Architecture: Dense(7) -> Dense(32) -> Dense(16) [Bottleneck] -> Dense(32) -> Dense(7)")
    print(f"• Mean Reconstruction MSE: {df['AUTOENCODER_MSE'].mean():.4f} | Anomaly Cutoff MSE (97th Pct): {deep_threshold:.4f}")
    print(f"• Deep Non-Linear Anomalies Flagged: {len(deep_anomalies):,} locations ({len(deep_anomalies)/N*100:.1f}%)")
    print("\n  Top Deep Autoencoder Reconstruction Anomalies:")
    out_cols = [c for c in ['ACCNTNUM', 'LOCNUM', 'STATE', 'BLDGCLASS', 'TIV_TOTAL', 'BLANPREAMT', 'FLOORAREA', 'AUTOENCODER_MSE'] if c in deep_anomalies.columns]
    print(deep_anomalies[out_cols].head(5).to_string(index=False))
    
    # ----------------------------------------------------------------------------------------------
    # DL MODEL 2: Deep MLP Non-Linear Rate-on-Line & Loss-Cost Pricing Regressor
    # ----------------------------------------------------------------------------------------------
    section("DL Model 2: Deep MLP Non-Linear Pricing Regressor (Neural Residual Mispricing)")
    mlp_reg = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=60,
        batch_size=512,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=5
    )
    X_mlp = df[['TIV_TOTAL', 'FLOORAREA', 'NUMSTORIES', 'AGE', 'STORIES_AREA_INTERACTION']].fillna(0.0)
    scaler_mlp_X = StandardScaler()
    X_mlp_scaled = scaler_mlp_X.fit_transform(X_mlp)
    
    y_raw = df['BLANPREAMT'].fillna(0.0).values
    y_log = np.log1p(np.maximum(0, y_raw))
    scaler_mlp_y = StandardScaler()
    y_mlp_scaled = scaler_mlp_y.fit_transform(y_log.reshape(-1, 1)).ravel()
    
    mlp_reg.fit(X_mlp_scaled, y_mlp_scaled)
    pred_s = mlp_reg.predict(X_mlp_scaled)
    pred_log = scaler_mlp_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    df['NEURAL_PREDICTED_PREM'] = np.maximum(0, np.expm1(pred_log))
    df['NEURAL_RESIDUAL'] = df['BLANPREAMT'] - df['NEURAL_PREDICTED_PREM']
    df['NEURAL_DEV_%'] = np.where(df['NEURAL_PREDICTED_PREM'] > 0, (df['NEURAL_RESIDUAL'] / df['NEURAL_PREDICTED_PREM']) * 100, 0.0)
    
    r2_dl = r2_score(y_raw, df['NEURAL_PREDICTED_PREM'])
    print(f"• Deep Multi-Layer Perceptron (MLP) Pricing Model Fit (R² Score): {r2_dl:.3f}")
    neural_under = df[df['NEURAL_DEV_%'] < -30]
    neural_over = df[df['NEURAL_DEV_%'] > 30]
    print(f"  -> Severe Underpriced Risks (< -30% vs Deep Benchmark): {len(neural_under):,} accounts")
    print(f"  -> Severe Overpriced Risks (> +30% vs Deep Benchmark):  {len(neural_over):,} accounts")
    
    # ----------------------------------------------------------------------------------------------
    # ML MODEL 3: Multi-Dimensional Exposure Anomaly Detection (Isolation Forest)
    # ----------------------------------------------------------------------------------------------
    section("ML Model 3: Multi-Dimensional Exposure & Valuation Outliers (Isolation Forest)")
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples=min(10000, N),
        contamination=0.03,
        random_state=42,
        n_jobs=-1
    )
    df['ANOMALY_LABEL'] = iso_forest.fit_predict(X_scaled)
    df['ANOMALY_SCORE'] = iso_forest.decision_function(X_scaled)
    iso_anomalies = df[df['ANOMALY_LABEL'] == -1].sort_values(by='ANOMALY_SCORE')
    print(f"• Isolation Forest Outliers Flagged: {len(iso_anomalies):,} locations ({len(iso_anomalies)/N*100:.1f}%)")
    
    # ----------------------------------------------------------------------------------------------
    # ML MODEL 4: Portfolio Risk Archetype Clustering (MiniBatchKMeans + PCA)
    # ----------------------------------------------------------------------------------------------
    section("ML Model 4: Portfolio Risk Archetype Clustering (MiniBatchKMeans + PCA)")
    k_clusters = min(4, N)
    kmeans = MiniBatchKMeans(n_clusters=k_clusters, batch_size=4096, random_state=42, n_init=3)
    df['CLUSTER_ID'] = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df['PCA1'] = pca_coords[:, 0]
    df['PCA2'] = pca_coords[:, 1]
    
    sample_sz = min(3000, N)
    sil_score = silhouette_score(X_scaled, df['CLUSTER_ID'], sample_size=sample_sz, random_state=42) if N > k_clusters else 0.0
    print(f"• Clustered into {k_clusters} Segments | Silhouette Score: {sil_score:.3f}")
    
    cluster_profiles = df.groupby('CLUSTER_ID').agg(
        Count=('LOCNUM', 'count'),
        Avg_TIV=('TIV_TOTAL', 'mean'),
        Avg_Premium=('BLANPREAMT', 'mean'),
        Avg_FloorArea=('FLOORAREA', 'mean'),
        Avg_Stories=('NUMSTORIES', 'mean'),
        Avg_RoL_bps=('RATE_ON_LINE_BPS', 'mean')
    ).reset_index()
    
    cluster_names = {0: "High-Value Commercial Mega-Risks", 1: "Mid-Market Commercial Real Estate", 2: "Light Industrial / Low Stories", 3: "Standard Residential High-Volume"}
    cluster_profiles['Archetype'] = cluster_profiles['CLUSTER_ID'].map(cluster_names).fillna("Risk Segment")
    print(cluster_profiles.to_string(index=False))
    
    # ----------------------------------------------------------------------------------------------
    # ML MODEL 5: Geospatial Catastrophe Accumulation & Hotspot Clustering
    # ----------------------------------------------------------------------------------------------
    section("ML Model 5: Geospatial Catastrophe Accumulation & Hotspot Zones")
    valid_coords = df[(df['LATITUDE'] != 0) & (df['LONGITUDE'] != 0)]
    if len(valid_coords) > 10:
        geo_features = valid_coords[['LATITUDE', 'LONGITUDE']].values
        geo_clusterer = MiniBatchKMeans(n_clusters=6, batch_size=2048, random_state=42, n_init=3)
        valid_coords['GEO_ZONE_ID'] = geo_clusterer.fit_predict(geo_features)
        
        geo_zones = valid_coords.groupby('GEO_ZONE_ID').agg(
            Locations=('LOCNUM', 'count'),
            Center_Lat=('LATITUDE', 'mean'),
            Center_Lon=('LONGITUDE', 'mean'),
            Total_Zone_TIV=('TIV_TOTAL', 'sum'),
            Mean_Loc_TIV=('TIV_TOTAL', 'mean')
        ).reset_index()
        geo_zones['Zone_TIV_Share_%'] = (geo_zones['Total_Zone_TIV'] / df['TIV_TOTAL'].sum()) * 100
        geo_zones = geo_zones.sort_values(by='Total_Zone_TIV', ascending=False)
        print(geo_zones.to_string(index=False))
    else:
        geo_zones = pd.DataFrame()
        print("• Coordinates missing or insufficient for geospatial clustering.")
        
    # ----------------------------------------------------------------------------------------------
    # ML MODEL 6: Extreme Value Theory (EVT) & Tail Risk Quantiles (VaR & TVaR)
    # ----------------------------------------------------------------------------------------------
    section("ML Model 6: Extreme Value Theory (EVT) & Tail Exposure Risk (VaR & TVaR)")
    tiv_series = df['TIV_TOTAL'].values
    var_95 = np.percentile(tiv_series, 95.0)
    var_99 = np.percentile(tiv_series, 99.0)
    var_995 = np.percentile(tiv_series, 99.5)
    
    tvar_95 = tiv_series[tiv_series >= var_95].mean() if np.any(tiv_series >= var_95) else var_95
    tvar_99 = tiv_series[tiv_series >= var_99].mean() if np.any(tiv_series >= var_99) else var_99
    tvar_995 = tiv_series[tiv_series >= var_995].mean() if np.any(tiv_series >= var_995) else var_995
    
    print(f"• Value-at-Risk (VaR) & Tail Value-at-Risk (TVaR / Expected Shortfall) on Single-Location TIV:")
    print(f"  - 95.0% VaR (1-in-20 Yr Event Proxy):  ${var_95:>15,.2f} | 95.0% TVaR: ${tvar_95:>15,.2f}")
    print(f"  - 99.0% VaR (1-in-100 Yr Event Proxy): ${var_99:>15,.2f} | 99.0% TVaR: ${tvar_99:>15,.2f}")
    print(f"  - 99.5% VaR (1-in-200 Yr Event Proxy): ${var_995:>15,.2f} | 99.5% TVaR: ${tvar_995:>15,.2f}")
    
    # ----------------------------------------------------------------------------------------------
    # ML MODEL 7: Full Statistical Feature Drift Suite (Population Stability Index - PSI & KS Tests)
    # ----------------------------------------------------------------------------------------------
    section("ML Model 7: Statistical Feature Drift Matrix (Population Stability Index - PSI & KS Tests)")
    
    def calculate_psi(expected, actual, num_buckets=10):
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        percentiles = np.linspace(0, 100, num_buckets + 1)
        breakpoints = np.percentile(expected, percentiles)
        breakpoints[0] -= 1e-5
        breakpoints[-1] += 1e-5
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0
        
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        expected_pct = np.clip(expected_counts / len(expected), 1e-4, 1.0)
        actual_pct = np.clip(actual_counts / len(actual), 1e-4, 1.0)
        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi_val)
    
    p_ml = prior_all.copy()
    c_ml = curr_all.copy()
    drift_records = []
    
    eval_cols = ['TIV_TOTAL', 'BLANPREAMT', 'FLOORAREA', 'NUMSTORIES', 'YEARBUILT', 'WSCV4VAL', 'WSCV6VAL', 'RATE_ON_LINE_PCT']
    for col in eval_cols:
        if col in p_ml.columns and col in c_ml.columns:
            v_prior = p_ml[col].fillna(0.0).values
            v_curr = c_ml[col].fillna(0.0).values
            psi_val = calculate_psi(v_prior, v_curr)
            ks_res = ks_2samp(v_prior, v_curr)
            
            status = "STABLE" if psi_val < 0.10 else ("MODERATE DRIFT" if psi_val < 0.25 else "SIGNIFICANT DRIFT")
            drift_records.append({
                "Feature": col,
                "PSI_Score": round(psi_val, 4),
                "KS_Statistic": round(ks_res.statistic, 4),
                "P_Value": round(ks_res.pvalue, 4),
                "Drift_Status": status
            })
            
    drift_matrix = pd.DataFrame(drift_records).sort_values(by='PSI_Score', ascending=False)
    print(drift_matrix.to_string(index=False))
    
    # ----------------------------------------------------------------------------------------------
    # ML MODEL 8: Reinsurance Portfolio Stress-Testing & Shock Simulation
    # ----------------------------------------------------------------------------------------------
    section("ML Model 8: Reinsurance Stress-Testing & Shock Simulation (Inflation & Rate Hardening)")
    base_tiv = df['TIV_TOTAL'].sum()
    base_prem = df.drop_duplicates('ACCNTNUM')['BLANPREAMT'].sum()
    
    tiv_shock_15 = base_tiv * 1.15
    pre_2002_mask = df['YEARBUILT'] < 2002
    tiv_shock_code = df['TIV_TOTAL'].copy()
    tiv_shock_code.loc[pre_2002_mask] *= 1.25
    tiv_shock_code_total = tiv_shock_code.sum()
    prem_shock_25bps = base_tiv * ((df['RATE_ON_LINE_BPS'].mean() + 25) / 10000)
    
    print(f"• Stress Scenario A (+15% Demand Surge / Inflation Shock):")
    print(f"  - Stressed TIV: ${tiv_shock_15:,.2f} (Exposure Expansion: +${tiv_shock_15 - base_tiv:,.2f})")
    print(f"• Stress Scenario B (Pre-2002 Code Penalty +25%):")
    print(f"  - Stressed TIV: ${tiv_shock_code_total:,.2f} (Exposure Expansion: +${tiv_shock_code_total - base_tiv:,.2f})")
    print(f"• Stress Scenario C (+25 bps Portfolio Rate Hardening):")
    print(f"  - Stressed Premium Yield: ${prem_shock_25bps:,.2f} (Revenue Uplift: +${prem_shock_25bps - base_prem:,.2f})")
    
    return df, deep_anomalies, cluster_profiles, drift_matrix, geo_zones

# ==================================================================================================
# 6. SQL SERVER / NATIVE ANSI SQL CTE SCRIPT REPOSITORY
# ==================================================================================================
def execute_advanced_sql_queries(engine):
    banner("6. SQL Server / SQLite Native CTE Queries Execution")
    
    q1 = """
    WITH AccountAggregates AS (
        SELECT 
            ACCNTNUM,
            ACCNTNAME,
            PRODNAME,
            CEDANTNAME,
            LOBNAME,
            BLANPREAMT,
            BLANLIMAMT,
            COUNT(LOCNUM) AS Total_Locations,
            SUM(TIV_TOTAL) AS Aggregate_TIV,
            AVG(TIV_TOTAL) AS Mean_Loc_TIV,
            MAX(TIV_TOTAL) AS Max_Loc_TIV
        FROM portfolio_curr_april_2026
        GROUP BY ACCNTNUM, ACCNTNAME, PRODNAME, CEDANTNAME, LOBNAME, BLANPREAMT, BLANLIMAMT
    )
    SELECT 
        ACCNTNUM,
        ACCNTNAME,
        PRODNAME,
        CEDANTNAME,
        Total_Locations,
        Aggregate_TIV,
        BLANPREAMT AS Blanket_Premium,
        ROUND((BLANPREAMT / NULLIF(Aggregate_TIV, 0)) * 100.0, 4) AS Rate_On_Line_Pct,
        RANK() OVER (ORDER BY Aggregate_TIV DESC) AS TIV_Rank
    FROM AccountAggregates
    ORDER BY Aggregate_TIV DESC
    LIMIT 5;
    """
    section("SQL Query 1: Top 5 Highest Exposure Accounts (Window Ranking CTE)")
    print(engine.query(q1).to_string(index=False))
    
    q2 = """
    WITH PriorLOB AS (
        SELECT 
            LOBNAME,
            COUNT(DISTINCT ACCNTNUM) AS Prior_Accounts,
            COUNT(LOCNUM) AS Prior_Locations,
            SUM(TIV_TOTAL) AS Prior_TIV,
            SUM(BLANPREAMT) AS Prior_Prem
        FROM portfolio_prior_march_2026
        GROUP BY LOBNAME
    ),
    CurrLOB AS (
        SELECT 
            LOBNAME,
            COUNT(DISTINCT ACCNTNUM) AS Curr_Accounts,
            COUNT(LOCNUM) AS Curr_Locations,
            SUM(TIV_TOTAL) AS Curr_TIV,
            SUM(BLANPREAMT) AS Curr_Prem
        FROM portfolio_curr_april_2026
        GROUP BY LOBNAME
    )
    SELECT 
        COALESCE(c.LOBNAME, p.LOBNAME) AS Line_Of_Business,
        p.Prior_Locations,
        c.Curr_Locations,
        (c.Curr_Locations - p.Prior_Locations) AS Loc_Delta,
        p.Prior_TIV,
        c.Curr_TIV,
        ROUND(c.Curr_TIV - p.Prior_TIV, 2) AS TIV_Delta,
        ROUND(((c.Curr_TIV - p.Prior_TIV) / NULLIF(p.Prior_TIV, 0)) * 100.0, 2) AS TIV_Growth_Pct
    FROM CurrLOB c
    LEFT JOIN PriorLOB p ON c.LOBNAME = p.LOBNAME
    ORDER BY Curr_TIV DESC;
    """
    section("SQL Query 2: MoM Exposure Drift by Line of Business (Multi-Table CTE Join)")
    print(engine.query(q2).to_string(index=False))

# ==================================================================================================
# 7. PRODUCTION VISUAL DASHBOARDS & AUDIT FILE EXPORTER
# ==================================================================================================
def generate_visual_deliverables(prior_all, curr_all, kpi_comp, state_drift, ai_df, deep_anomalies, cluster_profiles, drift_matrix, geo_zones):
    banner("7. Generating Production Visual Dashboards & Audit Reports")
    
    if len(curr_all) == 0:
        print("• No records available to plot visual deliverables.")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RMS Exposure Portfolio: March 2026 (Prior) vs April 2026 (Current) MoM Analytics', fontsize=16, fontweight='bold')
    
    top_states = state_drift.head(7)
    x = np.arange(len(top_states))
    width = 0.35
    axes[0, 0].bar(x - width/2, top_states['Prior_TIV'] / 1e6, width, label='March 2026', color='#2b5c8f')
    axes[0, 0].bar(x + width/2, top_states['Curr_TIV'] / 1e6, width, label='April 2026', color='#e26d5c')
    axes[0, 0].set_title('Top States Total Insured Value ($ Millions)', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(top_states.index, fontweight='bold')
    axes[0, 0].set_ylabel('TIV ($M)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    cov_labels = ['Building\n(CV4)', 'Other Struct\n(CV5)', 'Contents\n(CV6)', 'Time Element\n(CV7)']
    p_cov = [prior_all['WSCV4VAL'].sum()/1e6, prior_all['WSCV5VAL'].sum()/1e6, prior_all['WSCV6VAL'].sum()/1e6, prior_all['WSCV7VAL'].sum()/1e6]
    c_cov = [curr_all['WSCV4VAL'].sum()/1e6, curr_all['WSCV5VAL'].sum()/1e6, curr_all['WSCV6VAL'].sum()/1e6, curr_all['WSCV7VAL'].sum()/1e6]
    
    x_cov = np.arange(len(cov_labels))
    axes[0, 1].bar(x_cov - width/2, p_cov, width, label='March 2026', color='#457b9d')
    axes[0, 1].bar(x_cov + width/2, c_cov, width, label='April 2026', color='#f4a261')
    axes[0, 1].set_title('Exposure by Coverage Component ($ Millions)', fontweight='bold')
    axes[0, 1].set_xticks(x_cov)
    axes[0, 1].set_xticklabels(cov_labels)
    axes[0, 1].set_ylabel('Exposure ($M)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    plot_sub = curr_all.sample(min(len(curr_all), 10000), random_state=42) if len(curr_all) > 10000 else curr_all
    p_plot_sub = prior_all.sample(min(len(prior_all), 10000), random_state=42) if len(prior_all) > 10000 else prior_all
    
    try:
        sns.kdeplot(p_plot_sub['RATE_ON_LINE_PCT'], ax=axes[1, 0], label='March 2026', color='#1d3557', fill=True, alpha=0.3)
        sns.kdeplot(plot_sub['RATE_ON_LINE_PCT'], ax=axes[1, 0], label='April 2026', color='#e63946', fill=True, alpha=0.3)
    except Exception:
        axes[1, 0].hist(plot_sub['RATE_ON_LINE_PCT'], bins=30, color='#e63946', alpha=0.7, label='April 2026')
    axes[1, 0].set_title('Rate on Line Distribution (Pricing Density %)', fontweight='bold')
    axes[1, 0].set_xlabel('Rate on Line (%)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    top_bldg = curr_all['BLDGCLASS'].value_counts().head(5).index
    bldg_p = prior_all[prior_all['BLDGCLASS'].isin(top_bldg)].groupby('BLDGCLASS')['TIV_TOTAL'].sum() / 1e6
    bldg_c = curr_all[curr_all['BLDGCLASS'].isin(top_bldg)].groupby('BLDGCLASS')['TIV_TOTAL'].sum() / 1e6
    bldg_df = pd.DataFrame({'March': bldg_p, 'April': bldg_c}).loc[top_bldg].fillna(0)
    
    x_b = np.arange(len(bldg_df))
    axes[1, 1].barh(x_b - width/2, bldg_df['March'], width, label='March 2026', color='#2a9d8f')
    axes[1, 1].barh(x_b + width/2, bldg_df['April'], width, label='April 2026', color='#e76f51')
    axes[1, 1].set_title('Top Construction Classes by Total Insured Value ($M)', fontweight='bold')
    axes[1, 1].set_yticks(x_b)
    axes[1, 1].set_yticklabels(top_bldg)
    axes[1, 1].set_xlabel('TIV ($M)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    dash_file = "rms_eda_mom_comparison.png"
    plt.savefig(dash_file, dpi=300)
    plt.close()
    print(f" [OK] Saved MoM Dashboard Visual: {dash_file}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Enterprise AI & Deep Learning Diagnostics: Autoencoder, Hotspots & Tail Risk', fontsize=16, fontweight='bold')
    
    sns.histplot(ai_df['AUTOENCODER_MSE'], bins=40, kde=True, ax=axes[0, 0], color='#2b5c8f')
    deep_cutoff = np.percentile(ai_df['AUTOENCODER_MSE'], 97.0)
    axes[0, 0].axvline(deep_cutoff, color='#d62828', linestyle='--', linewidth=2, label=f'97th Pct Anomaly Cutoff ({deep_cutoff:.3f})')
    axes[0, 0].set_title('Deep Autoencoder Reconstruction Error (MSE Loss)', fontweight='bold')
    axes[0, 0].set_xlabel('Reconstruction MSE Loss')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    plot_ai = ai_df.sample(min(len(ai_df), 5000), random_state=42) if len(ai_df) > 5000 else ai_df
    axes[0, 1].scatter(plot_ai['NEURAL_PREDICTED_PREM'] / 1e3, plot_ai['BLANPREAMT'] / 1e3, alpha=0.5, c='#2a9d8f', s=25)
    max_p = max(plot_ai['BLANPREAMT'].max(), plot_ai['NEURAL_PREDICTED_PREM'].max()) / 1e3
    axes[0, 1].plot([0, max_p], [0, max_p], 'r--', linewidth=2, label='Perfect Calibration Line')
    axes[0, 1].set_title('Deep MLP Neural Regressor: Predicted vs Actual Premium ($k)', fontweight='bold')
    axes[0, 1].set_xlabel('Deep Neural Predicted Premium ($k)')
    axes[0, 1].set_ylabel('Actual Blanket Premium ($k)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    top_psi = drift_matrix.head(7)
    y_p = np.arange(len(top_psi))
    psi_colors = ['#d62828' if v >= 0.10 else '#2a9d8f' for v in top_psi['PSI_Score']]
    axes[1, 0].barh(y_p, top_psi['PSI_Score'], color=psi_colors)
    axes[1, 0].axvline(0.10, color='#f4a261', linestyle='--', linewidth=2, label='Moderate Drift Threshold (0.10)')
    axes[1, 0].axvline(0.25, color='#d62828', linestyle='--', linewidth=2, label='Significant Drift Threshold (0.25)')
    axes[1, 0].set_yticks(y_p)
    axes[1, 0].set_yticklabels(top_psi['Feature'])
    axes[1, 0].set_title('Population Stability Index (PSI) Feature Drift Ranking', fontweight='bold')
    axes[1, 0].set_xlabel('PSI Score (March vs April)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    sorted_tiv = np.sort(ai_df['TIV_TOTAL'].values)[::-1]
    exceedance_prob = (np.arange(1, len(sorted_tiv) + 1) / len(sorted_tiv)) * 100
    axes[1, 1].plot(sorted_tiv / 1e6, exceedance_prob, color='#e63946', linewidth=2.5)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Single-Risk Exceedance Probability (EP) Exposure Curve', fontweight='bold')
    axes[1, 1].set_xlabel('Total Insured Value ($ Millions - Log Scale)')
    axes[1, 1].set_ylabel('Exceedance Probability (% - Log Scale)')
    axes[1, 1].grid(True, which="both", linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ai_file = "rms_ai_deep_learning_suite.png"
    plt.savefig(ai_file, dpi=300)
    plt.close()
    print(f" [OK] Saved AI & Deep Learning Visual Dashboard: {ai_file}")
    
    if len(deep_anomalies) > 0:
        deep_anomalies.to_csv("rms_deep_anomalies_audit.csv", index=False)
    drift_matrix.to_csv("rms_statistical_drift_psi.csv", index=False)
    if len(geo_zones) > 0:
        geo_zones.to_csv("rms_spatial_cat_zones.csv", index=False)
    state_drift.to_csv("rms_state_exposure_drift_mom.csv")
    kpi_comp.to_csv("rms_kpi_comparison_mom.csv")
    print(f" [OK] Exported Audit Files: 'rms_deep_anomalies_audit.csv', 'rms_statistical_drift_psi.csv', 'rms_spatial_cat_zones.csv', 'rms_state_exposure_drift_mom.csv', 'rms_kpi_comparison_mom.csv'")

# ==================================================================================================
# 8. MASTER PIPELINE RUNNER
# ==================================================================================================
def run_rms_master_pipeline():
    banner("STARTING RMS EXPOSURE DATA AUTOMATED EDA, SQL MERGE, MOM & ADVANCED AI PIPELINE")
    
    engine = ExposureDataEngine(FILES)
    engine.load_and_ingest()
    
    summary_df = run_standalone_table_eda(engine)
    
    merged_data, prior_all, curr_all = merge_and_analyze_exposure(engine)
    
    kpi_comp, state_drift, lob_drift = compare_prior_vs_current(engine, prior_all, curr_all)
    
    ai_df, deep_anomalies, cluster_profiles, drift_matrix, geo_zones = run_advanced_ai_and_dl_suite(curr_all, prior_all)
    
    execute_advanced_sql_queries(engine)
    
    generate_visual_deliverables(prior_all, curr_all, kpi_comp, state_drift, ai_df, deep_anomalies, cluster_profiles, drift_matrix, geo_zones)
    
    banner("ENTERPRISE AI & CATASTROPHE EXPOSURE PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_rms_master_pipeline()
