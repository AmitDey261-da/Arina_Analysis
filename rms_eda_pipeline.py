"""
====================================================================================================
RMS CATASTROPHE EXPOSURE & POLICY ANALYTICS PIPELINE (PYTHON / SQL RELATIONAL ENGINE)
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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# ==================================================================================================
# 0. CONFIGURATION & SCHEMA DEFINITIONS
# ==================================================================================================
DATA_DIR = "."  # <-- Set your folder path here (e.g. r"C:\ExposureData" or ".")

FILES = {
    # Temporary test using April files for both prior and current
    "prior_com_loc": "/Users/amitdey/Downloads/AIIC-raw-data-main/COM RMS_Loc_April_2026.csv",
    "prior_com_pol": "/Users/amitdey/Downloads/AIIC-raw-data-main/COM RMS_Pol_April_2026.csv",
    "prior_std_loc": "/Users/amitdey/Downloads/AIIC-raw-data-main/RMS_Loc_April_2026.csv",
    "prior_std_pol": "/Users/amitdey/Downloads/AIIC-raw-data-main/RMS_Pol_April_2026.csv",
    
    "curr_com_loc":  "/Users/amitdey/Downloads/AIIC-raw-data-main/COM RMS_Loc_April_2026.csv",
    "curr_com_pol":  "/Users/amitdey/Downloads/AIIC-raw-data-main/COM RMS_Pol_April_2026.csv",
    "curr_std_loc":  "/Users/amitdey/Downloads/AIIC-raw-data-main/RMS_Loc_April_2026.csv",
    "curr_std_pol":  "/Users/amitdey/Downloads/AIIC-raw-data-main/RMS_Pol_April_2026.csv",
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

def print_banner(title):
    print("\n" + "=" * 96)
    print(f" {title.upper()}")
    print("=" * 96)

def print_section(title):
    print("\n" + "-" * 80)
    print(f" >>> {title}")
    print("-" * 80)

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
        print_banner("1. Ingestion & Case-Insensitive Schema Normalization into SQL Engine")
        for key, filepath in self.file_dict.items():
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Required file not found: '{filepath}'. Please check the path in FILES dictionary.")
            
            df = pd.read_csv(filepath, low_memory=False, encoding='latin')
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
            print(f" [OK] Table: '{table_name:<15}' | Type: {t_type:<8} | Rows: {len(norm_df):>6,} | Cols: {norm_df.shape[1]:>2} | Path: {filepath}")
            
        print(f"\nAll 8 tables loaded into relational SQL engine. In-memory SQL connectivity active.")
        return self
        
    def query(self, sql):
        return pd.read_sql_query(sql, self.conn)

# ==================================================================================================
# 2. STANDALONE TABLE EDA
# ==================================================================================================
def run_standalone_table_eda(engine):
    print_banner("2. Standalone Exploratory Data Analysis (EDA) for Each Dataset")
    eda_summary = []
    
    for name, df in engine.normalized_dfs.items():
        is_loc = 'LOCNUM' in df.columns
        table_type = "Location Table" if is_loc else "Policy Table"
        print_section(f"Dataset: {name} ({table_type}) - Source: {engine.file_dict[name]}")
        
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
            p99_tiv = df['TIV_TOTAL'].quantile(0.99) if len(df) > 0 else 0.0
            
            cv4 = df['WSCV4VAL'].sum() if 'WSCV4VAL' in df.columns else 0
            cv5 = df['WSCV5VAL'].sum() if 'WSCV5VAL' in df.columns else 0
            cv6 = df['WSCV6VAL'].sum() if 'WSCV6VAL' in df.columns else 0
            cv7 = df['WSCV7VAL'].sum() if 'WSCV7VAL' in df.columns else 0
            
            cv4_pct = (cv4 / tot_tiv * 100) if tot_tiv > 0 else 0
            cv5_pct = (cv5 / tot_tiv * 100) if tot_tiv > 0 else 0
            cv6_pct = (cv6 / tot_tiv * 100) if tot_tiv > 0 else 0
            cv7_pct = (cv7 / tot_tiv * 100) if tot_tiv > 0 else 0
            
            print(f"• Valuation Summary -> Total Insured Value (TIV): ${tot_tiv:,.2f}")
            print(f"  TIV Distribution: Mean = ${mean_tiv:,.2f} | Median = ${median_tiv:,.2f} | 99th Pct = ${p99_tiv:,.2f}")
            print(f"  Coverage Split (WS) -> Bldg (CV4): ${cv4:,.2f} ({cv4_pct:.1f}%), "
                  f"Other Struct (CV5): ${cv5:,.2f} ({cv5_pct:.1f}%), "
                  f"Contents (CV6): ${cv6:,.2f} ({cv6_pct:.1f}%), "
                  f"Time Element (CV7): ${cv7:,.2f} ({cv7_pct:.1f}%)")
                  
            fl_mean = df['FLOORAREA'].dropna().mean() if 'FLOORAREA' in df.columns else np.nan
            st_mean = df['NUMSTORIES'].dropna().mean() if 'NUMSTORIES' in df.columns else np.nan
            
            if 'YEARBUILT' in df.columns:
                valid_yr = df[df['YEARBUILT'] > 1800]['YEARBUILT'].dropna()
                avg_yr_str = f"{int(valid_yr.mean())}" if len(valid_yr) > 0 and not np.isnan(valid_yr.mean()) else "N/A"
            else:
                avg_yr_str = "N/A"
                
            fl_str = f"{fl_mean:,.1f} sqft" if pd.notnull(fl_mean) else "N/A"
            st_str = f"{st_mean:.1f}" if pd.notnull(st_mean) else "N/A"
            print(f"  Physical Profile -> Avg Floor Area: {fl_str} | Avg Stories: {st_str} | Avg Year Built: {avg_yr_str}")
                
            if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
                valid_geo = df[(df['LATITUDE'].between(-90, 90)) & (df['LONGITUDE'].between(-180, 180)) & (df['LATITUDE'] != 0)]
                valid_pct = (len(valid_geo) / len(df) * 100) if len(df) > 0 else 0
                lat_min, lat_max = (df['LATITUDE'].min(), df['LATITUDE'].max()) if len(df) > 0 else (0, 0)
                lon_min, lon_max = (df['LONGITUDE'].min(), df['LONGITUDE'].max()) if len(df) > 0 else (0, 0)
                print(f"  Geocoding -> Valid Coordinates: {len(valid_geo):,}/{len(df):,} ({valid_pct:.1f}%) | Lat: [{lat_min:.2f}, {lat_max:.2f}], Lon: [{lon_min:.2f}, {lon_max:.2f}]")
                
            if 'STATE' in df.columns:
                top_st = {k: int(v) for k, v in df['STATE'].value_counts().head(3).items() if str(k).strip()}
                print(f"  Top States by Location Count: {top_st}")
            if 'BLDGCLASS' in df.columns:
                top_bc = {k: int(v) for k, v in df['BLDGCLASS'].value_counts().head(3).items() if str(k).strip()}
                print(f"  Top Building Classes: {top_bc}")
            if 'ROOFSYS' in df.columns:
                top_rs = {k: int(v) for k, v in df['ROOFSYS'].value_counts().head(3).items() if str(k).strip()}
                print(f"  Top Roof Systems: {top_rs}")
                
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
            mean_lim = df['BLANLIMAMT'].mean() if len(df) > 0 else 0.0
            prem_rate = (tot_prem / tot_lim * 100) if tot_lim > 0 else 0.0
            
            print(f"• Policy Financials -> Total Blanket Premium: ${tot_prem:,.2f} | Total Blanket Limit: ${tot_lim:,.2f}")
            print(f"  Premium Distribution: Mean = ${mean_prem:,.2f} | Median = ${median_prem:,.2f} | Avg Limit = ${mean_lim:,.2f}")
            print(f"  Aggregate Rate on Blanket Limit: {prem_rate:.3f}%")
            
            if 'LOBNAME' in df.columns:
                top_lob = {k: int(v) for k, v in df['LOBNAME'].value_counts().head(3).items() if str(k).strip()}
                print(f"  Top Lines of Business (LOB): {top_lob}")
            if 'CEDANTNAME' in df.columns:
                top_ced = {k: int(v) for k, v in df['CEDANTNAME'].value_counts().head(3).items() if str(k).strip()}
                print(f"  Top Cedants: {top_ced}")
            if 'PRODNAME' in df.columns:
                top_prod = {k: int(v) for k, v in df['PRODNAME'].value_counts().head(3).items() if str(k).strip()}
                print(f"  Top Producers: {top_prod}")
            if 'POLICYTYPE' in df.columns:
                top_pt = {k: int(v) for k, v in df['POLICYTYPE'].value_counts().head(3).items() if str(k).strip()}
                print(f"  Policy Types: {top_pt}")
                
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
    print_banner("3. Relational Merging (Location ⨝ Policy on ACCNTNUM) & Multi-Tier EDA")
    
    pairs = [
        ("prior_com", "prior_com_loc", "prior_com_pol", "Prior Commercial (March 2026)"),
        ("prior_std", "prior_std_loc", "prior_std_pol", "Prior Standard (March 2026)"),
        ("curr_com", "curr_com_loc", "curr_com_pol", "Current Commercial (April 2026)"),
        ("curr_std", "curr_std_loc", "curr_std_pol", "Current Standard (April 2026)"),
    ]
    
    merged_data = {}
    
    for key, loc_key, pol_key, label in pairs:
        print_section(f"Merge Integrity & Exposure Analysis: {label}")
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
        print(f"• Match Rates -> Policy Match: {pol_match_rate:.1f}% | Location Match: {loc_match_rate:.1f}%")
        print(f"• Orphan Checks -> Orphan Locations (no policy): {len(orphan_loc_accnts)} | Orphan Policies (no locs): {len(orphan_pol_accnts)}")
        
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
        
        print(f"• Exposure Financials -> Total TIV: ${tot_tiv:,.2f} | Blanket Premium: ${tot_prem:,.2f} | Blanket Limit: ${tot_lim:,.2f}")
        print(f"• Rate on Line (RoL): {avg_rol:.4f}% ({avg_rol * 100:.1f} bps) | Limit / TIV: {(tot_lim/tot_tiv)*100 if tot_tiv>0 else 0:.1f}%")
        print(f"• Density: Avg Locations / Account: {merged_df['LOC_COUNT'].mean() if len(merged_df)>0 else 0:.2f} | Avg TIV / Loc: ${merged_df['TIV_TOTAL'].mean() if len(merged_df)>0 else 0:,.2f}")
        
        state_tiv = merged_df.groupby('STATE')['TIV_TOTAL'].sum()
        if len(state_tiv) > 0 and state_tiv.sum() > 0:
            state_shares = (state_tiv / state_tiv.sum()) * 100
            hhi_state = (state_shares ** 2).sum()
            print(f"• Geographic Concentration -> State HHI Index: {hhi_state:,.1f} | Top Exposure State: {state_shares.idxmax()} ({state_shares.max():.1f}% share)")
        else:
            print(f"• Geographic Concentration -> State HHI Index: N/A")
            
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
    print_banner("4. Prior Month (March 2026) vs Current Month (April 2026) Comparison Engine")
    
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
    
    print_section("Executive Portfolio KPI MoM Comparison Table")
    print(kpi_comp.to_string())
    
    print_section("Executive Roll-Forward Bridge (TIV & Premium Waterfalls)")
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
    
    print_section("Geographic Exposure Drift: State Breakdown")
    prior_state = prior_all.groupby('STATE').agg(Prior_TIV=('TIV_TOTAL', 'sum'), Prior_Locs=('LOCNUM', 'count'))
    curr_state = curr_all.groupby('STATE').agg(Curr_TIV=('TIV_TOTAL', 'sum'), Curr_Locs=('LOCNUM', 'count'))
    
    state_drift = pd.concat([prior_state, curr_state], axis=1).fillna(0)
    state_drift['TIV_Delta_USD'] = state_drift['Curr_TIV'] - state_drift['Prior_TIV']
    state_drift['TIV_Growth_%'] = np.where(state_drift['Prior_TIV'] > 0, (state_drift['TIV_Delta_USD'] / state_drift['Prior_TIV']) * 100, 0.0)
    state_drift['Loc_Delta'] = state_drift['Curr_Locs'] - state_drift['Prior_Locs']
    state_drift = state_drift.sort_values(by='Curr_TIV', ascending=False)
    print(state_drift.to_string())
    
    print_section("Line of Business (LOB) Exposure Drift")
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
# 5. LIGHTNING-FAST MACHINE LEARNING SUITE
# ==================================================================================================
def run_machine_learning_suite(curr_all, prior_all):
    print_banner("5. Machine Learning Suite (Anomalies, Clustering, Covariate Drift & Valuation)")
    
    df = curr_all.copy()
    N = len(df)
    
    # Safe Feature Engineering
    valid_yb = df.loc[df['YEARBUILT'] > 1800, 'YEARBUILT']
    median_yb = valid_yb.median() if len(valid_yb) > 0 else 1995
    imputed_yb = np.where((df['YEARBUILT'] < 1800) | (df['YEARBUILT'].isna()), median_yb, df['YEARBUILT'])
    df['AGE'] = np.clip(2026 - imputed_yb, 0, 150)
    
    df['TIV_PER_SQFT'] = np.where(df['FLOORAREA'] > 0, df['TIV_TOTAL'] / df['FLOORAREA'], 0.0)
    df['RATE_ON_LINE_BPS'] = df['RATE_ON_LINE_PCT'] * 100
    
    feature_cols = ['TIV_TOTAL', 'BLANPREAMT', 'FLOORAREA', 'NUMSTORIES', 'AGE', 'TIV_PER_SQFT', 'RATE_ON_LINE_BPS']
    X = df[feature_cols].copy().fillna(0.0)
    
    if N < 5:
        print("• Insufficient records for ML Suite. Skipping ML modeling.")
        return df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    # Model 1: Isolation Forest (Parallelized, Scalable)
    print_section("ML Model 1: Exposure & Pricing Anomaly Detection (Isolation Forest)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples=min(10000, N),
        contamination=0.03,
        random_state=42,
        n_jobs=-1
    )
    df['ANOMALY_LABEL'] = iso_forest.fit_predict(X_scaled)
    df['ANOMALY_SCORE'] = iso_forest.decision_function(X_scaled)
    
    anomalies = df[df['ANOMALY_LABEL'] == -1].sort_values(by='ANOMALY_SCORE')
    print(f"• Total Evaluated Records: {N:,}")
    print(f"• Flagged Outliers / High-Risk Anomalies: {len(anomalies)} locations ({len(anomalies)/N*100:.1f}%)")
    
    # Model 2: MiniBatchKMeans + PCA (Sub-Second Execution)
    print_section("ML Model 2: Portfolio Risk Segmentation & Archetype Clustering (K-Means + PCA)")
    k_clusters = min(4, N)
    
    kmeans = MiniBatchKMeans(n_clusters=k_clusters, batch_size=4096, random_state=42, n_init=3)
    df['CLUSTER_ID'] = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df['PCA1'] = pca_coords[:, 0]
    df['PCA2'] = pca_coords[:, 1]
    
    # Fast sub-sampled silhouette evaluation (avoids freezing)
    if N > k_clusters:
        sample_sz = min(3000, N)
        sil_score = silhouette_score(X_scaled, df['CLUSTER_ID'], sample_size=sample_sz, random_state=42)
    else:
        sil_score = 0.0
        
    print(f"• Fast Clustered into {k_clusters} Segments | Sampled Silhouette Score: {sil_score:.3f}")
    print(f"• PCA Explained Variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")
    
    cluster_profiles = df.groupby('CLUSTER_ID').agg(
        Count=('LOCNUM', 'count'),
        Avg_TIV=('TIV_TOTAL', 'mean'),
        Avg_Premium=('BLANPREAMT', 'mean'),
        Avg_FloorArea=('FLOORAREA', 'mean'),
        Avg_Stories=('NUMSTORIES', 'mean'),
        Avg_RoL_bps=('RATE_ON_LINE_BPS', 'mean')
    ).reset_index()
    
    cluster_names = {
        0: "High-Value Commercial Complex",
        1: "Mid-Market Commercial Real Estate",
        2: "Light Industrial / Low Stories",
        3: "Standard Residential High-Volume"
    }
    cluster_profiles['Archetype'] = cluster_profiles['CLUSTER_ID'].map(cluster_names).fillna("Risk Segment")
    print("\n  Portfolio Cluster Archetypes:")
    print(cluster_profiles.to_string(index=False))
    
    # Model 3: Drift Classifier
    print_section("ML Model 3: Month-over-Month Covariate Portfolio Drift Classifier (Random Forest)")
    p_ml = prior_all.copy().assign(PERIOD_LABEL=0)
    c_ml = curr_all.copy().assign(PERIOD_LABEL=1)
    
    for sub_df in [p_ml, c_ml]:
        sub_valid_yb = sub_df.loc[sub_df['YEARBUILT'] > 1800, 'YEARBUILT']
        sub_med_yb = sub_valid_yb.median() if len(sub_valid_yb) > 0 else 1995
        sub_imp_yb = np.where((sub_df['YEARBUILT'] < 1800) | (sub_df['YEARBUILT'].isna()), sub_med_yb, sub_df['YEARBUILT'])
        sub_df['AGE'] = np.clip(2026 - sub_imp_yb, 0, 150)
        sub_df['TIV_PER_SQFT'] = np.where(sub_df['FLOORAREA'] > 0, sub_df['TIV_TOTAL'] / sub_df['FLOORAREA'], 0.0)
        sub_df['RATE_ON_LINE_BPS'] = sub_df['RATE_ON_LINE_PCT'] * 100
    
    drift_df = pd.concat([p_ml, c_ml], ignore_index=True)
    X_drift = drift_df[feature_cols].fillna(0.0)
    y_drift = drift_df['PERIOD_LABEL']
    
    rf_drift = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        max_samples=min(10000, len(X_drift)),
        random_state=42,
        n_jobs=-1
    )
    rf_drift.fit(X_drift, y_drift)
    
    drift_importances = pd.DataFrame({
        'Feature': feature_cols,
        'Drift_Importance_%': rf_drift.feature_importances_ * 100
    }).sort_values(by='Drift_Importance_%', ascending=False)
    
    print("• Covariate Drift Feature Importance (March vs April):")
    print(drift_importances.to_string(index=False))
    
    # Model 4: Regressor
    print_section("ML Model 4: Premium Benchmark Regressor & Residual Mispricing Analysis")
    X_reg = df[['TIV_TOTAL', 'FLOORAREA', 'NUMSTORIES', 'AGE']].fillna(0.0)
    y_reg = df['BLANPREAMT'].fillna(0.0)
    
    rf_reg = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        max_samples=min(10000, N),
        random_state=42,
        n_jobs=-1
    )
    rf_reg.fit(X_reg, y_reg)
    
    df['BENCHMARK_PREMIUM'] = rf_reg.predict(X_reg)
    df['PREMIUM_RESIDUAL'] = df['BLANPREAMT'] - df['BENCHMARK_PREMIUM']
    df['PRICING_DEVIATION_%'] = np.where(df['BENCHMARK_PREMIUM'] > 0, (df['PREMIUM_RESIDUAL'] / df['BENCHMARK_PREMIUM']) * 100, 0.0)
    
    r2 = rf_reg.score(X_reg, y_reg)
    print(f"• Benchmark Regressor Model Fit (R² Score): {r2:.3f}")
    
    return df, anomalies, cluster_profiles, drift_importances

# ==================================================================================================
# 6. SQL SERVER / NATIVE ANSI SQL CTE SCRIPT REPOSITORY
# ==================================================================================================
def execute_advanced_sql_queries(engine):
    print_banner("6. SQL Server / SQLite Native CTE Queries Execution")
    
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
    print_section("SQL Query 1: Top 5 Highest Exposure Accounts (Window Ranking CTE)")
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
    print_section("SQL Query 2: MoM Exposure Drift by Line of Business (Multi-Table CTE Join)")
    print(engine.query(q2).to_string(index=False))

# ==================================================================================================
# 7. PRODUCTION VISUAL DASHBOARD & AUDIT FILE EXPORTER
# ==================================================================================================
def generate_visual_deliverables(prior_all, curr_all, kpi_comp, state_drift, ml_df, anomalies, cluster_profiles):
    print_banner("7. Generating Production Visual Dashboards & Audit Reports")
    
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
    
    try:
        sns.kdeplot(prior_all['RATE_ON_LINE_PCT'], ax=axes[1, 0], label='March 2026', color='#1d3557', fill=True, alpha=0.3)
        sns.kdeplot(curr_all['RATE_ON_LINE_PCT'], ax=axes[1, 0], label='April 2026', color='#e63946', fill=True, alpha=0.3)
    except Exception:
        axes[1, 0].hist(curr_all['RATE_ON_LINE_PCT'], bins=30, color='#e63946', alpha=0.7, label='April 2026')
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
    
    if 'ANOMALY_LABEL' in ml_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Machine Learning Portfolio Diagnostics: Anomaly Flags & Risk Clusters', fontsize=16, fontweight='bold')
        
        norm_recs = ml_df[ml_df['ANOMALY_LABEL'] == 1]
        anom_recs = ml_df[ml_df['ANOMALY_LABEL'] == -1]
        
        axes[0].scatter(norm_recs['FLOORAREA'] / 1e3, norm_recs['TIV_TOTAL'] / 1e6, c='#457b9d', alpha=0.6, s=30, label='Normal Policies')
        axes[0].scatter(anom_recs['FLOORAREA'] / 1e3, anom_recs['TIV_TOTAL'] / 1e6, c='#d62828', alpha=0.9, s=90, marker='X', label=f'Flagged Anomalies (n={len(anom_recs)})')
        axes[0].set_title('ML Anomaly Detection: Floor Area vs TIV', fontweight='bold')
        axes[0].set_xlabel('Floor Area (Thousand Sq Ft)')
        axes[0].set_ylabel('Total Insured Value ($ Millions)')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)
        
        cluster_colors = ['#264653', '#2a9d8f', '#e76f51', '#f4a261']
        for cid in range(min(4, len(cluster_profiles))):
            c_sub = ml_df[ml_df['CLUSTER_ID'] == cid]
            archetype_name = cluster_profiles.loc[cid, "Archetype"] if cid in cluster_profiles.index else f"Cluster {cid}"
            axes[1].scatter(c_sub['PCA1'], c_sub['PCA2'], s=35, alpha=0.7, color=cluster_colors[cid % len(cluster_colors)], label=f'Cluster {cid}: {archetype_name[:20]}...')
        axes[1].set_title('PCA 2D Projection of Exposure Risk Clusters', fontweight='bold')
        axes[1].set_xlabel('Principal Component 1')
        axes[1].set_ylabel('Principal Component 2')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        ml_file = "rms_ml_diagnostics.png"
        plt.savefig(ml_file, dpi=300)
        plt.close()
        print(f" [OK] Saved ML Diagnostics Visual: {ml_file}")
        
    if len(anomalies) > 0:
        anomalies.to_csv("rms_flagged_anomalies_audit.csv", index=False)
    state_drift.to_csv("rms_state_exposure_drift_mom.csv")
    kpi_comp.to_csv("rms_kpi_comparison_mom.csv")
    print(f" [OK] Exported Audit CSV Files: 'rms_flagged_anomalies_audit.csv', 'rms_state_exposure_drift_mom.csv', 'rms_kpi_comparison_mom.csv'")

# ==================================================================================================
# 8. MASTER AUTOMATION RUNNER
# ==================================================================================================
def run_rms_master_pipeline():
    print_banner("STARTING RMS EXPOSURE DATA AUTOMATED EDA, SQL MERGE, MOM & ML PIPELINE")
    
    engine = ExposureDataEngine(FILES)
    engine.load_and_ingest()
    
    summary_df = run_standalone_table_eda(engine)
    
    merged_data, prior_all, curr_all = merge_and_analyze_exposure(engine)
    
    kpi_comp, state_drift, lob_drift = compare_prior_vs_current(engine, prior_all, curr_all)
    
    ml_df, anomalies, cluster_profiles, drift_importances = run_machine_learning_suite(curr_all, prior_all)
    
    execute_advanced_sql_queries(engine)
    
    generate_visual_deliverables(prior_all, curr_all, kpi_comp, state_drift, ml_df, anomalies, cluster_profiles)
    
    print_banner("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_rms_master_pipeline()