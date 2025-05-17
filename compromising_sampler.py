import numba
import pandas as pd
import numpy as np
import io
import time
import argparse # Added for command-line arguments
from scipy.optimize import minimize

# --- Model Parameters ---
INITIAL_ALPHA = 0.23 # Initial guess for optimizer
INITIAL_K = 9      # Initial guess for optimizer
N_IDS = 200000
SEED = 42

# --- Initialize RNGs ---
# Seed the legacy RNG for Numba compatibility
np.random.seed(SEED)
# Initialize the modern RNG if needed elsewhere (currently not used after modification)
rng = np.random.default_rng(SEED)

# --- Numba Optimized Simulation Function ---
@numba.njit
def run_simulation_numba(num_rows, K,
                         pa_vals, pb_vals, pc_vals, pd_vals,
                         va1_vals, va2_vals, vb1_vals, vb2_vals,
                         vc1_vals, vc2_vals, vd1_vals, vd2_vals,
                         pcomp_vals):
    """
    Runs the simulation using Numba-compatible legacy np.random.*
    Uses SAS-like logic where rnd == p results in no action.
    K must be an integer.
    """
    # Numba uses its own internal state for np.random functions,
    # seeded by np.random.seed() called outside the function.

    babc_results = np.zeros(num_rows, dtype=np.int32)
    cabc_results = np.zeros(num_rows, dtype=np.int32)
    bbcd_results = np.zeros(num_rows, dtype=np.int32)
    cbcd_results = np.zeros(num_rows, dtype=np.int32)

    # Ensure K is integer for range loop
    K_int = int(round(K))
    if K_int < 1: K_int = 1 # Ensure K is at least 1

    for r_idx in range(num_rows):
        pa, pb, pc, pd_ = pa_vals[r_idx], pb_vals[r_idx], pc_vals[r_idx], pd_vals[r_idx]
        va1, va2 = va1_vals[r_idx], va2_vals[r_idx]
        vb1, vb2 = vb1_vals[r_idx], vb2_vals[r_idx]
        vc1, vc2 = vc1_vals[r_idx], vc2_vals[r_idx]
        vd1, vd2 = vd1_vals[r_idx], vd2_vals[r_idx]
        pcomp = pcomp_vals[r_idx] # alpha passed via pcomp_vals

        # --- 'abc' block ---
        babc_flag_pre, cabc_flag_pre = 0, 0
        # Use Numba-compatible legacy random functions
        suma = np.random.standard_normal() * 1e-8
        sumb = np.random.standard_normal() * 1e-8
        sumc = np.random.standard_normal() * 1e-8
        for _ in range(K_int): # Use integer K
            rnd = np.random.random()
            if rnd < pa: suma += va1
            elif rnd > pa: suma += va2
            if rnd < pb: sumb += vb1
            elif rnd > pb: sumb += vb2
            if rnd < pc: sumc += vc1
            elif rnd > pc: sumc += vc2
        # Determine pre-compromise winner
        if sumb > suma and sumb > sumc: babc_flag_pre = 1
        elif sumc > suma and sumc > sumb: cabc_flag_pre = 1
        # Compromise check
        if np.random.random() < pcomp:
            babc_results[r_idx] = 1
            cabc_results[r_idx] = 0
        else:
            babc_results[r_idx] = babc_flag_pre
            cabc_results[r_idx] = cabc_flag_pre

        # --- 'bcd' block ---
        bbcd_flag_pre, cbcd_flag_pre = 0, 0
        sumb = np.random.standard_normal() * 1e-8
        sumc = np.random.standard_normal() * 1e-8
        sumd = np.random.standard_normal() * 1e-8
        for _ in range(K_int): # Use integer K
            rnd = np.random.random()
            if rnd < pb: sumb += vb1
            elif rnd > pb: sumb += vb2
            if rnd < pc: sumc += vc1
            elif rnd > pc: sumc += vc2
            if rnd < pd_: sumd += vd1
            elif rnd > pd_: sumd += vd2
        # Determine pre-compromise winner
        if sumb > sumd and sumb > sumc: bbcd_flag_pre = 1
        elif sumc > sumd and sumc > sumb: cbcd_flag_pre = 1
        # Compromise check
        if np.random.random() < pcomp:
            bbcd_results[r_idx] = 0
            cbcd_results[r_idx] = 1
        else:
            bbcd_results[r_idx] = bbcd_flag_pre
            cbcd_results[r_idx] = cbcd_flag_pre

    return babc_results, cabc_results, bbcd_results, cbcd_results

# --- Simulation and Calculation Function ---
def run_full_simulation(alpha, k, df_a_initial, n_ids):
    """
    Runs the full simulation pipeline for a given alpha and k.
    Returns the aggregated results DataFrame (df_o) and the mean MSD.
    """
    # Ensure k is treated as integer internally
    k_int = int(round(k))
    if k_int < 1: k_int = 1

    # --- Data Expansion ---
    df_a = pd.DataFrame(np.repeat(df_a_initial.values, n_ids, axis=0), columns=df_a_initial.columns)
    df_a['pcomp'] = alpha # Use passed alpha
    df_a['kapa'] = k_int   # Use passed (rounded) k
    df_a['id'] = np.tile(np.arange(1, n_ids + 1), len(df_a_initial))

    # Cast columns (ensure consistency)
    vvf_cols = ['va1', 'vb1', 'vc1', 'vd1']
    vvs_cols = ['va2', 'vb2', 'vc2', 'vd2']
    pp_cols = ['pa', 'pb', 'pc', 'pd']
    numeric_cols = vvf_cols + vvs_cols + pp_cols + ['pcomp', 'kapa', 'oce']
    for col in numeric_cols:
        if col in df_a.columns:
            df_a[col] = pd.to_numeric(df_a[col], errors='coerce') # Robust casting
    if 'superset' in df_a.columns:
        df_a['superset'] = df_a['superset'].astype(int)
    # Drop rows with NaN in essential columns if necessary, or handle imputation
    # df_a.dropna(subset=numeric_cols, inplace=True) # Optional: if NaNs cause issues

    # --- Prepare data for Numba function ---
    num_rows = len(df_a)
    if num_rows == 0: # Handle cases where df_a might become empty
        return pd.DataFrame(), np.inf # Return empty DF and infinite MSD

    pa_vals = np.ascontiguousarray(df_a['pa'].values)
    pb_vals = np.ascontiguousarray(df_a['pb'].values)
    pc_vals = np.ascontiguousarray(df_a['pc'].values)
    pd_vals = np.ascontiguousarray(df_a['pd'].values)
    va1_vals = np.ascontiguousarray(df_a['va1'].values)
    va2_vals = np.ascontiguousarray(df_a['va2'].values)
    vb1_vals = np.ascontiguousarray(df_a['vb1'].values)
    vb2_vals = np.ascontiguousarray(df_a['vb2'].values)
    vc1_vals = np.ascontiguousarray(df_a['vc1'].values)
    vc2_vals = np.ascontiguousarray(df_a['vc2'].values)
    vd1_vals = np.ascontiguousarray(df_a['vd1'].values)
    vd2_vals = np.ascontiguousarray(df_a['vd2'].values)
    pcomp_vals = np.ascontiguousarray(df_a['pcomp'].values)

    # --- Run the Numba Simulation ---
    # Numba function now uses internal np.random state
    babc_res, cabc_res, bbcd_res, cbcd_res = run_simulation_numba(
        num_rows, k_int, # K is passed, rng is not
        pa_vals, pb_vals, pc_vals, pd_vals,
        va1_vals, va2_vals, vb1_vals, vb2_vals,
        vc1_vals, vc2_vals, vd1_vals, vd2_vals,
        pcomp_vals
    )

    # Add results back to DataFrame
    df_a['babc'] = babc_res
    df_a['cabc'] = cabc_res
    df_a['bbcd'] = bbcd_res
    df_a['cbcd'] = cbcd_res

    # --- Aggregation and Results ---
    by_vars = ['superset', 'va1', 'pa', 'va2', 'vb1', 'pb', 'vb2', 'vc1', 'pc', 'vc2', 'vd1', 'pd', 'vd2', 'eva', 'evb', 'evc', 'evd', 'oce']
    valid_by_vars = [var for var in by_vars if var in df_a.columns]

    # Aggregate results
    df_o = df_a.groupby(valid_by_vars, observed=True, sort=False).agg(
        babc=('babc', 'mean'),
        cabc=('cabc', 'mean'),
        bbcd=('bbcd', 'mean'),
        cbcd=('cbcd', 'mean')
    ).reset_index()

    # Add parameters to the output dataframe
    df_o['ALPHA'] = alpha
    df_o['K'] = k_int # Store the integer K used

    # Calculate CONDITIONAL CE
    denom_abc = df_o['babc'] + df_o['cabc']
    denom_bcd = df_o['bbcd'] + df_o['cbcd']
    # Avoid division by zero, result is NaN if denominator is zero
    pb_cond_abc = np.where(denom_abc == 0, np.nan, df_o['babc'] / denom_abc)
    pb_cond_bcd = np.where(denom_bcd == 0, np.nan, df_o['bbcd'] / denom_bcd)
    df_o['ce'] = (pb_cond_abc - pb_cond_bcd).round(3)

    # Calculate MSD
    df_o['oce'] = df_o['oce'].astype(float)
    df_o['msd'] = (df_o['ce'] - df_o['oce'])**2

    mean_msd = df_o['msd'].mean()

    # Handle potential NaN in mean_msd if all 'ce' or 'oce' were NaN
    if pd.isna(mean_msd):
        mean_msd = np.inf # Return infinity if MSD cannot be calculated

    return df_o, mean_msd

# --- Objective Function for Optimizer ---
def objective_function(params, df_a_initial, n_ids):
    """
    Objective function for scipy.optimize.minimize.
    Takes parameters [alpha, k] and returns the mean MSD.
    """
    alpha, k = params
    # Run simulation and get MSD (rng_instance removed)
    _, mean_msd = run_full_simulation(alpha, k, df_a_initial, n_ids)
    # Optional: Print progress
    # k_int = int(round(k)) if k >=1 else 1
    # print(f"Trying Alpha={alpha:.4f}, K={k_int}, MSD={mean_msd:.6f}")
    return mean_msd

def grid_search(df_a_initial, n_ids):
    print("--- Starting Grid Search ---")
    # Define the ranges and step sizes
    alpha_range = np.arange(0.13, 0.28, 0.01)  # Step size of 0.01
    k_range = np.arange(3, 15, 1)              # Step size of 1
    
    best_msd = float('inf')
    best_params = None
    results = []
    
    total_combinations = len(alpha_range) * len(k_range)
    current = 0
    
    for alpha in alpha_range:
        for k in k_range:
            current += 1
            _, mean_msd = run_full_simulation(alpha, k, df_a_initial, n_ids)
            results.append((alpha, k, mean_msd))
            
            if mean_msd < best_msd:
                best_msd = mean_msd
                best_params = (alpha, k)
            
            print(f"Progress: {current}/{total_combinations} - Alpha={alpha:.2f}, K={k}, MSD={mean_msd:.6f}")
    
    # Sort results by MSD
    results.sort(key=lambda x: x[2])
    
    print("\n--- Grid Search Results ---")
    print("Top 5 combinations:")
    for alpha, k, msd in results[:5]:
        print(f"Alpha={alpha:.2f}, K={k}, MSD={msd:.6f}")
    
    return best_params, best_msd

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run compromising sampler simulation with optional parameter search.")
    parser.add_argument(
        '--mode', type=str, default='search',
        choices=['search', 'execute'],
        help="'search': optimize ALPHA and K (default). 'execute': run with INITIAL_ALPHA and INITIAL_K."
    )
    args = parser.parse_args()

    start_time = time.time()
    print("--- Preparing Initial Data ---")

    # --- Data Creation (Only Once) ---
    data_string = """
     1 8 .5 1 7 .5 2 6 .5 3 5 .5 4 -.270
     2 6 .55 22 9 .55 18 12 .55 14 10 .45 15 -.142
     3 8 .55 1 7 .55 2 6 .55 3 5 .55 4 -.249
     4 6 .5 22 9 .5 18 12 .5 14 10 .5 15 .057
     5 100 .3 0 75 .4 0 38 .8 0 33 .9 0 .154
     6 150 .2 0 100 .3 0 43 .7 0 33 .9 0 -.018
     7 150 .2 0 100 .3 0 60 .5 0 50 .6 0 .111
     8 33 .9 0 38 .8 0 43 .7 0 50 .6 0 .291
     9 50 .6 0 60 .5 0 75 .4 0 100 .3 0 .326
    """
    cols = ['superset', 'va1', 'pa', 'va2', 'vb1', 'pb', 'vb2', 'vc1', 'pc', 'vc2', 'vd1', 'pd', 'vd2', 'oce']
    df_a_initial = pd.read_csv(io.StringIO(data_string.strip()), sep=r'\s+', header=None, names=cols)

    # --- EV Calculation (Only Once) ---
    vvf_cols = ['va1', 'vb1', 'vc1', 'vd1']
    vvs_cols = ['va2', 'vb2', 'vc2', 'vd2']
    pp_cols = ['pa', 'pb', 'pc', 'pd']
    ev_cols = ['eva', 'evb', 'evc', 'evd']
    for col_list in [vvf_cols, vvs_cols, pp_cols]:
        if all(col in df_a_initial.columns for col in col_list):
            df_a_initial[col_list] = df_a_initial[col_list].astype(float)
    for i in range(4):
        if all(col in df_a_initial.columns for col in [vvf_cols[i], pp_cols[i], vvs_cols[i]]):
            df_a_initial[ev_cols[i]] = (df_a_initial[vvf_cols[i]] * df_a_initial[pp_cols[i]] +
                                      df_a_initial[vvs_cols[i]] * (1 - df_a_initial[pp_cols[i]])).round(2)

    # --- Parameter Determination (Based on Mode) ---
    if args.mode == 'search':
        print("--- Starting Grid Search ---")
        best_params, min_msd = grid_search(df_a_initial, N_IDS)
        optimal_alpha, optimal_k = best_params
        print("\n--- Grid Search Complete ---")
        print(f"Best ALPHA: {optimal_alpha:.4f}")
        print(f"Best K: {optimal_k}")
        print(f"Minimum Mean MSD: {min_msd:.6f}")

    elif args.mode == 'execute':
        print("--- Running in Execute Mode (Using Initial Parameters) ---")
        optimal_alpha = INITIAL_ALPHA
        optimal_k = INITIAL_K
        print(f"Using ALPHA: {optimal_alpha:.4f}")
        print(f"Using K: {optimal_k}")

    print("\n--- Running Final Simulation ---")
    # Run the simulation one last time with the determined parameters
    final_df_o, final_mean_msd = run_full_simulation(
        optimal_alpha, optimal_k, df_a_initial, N_IDS
    )

    # --- Final Results Table ---
    if not final_df_o.empty:
        # Define all columns to display in the desired order
        display_cols = [
            'superset', 'ALPHA', 'K',
            'va1', 'pa', 'va2', # Option A
            'vb1', 'pb', 'vb2', # Option B
            'vc1', 'pc', 'vc2', # Option C
            'vd1', 'pd', 'vd2', # Option D
            'eva', 'evb', 'evc', 'evd', # EVs
            'babc', 'cabc', 'bbcd', 'cbcd', # Choice Rates
            'ce', 'oce', 'msd' # CE and MSD
        ]
        final_display_cols = [col for col in display_cols if col in final_df_o.columns]

        # Define renaming map for clarity
        display_cols_map = {
            'superset': 'Superset', 'ALPHA': 'Alpha', 'K': 'K',
            'va1': 'V(A1)', 'pa': 'P(A)', 'va2': 'V(A2)',
            'vb1': 'V(B1)', 'pb': 'P(B)', 'vb2': 'V(B2)',
            'vc1': 'V(C1)', 'pc': 'P(C)', 'vc2': 'V(C2)',
            'vd1': 'V(D1)', 'pd': 'P(D)', 'vd2': 'V(D2)',
            'eva': 'EV(A)', 'evb': 'EV(B)', 'evc': 'EV(C)', 'evd': 'EV(D)',
            'babc': 'P(B|ABC)', 'cabc': 'P(C|ABC)', 'bbcd': 'P(B|BCD)', 'cbcd': 'P(C|BCD)',
            'ce': 'CE_Pred', 'oce': 'CE_Obs', 'msd': 'MSD'
        }

        results_table = final_df_o[final_display_cols].copy()
        results_table.rename(columns=display_cols_map, inplace=True)

        # Ensure correct types and sort
        if 'Superset' in results_table.columns:
            results_table['Superset'] = results_table['Superset'].astype(int)
            results_table = results_table.sort_values('Superset')

        # Define columns for formatting (original keys)
        format_2dp_keys = ['pa', 'pb', 'pc', 'pd', 'eva', 'evb', 'evc', 'evd']
        format_3dp_keys = ['babc', 'cabc', 'bbcd', 'cbcd', 'ce', 'oce', 'msd']

        # Map original keys to renamed keys in results_table
        format_2dp_cols = [display_cols_map[key] for key in format_2dp_keys if key in display_cols_map and display_cols_map[key] in results_table.columns]
        format_3dp_cols = [display_cols_map[key] for key in format_3dp_keys if key in display_cols_map and display_cols_map[key] in results_table.columns]

        # Apply formatting
        for col in format_2dp_cols:
            results_table[col] = results_table[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NaN")
        for col in format_3dp_cols:
            results_table[col] = results_table[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")

        print("\n--- Final Results (Optimal Parameters) ---")
        print(f"Using Alpha={optimal_alpha:.4f}, K={optimal_k}")
        print(results_table.to_string(index=False))

        # --- Final Correlation Analysis ---
        valid_corr_data = final_df_o[['ce', 'oce']].copy()
        valid_corr_data['ce'] = pd.to_numeric(valid_corr_data['ce'], errors='coerce')
        valid_corr_data['oce'] = pd.to_numeric(valid_corr_data['oce'], errors='coerce')
        valid_corr_data.dropna(inplace=True)

        if not valid_corr_data.empty and len(valid_corr_data) > 1:
             correlation = valid_corr_data['ce'].corr(valid_corr_data['oce'])
             print(f"\nCorrelation (OCE vs CE_Pred): {correlation:.4f}")
        else:
             print("\nCorrelation could not be computed (insufficient data).")

        print(f"Mean MSD: {final_mean_msd:.6f}") # Use the final calculated MSD
    else:
        print("\n--- Final simulation failed or produced no results ---")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")