import numba
import pandas as pd
import numpy as np
import io
import time

# --- Model Parameters ---
ALPHA = 0.23
K = 9
N_IDS = 400000
SEED = 42

# --- Initialize Standard NumPy RNG ---
rng = np.random.default_rng(SEED)

# --- Numba Optimized Simulation Function ---
@numba.njit
def run_simulation_numpy_rng(num_rows, K, rng,
                             pa_vals, pb_vals, pc_vals, pd_vals,
                             va1_vals, va2_vals, vb1_vals, vb2_vals,
                             vc1_vals, vc2_vals, vd1_vals, vd2_vals,
                             pcomp_vals):
    """
    Runs the simulation using standard NumPy RNG passed via rng instance.
    Uses SAS-like logic where rnd == p results in no action.
    """
    babc_results = np.zeros(num_rows, dtype=np.int32)
    cabc_results = np.zeros(num_rows, dtype=np.int32)
    bbcd_results = np.zeros(num_rows, dtype=np.int32)
    cbcd_results = np.zeros(num_rows, dtype=np.int32)

    for r_idx in range(num_rows):
        pa, pb, pc, pd_ = pa_vals[r_idx], pb_vals[r_idx], pc_vals[r_idx], pd_vals[r_idx]
        va1, va2 = va1_vals[r_idx], va2_vals[r_idx]
        vb1, vb2 = vb1_vals[r_idx], vb2_vals[r_idx]
        vc1, vc2 = vc1_vals[r_idx], vc2_vals[r_idx]
        vd1, vd2 = vd1_vals[r_idx], vd2_vals[r_idx]
        pcomp = pcomp_vals[r_idx]

        # --- 'abc' block ---
        babc_flag_pre, cabc_flag_pre = 0, 0
        suma = rng.standard_normal() / 1e8
        sumb = rng.standard_normal() / 1e8
        sumc = rng.standard_normal() / 1e8
        for _ in range(K):
            rnd = rng.random()
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
        if rng.random() < pcomp:
            babc_results[r_idx] = 1
            cabc_results[r_idx] = 0
        else:
            babc_results[r_idx] = babc_flag_pre
            cabc_results[r_idx] = cabc_flag_pre

        # --- 'bcd' block ---
        bbcd_flag_pre, cbcd_flag_pre = 0, 0
        sumb = rng.standard_normal() / 1e8
        sumc = rng.standard_normal() / 1e8
        sumd = rng.standard_normal() / 1e8
        for _ in range(K):
            rnd = rng.random()
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
        if rng.random() < pcomp:
            bbcd_results[r_idx] = 0
            cbcd_results[r_idx] = 1
        else:
            bbcd_results[r_idx] = bbcd_flag_pre
            cbcd_results[r_idx] = cbcd_flag_pre

    return babc_results, cabc_results, bbcd_results, cbcd_results

# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"--- Starting Simulation ---")
    print(f"Alpha: {ALPHA}, K: {K}, Runs: {N_IDS}, Seed: {SEED}")
    start_time = time.time()

    # --- Data Creation ---
    data_string = """
     1 8 .5 1 7 .5 2 6 .5 3 5 .5 4 -.195
     2 6 .55 22 9 .55 18 12 .55 14 10 .45 15 -.074
     3 8 .55 1 7 .55 2 6 .55 3 5 .55 4 -.321
     4 6 .5 22 9 .5 18 12 .5 14 10 .5 15 .009
     5 100 .3 0 75 .4 0 38 .8 0 33 .9 0 .154
     6 150 .2 0 100 .3 0 43 .7 0 33 .9 0 -.018
     7 150 .2 0 100 .3 0 60 .5 0 50 .6 0 .111
     8 33 .9 0 38 .8 0 43 .7 0 50 .6 0 .291
     9 50 .6 0 60 .5 0 75 .4 0 100 .3 0 .326
    """
    cols = ['superset', 'va1', 'pa', 'va2', 'vb1', 'pb', 'vb2', 'vc1', 'pc', 'vc2', 'vd1', 'pd', 'vd2', 'oce']
    df_a_initial = pd.read_csv(io.StringIO(data_string.strip()), sep=r'\s+', header=None, names=cols)

    # --- EV Calculation ---
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

    # --- Data Expansion ---
    df_a = pd.DataFrame(np.repeat(df_a_initial.values, N_IDS, axis=0), columns=df_a_initial.columns)
    df_a['pcomp'] = ALPHA
    df_a['kapa'] = K
    df_a['id'] = np.tile(np.arange(1, N_IDS + 1), len(df_a_initial))
    # Cast columns
    for col_list in [vvf_cols, vvs_cols, pp_cols, ['pcomp', 'kapa', 'oce']]:
         for col in col_list:
            if col in df_a.columns:
                 df_a[col] = df_a[col].astype(float)
    if 'superset' in df_a.columns:
        df_a['superset'] = df_a['superset'].astype(int)

    # --- Prepare data for Numba function ---
    num_rows = len(df_a)
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

    # --- Run the Simulation ---
    babc_res, cabc_res, bbcd_res, cbcd_res = run_simulation_numpy_rng(
        num_rows, K, rng,
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
    df_o['ALPHA'] = ALPHA
    df_o['K'] = K

    # Calculate CONDITIONAL CE
    denom_abc = df_o['babc'] + df_o['cabc']
    denom_bcd = df_o['bbcd'] + df_o['cbcd']
    pb_cond_abc = np.where(denom_abc == 0, np.nan, df_o['babc'] / denom_abc)
    pb_cond_bcd = np.where(denom_bcd == 0, np.nan, df_o['bbcd'] / denom_bcd)
    df_o['ce'] = (pb_cond_abc - pb_cond_bcd).round(3)

    # Calculate MSD
    df_o['oce'] = df_o['oce'].astype(float)
    df_o['msd'] = (df_o['ce'] - df_o['oce'])**2

    # --- Final Results Table ---
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
    # Ensure only columns present in df_o are selected
    final_display_cols = [col for col in display_cols if col in df_o.columns]

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

    results_table = df_o[final_display_cols].copy()
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

    print("\n--- Results ---")
    print(results_table.to_string(index=False))

    # --- Correlation Analysis ---
    valid_corr_data = df_o[['ce', 'oce']].copy()
    valid_corr_data['ce'] = pd.to_numeric(valid_corr_data['ce'], errors='coerce')
    valid_corr_data['oce'] = pd.to_numeric(valid_corr_data['oce'], errors='coerce')
    valid_corr_data.dropna(inplace=True)

    correlation = valid_corr_data['ce'].corr(valid_corr_data['oce'])
    mean_msd = df_o['msd'].mean()

    print(f"\nCorrelation (OCE vs CE): {correlation:.4f}")
    print(f"Mean MSD: {mean_msd:.6f}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")