import pandas as pd
import numpy as np
import io
import time # To time the execution
import numba # For accelerating the simulation loop

# --- Numba Simulation Function for a Chunk (Original SAS Logic) ---
@numba.njit
def run_chunk_simulation_numpy_rng(
    num_rows_in_chunk, K, pcomp_val,
    rng, # Standard NumPy RNG instance
    pa_vals, pb_vals, pc_vals, pd_vals,
    va1_vals, va2_vals, vb1_vals, vb2_vals,
    vc1_vals, vc2_vals, vd1_vals, vd2_vals
    ):
    """
    Runs the simulation for a chunk using standard NumPy RNG:
    - Uses the provided rng instance for all random numbers.
    - Calculates 0/1 flags for *all* runs in the chunk.
    """
    # Output arrays for the chunk
    babc_results = np.zeros(num_rows_in_chunk, dtype=np.int32)
    cabc_results = np.zeros(num_rows_in_chunk, dtype=np.int32)
    bbcd_results = np.zeros(num_rows_in_chunk, dtype=np.int32)
    cbcd_results = np.zeros(num_rows_in_chunk, dtype=np.int32)

    # Loop through rows in the chunk
    for r_idx in range(num_rows_in_chunk):
        # Extract parameters for the row (all rows in chunk have same K, pcomp)
        pa, pb, pc, pd_ = pa_vals[r_idx], pb_vals[r_idx], pc_vals[r_idx], pd_vals[r_idx]
        va1, va2 = va1_vals[r_idx], va2_vals[r_idx]
        vb1, vb2 = vb1_vals[r_idx], vb2_vals[r_idx]
        vc1, vc2 = vc1_vals[r_idx], vc2_vals[r_idx]
        vd1, vd2 = vd1_vals[r_idx], vd2_vals[r_idx]

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
        if sumb > suma and sumb > sumc: babc_flag_pre = 1
        elif sumc > suma and sumc > sumb: cabc_flag_pre = 1

        # Compromise check using the same RNG
        if rng.random() < pcomp_val:
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
        if sumb > sumd and sumb > sumc: bbcd_flag_pre = 1
        elif sumc > sumd and sumc > sumb: cbcd_flag_pre = 1

        # Compromise check using the same RNG
        if rng.random() < pcomp_val: # Compromise chooses C for bcd
            bbcd_results[r_idx] = 0
            cbcd_results[r_idx] = 1
        else:
            bbcd_results[r_idx] = bbcd_flag_pre
            cbcd_results[r_idx] = cbcd_flag_pre

    return babc_results, cabc_results, bbcd_results, cbcd_results


# --- Configuration ---
# Define parameter ranges to search
PCOMP_SEARCH_RANGE = np.round(np.arange(0.23, 0.26, 0.01), 2) # Explicit rounding
KAPA_SEARCH_RANGE = np.arange(7, 10, 1)        # K=1 to 26

# Simulation settings
N_IDS = 400000 # Simulation runs per parameter setting
CHUNK_SIZE = 40000  # Process N_IDS in chunks (adjust based on memory)
SEED = 42 # Seed for the main sampling RNG (matching SAS seed=42)

# --- Initialize RNGs ---
# 1. Standard NumPy RNG (seeded)
if not isinstance(SEED, int) or SEED <= 0:
    raise ValueError("SEED must be a positive integer.")
rng_instance = np.random.default_rng(seed=SEED)

print(f"--- Starting Optimization (Using NumPy RNG) ---")
print(f"Searching pcomp in: {PCOMP_SEARCH_RANGE}")
print(f"Searching kapa in: {KAPA_SEARCH_RANGE}")
print(f"Using N_IDS (sim runs per setting): {N_IDS}")
print(f"Using chunk size: {CHUNK_SIZE}")
print(f"Using FIXED SEED {SEED} for main sampling")

# --- Step 1: Initial Data Creation ---
print("\nStep 1: Creating initial DataFrame 'df_a_initial'...")
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
num_supersets = len(df_a_initial)
print(f"Initial df_a_initial shape: {df_a_initial.shape}")

# --- Step 2: EV Calculation (Done once) ---
print("\nStep 2: Calculating EV variables...")
vvf_cols = ['va1', 'vb1', 'vc1', 'vd1']
vvs_cols = ['va2', 'vb2', 'vc2', 'vd2']
pp_cols = ['pa', 'pb', 'pc', 'pd']
ev_cols = ['eva', 'evb', 'evc', 'evd']
for col_list in [vvf_cols, vvs_cols, pp_cols]:
    df_a_initial[col_list] = df_a_initial[col_list].astype(float)
for i in range(4):
    df_a_initial[ev_cols[i]] = (df_a_initial[vvf_cols[i]] * df_a_initial[pp_cols[i]] +
                              df_a_initial[vvs_cols[i]] * (1 - df_a_initial[pp_cols[i]])).round(2)

# Select only necessary columns for processing
base_cols_to_keep = ['superset'] + vvf_cols + vvs_cols + pp_cols + ['oce']
df_base_data = df_a_initial[base_cols_to_keep].copy()


# --- Step 3: Parameter Search with Chunked Simulation ---
print("\nStep 3: Starting parameter search with chunked processing...")
start_search_time = time.time()

# Results storage
all_results_m = [] # Store final aggregated MSD per parameter set

# Process each parameter combination
param_count = 0
total_params = len(PCOMP_SEARCH_RANGE) * len(KAPA_SEARCH_RANGE)
for pcomp in PCOMP_SEARCH_RANGE:
    for kapa in KAPA_SEARCH_RANGE:
        param_count += 1
        print(f"Processing Param Set {param_count}/{total_params}: pcomp={pcomp:.2f}, kapa={kapa} ...")

        # Store results for this parameter set across all chunks
        chunk_results_list = []

        # Process N_IDS in chunks
        for chunk_start in range(0, N_IDS, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, N_IDS)
            current_chunk_size_sim = chunk_end - chunk_start # Number of simulations in this chunk
            num_rows_in_df_chunk = current_chunk_size_sim * num_supersets # Total rows needed

            # --- Prepare data for the Numba function for this chunk ---
            # Repeat base data for each simulation run in the chunk
            df_chunk = pd.DataFrame(np.repeat(df_base_data.values, current_chunk_size_sim, axis=0),
                                  columns=df_base_data.columns)

            # Prepare NumPy arrays from the chunk's DataFrame columns
            pa_chunk = np.ascontiguousarray(df_chunk['pa'].values)
            pb_chunk = np.ascontiguousarray(df_chunk['pb'].values)
            pc_chunk = np.ascontiguousarray(df_chunk['pc'].values)
            pd_chunk = np.ascontiguousarray(df_chunk['pd'].values)
            va1_chunk = np.ascontiguousarray(df_chunk['va1'].values)
            va2_chunk = np.ascontiguousarray(df_chunk['va2'].values)
            vb1_chunk = np.ascontiguousarray(df_chunk['vb1'].values)
            vb2_chunk = np.ascontiguousarray(df_chunk['vb2'].values)
            vc1_chunk = np.ascontiguousarray(df_chunk['vc1'].values)
            vc2_chunk = np.ascontiguousarray(df_chunk['vc2'].values)
            vd1_chunk = np.ascontiguousarray(df_chunk['vd1'].values)
            vd2_chunk = np.ascontiguousarray(df_chunk['vd2'].values)
            oce_chunk = np.ascontiguousarray(df_chunk['oce'].values) # Needed for MSD calc later

            # --- Run Numba simulation for the chunk ---
            babc_res, cabc_res, bbcd_res, cbcd_res = run_chunk_simulation_numpy_rng(
                num_rows_in_df_chunk, int(kapa), float(pcomp), # Ensure types match Numba expectations
                rng_instance, # The single seeded NumPy RNG instance
                pa_chunk, pb_chunk, pc_chunk, pd_chunk,
                va1_chunk, va2_chunk, vb1_chunk, vb2_chunk,
                vc1_chunk, vc2_chunk, vd1_chunk, vd2_chunk
            )

            # --- Calculate Conditional CE and MSD for the chunk ---
            # Add results back to a temporary structure or directly calculate means
            # To avoid large intermediate dataframes, calculate necessary means directly

            # Calculate sums needed for means
            # Create temporary df to leverage groupby (efficient for sums over N_IDS)
            temp_df = pd.DataFrame({
                'superset': df_chunk['superset'].values,
                'babc': babc_res,
                'cabc': cabc_res,
                'bbcd': bbcd_res,
                'cbcd': cbcd_res,
                'oce': oce_chunk # Carry OCE for MSD calculation
            })

            # Aggregate sums within the chunk (grouped by superset)
            agg_sums = temp_df.groupby('superset').agg(
                sum_babc=('babc', 'sum'),
                sum_cabc=('cabc', 'sum'),
                sum_bbcd=('bbcd', 'sum'),
                sum_cbcd=('cbcd', 'sum'),
                count=('babc', 'size'), # Should be current_chunk_size_sim per superset
                oce = ('oce', 'first') # OCE is constant per superset
            ).reset_index()

            chunk_results_list.append(agg_sums)
            # Optional: Memory cleanup if needed, though agg_sums is small
            del df_chunk, temp_df, babc_res, cabc_res, bbcd_res, cbcd_res
            del pa_chunk, pb_chunk, pc_chunk, pd_chunk, va1_chunk, va2_chunk
            del vb1_chunk, vb2_chunk, vc1_chunk, vc2_chunk, vd1_chunk, vd2_chunk
            del oce_chunk

        # --- Aggregate results across all chunks for the current parameter set ---
        df_param_results = pd.concat(chunk_results_list, ignore_index=True)

        # Sum the sums and counts across chunks for each superset
        final_sums = df_param_results.groupby('superset').agg(
            total_babc = ('sum_babc', 'sum'),
            total_cabc = ('sum_cabc', 'sum'),
            total_bbcd = ('sum_bbcd', 'sum'),
            total_cbcd = ('sum_cbcd', 'sum'),
            total_count = ('count', 'sum'), # Should equal N_IDS per superset
            oce = ('oce', 'first')
        ).reset_index()

        # --- Calculate Final Conditional CE and MSD for this parameter set ---
        # Calculate mean probabilities (equivalent to mean of 0/1 flags over N_IDS)
        final_sums['mean_babc'] = final_sums['total_babc'] / final_sums['total_count']
        final_sums['mean_cabc'] = final_sums['total_cabc'] / final_sums['total_count']
        final_sums['mean_bbcd'] = final_sums['total_bbcd'] / final_sums['total_count']
        final_sums['mean_cbcd'] = final_sums['total_cbcd'] / final_sums['total_count']

        # Calculate Conditional CE (P(B|{B,C}) Definition)
        denom_abc = final_sums['mean_babc'] + final_sums['mean_cabc']
        denom_bcd = final_sums['mean_bbcd'] + final_sums['mean_cbcd']
        pb_cond_abc = np.where(denom_abc == 0, 0.0, final_sums['mean_babc'] / denom_abc) # Assign 0 if denom is 0
        pb_cond_bcd = np.where(denom_bcd == 0, 0.0, final_sums['mean_bbcd'] / denom_bcd) # Assign 0 if denom is 0
        final_sums['ce_conditional'] = pb_cond_abc - pb_cond_bcd

        # Calculate MSD based on this Conditional CE (to match original SAS objective)
        final_sums['msd'] = (final_sums['ce_conditional'] - final_sums['oce'])**2

        # Calculate the overall Mean MSD for this parameter setting
        mean_msd_for_params = final_sums['msd'].mean()

        # Store the result for this parameter combination
        all_results_m.append({'pcomp': pcomp, 'kapa': kapa, 'msd': mean_msd_for_params})

# --- Final Analysis ---
end_search_time = time.time()
print(f"\nParameter search processing took {end_search_time - start_search_time:.2f} seconds.")

print("\n--- Final results ranked by MSD (based on Conditional CE) ---")
df_m_final = pd.DataFrame(all_results_m)

if not df_m_final.empty:
    df_m_final = df_m_final.sort_values(by='msd').reset_index(drop=True)
    print(df_m_final.to_string(index=False, float_format="%.6f")) # More precision for MSD

    # Find and print the best parameters
    best_params = df_m_final.iloc[0] # First row after sorting by MSD
    print("\n--- Best Parameters Found (based on Conditional CE MSD) ---")
    print(f"Minimum MSD achieved: {best_params['msd']:.6f}")
    print(f"Optimal pcomp (alpha): {best_params['pcomp']:.2f}")
    print(f"Optimal kapa (kappa): {int(best_params['kapa'])}")
else:
    print("\nNo results found.")

end_total_time = time.time()
print(f"\nTotal execution time: {end_total_time - start_search_time:.2f} seconds.")
print("Python script finished.")