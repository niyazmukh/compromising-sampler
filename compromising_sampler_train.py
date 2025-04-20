import pandas as pd
import numpy as np
import io
import time # To time the execution
import itertools # Useful for parameter combinations if looping outside

# --- Configuration ---
# Define parameter ranges to search
# pcomp (alpha) is a probability [0, 1]
# kapa (kappa) is the number of samples (positive integer)
PCOMP_SEARCH_RANGE = np.arange(0.1, 0.36, 0.01)  # Example: 0.0, 0.1, ..., 1.0
KAPA_SEARCH_RANGE = np.arange(1, 27, 1)        # Example: 1, 2, ..., 15
rng = np.random.default_rng(seed=42) # Pass seed here

# Reduce n_ids for faster parameter sweep? (Original was 200000)
# Trades precision of MSD estimate for speed. Adjust as needed.
N_IDS = 200000 # Example: Reduced for feasibility
CHUNK_SIZE = 10000  # Process 10,000 rows at a time

# Set seed for reproducibility across the entire run
np.random.seed(42)

print(f"--- Starting Optimization ---")
print(f"Searching pcomp in: {PCOMP_SEARCH_RANGE}")
print(f"Searching kapa in: {KAPA_SEARCH_RANGE}")
print(f"Using N_IDS (simulation runs per setting): {N_IDS}")
print(f"Using chunk size: {CHUNK_SIZE}")
start_total_time = time.time()

# --- SAS Step 1: Initial Data Creation ---
print("\nStep 1: Creating initial DataFrame 'df_a'...")
data_string = """
 1 8 .5 1 7 .5 2 6 .5 3 5 .5 4 -.195
 2 6 .55 22 9 .55 18 12 .55 14 15 .55 10 -.074
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
print(f"Initial df_a_initial shape: {df_a_initial.shape}")

# --- SAS Step 2: EV Calculation (Done once on initial data) ---
print("\nStep 2: Calculating EV variables...")
vvf_cols = ['va1', 'vb1', 'vc1', 'vd1']
vvs_cols = ['va2', 'vb2', 'vc2', 'vd2']
pp_cols = ['pa', 'pb', 'pc', 'pd']
ev_cols = ['eva', 'evb', 'evc', 'evd']

# Ensure correct types for calculation
for col_list in [vvf_cols, vvs_cols, pp_cols]:
    df_a_initial[col_list] = df_a_initial[col_list].astype(float)

for i in range(4):
    df_a_initial[ev_cols[i]] = (df_a_initial[vvf_cols[i]] * df_a_initial[pp_cols[i]] +
                              df_a_initial[vvs_cols[i]] * (1 - df_a_initial[pp_cols[i]])).round(2)

print("df_a_initial shape after EV calculation:", df_a_initial.shape)

# --- SAS Step 3: Data Expansion and Simulation (Now with chunked processing) ---
print("\nStep 3: Starting chunked processing of data expansion and simulation...")
start_expansion_time = time.time()

# Initialize results storage
results = []

# Process each parameter combination
for pcomp in PCOMP_SEARCH_RANGE:
    for kapa in KAPA_SEARCH_RANGE:
        print(f"\nProcessing pcomp={pcomp:.2f}, kapa={kapa}")
        
        # Process in chunks
        for chunk_start in range(0, N_IDS, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, N_IDS)
            chunk_size = chunk_end - chunk_start
            
            # Create chunk of data
            df_chunk = pd.DataFrame(np.repeat(df_a_initial.values, chunk_size, axis=0), 
                                  columns=df_a_initial.columns)
            
            # Add parameters
            df_chunk['pcomp'] = pcomp
            df_chunk['kapa'] = kapa
            df_chunk['id'] = np.tile(np.arange(chunk_start + 1, chunk_end + 1), len(df_a_initial))
            
            # --- Simulation for this chunk ---
            sum_cols = ['suma', 'sumb', 'sumc', 'sumd']
            
            # 'abc' Simulation Block
            df_chunk['suma'] = rng.standard_normal(len(df_chunk)) / 100000000
            df_chunk['sumb'] = rng.standard_normal(len(df_chunk)) / 100000000
            df_chunk['sumc'] = rng.standard_normal(len(df_chunk)) / 100000000
            
            for i in range(1, kapa + 1):
                rnd = rng.random(len(df_chunk))
                for s in range(3):  # Options A, B, C
                    value_to_add = np.where(rnd < df_chunk[pp_cols[s]], 
                                          df_chunk[vvf_cols[s]], 
                                          df_chunk[vvs_cols[s]])
                    df_chunk[sum_cols[s]] += value_to_add
            
            # Determine 'babc', 'cabc'
            df_chunk['babc'] = 0
            df_chunk['cabc'] = 0
            cond_b_max_abc = (df_chunk['suma'] < df_chunk['sumb']) & (df_chunk['sumc'] < df_chunk['sumb'])
            df_chunk.loc[cond_b_max_abc, ['babc', 'cabc']] = [1, 0]
            cond_c_max_abc = (df_chunk['suma'] < df_chunk['sumc']) & (df_chunk['sumb'] < df_chunk['sumc'])
            df_chunk.loc[cond_c_max_abc, ['babc', 'cabc']] = [0, 1]
            
            # Compromise Override for 'abc'
            override_rnd_abc = rng.random(len(df_chunk))
            cond_override_abc = override_rnd_abc < df_chunk['pcomp']
            df_chunk.loc[cond_override_abc, ['babc', 'cabc']] = [1, 0]
            
            # 'bcd' Simulation Block
            df_chunk['sumb'] = rng.standard_normal(len(df_chunk)) / 100000000
            df_chunk['sumc'] = rng.standard_normal(len(df_chunk)) / 100000000
            df_chunk['sumd'] = rng.standard_normal(len(df_chunk)) / 100000000
            
            for i in range(1, kapa + 1):
                rnd = rng.random(len(df_chunk))
                for s in range(1, 4):  # Options B, C, D
                    value_to_add = np.where(rnd < df_chunk[pp_cols[s]], 
                                          df_chunk[vvf_cols[s]], 
                                          df_chunk[vvs_cols[s]])
                    df_chunk[sum_cols[s]] += value_to_add
            
            # Determine 'bbcd', 'cbcd'
            df_chunk['bbcd'] = 0
            df_chunk['cbcd'] = 0
            cond_b_max_bcd = (df_chunk['sumd'] < df_chunk['sumb']) & (df_chunk['sumc'] < df_chunk['sumb'])
            df_chunk.loc[cond_b_max_bcd, ['bbcd', 'cbcd']] = [1, 0]
            cond_c_max_bcd = (df_chunk['sumd'] < df_chunk['sumc']) & (df_chunk['sumb'] < df_chunk['sumc'])
            df_chunk.loc[cond_c_max_bcd, ['bbcd', 'cbcd']] = [0, 1]
            
            # Compromise Override for 'bcd'
            override_rnd_bcd = rng.random(len(df_chunk))
            cond_override_bcd = override_rnd_bcd < df_chunk['pcomp']
            df_chunk.loc[cond_override_bcd, ['bbcd', 'cbcd']] = [0, 1]
            
            # Calculate metrics for this chunk
            df_chunk['ce'] = (df_chunk['babc'] - df_chunk['bbcd']).round(3)
            df_chunk['msd'] = (df_chunk['ce'] - df_chunk['oce'])**2
            
            # Aggregate results for this chunk
            chunk_results = df_chunk.groupby(['pcomp', 'kapa', 'superset']).agg({
                'msd': 'mean'
            }).reset_index()
            
            results.append(chunk_results)
            
            # Free memory
            del df_chunk
            
            print(f"  Processed chunk {chunk_start//CHUNK_SIZE + 1}/{(N_IDS + CHUNK_SIZE - 1)//CHUNK_SIZE}")

# Combine all results
df_m = pd.concat(results, ignore_index=True)
df_m = df_m.groupby(['pcomp', 'kapa']).agg({'msd': 'mean'}).reset_index()

end_expansion_time = time.time()
print(f"Processing took {end_expansion_time - start_expansion_time:.2f} seconds.")

# --- Final Analysis ---
print("\n--- Final results ranked by MSD ---")
df_m = df_m.sort_values(by='msd').reset_index(drop=True)
print(df_m.to_string(index=False))

# Find and print the best parameters
if not df_m.empty:
    best_params = df_m.loc[df_m['msd'].idxmin()]
    print("\n--- Best Parameters Found ---")
    print(f"Minimum MSD achieved: {best_params['msd']:.6f}")
    print(f"Optimal pcomp (alpha): {best_params['pcomp']:.2f}")
    print(f"Optimal kapa (kappa): {int(best_params['kapa'])}")
else:
    print("\nNo results found in df_m.")

end_total_time = time.time()
print(f"\nTotal execution time: {end_total_time - start_total_time:.2f} seconds.")
print("Python script finished.")