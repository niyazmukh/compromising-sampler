import pandas as pd
import numpy as np
import io
import time

# Fixed parameters
ALPHA = 0.25
K = 3
N_IDS = 400000  # Number of simulation runs
rng = np.random.default_rng(seed=42)

print(f"--- Starting Simulation ---")
print(f"Using alpha: {ALPHA}")
print(f"Using k: {K}")
print(f"Number of simulation runs: {N_IDS}")
start_time = time.time()

# --- Data Creation ---
print("\nStep 1: Creating initial DataFrame...")
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

# --- EV Calculation ---
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

# --- Data Expansion ---
print("\nStep 3: Expanding DataFrame for simulation...")
df_expanded = pd.DataFrame(np.repeat(df_a_initial.values, N_IDS, axis=0), columns=df_a_initial.columns)
df_expanded['pcomp'] = ALPHA
df_expanded['kapa'] = K
df_expanded['id'] = np.tile(np.arange(1, N_IDS + 1), len(df_a_initial))

df_a = df_expanded
del df_expanded

# --- Simulation ---
print("\nStep 4: Running simulation...")
sum_cols = ['suma', 'sumb', 'sumc', 'sumd']

# 'abc' Simulation Block
df_a['suma'] = rng.standard_normal(len(df_a)) / 100000000
df_a['sumb'] = rng.standard_normal(len(df_a)) / 100000000
df_a['sumc'] = rng.standard_normal(len(df_a)) / 100000000

for i in range(1, K + 1):
    rnd = rng.random(len(df_a))
    for s in range(3):  # Options A, B, C
        pp_s = df_a[pp_cols[s]]
        vvf_s = df_a[vvf_cols[s]]
        vvs_s = df_a[vvs_cols[s]]
        value_to_add = np.where(rnd < pp_s, vvf_s, vvs_s)
        df_a[sum_cols[s]] += value_to_add

df_a['babc'] = 0
df_a['cabc'] = 0
cond_b_max_abc = (df_a['suma'] < df_a['sumb']) & (df_a['sumc'] < df_a['sumb'])
df_a.loc[cond_b_max_abc, ['babc', 'cabc']] = [1, 0]
cond_c_max_abc = (df_a['suma'] < df_a['sumc']) & (df_a['sumb'] < df_a['sumc'])
df_a.loc[cond_c_max_abc, ['babc', 'cabc']] = [0, 1]

override_rnd_abc = rng.random(len(df_a))
cond_override_abc = override_rnd_abc < df_a['pcomp']
df_a.loc[cond_override_abc, ['babc', 'cabc']] = [1, 0]

# 'bcd' Simulation Block
df_a['sumb'] = rng.standard_normal(len(df_a)) / 100000000
df_a['sumc'] = rng.standard_normal(len(df_a)) / 100000000
df_a['sumd'] = rng.standard_normal(len(df_a)) / 100000000

for i in range(1, K + 1):
    rnd = rng.random(len(df_a))
    for s in range(1, 4):  # Options B, C, D
        pp_s = df_a[pp_cols[s]]
        vvf_s = df_a[vvf_cols[s]]
        vvs_s = df_a[vvs_cols[s]]
        value_to_add = np.where(rnd < pp_s, vvf_s, vvs_s)
        df_a[sum_cols[s]] += value_to_add

df_a['bbcd'] = 0
df_a['cbcd'] = 0
cond_b_max_bcd = (df_a['sumd'] < df_a['sumb']) & (df_a['sumc'] < df_a['sumb'])
df_a.loc[cond_b_max_bcd, ['bbcd', 'cbcd']] = [1, 0]
cond_c_max_bcd = (df_a['sumd'] < df_a['sumc']) & (df_a['sumb'] < df_a['sumc'])
df_a.loc[cond_c_max_bcd, ['bbcd', 'cbcd']] = [0, 1]

override_rnd_bcd = rng.random(len(df_a))
cond_override_bcd = override_rnd_bcd < df_a['pcomp']
df_a.loc[cond_override_bcd, ['bbcd', 'cbcd']] = [0, 1]

# --- Aggregation and Results ---
print("\nStep 5: Calculating final results...")
by_vars = ['superset', 'va1', 'pa', 'va2', 'vb1', 'pb', 'vb2', 'vc1', 'pc', 'vc2', 'vd1', 'pd', 'vd2', 'oce']

df_o = df_a.groupby(by_vars, observed=True).agg({
    'babc': 'mean',
    'cabc': 'mean',
    'bbcd': 'mean',
    'cbcd': 'mean'
}).reset_index()

df_o['ce'] = (df_o['babc'] - df_o['bbcd']).round(3)
df_o['msd'] = (df_o['ce'] - df_o['oce'])**2

# --- Final Results Table ---
print("\n--- Final Results Table ---")
results_table = df_o[['superset', 'ce', 'oce', 'msd']].copy()
results_table.columns = ['Superset', 'CE', 'OCE', 'MSD']
results_table['Superset'] = results_table['Superset'].astype(int)
results_table = results_table.sort_values('Superset')
print(results_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Calculate and print correlation
correlation = df_o['ce'].corr(df_o['oce'])
print("\n" + "="*50)
print("CORRELATION ANALYSIS")
print("="*50)
print(f"Pearson correlation between observed and fitted CE: {correlation:.4f}")
print("="*50)

# --- Summary Statistics ---
print("\n--- Summary Statistics ---")
print(f"Mean MSD: {df_o['msd'].mean():.6f}")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")