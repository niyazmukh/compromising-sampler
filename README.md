# Compromise Effect Sampler Simulation

This Python script simulates a choice experiment designed to model the "compromise effect" in decision-making. It uses a stochastic accumulator model and includes a mechanism for probabilistic compromise choices. The script can operate in two modes:

1.  **Search Mode:** Uses optimization (`scipy.optimize.minimize` with Nelder-Mead) to find the `ALPHA` (compromise probability factor) and `K` (number of evidence accumulation steps) parameters that best fit observed data by minimizing the Mean Squared Deviation (MSD) between predicted and observed compromise effects (`oce`).
2.  **Execute Mode:** Runs the simulation directly with pre-defined `INITIAL_ALPHA` and `INITIAL_K` values set within the script.

## Features

*   Simulates choices between three options (ABC and BCD contexts).
*   Implements a probabilistic compromise choice mechanism.
*   Uses Numba (`@numba.njit`) for significantly accelerated simulation performance.
*   Calculates Expected Values (EVs) for options.
*   Aggregates results across multiple simulation runs (`N_IDS`).
*   Calculates predicted compromise effect (`ce`) and Mean Squared Deviation (`msd`) against observed data (`oce`).
*   Provides command-line options to switch between parameter search and direct execution.
*   Outputs a formatted table summarizing parameters, choice probabilities, EVs, and fit metrics (CE, OCE, MSD).
*   Calculates the correlation between observed and predicted compromise effects.

## Requirements

*   Python 3.x
*   NumPy
*   Pandas
*   SciPy
*   Numba

You can install the required libraries using pip:

```bash
pip install numpy pandas scipy numba
```

## Usage

The script is run from the command line.

**1. Parameter Search Mode (Default):**

This mode runs the optimization algorithm to find the best-fitting `ALPHA` and `K` based on the hardcoded initial data and observed compromise effect (`oce`).

```bash
python compromising_sampler.py
```

or explicitly:

```bash
python compromising_sampler.py --mode search
```

The optimizer (Nelder-Mead) will start its search from the `INITIAL_ALPHA` and `INITIAL_K` values defined near the top of the script.

**2. Execute Mode:**

This mode skips the optimization and runs the simulation directly using the `INITIAL_ALPHA` and `INITIAL_K` values defined in the script.

```bash
python compromising_sampler.py --mode execute
```

## Input Data

Currently, the input parameters for the different choice problems (`supersets`) and their corresponding observed compromise effects (`oce`) are hardcoded within the `data_string` variable inside the `if __name__ == "__main__":` block.

## Output

The script prints the following to the console:

1.  Status messages indicating preparation, optimization steps (if in search mode), and final simulation run.
2.  If in search mode, the results of the optimization (optimal parameters found, minimum MSD).
3.  A formatted table (`results_table`) showing:
    *   Input parameters (`Superset`, `Alpha`, `K`, option values/probabilities).
    *   Calculated Expected Values (`EV(A)`, `EV(B)`, etc.).
    *   Simulated choice probabilities (`P(B|ABC)`, `P(C|ABC)`, etc.).
    *   Predicted Compromise Effect (`CE_Pred`).
    *   Observed Compromise Effect (`CE_Obs`).
    *   Mean Squared Deviation (`MSD`) for each superset.
4.  The overall correlation between `CE_Obs` and `CE_Pred`.
5.  The final mean MSD across all supersets.
6.  Total execution time.

## Key Parameters (Script Constants)

*   `INITIAL_ALPHA`: Initial guess for the compromise probability factor (used as starting point for search mode, or directly in execute mode).
*   `INITIAL_K`: Initial guess for the number of simulation steps (used as starting point for search mode, or directly in execute mode).
*   `N_IDS`: Number of simulation runs per parameter set (superset). Higher values reduce simulation noise but increase runtime.
*   `SEED`: Seed for the random number generator to ensure reproducibility. 