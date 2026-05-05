# Deal or No Deal — Banker Strategy Simulation

A complete Monte Carlo simulation project modeling the TV game *Deal or No Deal* from the **Banker's perspective**.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Project Structure

```
deal_or_no_deal_banker_project/
├── main.py                         # Entry point
├── requirements.txt
├── README.md
├── src/
│   ├── game.py                     # Game engine (26 cases, realistic opening schedule)
│   ├── policies.py                 # 4 banker offer policies
│   ├── players.py                  # Logistic acceptance model (3 player types)
│   ├── simulation.py               # Monte Carlo simulation engine
│   ├── analysis.py                 # Statistics & convergence analysis
│   └── visualization.py            # Matplotlib figure generation
├── outputs/
│   ├── results_summary.csv         # Summary statistics per policy × player type
│   ├── raw_simulation_results.csv  # All 120,000 game results
│   ├── convergence_results.csv     # Convergence analysis table
│   └── figures/
│       ├── expected_profit_comparison.png
│       ├── profit_distribution_boxplot.png
│       ├── convergence_plot.png
│       └── acceptance_rate_comparison.png
└── report/
    ├── report.md                   # Full academic report
    └── ai_usage_log.md             # AI tool transparency log
```

## Banker Policies

| Policy | Formula |
|--------|---------|
| Baseline | `0.8 × mean(remaining)` |
| Risk-Adjusted | `0.85 × mean − 0.05 × std` |
| Dynamic | `(0.65 + progress × 0.30) × mean` |
| Player-Adaptive | `base_mult × mean`, adjusted by rejection history |

## Player Types

| Type | Accept Threshold | Behavior |
|------|-----------------|---------|
| Risk-Averse | 70% of EV | Accepts below expected value |
| Risk-Neutral | 90% of EV | Accepts near expected value |
| Risk-Seeking | 110% of EV | Requires above expected value |

## Results Summary

All three alternative policies outperform the baseline:

| Policy | Avg Banker Profit |
|--------|-----------------|
| Player-Adaptive | $25,057 |
| Dynamic Round-Based | $24,067 |
| Risk-Adjusted | $23,616 |
| **Baseline (0.8×EV)** | **$19,715** |

## Reproducibility

Set `RANDOM_SEED = 42` in `main.py`. All results are fully reproducible.
