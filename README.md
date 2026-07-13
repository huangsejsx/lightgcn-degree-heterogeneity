# Degree Heterogeneity and Curvature-Based Structural Analysis in LightGCN Recommendation

This repository contains the code, processed structural statistics, simulation data, experimental results, and figures used in the MSc dissertation:

**Degree Heterogeneity and Curvature-Based Structural Analysis in LightGCN Recommendation**

The project investigates how degree heterogeneity in user-item interaction graphs is associated with LightGCN recommendation performance and learned embedding geometry. Forman-Ricci curvature magnitude is used as an edge-level structural descriptor of the local degree scale surrounding user-item interactions.

The empirical analysis is conducted on MovieLens-1M. A controlled simulation study is additionally used to examine how LightGCN behaviour changes as item-degree heterogeneity is systematically increased.

## Project Structure

```text
.
├── data/
│   ├── processed/
│   │   ├── graph_stats.csv
│   │   ├── item_degrees.csv
│   │   ├── item_mapping.csv
│   │   ├── user_degrees.csv
│   │   └── user_mapping.csv
│   ├── raw/
│   │   └── ml-1m/
│   └── simulated/
│       ├── low_heterogeneity_interactions.csv
│       ├── low_heterogeneity_item_degrees.csv
│       ├── medium_heterogeneity_interactions.csv
│       ├── medium_heterogeneity_item_degrees.csv
│       ├── high_heterogeneity_interactions.csv
│       └── high_heterogeneity_item_degrees.csv
├── results/
│   ├── *.csv
│   └── *.png
├── src/
│   ├── preprocess.py
│   ├── split.py
│   ├── analyze_graph.py
│   ├── baseline_popularity.py
│   ├── baseline_itemcf.py
│   ├── lightgcn.py
│   ├── evaluate_groups.py
│   ├── analyze_edge_structure.py
│   ├── analyze_embeddings.py
│   ├── analyze_test_item_composition.py
│   ├── simulate_degree_heterogeneity.py
│   ├── run_simulation_lightgcn.py
│   ├── analyze_simulation_test_composition.py
│   └── plot_*.py
├── analysis.md
├── requirements.txt
└── README.md
```

The `src/` directory contains the main experimental and analysis scripts. Generated numerical results and dissertation figures are stored in `results/`.

Raw MovieLens data and generated embedding arrays are excluded from the repository.

## Dataset

The empirical analysis uses the **MovieLens-1M** dataset provided by GroupLens Research.

The raw dataset is not redistributed in this repository. Download MovieLens-1M from the official GroupLens source and extract it so that the ratings file is located at:

```text
data/raw/ml-1m/ratings.dat
```

The preprocessing pipeline treats ratings greater than or equal to 4 as positive implicit-feedback interactions. Ratings below 4 are discarded.

Users with fewer than three positive interactions are excluded.

## Environment Setup

The project was developed using Python in Visual Studio Code.

It is recommended to create a virtual environment before installing the required packages.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Reproducing the MovieLens Experiments

The empirical analysis can be reproduced from the raw MovieLens-1M ratings file.

Run the following commands from the project root directory.

### 1. Preprocess MovieLens-1M

```bash
python src/preprocess.py
```

This converts explicit MovieLens ratings into positive implicit-feedback interactions.

### 2. Create the temporal train-validation-test split

```bash
python src/split.py
```

For each user, positive interactions are ordered by timestamp. The final interaction is used for testing, the second-last interaction is used for validation, and all earlier interactions are used for training.

### 3. Analyse the training graph

```bash
python src/analyze_graph.py
```

This computes graph statistics and user and item degree information.

### 4. Run recommendation baselines

```bash
python src/baseline_popularity.py
python src/baseline_itemcf.py
```

The two baseline models are global item popularity and item-based collaborative filtering.

### 5. Train and evaluate LightGCN

```bash
python src/lightgcn.py
```

The final LightGCN configuration used in the dissertation is:

- embedding dimension: 64
- propagation layers: 2
- batch size: 2048
- learning rate: 0.001
- training epochs: 20
- optimisation objective: Bayesian Personalised Ranking (BPR)
- negative sampling: one uniformly sampled unobserved item per positive training interaction

Full-ranking evaluation is performed over all items not present in each user's training history.

### 6. Run grouped evaluation

```bash
python src/evaluate_groups.py
```

Users and items are divided into degree groups using the 20th and 80th percentiles of their training degree distributions.

The script evaluates LightGCN performance by user degree group and by the training degree group of the held-out test item.

### 7. Analyse edge-level graph structure

```bash
python src/analyze_edge_structure.py
```

For each training edge $begin:math:text$e\=\(u\,i\)$end:math:text$, the analysis computes:

```text
Forman-Ricci curvature:       F(e) = 4 - d_u - d_i
Curvature magnitude:          M(e) = |F(e)|
Endpoint degree sum:          S(e) = d_u + d_i
Endpoint degree difference:   D(e) = |d_u - d_i|
```

In the unweighted graph considered in this project, Forman-Ricci curvature magnitude is algebraically related to endpoint degree sum. It is therefore interpreted as an edge-level structural descriptor rather than as an independent causal variable.

### 8. Analyse learned embeddings

```bash
python src/analyze_embeddings.py
```

This analysis compares LightGCN embedding norms across degree groups and calculates Pearson correlations between node degree and embedding norm.

### 9. Analyse held-out test-item composition

```bash
python src/analyze_test_item_composition.py
```

This diagnostic analysis examines the degree-group composition of held-out test interactions across user degree groups.

### 10. Generate empirical figures

Run the relevant plotting scripts:

```bash
python src/plot_model_comparison.py
python src/plot_grouped_evaluation.py
python src/plot_edge_structure.py
python src/plot_forman_ricci_magnitude.py
python src/plot_embedding_analysis.py
```

Generated figures are saved in the `results/` directory.

## Reproducing the Controlled Simulation Study

The simulation study systematically varies item-degree heterogeneity while holding the numbers of users, items, and interactions fixed.

Each synthetic dataset contains:

- 1,000 users
- 600 items
- 20,000 positive interactions
- 20 interactions per user
- 5 latent user-item communities

The three scenarios use the following item-popularity concentration parameters:

| Scenario | Alpha |
| --- | ---: |
| Low heterogeneity | 0.0 |
| Medium heterogeneity | 0.7 |
| High heterogeneity | 1.3 |

### 1. Generate the synthetic interaction graphs

```bash
python src/simulate_degree_heterogeneity.py
```

All three scenarios use graph-generation seed 42.

The generated synthetic datasets are stored in:

```text
data/simulated/
```

### 2. Train LightGCN on the simulated graphs

```bash
python src/run_simulation_lightgcn.py
```

For each fixed synthetic graph, LightGCN is independently trained using optimisation seeds:

```text
42
43
44
```

The same LightGCN architecture and training configuration used in the empirical MovieLens experiment are applied to all three simulation scenarios.

Simulation recommendation and embedding statistics are summarised using the mean and standard deviation across the three optimisation runs.

### 3. Analyse simulation test-item composition

```bash
python src/analyze_simulation_test_composition.py
```

This script examines the training-degree composition of held-out test items and checks the proportion of test interactions involving zero-training-degree items.

### 4. Generate simulation figures

```bash
python src/plot_simulation_results.py
```

The generated simulation figures are stored in the `results/` directory.

## Main Outputs

The main numerical results are stored as CSV files in `results/`.

These include:

- aggregate recommendation metrics
- grouped LightGCN evaluation results
- edge-level structural summaries
- degree-embedding norm correlations
- embedding norm summaries
- test-item composition diagnostics
- simulation degree-heterogeneity statistics
- simulation results by optimisation seed
- simulation summary statistics

Figures used in the dissertation are also generated in the `results/` directory.

## Reproducibility Notes

The empirical analysis uses the fixed LightGCN configuration documented above.

For the controlled simulation study:

- graph-generation seed: 42
- LightGCN optimisation seeds: 42, 43, and 44

Each heterogeneity scenario is represented by one synthetic graph generated using the fixed graph-generation seed. The three optimisation seeds assess robustness to model-training randomness, but not to alternative synthetic graph realisations.

The simulation should therefore be interpreted as a controlled structural experiment rather than as evidence of an independent causal effect of Forman-Ricci curvature.

## Software Assistance

Visual Studio Code was used as the primary code-development environment, with GitHub Copilot enabled as a coding assistant. Coding suggestions and corrections were reviewed before being accepted or incorporated into the project.

Generative AI tools were used to assist with code development and debugging, language editing, LaTeX formatting, and discussion of experimental design and interpretation. All generated suggestions were reviewed, adapted, and verified by the author.

## Author

MSc Statistics  
Imperial College London
