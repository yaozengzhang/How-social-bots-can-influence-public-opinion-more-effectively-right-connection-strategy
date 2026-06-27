# How Social Bots Can Influence Public Opinion More Effectively

This repository contains Python code for the simulation study associated with:

Zhang, Y., Ma, J., & Fang, F. (2024). How social bots can influence public opinion more effectively: Right connection strategy. Physica A: Statistical Mechanics and its Applications, 633, 129386.

DOI: https://doi.org/10.1016/j.physa.2023.129386

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── social_bot.py
└── more data of fig8 fig11 fig12/
    ├── fig8more.tif
    ├── fig11more.tif
    └── fig12more.tif
```

- `social_bot.py`: main simulation script. It generates BA scale-free networks, initializes users and social bots, runs the opinion evolution process, and exports a heatmap and CSV results.
- `more data of fig8 fig11 fig12/`: additional figure files related to the paper.

## Code Flow

```text
BA network generation
-> user opinion and corpus initialization
-> social bot connection initialization
-> social bot activation
-> bot insertion into the network
-> user opinion interaction
-> activity decay
-> repeated simulations
-> CSV and heatmap output
```

The code keeps the random behavior of the original MATLAB-style script. By default, no random seed is fixed. Use `--seed` only when a deterministic Python run is needed for checking.

## Installation

```bash
pip install -r requirements.txt
```

## Run

Full run with default parameters:

```bash
python social_bot.py
```

Quick check run:

```bash
python social_bot.py --seed 1 --network-size 30 --m-values 1 2 --replications 1 --tolerance-steps 3 --simulation-steps 5 --robots 3 --robot-links 3 --output-dir outputs_smoke
```

The quick check is only for verifying that the program runs and writes outputs.

## Main Parameters

```text
--initial-nodes          initial BA network node count, default 6
--network-size           final BA network size, default 1000
--m-values               BA m parameters, default 1 2 3 4 5 6
--replications           repeated runs per setting, default 10
--tolerance-steps        tolerance grid size, default 31
--tolerance-step-size    tolerance step, default 0.01
--simulation-steps       opinion evolution steps, default 500
--robots                 number of social bots, default 50
--robot-links            links per social bot, default 50
--seed                   optional seed for deterministic Python checks
--output-dir             output directory, default outputs
--show                   show the heatmap window after saving
```

## Outputs

```text
outputs/
├── simulation_results.csv
└── simulation_heatmap.png
```

- `simulation_results.csv`: averaged simulation result matrix.
- `simulation_heatmap.png`: heatmap for the result matrix.
