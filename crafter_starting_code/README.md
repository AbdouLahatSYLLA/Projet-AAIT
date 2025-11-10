-----

# Crafter Assignment: DQN vs. Double DQN

This repository contains the implementation of two Deep Reinforcement Learning agents, **DQN** and **Double DQN**, for the Crafter assignment.

The primary goal was to implement these agents from scratch and analyze their comparative performance, specifically focusing on learning stability and final average return.

## File Structure

  * `train.py`: A configurable training loop to run `random`, `dqn`, or `double_dqn` agents.
  * `multiAgents.py`: Contains the core agent logic, including a base class `DQNAgentBase` and the specific implementations for `DQNAgent` and `DoubleDQNAgent` to avoid code duplication.
  * `src/crafter_wrapper.py`: The provided environment wrapper for observation preprocessing and logging.
  * `analysis/plot_eval_performance.py`: A script to generate a comparative performance plot (mean + standard deviation) for multiple agent types.

## Setup & Installation

1.  **Clone the repository** and navigate to the folder.
2.  **Create a virtual environment** (e.g., using conda):
    ```bash
    conda create -n crafter python=3.9
    conda activate crafter
    ```
3.  **Install dependencies**: Follow the installation instructions in the [Crafter repository](https://github.com/danijar/crafter). Then, install the required Python packages:
    ```bash
    pip install torch pandas seaborn matplotlib
    ```

## How to Run

The `train.py` script accepts an `--agent` argument to select the agent (`random`, `dqn`, or `double_dqn`).

### 1\. Baseline (Random Agent)

To generate baseline data, run the random agent several times with different seeds.

```bash
# Example: 5 runs in parallel
for i in $(seq 0 4); do \
    python train.py \
        --agent random \
        --logdir logdir/random_agent/$i \
        --steps 1_000_000 \
        --eval-interval 25_000 & \
done
```

### 2\. DQN Agent

To train the `DQNAgent`, specify `--agent dqn`.

```bash
# Example: 5 runs in parallel
for i in $(seq 0 4); do \
    python train.py \
        --agent dqn \
        --logdir logdir/dqn_agent/$i \
        --steps 1_000_000 \
        --eval-interval 25_000 & \
done
```

### 3\. Double DQN Agent

To train the `DoubleDQNAgent`, specify `--agent double_dqn`.

```bash
# Example: 5 runs in parallel
for i in $(seq 0 4); do \
    python train.py \
        --agent double_dqn \
        --logdir logdir/double_dqn_agent/$i \
        --steps 1_000_000 \
        --eval-interval 25_000 & \
done
```

### Agent Hyperparameters

All key hyperparameters are exposed in `train.py` and can be overridden from the command line. The defaults (used in these experiments) are:

  * `--lr`: 1e-5
  * `--batch-size`: 32
  * `--buffer-size`: 100,000
  * `--gamma`: 0.99
  * `--target-update-interval`: 1000

-----

## Visualization & Analysis
```bash
python3 analysis/plot_eval_performance.py --logdir logdir/random_agent logdir/dqn_agent logdir/double_dqn_agent
```
### Generating the Comparison Plot

To generate the final comparative performance plot (which aggregates all runs for each agent), provide the parent directories to the analysis script:

```bash
python analysis/plot_eval_performance.py --logdir logdir/random_agent logdir/dqn_agent logdir/double_dqn_agent
```

This will save the aggregated plot as `performance_plot.png`.

### Analysis of Results

The following graph presents the aggregated comparison of the three agents, trained for 1 million steps. The solid line represents the mean performance across all training runs, and the shaded area indicates the standard deviation.

This graph provides a clear summary of the agents' capabilities:

  * **Random Agent (Baseline):** The `random_agent` (blue) exhibits a stable, low-variance performance, averaging an episodic return of approximately 1.4. This confirms it is a reliable baseline.

  * **DQN Agent:** The `dqn_agent` (orange) successfully learns a policy that significantly outperforms the baseline, achieving a final mean return of \~4.8. However, its most notable characteristic is the **pronounced instability and high variance**, visualized by the expansive shaded region. This empirical result is a classic demonstration of the overestimation bias inherent in standard Q-learning, which leads to erratic policy updates and high variance between independent training runs.

  * **Double DQN Agent:** The `double_dqn_agent` (green) provides the most compelling results.

    1.  **Superior Stability:** Its exceptionally low variance (the narrow green shaded band) proves that it mitigates the overestimation bias, leading to highly stable and reliable learning.
    2.  **Rapid Convergence:** The Double DQN agent also exhibits a faster and more consistent initial learning curve, converging to a high-performance policy significantly earlier than the standard DQN agent.

### Conclusion

These results empirically validate that **Double DQN is a significant and robust improvement over standard DQN**. While both agents learn, the Double DQN implementation converges faster and with far greater stability, making it the more reliable and effective algorithm for this environment.