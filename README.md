# AutoLab - LLM-Driven Automated ML Experiment

An automated machine learning experimentation system that uses an LLM to intelligently design, run, and analyze CNN experiments on MNIST. The system closes the loop between hypothesis generation and empirical validation - the LLM proposes experiments, a bandit strategy prioritizes them, and results feed back into the next planning cycle.

Inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch).

## Architecture

```
┌─────────────┐     ┌───────────────┐     ┌──────────────────┐
│  LLM Planner│────▶│ Bandit Scorer │────▶│ Experiment Runner│
│  (propose)  │     │  (UCB rank)   │     │  (train & eval)  │
└─────────────┘     └───────────────┘     └────────┬─────────┘
       ▲                                           │
       │            ┌───────────────┐              │
       └────────────│   Analyzer    │◀─────────────┘
                    │  (summarize)  │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │    SQLite DB  │
                    └───────────────┘
```

`main_loop.py` orchestrates 5 cycles of **propose → score → train → analyze**.

## Components

| File | Role |
|---|---|
| `planner_llm.py` | Calls the LLM (via OpenRouter) to reflect on past results, form a hypothesis, and propose a batch of 3 experiments |
| `bandit.py` | Scores each proposed config with UCB (Upper Confidence Bound) to balance exploration vs exploitation |
| `experiment_runner.py` | Trains a configurable CNN on MNIST for 1 epoch and returns test accuracy |
| `database.py` | SQLite storage for experiment configs, accuracies, hypotheses, and cycle metadata |
| `analyzer.py` | Aggregates results by config, ranks them, and prints the best configuration found so far |
| `models/cnn.py` | Dynamically-built CNN - conv layers, filters, and depth are all configurable |

## Hyperparameter Search Space

| Parameter | Options |
|---|---|
| `filters` | 16, 32, 64, 128 |
| `num_layers` | 1, 2, 3 |
| `learning_rate` | 0.01, 0.001 |
| `optimizer` | adam, sgd |

The total grid has 48 combinations. The LLM doesn't enumerate them - it reasons about which regions to explore next based on accumulated evidence.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the `autolab/` directory:

```
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-5.3-chat  # or any model available on OpenRouter
```

## Run

```bash
cd autolab
python main_loop.py
```

The system will run 5 agent cycles. Each cycle proposes up to 3 experiments, trains them, and prints an updated leaderboard. All results are persisted to `experiments.db`.

## Requirements

- Python 3.10+
- PyTorch, torchvision
- pandas, plotly
- openai (used as the OpenRouter client)
- python-dotenv

See `requirements.txt` for the full list.

## Acknowledgments

- [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy - the direct inspiration for this project

