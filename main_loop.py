import os
from planner_llm import propose_batch
from experiment_runner import run_experiment
from database import save_experiment, init_db, experiment_exists
from bandit import bandit_score
from analyzer import analyze_experiments

init_db()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("Error: OPENROUTER_API_KEY not set. Create a .env file with your key.")
    exit(1)

max_cycles = 5

for cycle in range(max_cycles):

    print(f"\n===== AGENT CYCLE {cycle+1}/{max_cycles} =====")

    try:
        decision = propose_batch()
    except Exception as e:
        print(f"Error proposing batch: {e}")
        continue

    hypothesis = decision["hypothesis"]

    print("\nAgent Reflection:")
    print(decision["reflection"])

    print("\nAgent Hypothesis:")
    print(hypothesis)

    # Score experiments with bandit strategy and sort by priority
    experiments = decision["experiments"]
    scored = []

    for config in experiments:
        config_fixed = {
            "filters": config["filters"],
            "lr": config["learning_rate"],
            "optimizer": config["optimizer"],
            "num_layers": config.get("num_layers", 2)
        }
        score = bandit_score(config_fixed)
        scored.append((score, config_fixed))

    scored.sort(key=lambda x: x[0], reverse=True)

    for score, config_fixed in scored:

        if experiment_exists(config_fixed):
            print(f"Skipping duplicate experiment: {config_fixed}")
            continue

        print(f"\nRunning experiment (bandit score: {score:.4f}): {config_fixed}")

        try:
            accuracy = run_experiment(config_fixed)
        except Exception as e:
            print(f"Experiment failed: {e}")
            continue

        save_experiment(config_fixed, accuracy, hypothesis, cycle+1)

        print(f"Accuracy: {accuracy:.2f}%")

    # Show analysis after each cycle
    print(f"\n--- Analysis after Cycle {cycle+1} ---")
    analyze_experiments()

print("\n===== EXPERIMENT LOOP COMPLETE =====")
analyze_experiments()