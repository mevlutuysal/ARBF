import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def analyze_results(csv_file='evaluation_results.csv'):
    """Analyzes the CSV file to compute and print final metrics."""
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please run the evaluation first.")
        return

    print(f"\n--- 📊 Analysis of Evaluation Results from '{csv_file}' 📊 ---")

    # --- 1. Effectiveness Metrics (License Selection) ---
    y_true = np.ones(len(df)) # Ground truth is that all should be correct
    y_pred = df['is_correct'].values

    num_correct = df['is_correct'].sum()
    total_scenarios = len(df)
    accuracy = num_correct / total_scenarios if total_scenarios > 0 else 0

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n## 🎯 Effectiveness Metrics")
    print("-" * 25)
    print(f"Total Scenarios Run:      {total_scenarios}")
    print(f"Correctly Classified:     {num_correct}")
    print(f"Accuracy:                 {accuracy:.2%}")
    print(f"Precision:                {precision:.4f}")
    print(f"Recall:                   {recall:.4f}")
    print(f"F1-Score:                 {f1:.4f}")

    # --- 2. Performance (Latency) ---
    latencies = df['latency_seconds'].dropna()
    avg_latency = latencies.mean()
    std_latency = latencies.std()
    ci_latency = 1.96 * (std_latency / np.sqrt(len(latencies))) if len(latencies) > 0 else 0

    print("\n## ⏱️ Performance Metrics (Latency)")
    print("-" * 25)
    print(f"Average Latency:          {avg_latency:.2f} seconds")
    print(f"Latency 95% CI:           [{avg_latency - ci_latency:.2f}s, {avg_latency + ci_latency:.2f}s]")
    print(f"Min | Max Latency:        {latencies.min():.2f}s | {latencies.max():.2f}s")


    # --- 3. Cost-Efficiency (Gas) ---
    gas_used = df['gas_used'].dropna()
    avg_gas_used = gas_used.mean()

    gas_cost_eth = df['gas_cost_eth'].dropna()
    avg_gas_cost_eth = gas_cost_eth.mean()

    print("\n## ⛽ Cost-Efficiency Metrics (Sepolia Testnet)")
    print("-" * 25)
    print(f"Average Gas Used:         {avg_gas_used:,.0f}")
    print(f"Average On-Chain Cost:    {avg_gas_cost_eth:.8f} Sepolia ETH")
    print("-" * 25)

if __name__ == '__main__':
    analyze_results()