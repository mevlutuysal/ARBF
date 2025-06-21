# orchestrator.py

import os
import csv
import time
from pathlib import Path

# Import agents and shared utilities
import agent_ingestion
import agent_policy
import agent_registration
import shared_utils
import json
from agent_policy import run_policy_selection # Using your new rewritten agent
from shared_utils import PolicyData


def load_scenarios_from_csv(file_path='scenarios.csv'):
    """Loads evaluation scenarios from a CSV file."""
    # This logic is from your original run_evaluation.py
    scenarios = []
    with open(file_path, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            scenarios.append(row)
    return scenarios


def load_policies_from_file(filepath: str) -> list:
    """Loads policy data from a JSON file."""
    # This function should load your list of all policy objects.
    # The expert suggestion relies on having the full text for each policy.
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_dummy_content_file(file_path: str):
    """Creates a non-empty dummy file if it doesn't exist."""
    # This logic is from your original run_evaluation.py
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with open(path, 'w') as f:
            f.write(f"This is a dummy content file for evaluation: {path.name}\n")


def main():
    """Main orchestration logic to run the evaluation suite."""
    print("======== 🚀 Starting Refactored AIGC Framework Evaluation 🚀 ========")

    # 1. Initialize all shared components once
    try:
        components = shared_utils.initialize_components()
    except (ValueError, FileNotFoundError, ConnectionError) as e:
        print(f"!!!!!!!! ❌ CRITICAL ERROR during initialization !!!!!!!!")
        print(e)
        return

    # 2. Load evaluation scenarios
    scenarios = load_scenarios_from_csv()
    results_data = []
    print("Loading policies from source file...")
    all_policies = load_policies_from_file("policies.json")

    # Get the web3 instance for unit conversions
    w3 = components["w3"]


    # 3. Run the workflow for each scenario
    for i, scenario in enumerate(scenarios):
        print(f"\n--- 🏃 Running Scenario {i + 1}/{len(scenarios)}: {scenario['scenario_id']} ---")

        # Add agent's wallet address to scenario data
        scenario['creator_wallet'] = components['agent_account'].address
        ensure_dummy_content_file(scenario["content_file"])

        start_time = time.time()
        final_status = "Success"
        error_message = ""
        gas_price_gwei = 0
        cost_eth = 0

        try:
            # --- AGENT WORKFLOW ---
            # Step 1: Ingestion Agent
            ingestion_result = agent_ingestion.run_ingestion(
                file_path=scenario["content_file"],
                ipfs_manager=components["ipfs_storage"]
            )

            # Step 2: Policy Agent
            # policy_result = agent_policy.run_policy_selection(
            #     asset_description=scenario["asset_description"],
            #     vector_db=components["vector_db"],
            #     llm=components["llm"]
            # )

            policy_result = run_policy_selection(
                asset_description=scenario["asset_description"],
                all_policies=all_policies,  # Pass the list here
                llm=components["llm"]
            )

            # Step 3: Registration Agent
            registration_result = agent_registration.run_registration(
                ingestion_data=ingestion_result,
                policy_data=policy_result,
                scenario_data=scenario,
                components=components
            )

            # --- Cost Calculation ---
            if registration_result.status == "Success":
                gas_price_wei = registration_result.effective_gas_price
                gas_used = registration_result.gas_used
                # Convert Wei to Gwei for readability
                gas_price_gwei = w3.from_wei(gas_price_wei, 'gwei')
                # Calculate total cost in ETH
                cost_eth = w3.from_wei(gas_used * gas_price_wei, 'ether')


        except Exception as e:
            final_status = "Failed"
            error_message = str(e)
            print(f"!!!!!!!! ❌ SCENARIO FAILED: {error_message} !!!!!!!!")
            # In a real system, you might not have these results, so fill with placeholders
            policy_result = shared_utils.PolicyData(selected_license_uri="N/A (due to error)")
            registration_result = shared_utils.RegistrationData(metadata_cid="N/A", transaction_hash="N/A",
                                                                token_id="N/A", block_number=0, status="Failed",
                                                                gas_used=0, effective_gas_price=0)


        latency = time.time() - start_time

        # 4. Record results
        is_correct = int(
            policy_result.selected_license_uri is not None and
            policy_result.selected_license_uri.strip().lower() == scenario["expected_license_uri"].strip().lower()
        )

        results_data.append({
            "scenario_id": scenario["scenario_id"],
            "final_status": final_status,
            "latency_seconds": f"{latency:.2f}",
            "gas_used": registration_result.gas_used,
            "gas_price_gwei": f"{gas_price_gwei:.4f}",
            "cost_eth": f"{cost_eth:.8f}",
            "expected_license": scenario["expected_license_uri"],
            "selected_license": policy_result.selected_license_uri,
            "is_correct": is_correct,
            # "tx_hash": registration_result.transaction_hash,
            "error_message": error_message
        })
        print(
            f"✅ Result: Status={final_status}, CorrectLicense={bool(is_correct)}, Latency={latency:.2f}s, GasUsed={registration_result.gas_used}, GasPrice={gas_price_gwei:.4f} Gwei, Cost={cost_eth:.8f} ETH")


    # 5. Save results to CSV
    csv_file = 'evaluation_results_refactored.csv'
    if results_data:
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results_data[0].keys())
            writer.writeheader()
            writer.writerows(results_data)
        print(f"\n======== 🎉 Evaluation Complete. Results saved to {csv_file} 🎉 ========")


if __name__ == "__main__":
    main()