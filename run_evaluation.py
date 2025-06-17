# In run_evaluation.py

import os
import sys
import time
import json
import re
import csv
from pathlib import Path

# --- Import your agent and web3 instance ---
try:
    import agent_main
    from agent_main import _AgentState  # Add this import

    w3 = agent_main.w3
    agent_executor = agent_main.agent_executor
    agent_address = agent_main.agent_account.address
except ImportError as e:
    print(f"Error: Could not import from agent_main.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during import from agent_main: {e}")
    sys.exit(1)


def load_scenarios_from_csv(file_path='scenarios.csv'):
    """Loads evaluation scenarios from a CSV file."""
    scenarios = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                row['creator_wallet'] = agent_address
                scenarios.append(row)
        print(f"Successfully loaded {len(scenarios)} scenarios from {file_path}.")
        return scenarios
    except FileNotFoundError:
        print(f"Error: Scenarios file not found at {file_path}. Aborting.")
        sys.exit(1)


def ensure_dummy_content_file(file_path: str):
    """Creates a non-empty dummy file and parent directories if they don't exist."""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Always write to the file to ensure it's not 0-bytes, which can cause upload hangs.
        with open(path, 'w') as f:
            f.write(f"This is a dummy content file for: {path.name}\n")
    except Exception as e:
        print(f"Warning: Could not create dummy file {file_path}. Error: {e}")


def parse_agent_output(response_text: str):
    """
    Parses the agent's final JSON output to extract the license URI and transaction hash.
    This version is robust and can handle multiple JSON formats the agent might output.
    """
    license_uri, tx_hash = None, None
    try:
        # The agent's final answer should be a JSON blob. Find and parse it.
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            print(f"Warning: No JSON object found in the final output.")
            return None, None

        data = json.loads(json_match.group(0))

        # --- NEW ROBUST LOGIC ---
        # Case 1: The agent followed instructions (preferred format)
        if "license_uri" in data and "transaction_details" in data:
            license_uri = data.get("license_uri")
            tx_details = data.get("transaction_details", {})
            tx_hash = tx_details.get("transaction_hash")
        # Case 2: The agent returned the raw tool output (flat format)
        elif "transaction_hash" in data:
            tx_hash = data.get("transaction_hash")
            # License URI is missing in this format, so it remains None.
            # The evaluation will correctly mark this as an error.
        else:
            print(f"Warning: Could not find 'transaction_hash' in the final JSON output.")

        # Ensure the hash has the '0x' prefix for web3.py
        if tx_hash and not tx_hash.startswith('0x'):
            tx_hash = '0x' + tx_hash

    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Warning: Could not parse final JSON output. Error: {e}.")
        print(f"--- Raw Text Start ---\n{response_text}\n--- Raw Text End ---")

    return license_uri, tx_hash


# --- REPLACE THIS FUNCTION WITH THE NEW VERSION ---
def get_tx_details(tx_hash_str: str):
    """Fetches transaction receipt with retries to get gas usage and cost."""
    if not isinstance(tx_hash_str, str) or not tx_hash_str.startswith('0x'):
        return None, None

    max_retries = 3
    retry_delay_seconds = 10
    for attempt in range(max_retries):
        try:
            # Pass the hex string directly. Do NOT convert to bytes.
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash_str, timeout=180)

            gas_used = receipt.get('gasUsed')
            effective_gas_price = receipt.get('effectiveGasPrice')

            if gas_used is not None and effective_gas_price is not None:
                cost_in_wei = gas_used * effective_gas_price
                cost_in_eth = w3.from_wei(cost_in_wei, 'ether')
                return gas_used, cost_in_eth
            else:
                return gas_used, None
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for {tx_hash_str}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)
            else:
                print(f"All retries failed for {tx_hash_str}.")
                return None, None


# --- END OF REPLACEMENT ---


# --- REPLACE THIS FUNCTION WITH THE NEW VERSION ---
def run_evaluation_suite(scenarios):
    """Runs the full evaluation suite and saves results to a CSV."""
    print("\n======== 🚀 Starting AIGC Framework Evaluation Suite 🚀 ========")
    results_data = []
    agent_state = _AgentState.get()
    for scenario in scenarios:
        ensure_dummy_content_file(scenario["content_file"])

    for i, scenario in enumerate(scenarios):
        agent_state.reset()
        print(f"\n--- 🏃 Running Scenario {i + 1}/{len(scenarios)}: {scenario['scenario_id']} ---")
        agent_input_query = f"""
                Register the AIGC file: '{Path(scenario["content_file"]).as_posix()}'.
                Creator Address: {scenario['creator_wallet']}.
                Asset Name: '{scenario['asset_name']}'.
                Asset Description: '{scenario['asset_description']}'.
                Use the standard tool sequence to perform the registration. For the license query, extract the most specific and restrictive terms from the asset description (e.g., 'exclusive rights', 'non-commercial', 'public domain') to create a precise search query.
                """
        start_time = time.time()
        final_output_text = ""
        failed_step = "Step 1: Upload"  # Default failure step

        try:
            response = agent_executor.invoke({"input": agent_input_query})
            final_output_text = response.get("output", "")
            # Check state to determine success level
            if agent_state.has_minted_nft:
                failed_step = "Success"
            elif agent_state.has_created_metadata:
                failed_step = "Failed at Step 4: Mint"
            elif agent_state.has_queried_policies:
                failed_step = "Failed at Step 3: Metadata"
            elif agent_state.has_uploaded_content:
                failed_step = "Failed at Step 2: Query"

        except Exception as e:
            print(f"!!!!!!!! ❌ ERROR during agent execution for {scenario['scenario_id']} !!!!!!!!")
            print(f"Exception Type: {type(e).__name__}, Message: {e}")
            final_output_text = f"Agent execution failed: {e}"
            # Check state to see where it failed
            if agent_state.has_created_metadata:
                failed_step = "Failed at Step 4: Mint"
            elif agent_state.has_queried_policies:
                failed_step = "Failed at Step 3: Metadata"
            elif agent_state.has_uploaded_content:
                failed_step = "Failed at Step 2: Query"

        end_time = time.time()
        latency = end_time - start_time
        selected_license, tx_hash = parse_agent_output(final_output_text)
        gas_used, gas_cost_eth = get_tx_details(tx_hash)

        def _canon(uri: str) -> str:
            return uri.strip().lower().rstrip('/')

        is_correct = int(
            selected_license is not None and
            _canon(selected_license) == _canon(scenario["expected_license_uri"])
        )

        results_data.append({
            "scenario_id": scenario["scenario_id"],
            "final_status": failed_step,  # Add the new status field
            "latency_seconds": latency,
            "expected_license": scenario["expected_license_uri"],
            "selected_license": selected_license,
            "is_correct": is_correct,
            "tx_hash": tx_hash,
            "gas_used": gas_used,
            "gas_cost_eth": gas_cost_eth
        })
        print(
            f"✅ Result: Status={failed_step}, CorrectLicense={bool(is_correct)}, Latency={latency:.2f}s, Gas Used={gas_used}")

    csv_file = 'evaluation_results.csv'
    # This will dynamically get all columns, including the new 'final_status'
    csv_columns = results_data[0].keys() if results_data else []
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(results_data)
        print(f"\n======== 🎉 Evaluation Complete. Results saved to {csv_file} 🎉 ========")
    except IOError:
        print("I/O error writing CSV file.")


# --- END OF REPLACEMENT ---


if __name__ == "__main__":
    synthetic_dataset = load_scenarios_from_csv()
    if synthetic_dataset:
        run_evaluation_suite(synthetic_dataset)