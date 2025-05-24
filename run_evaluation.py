import os
import sys
import time  # Added for potential delays or unique naming
from pathlib import Path

# Ensure the 'integration' package and other modules from agent_main can be found.
# This assumes the test script is in the same directory as agent_main.py
# or that the project structure allows Python to find agent_main and its imports.
try:
    import agent_main  # This will run all initializations in agent_main.py
except ImportError as e:
    print(f"Error: Could not import agent_main.py. Ensure it's in the Python path: {e}")
    print("Make sure you run this test script from the project root directory where agent_main.py is located.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during the import and initialization of agent_main.py: {e}")
    print(
        "Please check your .env configuration and ensure all services (Ollama, ChromaDB, Sepolia RPC) are accessible.")
    sys.exit(1)

# --- Configuration & Dataset ---

# Use the agent's address from the initialized agent_main module
# This replaces the EVAL_AGENT_WALLET_ADDRESS placeholder from the dataset.txt
EVAL_AGENT_WALLET_ADDRESS = agent_main.agent_account.address
print(f"Using Evaluation Agent Wallet Address: {EVAL_AGENT_WALLET_ADDRESS}")

# The synthetic dataset from your dataset.txt
# EVAL_AGENT_WALLET_ADDRESS will be substituted by the actual address above.
synthetic_dataset = [
    {
        "scenario_id": "SCENARIO-001",
        "content_file": "test_content_files/scenario001_wearable.png",
        "asset_name": "Nova Prime Virtual Jacket",
        "asset_description": "A futuristic virtual jacket for avatars, intended for multiple sales on the marketplace. Standard commercial terms for digital assets should apply, allowing buyers to use but not resell the design IP.",
        "creator_wallet": EVAL_AGENT_WALLET_ADDRESS,  # Placeholder will be replaced
        "expected_license_uri": "https://metaverse.yourplatform.com/licenses/standard-commercial-v1"
    },
    {
        "scenario_id": "SCENARIO-002",
        "content_file": "test_content_files/scenario002_event_art.jpg",
        "asset_name": "MetaFest 2025 Promo Art",
        "asset_description": "Promotional artwork for the MetaFest 2025 community event. This image is free to share and display widely for any purpose, even commercially. Please provide attribution to the original creator if used.",
        "creator_wallet": EVAL_AGENT_WALLET_ADDRESS,
        "expected_license_uri": "https://creativecommons.org/licenses/by/4.0/"
    },
    {
        "scenario_id": "SCENARIO-003",
        "content_file": "test_content_files/scenario003_npc_dialogue.txt",
        "asset_name": "NPC Quest Giver Dialogue - The Lost Artifact",
        "asset_description": "Exclusive interactive dialogue script for the NPC 'Guardian Alatar' in the 'Chronicles of Etheria' game. This content is licensed exclusively to 'Etheria Game Studios' for their commercial game, with royalties based on game sales.",
        "creator_wallet": EVAL_AGENT_WALLET_ADDRESS,
        "expected_license_uri": "https://metaverse.yourplatform.com/licenses/exclusive-commercial-royalty-v1"
    },
    {
        "scenario_id": "SCENARIO-004",
        "content_file": "test_content_files/scenario004_arch_element.glb",  # .glb is a 3D model format
        "asset_name": "Modular Sci-Fi Corridor Section",
        "asset_description": "A 3D model of a modular sci-fi corridor section. Offered completely free for any use, including commercial projects and modifications, with no restrictions. Public domain dedication intended.",
        "creator_wallet": EVAL_AGENT_WALLET_ADDRESS,
        "expected_license_uri": "https://creativecommons.org/publicdomain/zero/1.0/"
    },
    {
        "scenario_id": "SCENARIO-005",
        "content_file": "test_content_files/scenario005_music_loop.mp3",
        "asset_name": "Cyberpunk Alley Ambient Loop",
        "asset_description": "A seamless ambient background music loop with a cyberpunk theme. For sale as a stock audio asset for non-exclusive use in various Metaverse experiences and games. Royalty-free use after initial purchase.",
        "creator_wallet": EVAL_AGENT_WALLET_ADDRESS,
        "expected_license_uri": "https://metaverse.yourplatform.com/licenses/nonexclusive-royaltyfree-music-v1"
    },
    {
        "scenario_id": "SCENARIO-006",
        "content_file": "test_content_files/scenario006_profile_pic.png",
        "asset_name": "My Custom Avatar Pic",
        "asset_description": "A custom-generated profile picture for my personal avatar. This image is for my personal use only and is not intended for sale, public distribution, or any commercial activity.",
        "creator_wallet": EVAL_AGENT_WALLET_ADDRESS,
        "expected_license_uri": "https://metaverse.yourplatform.com/licenses/personal-use-only-v1"
    },
    {
        "scenario_id": "SCENARIO-007",
        "content_file": "test_content_files/scenario007_unique_sword.glb",  # .glb is a 3D model format
        "asset_name": "Blade of the Cosmos (Unique NFT)",
        "asset_description": "A unique, one-of-a-kind 3D model of the legendary 'Blade of the Cosmos'. This item is sold as a unique digital collectible (NFT). The buyer will own this specific instance and can use it commercially, for example, in monetized game streams or as a key item in their own game.",
        "creator_wallet": EVAL_AGENT_WALLET_ADDRESS,
        "expected_license_uri": "https://metaverse.yourplatform.com/licenses/unique-collectible-commercial-owner-v1"
    }
]

# Update creator_wallet in the dataset to the actual address
for scenario_data in synthetic_dataset:
    scenario_data["creator_wallet"] = EVAL_AGENT_WALLET_ADDRESS


# --- Helper Function to Create Dummy Content Files ---
def ensure_dummy_content_file(file_path: str):
    """
    Creates an empty dummy file if it doesn't exist.
    Creates parent directories if they don't exist.
    """
    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                if file_path.endswith((".png", ".jpg", ".jpeg", ".gif")):
                    f.write("This is a dummy image file.")  # Simple text for non-binary
                elif file_path.endswith(".glb"):
                    f.write("This is a dummy GLB 3D model file.")  # Simple text
                elif file_path.endswith(".mp3"):
                    f.write("This is a dummy MP3 audio file.")  # Simple text
                else:
                    f.write(f"This is a dummy content file for testing: {os.path.basename(file_path)}\n")
            print(f"Created dummy file: {file_path}")
        # else:
        #     print(f"Dummy file already exists: {file_path}") # Optional: for verbosity
    except Exception as e:
        print(f"Warning: Could not create dummy file {file_path}. Error: {e}")
        print("Please ensure you have write permissions and the path is valid.")
        print("The agent might fail if it cannot find the content file.")


# --- Test Execution ---
def run_tests():
    """
    Runs the AIGC registration agent for each scenario in the dataset.
    """
    print("\n======== Starting AIGC Framework Test Suite ========")
    if not agent_main.agent_executor:
        print("Error: agent_executor from agent_main.py is not available. Exiting.")
        return

    # Create the main directory for test content files
    base_content_dir = "test_content_files"
    if not os.path.exists(base_content_dir):
        os.makedirs(base_content_dir, exist_ok=True)
        print(f"Created base directory for test content: {base_content_dir}")

    for i, scenario in enumerate(synthetic_dataset):
        print(f"\n--- Test Scenario {i + 1}/{len(synthetic_dataset)}: {scenario['scenario_id']} ---")


        content_file_to_process = Path(scenario["content_file"]).as_posix()
        creator_wallet_address = scenario["creator_wallet"]  # Already updated
        asset_name_input = scenario["asset_name"]
        asset_description_input = scenario["asset_description"]
        # expected_license = scenario["expected_license_uri"] # For future validation

        # Ensure the dummy content file exists for the agent to pick up
        ensure_dummy_content_file(content_file_to_process)

        # Construct the input query for the agent, mirroring agent_main.py's example input format.
        # The agent's prompt in agent_main.py hardcodes "For the license query, search for 'default policy'".
        # This test script will adhere to that, so the agent's behavior regarding license selection
        # will depend on that hardcoded query and the contents of the vector DB for "default policy".
        agent_input_query = f"""
        Register the AIGC file: '{content_file_to_process}'.
        Creator Address: {creator_wallet_address}.
        Asset Name: '{asset_name_input}'.
        Asset Description: '{asset_description_input}' non-exclusive multiple sales.
        Use the standard tool sequence to perform the registration and report the outcome.
        For the license query (QueryGovernancePolicies tool - Step 2), use a query based on the asset description.
        """
        # Note: The 'expected_license_uri' is not directly used to guide the agent's policy query here,
        # as the agent_input format in agent_main.py specifies a fixed "default policy" search.
        # The 'expected_license_uri' could be used for manual verification of results or if the
        # agent/prompt is later updated to allow dynamic policy queries based on the description.

        print(f"Content File: {os.path.abspath(content_file_to_process)}")
        print(f"Creator: {creator_wallet_address}")
        print(f"Asset Name: {asset_name_input}")
        # print(f"Full Input to Agent:\n{agent_input_query}") # Uncomment for very verbose logging

        try:
            print(f"Invoking agent for scenario: {scenario['scenario_id']}...")
            # The agent_executor is already configured with verbose=True in agent_main.py
            response = agent_main.agent_executor.invoke({"input": agent_input_query})

            print(f"\n--- Agent Response for {scenario['scenario_id']} ---")
            if isinstance(response, dict) and "output" in response:
                print(response["output"])
            else:
                print(response)  # Print raw response if structure is unexpected

            # You could add assertions here later to check parts of the response,
            # e.g., if a transaction hash is present, or if the expected license (if determinable) was mentioned.
            # For now, this script focuses on running the pipeline and getting an output.

        except Exception as e:
            print(f"\n!!!!!!!! ERROR during agent execution for {scenario['scenario_id']} !!!!!!!!")
            print(f"An unexpected error occurred: {type(e).__name__}: {e}")
            # If agent_main.py's OutputParserException handling is robust, it might be caught there.
            # This catches other potential errors during the .invoke() call itself.

        print(f"--- Finished Test Scenario: {scenario['scenario_id']} ---")
        # Optional: Add a small delay if making many sequential blockchain transactions quickly
        # time.sleep(5) # e.g., 5 seconds

    print("\n======== AIGC Framework Test Suite Complete ========")


if __name__ == "__main__":
    # Before running tests, ensure all components from agent_main are ready.
    # The import of agent_main should have handled initializations.
    # A small delay might be good if some initializations are async or take time.
    print("Waiting a few seconds for all components from agent_main.py to fully initialize...")
    time.sleep(5)  # Give a moment for any background initializations if necessary

    # Check critical components from agent_main
    if not agent_main.llm:
        print("Error: LLM not initialized in agent_main. Cannot run tests.")
        sys.exit(1)
    if not agent_main.vector_db or not agent_main.vector_db.is_connected():
        print("Error: Vector DB not initialized or connected in agent_main. Cannot run tests.")
        sys.exit(1)
    if not agent_main.ipfs_storage:
        print("Error: IPFS Storage not initialized in agent_main. Cannot run tests.")
        sys.exit(1)
    if not agent_main.w3 or not agent_main.w3.is_connected():
        print("Error: Web3 not initialized or connected in agent_main. Cannot run tests.")
        sys.exit(1)
    if not agent_main.aigc_contract:
        print("Error: AIGC Contract not initialized in agent_main. Cannot run tests.")
        sys.exit(1)

    print("All prerequisite components from agent_main appear to be initialized.")
    run_tests()
