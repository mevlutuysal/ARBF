# agent_registration.py

import json
from web3 import Web3
from shared_utils import IngestionData, PolicyData, RegistrationData


def run_registration(
        ingestion_data: IngestionData,
        policy_data: PolicyData,
        scenario_data: dict,
        components: dict
) -> RegistrationData:
    """
    Performs the final registration and minting.
    This logic is adapted from `create_and_upload_metadata_tool` and `mint_nft_tool`
    in your original agent_main.py.

    Args:
        ingestion_data: The output from the Ingestion Agent.
        policy_data: The output from the Policy Agent.
        scenario_data: The original scenario data (for name, description, etc.).
        components: The dictionary of initialized components from shared_utils.

    Returns:
        A RegistrationData object with the final transaction details.
    """
    print("\n[Registration Agent] Starting metadata creation and minting...")

    # --- Unpack components ---
    ipfs_manager = components["ipfs_storage"]
    w3 = components["w3"]
    contract = components["aigc_contract"]
    agent_account = components["agent_account"]

    # --- 1. Create and Upload Metadata ---
    print("[Registration Agent] Assembling and uploading metadata...")
    metadata_dict = ipfs_manager.create_metadata(
        name=scenario_data['asset_name'],
        description=scenario_data['asset_description'],
        creator_address=scenario_data['creator_wallet'],
        content_cid=ingestion_data.content_cid,
        sha256_hash=ingestion_data.sha256_hash,
        phash=ingestion_data.phash,
        license_uri=policy_data.selected_license_uri
    )
    metadata_cid = ipfs_manager.upload_metadata_json(metadata_dict)
    if not metadata_cid:
        raise ConnectionError("Registration Agent Error: Failed to upload metadata to IPFS.")
    print(f"[Registration Agent] Metadata uploaded. CID: {metadata_cid}")



    # --- 2. Construct and Send Blockchain Transaction ---
    print("[Registration Agent] Building and sending blockchain transaction...")
    recipient_address = Web3.to_checksum_address(scenario_data['creator_wallet'])
    token_uri = f"ipfs://{metadata_cid}"

    # Convert CAR CID to bytes32 for the smart contract
    car_cid_bytes = ingestion_data.content_cid.encode('utf-8')
    if len(car_cid_bytes) > 32:
        car_cid_bytes32 = car_cid_bytes[:32]
    else:
        car_cid_bytes32 = car_cid_bytes.ljust(32, b'\0')

    # Build transaction using logic from original `mint_nft_tool`
    nonce = w3.eth.get_transaction_count(agent_account.address)
    tx_params = {'from': agent_account.address, 'nonce': nonce, 'chainId': w3.eth.chain_id}

    try:
        estimated_gas = contract.functions.safeMintWithCARCID(
            recipient_address, token_uri, car_cid_bytes32
        ).estimate_gas(tx_params)
        tx_params['gas'] = int(estimated_gas * 1.2)
    except Exception:
        tx_params['gas'] = 500000  # Fallback

    transaction = contract.functions.safeMintWithCARCID(
        recipient_address, token_uri, car_cid_bytes32
    ).build_transaction(tx_params)

    signed_tx = w3.eth.account.sign_transaction(transaction, agent_account.key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)

    print(f"[Registration Agent] Transaction sent. Hash: {tx_hash.hex()}. Waiting for receipt...")
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)

    # --- 3. Parse Receipt and Return ---
    status = "Success" if tx_receipt.status == 1 else "Failed"
    token_id = tx_receipt.get('tokenId', -1)
    gas_used = tx_receipt.get('gasUsed', 0)
    # Capture the effective gas price from the receipt (in Wei)
    effective_gas_price = tx_receipt.get('effectiveGasPrice', 0)


    # --- FIX START: Use robust event processing ---
    try:
        # Use the contract ABI to automatically find and parse the event
        processed_logs = contract.events.TokenRegistered().process_receipt(tx_receipt)
        if processed_logs:
            # Access the event argument by name, e.g., 'tokenId'
            # Note: The argument name must match what's in your Solidity event definition.
            token_id = str(processed_logs[0]['args']['tokenId'])
        else:
            print("Warning: TokenRegistered event not found in transaction receipt.")
    except Exception as e:
        print(f"Warning: Could not decode token ID from event logs using process_receipt: {e}")
    # --- FIX END ---

    print(f"[Registration Agent] Transaction confirmed. Status: {status}, Token ID: {token_id}")

    return RegistrationData(
        metadata_cid=metadata_cid,
        transaction_hash=tx_hash.hex(),
        token_id=token_id,
        block_number=tx_receipt.blockNumber,
        status=status,
        gas_used=gas_used,
        effective_gas_price=effective_gas_price
    )