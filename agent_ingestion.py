# agent_ingestion.py

import os
import json
from integration.ipfs_storage_manager import IPFSStorageManager
from integration import utils as integration_utils
from shared_utils import IngestionData


def run_ingestion(
        file_path: str,
        ipfs_manager: IPFSStorageManager
) -> IngestionData:
    """
    Performs the ingestion and packaging stage.
    This logic is adapted from the `upload_content_tool` in your original agent_main.py.

    Args:
        file_path: The local path to the AIGC asset.
        ipfs_manager: An initialized instance of the IPFSStorageManager.

    Returns:
        An IngestionData object with the results.
    """
    print(f"\n[Ingestion Agent] Processing file: {file_path}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Ingestion Agent Error: File not found at {file_path}")

    # 1. Calculate Hashes
    sha256_hash = integration_utils.calculate_sha256(file_path)
    phash_val = integration_utils.calculate_phash(file_path)
    print(f"[Ingestion Agent] Hashes calculated: SHA256={sha256_hash}, pHash={phash_val}")

    # 2. Upload content to IPFS (which also creates the CAR file)
    content_cid, _, _ = ipfs_manager.upload_content(file_path)
    if not content_cid:
        raise ConnectionError("Ingestion Agent Error: IPFS upload failed. Check Pinata JWT and network.")

    print(f"[Ingestion Agent] Content uploaded to IPFS. CID: {content_cid}")

    # 3. Return structured data
    return IngestionData(
        content_cid=content_cid,
        sha256_hash=sha256_hash,
        phash=phash_val
    )