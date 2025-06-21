# shared_utils.py

import os
import json
import sys

from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from web3 import Web3
import warnings
# Silence the noisy warning that Web3.py throws
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*MismatchedABI.*"
)

# LangChain/AI Components
from langchain_ollama import OllamaLLM

# Local Integration Components
try:
    from integration.ipfs_storage_manager import IPFSStorageManager
    from integration import utils as integration_utils
except ImportError:
    print("Error: Could not import from 'integration' package.")
    sys.exit(1)


# --- Pydantic Data Models for Agent Communication ---

class IngestionData(BaseModel):
    """Data output from the Ingestion Agent."""
    content_cid: str = Field(description="The CID of the content's CAR file from IPFS.")
    sha256_hash: str = Field(description="The SHA256 hash of the content file.")
    phash: Optional[str] = Field(description="The perceptual hash (phash) of the content, if applicable.")


class PolicyData(BaseModel):
    """Data output from the Policy Agent."""
    selected_license_uri: str = Field(description="The URI of the license selected via the RAG process.")


class RegistrationData(BaseModel):
    """Data output from the Registration Agent."""
    metadata_cid: str
    transaction_hash: str
    token_id: str
    block_number: int
    status: str
    gas_used : int
    effective_gas_price: int = Field(description="The final gas price per unit in Wei.")


# --- Centralized Component Initializer ---

def initialize_components():
    """Loads config from .env and initializes all shared components."""
    print("Loading configuration from .env file...")
    load_dotenv()

    # --- Configuration Loading ---
    SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL")
    AGENT_PRIVATE_KEY = os.getenv("AGENT_PRIVATE_KEY")
    PINATA_JWT_TOKEN = os.getenv("PINATA_JWT")
    AIGCREGISTRY_CONTRACT_ADDRESS = os.getenv("AIGCREGISTRY_CONTRACT_ADDRESS")
    ABI_FILE_PATH = "AIGCRegistry_abi.json"
   # LLM_MODEL = "llama3.1:8b-instruct-q6_K"
    LLM_MODEL = "qwen2.5:14b-instruct-q5_K_M"
    #LLM_MODEL = "llama3.3:70b-instruct-q5_K_M"

    # --- Validation ---
    if not all([SEPOLIA_RPC_URL, AGENT_PRIVATE_KEY, PINATA_JWT_TOKEN, AIGCREGISTRY_CONTRACT_ADDRESS]):
        raise ValueError("Missing one or more required environment variables in .env")
    if not os.path.exists(ABI_FILE_PATH):
        raise FileNotFoundError(f"ABI file not found at: {ABI_FILE_PATH}")
    print("Configuration loaded successfully.")

    # --- Component Initialization ---
    print("Initializing components...")

    # IPFS Manager
    ipfs_storage = IPFSStorageManager(jwt_token=PINATA_JWT_TOKEN)
    ipfs_storage.utils = integration_utils


    # Web3 / Blockchain
    w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))
    if not w3.is_connected():
        raise ConnectionError(f"Failed to connect to Sepolia RPC: {SEPOLIA_RPC_URL}")
    agent_account = w3.eth.account.from_key(AGENT_PRIVATE_KEY)
    w3.eth.default_account = agent_account.address
    with open(ABI_FILE_PATH, 'r') as f:
        contract_abi = json.load(f)
    checksum_address = Web3.to_checksum_address(AIGCREGISTRY_CONTRACT_ADDRESS)
    aigc_contract = w3.eth.contract(address=checksum_address, abi=contract_abi)

    # LLM (only for policy agent)
    llm = OllamaLLM(model=LLM_MODEL)
    print("All components initialized successfully.")

    return {
        "ipfs_storage": ipfs_storage,
        "w3": w3,
        "agent_account": agent_account,
        "aigc_contract": aigc_contract,
        "llm": llm
    }