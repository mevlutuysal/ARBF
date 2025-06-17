# In agent_main.py

import os
import json
import sys
from pydantic import BaseModel, Field
from typing import List, Dict
from typing import Optional
from dotenv import load_dotenv
from web3 import Web3
from web3.exceptions import TransactionNotFound, MismatchedABI
from langchain.agents.react.output_parser import ReActOutputParser
from langchain.agents.agent import AgentOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain.schema import AgentAction
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain.schema import AgentAction, AgentFinish
import re
from dataclasses import dataclass, field
from typing import Optional
_POLICIES_CACHE: List[Dict] | None = None
# ---------------------------------------------------------------------------
# 1. Simple process‑wide state holder
# ---------------------------------------------------------------------------
@dataclass
# In agent_main.py

@dataclass
class _AgentState:
    """A simple, process-wide state manager for the agent."""
    last_content_cid: Optional[str] = field(default=None)
    last_metadata_cid: Optional[str] = field(default=None)
    # Add flags to track completed steps
    has_uploaded_content: bool = False
    has_queried_policies: bool = False
    has_created_metadata: bool = False
    has_minted_nft: bool = False
    # singleton accessor
    @classmethod
    def get(cls) -> "_AgentState":
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    def reset(self):
        """Resets the state for a new evaluation scenario."""
        print("--- Resetting agent state for new scenario ---")
        self.last_content_cid = None
        self.last_metadata_cid = None
        self.has_uploaded_content = False
        self.has_queried_policies = False
        self.has_created_metadata = False
        self.has_minted_nft = False

# --- ADD THESE PYDANTIC MODELS ---
class MetadataInput(BaseModel):
    """Pydantic model for the CreateAndUploadMetadata tool's input."""
    name: str = Field(description="The name of the asset.")
    description: str = Field(description="A detailed description of the asset.")
    content_cid: str = Field(description="The CID of the content file from UploadContentToIPFS.")
    creator_address: str = Field(description="The wallet address of the creator.")
    license_uri: str = Field(description="The URI of the license selected from QueryGovernancePolicies.")
    sha256_hash: str = Field(description="The SHA256 hash from UploadContentToIPFS.")
    phash: Optional[str] = Field(description="The perceptual hash (phash) from UploadContentToIPFS, can be null.")

class MintInput(BaseModel):
    """Pydantic model for the MintAIGCNFT tool's input."""
    recipient_address: str = Field(description="The creator's wallet address who will receive the NFT.")
    metadata_cid: str = Field(description="The CID of the metadata JSON file from CreateAndUploadMetadata.")
    content_cid: str = Field(description="The original content CID (CAR file CID) from UploadContentToIPFS.")
# --- END OF ADDED SECTION ---


class CleanReActOutputParser(AgentOutputParser):
    """
    1. Strip random chatter the model sometimes inserts between
       `Action:` and `Action Input:` lines (or Observation → Thought).
    2. Try LangChain’s official ReActOutputParser.
    3. If that fails, first look for a lenient Action/Input pair.
    4. If no Action exists, look for a Final Answer block and finish.
    """
    _inner = ReActOutputParser()

    # remove “Thought:” lines that sneak between Action and Action Input
    _strip_between = re.compile(
        r"(Action:\s*\w+)\s*\n(?:[^\n]*\n)*?Action Input:",
        re.MULTILINE,
    )

    _action_fallback = re.compile(
        # This now captures a single word (\w+) as the action
        # and allows for trailing whitespace (\s*) before the newline.
        r"\*?\*?Action\*?\*?:\s*(?P<action>\w+)\s*\n"
        r"\*?\*?Action Input\*?\*?:\s*(?P<input>.+)",
        re.IGNORECASE | re.DOTALL,
    )

    _finish_fallback = re.compile(
        r"\*?\*?Final Answer\*?\*?:\s*(?P<answer>.+)",
        re.IGNORECASE | re.DOTALL,
    )

    def parse(self, text: str):
        clean = self._strip_between.sub(r"\1\nAction Input:", text)

        # 1) normal strict parse
        try:
            return self._inner.parse(clean)
        except OutputParserException:
            pass  # drop to our own fallbacks

        # 2) lenient Action / Action Input pair
        m = self._action_fallback.search(clean)
        if m:
            return AgentAction(
                tool=m.group("action").strip(),
                tool_input=m.group("input").strip(),
                log=text,
            )

        # 3) lenient Final Answer
        m = self._finish_fallback.search(clean)
        if m:
            return AgentFinish(
                return_values={"output": m.group("answer").strip()},
                log=text,
            )

        # still nothing? bubble the error up.
        raise OutputParserException(
            "Could not parse as Action or Final Answer:\n" + clean
        )
# Corrected import path for web3.py v6.x (standard location)
# from web3.middleware import geth_poa_middleware # For POA networks like Sepolia
# Note: User confirmed geth_poa_middleware import is not needed for web3 v7


# LangChain components
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent, Tool
# Using a more structured prompt format might help some models
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.exceptions import OutputParserException

# Import managers from the integration package
# Ensure the 'integration' directory is in the Python path
# If running agent_main.py from the root, this should work if integration is also in the root.
try:
    from integration.vector_db_manager import VectorDBManager
    from integration.ipfs_storage_manager import IPFSStorageManager
    # Assuming utils.py is inside integration package now based on previous steps
    from integration import utils as integration_utils
except ImportError:
    print("Error: Could not import from 'integration' package.")
    print("Ensure 'agent_main.py' is run from the project root directory")
    print("and the 'integration' directory exists with __init__.py and utils.py.")
    sys.exit(1)


def clean_llm_json_input(data):
    """Strips markdown backticks from a dictionary's string values."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                # Strip markdown and leading/trailing whitespace
                data[key] = value.strip().strip('`')
    return data


# --- Configuration Loading ---
print("Loading configuration from .env file...")
load_dotenv()

SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL")
AGENT_PRIVATE_KEY = os.getenv("AGENT_PRIVATE_KEY")
# Updated to load Pinata JWT
PINATA_JWT_TOKEN = os.getenv("PINATA_JWT")
AIGCREGISTRY_CONTRACT_ADDRESS = os.getenv("AIGCREGISTRY_CONTRACT_ADDRESS")
ABI_FILE_PATH = "AIGCRegistry_abi.json"  # Path to your ABI file

# --- Basic Configuration Validation ---
# Updated to check PINATA_JWT_TOKEN
if not all([SEPOLIA_RPC_URL, AGENT_PRIVATE_KEY, PINATA_JWT_TOKEN, AIGCREGISTRY_CONTRACT_ADDRESS]):
    raise ValueError(
        "Missing one or more required environment variables (SEPOLIA_RPC_URL, AGENT_PRIVATE_KEY, PINATA_JWT, AIGCREGISTRY_CONTRACT_ADDRESS)")

if not AGENT_PRIVATE_KEY.startswith("0x"):
    raise ValueError("AGENT_PRIVATE_KEY must start with 0x")

# Check Pinata JWT format (basic check)
if not isinstance(PINATA_JWT_TOKEN, str) or not PINATA_JWT_TOKEN.startswith("ey"):
    print(
        "Warning: PINATA_JWT in .env file doesn't look like a standard JWT key. Please double-check it on Pinata website.")

if not os.path.exists(ABI_FILE_PATH):
    raise FileNotFoundError(f"ABI file not found at: {ABI_FILE_PATH}")

print("Configuration loaded successfully.")

# --- Component Initialization ---
print("Initializing components...")

# LLM (Ollama)
# Make sure Ollama server is running and the model is downloaded
# docker exec ollama_server ollama list
LLM_MODEL = "llama3.1:8b-instruct-q6_K"  # User changed to llama3
print(f"Initializing LLM: {LLM_MODEL} via Ollama")
try:
    # Note: Suppressing the specific deprecation warning for clarity during execution
    import warnings
    from langchain_core._api.deprecation import LangChainDeprecationWarning

    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

    llm = Ollama(model=LLM_MODEL, timeout=60)
    # Test connection
    # llm.invoke("Respond with 'OK' if you are working.") # Keep this commented out for faster test runs
    print("LLM connection successful.")
except Exception as e:
    print(f"Error initializing or connecting to LLM via Ollama: {e}")
    print(
        f"Ensure Ollama Docker container is running and the model '{LLM_MODEL}' is available (use 'docker exec ollama_server ollama list').")
    sys.exit(1)

# Vector DB Manager
print("Initializing VectorDBManager...")
try:
    # 1. Initialize the manager. This connects to the DB service.
    vector_db = VectorDBManager(
        collection_name="aigc_governance_policies",
        host="localhost",
        port=8000
    )

    # 2. Get the collection object. The agent needs this to perform queries.
    #    The `chroma.py` script is responsible for creating and populating it.
    print(f"Attempting to get collection: '{vector_db.collection_name}'...")
    vector_db.collection = vector_db.client.get_collection(name=vector_db.collection_name)
    print("VectorDBManager initialized and collection handle retrieved successfully.")

except Exception as e:
    print(f"Error initializing VectorDBManager: {e}")
    print("Ensure the ChromaDB Docker container is running and accessible,")
    print("and that the 'chroma.py' setup script has been run successfully.")
    sys.exit(1)


# IPFS Storage Manager (Now using Pinata)
print("Initializing IPFSStorageManager for Pinata...")
try:
    # Pass the utils module to the storage manager if needed, or ensure imports work
    ipfs_storage = IPFSStorageManager(jwt_token=PINATA_JWT_TOKEN)  # MODIFIED HERE
    # Add utils reference if methods were moved there in IPFSStorageManager implementation
    ipfs_storage.utils = integration_utils  # Assuming your utils are here
    print("IPFSStorageManager (Pinata) initialized successfully.")
except Exception as e:
    print(f"Error initializing IPFSStorageManager (Pinata): {e}")
    sys.exit(1)

# Web3 / Blockchain Connection
print(f"Connecting to Sepolia network via: {SEPOLIA_RPC_URL}")
try:
    w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))
    # Inject POA middleware needed for Sepolia (User confirmed not needed for v7)
    # w3.middleware_onion.inject(geth_poa_middleware, layer=0)

    if not w3.is_connected():
        raise ConnectionError(f"Failed to connect to Sepolia RPC: {SEPOLIA_RPC_URL}")

    # Set up account using private key
    agent_account = w3.eth.account.from_key(AGENT_PRIVATE_KEY)
    w3.eth.default_account = agent_account.address
    print(f"Web3 connected. Agent Address: {agent_account.address}")

    # Load Contract ABI and create Contract instance
    print(f"Loading contract ABI from: {ABI_FILE_PATH}")
    with open(ABI_FILE_PATH, 'r') as f:
        contract_abi = json.load(f)

    # Validate and checksum the address
    checksum_contract_address = Web3.to_checksum_address(AIGCREGISTRY_CONTRACT_ADDRESS)
    print(f"Creating contract instance for address: {checksum_contract_address}")
    aigc_contract = w3.eth.contract(address=checksum_contract_address, abi=contract_abi)
    print("Contract instance created successfully.")

except FileNotFoundError:
    print(f"Error: ABI file not found at {ABI_FILE_PATH}")
    sys.exit(1)
except ConnectionError as e:
    print(f"Web3 connection error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error initializing Web3 or Contract: {e}")
    sys.exit(1)

# --- Tool Definitions ---
print("Defining agent tools...")


# Helper function to safely parse JSON observation from tools
def parse_tool_observation(observation_str: str) -> dict | None:
    try:
        # Try loading directly if it's already JSON
        if isinstance(observation_str, dict):
            return observation_str
        # Clean potential markdown code blocks
        cleaned_str = re.sub(r'^```json\s*|\s*```$', '', observation_str, flags=re.MULTILINE).strip()
        return json.loads(cleaned_str)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse tool observation as JSON: {observation_str}")
        # Attempt to extract JSON if it's embedded
        match = re.search(r'(\{.*\})', observation_str, re.DOTALL)
        if match:
            try:
                extracted_json_str = match.group(1)
                print(f"Attempting to parse extracted JSON substring: {extracted_json_str[:200]}...")
                return json.loads(extracted_json_str)
            except json.JSONDecodeError:
                print(f"Warning: Still could not parse extracted JSON substring.")
                return None
        return None
    except Exception as e:
        print(f"Warning: Unexpected error parsing tool observation: {e}")
        return None

# In agent_main.py, inside upload_content_tool
def _extract_file_path(raw: str) -> str:
    # Take only the first line in case of multi-line input from LLM
    cleaned = raw.splitlines()[0] if '\n' in raw else raw
    cleaned = cleaned.strip().strip("'\"")  # remove outer quotes
    cleaned = re.sub(r'\s*\(.*$', '', cleaned)  # drop parenthetical note
    cleaned = cleaned.rstrip("'\"")
    return cleaned.strip()

def upload_content_tool(file_path_input: str) -> str:
    """
    Tool to upload a content file (image, text, etc.) to IPFS via Pinata.
    It calculates SHA256 and pHash (for images).
    Returns a JSON string containing 'content_cid' (string or null), 'sha256_hash' (string or null), and 'phash' (string or null).
    Input MUST be the exact local file path string (e.g., 'my_image.png' or './path/to/file.txt').
    The tool will attempt to extract the path if extra text is included in the input.
    If upload fails (e.g., due to bad API key), 'content_cid' will be null and an 'error' key will be included.
    """
    state = _AgentState.get()
    if state.has_uploaded_content:
        return json.dumps(
            {"error": "Content has already been uploaded. Do not use this tool again. Proceed to the next step."})

    print("\n[Tool] === Uploading Content ===")
    print(f"Raw input: {file_path_input!r}")

    extracted_path = _extract_file_path(file_path_input)
    print(f"Using path: {extracted_path!r}")

    # Make an absolute path (helps when agent called from elsewhere)
    if not os.path.isabs(extracted_path):
        extracted_path = os.path.abspath(extracted_path)

    if not os.path.isfile(extracted_path):
        err = (f"File not found (or is a directory) at: {extracted_path}. "
               "Ensure the full path including filename is supplied.")
        print(err)
        return json.dumps({"error": err})

    # Proceed with the existing logic using the extracted_path
    sha256_hash = None
    phash_val = None
    content_cid = None
    error_msg = None
    try:
        sha256_hash = integration_utils.calculate_sha256(extracted_path)
        phash_val = integration_utils.calculate_phash(extracted_path)
        print(f"Calculated Hashes: SHA256={sha256_hash}, pHash={phash_val}")

        content_cid, _, _ = ipfs_storage.upload_content(extracted_path)

        if content_cid is None:
            error_msg = "IPFS upload failed (Pinata). upload_content returned None for CID. Check API Key (JWT) and network."
            print(f"Warning: {error_msg}")

    except FileNotFoundError:
        error_msg = f"Error: File not found at path during processing: {extracted_path}"
        print(error_msg)
    except Exception as e:
        error_msg = f"Error during content upload processing for '{extracted_path}': {type(e).__name__}: {e}"
        print(f"Error in upload_content_tool: {error_msg}")

    result = {
        "content_cid": content_cid,
        "sha256_hash": sha256_hash,
        "phash": phash_val,
        "error": error_msg if error_msg else None
    }
    if result["error"] is None:
        del result["error"]

    state = _AgentState.get()
    state.last_content_cid = content_cid
    state.last_sha256_hash = sha256_hash
    state.last_phash = phash_val
    state.has_uploaded_content = True
    print(f"UploadContentToIPFS Observation: {json.dumps(result)}")
    return json.dumps(result)


def query_policies_tool(query: str) -> str:
    """
    Tool to query the vector database for relevant governance policies or license information.
    Input is a natural language query about policies (e.g., 'default license for commercial art').
    Returns a JSON string representation of the query results (list of documents with id, document text, metadata, distance), or an empty list '[]' if no results, or a JSON object with an 'error' key on failure.
    """
    state = _AgentState.get()
    if not state.has_uploaded_content:
        return json.dumps({
                              "error": "Workflow violation. You must use UploadContentToIPFS (Step 1) before querying policies (Step 2)."})
    if state.has_queried_policies:
        return json.dumps({"error": "Policies already queried. Proceed to Step 3."})

    # Clean the query input to remove potential extraneous lines from LLM
    cleaned_query = query.splitlines()[0].strip() if '\n' in query else query.strip()
    cleaned_query = cleaned_query.strip("'\"") # Also remove quotes

    print(f"Cleaned Query: '{cleaned_query}'")

    if not isinstance(cleaned_query, str) or not cleaned_query:  # Check if cleaned_query is empty
        return json.dumps({"error": "Invalid or empty query provided for policies."})



    try:
        results = vector_db.query_policies(query_text=cleaned_query, n_results=5)  # Get top 5 results
        if results:
            processed_results = []
            ids_list = results.get('ids', [[]])[0]
            docs_list = results.get('documents', [[]])[0]
            metadatas_list = results.get('metadatas', [[]])[0]
            distances_list = results.get('distances', [[]])[0]

            if ids_list:
                for i, doc_id in enumerate(ids_list):
                    processed_results.append({
                        "id": doc_id,
                        "document": docs_list[i] if i < len(docs_list) else None,
                        "metadata": metadatas_list[i] if i < len(metadatas_list) else None,
                        "distance": distances_list[i] if i < len(distances_list) else None
                    })
                print(f"QueryPoliciesTool Observation: {json.dumps(processed_results)}")
                state.has_queried_policies = True
                _AgentState.get().policies_cache = processed_results
                return json.dumps(processed_results)
            else:
                print("QueryPoliciesTool Observation: [] (No results found)")
                return "[]"
        else:
            print("QueryPoliciesTool Observation: [] (Results object was None or empty)")
            return "[]"
    except Exception as e:
        error_msg = f"Error querying vector database: {type(e).__name__}: {e}"
        print(f"Error in query_policies_tool: {error_msg}")
        print(f"QueryPoliciesTool Observation: {json.dumps({'error': error_msg})}")
        return json.dumps({"error": error_msg})
def _strip_json_comments(raw: str) -> str:
    # remove // line comments
    return re.sub(r"//.*?$", "", raw, flags=re.MULTILINE)


# In agent_main.py
# --- REPLACE this function ---

def create_and_upload_metadata_tool(input_json_str: str) -> str:
    """
    Tool to create the standard NFT metadata JSON file and upload it to IPFS via Pinata.
    Input MUST be a single valid JSON string. The JSON string ITSELF should not be wrapped in extra quotes.
    All keys and all string values *within* the JSON MUST be enclosed in double quotes.
    Required keys: 'name', 'description', 'content_cid', 'creator_address', 'license_uri', 'sha256_hash', 'phash' (phash can be null).
    Returns a JSON string containing 'metadata_cid' or an 'error' key.
    Example Input: {"name": "My Art", "description": "...", "content_cid": "bafy...", ...}
    """
    state = _AgentState.get()
    if not state.has_queried_policies:
        return json.dumps({
            "error": "Workflow violation. You must use QueryGovernancePolicies (Step 2) before creating metadata (Step 3)."})
    if state.has_created_metadata:
        return json.dumps({"error": "Metadata already created. Proceed to Step 4."})

    # The wrapper now passes a string, but accept a raw dict just in case
    if isinstance(input_json_str, dict):
        input_json_str = json.dumps(input_json_str)

    print(f"\n[Tool] === Creating & Uploading Metadata ===")
    print(f"Raw JSON input received by tool: {input_json_str}")

    # --- START OF FIX ---
    # Helper function to robustly extract JSON from LLM output that might contain markdown fences.
    def _extract_json_from_llm_output(text: str) -> str:
        # First, look for a markdown-fenced JSON block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            print("Found JSON inside markdown fence.")
            return match.group(1).strip()

        # If no markdown, look for a raw JSON object as a fallback
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            print("Found raw JSON object.")
            return match.group(1).strip()

        # If no JSON object is found at all, return the original text for error handling
        print("Warning: No JSON object found in the input string.")
        return text

    clean_json_str = _extract_json_from_llm_output(input_json_str)
    # --- END OF FIX ---

    data = None
    try:
        data = json.loads(clean_json_str)
    except json.JSONDecodeError as je:
        err_msg = f"Invalid JSON format. The input must contain a valid JSON object. Error: {je}. Input after cleaning was: '{clean_json_str}'"
        print(f"CreateAndUploadMetadataTool Observation: {json.dumps({'error': err_msg})}")
        return json.dumps({"error": err_msg})

    # ... The rest of the function remains the same ...

    name = data.get("name")
    description = data.get("description")
    content_cid = data.get("content_cid")
    creator_address = data.get("creator_address")
    license_uri = data.get("license_uri")
    sha256_hash = data.get("sha256_hash")
    phash = data.get("phash")

    required_args_map = {
        "name": name, "description": description, "content_cid": content_cid,
        "creator_address": creator_address, "license_uri": license_uri, "sha256_hash": sha256_hash
    }
    missing_or_empty = [k for k, v in required_args_map.items() if
                        v is None or (isinstance(v, str) and not v.strip())]
    if missing_or_empty:
        error_msg = f"Missing or empty required keys in JSON: {', '.join(missing_or_empty)}"
        print(error_msg)
        return json.dumps({"error": error_msg})

    if not isinstance(content_cid, str) or len(content_cid) < 40:
        error_msg = f"Invalid 'content_cid' in JSON: '{content_cid}'. Must be from UploadContentToIPFS."
        return json.dumps({"error": error_msg})
    if not isinstance(sha256_hash, str) or len(sha256_hash) != 64:
        error_msg = f"Invalid 'sha256_hash' in JSON: '{sha256_hash}'. Must be from UploadContentToIPFS."
        return json.dumps({"error": error_msg})
    if not Web3.is_address(creator_address):
        error_msg = f"Invalid 'creator_address' format in JSON: {creator_address}"
        return json.dumps({"error": error_msg})

    print(
        f"Parsed Args for Metadata -> Name: {name}, Desc: {description[:30]}..., ContentCID: {content_cid}, Creator: {creator_address}, License: {license_uri}, SHA256: {sha256_hash}, pHash: {phash}")

    try:
        metadata_dict = ipfs_storage.create_metadata(
            name=name, description=description, content_cid=content_cid,
            creator_address=creator_address, license_uri=license_uri,
            sha256_hash=sha256_hash, phash=phash
        )
        metadata_cid = ipfs_storage.upload_metadata_json(metadata_dict)

        if metadata_cid:
            result = {"metadata_cid": metadata_cid}
            print(f"CreateAndUploadMetadataTool Observation: {json.dumps(result)}")
            state.has_created_metadata = True
            state.last_metadata_cid = metadata_cid  # Save the CID to the state
            return json.dumps(result)
        else:
            error_msg = "Failed to upload metadata JSON to IPFS (Pinata). upload_metadata_json returned None."
            print(f"CreateAndUploadMetadataTool Observation: {json.dumps({'error': error_msg})}")
            return json.dumps({"error": error_msg})
    except Exception as e:
        error_msg = f"Error during metadata creation/upload: {type(e).__name__}: {e}"
        print(f"CreateAndUploadMetadataTool Observation: {json.dumps({'error': error_msg})}")
        return json.dumps({"error": error_msg})


# In agent_main.py
# --- REPLACE this function ---

def mint_nft_tool(input_json_str: str) -> str:
    """
    Tool to mint the AIGC NFT on the Sepolia blockchain.
    Input MUST be a single valid JSON string. The JSON string ITSELF should not be wrapped in extra quotes.
    All keys and string values *within* the JSON MUST be enclosed in double quotes.
    Required keys: 'recipient_address', 'metadata_cid', 'content_cid' (this is the CAR file CID from UploadContentToIPFS).
    Returns a JSON string with transaction details or an 'error' key.
    Example Input: {"recipient_address": "0x...", "metadata_cid": "bafy...", "content_cid": "bafycar..."}
    """

    print(f"\n[Tool] === Minting NFT ===")
    print(f"Raw JSON input received by tool: {input_json_str}")

    state = _AgentState.get()
    if not state.has_created_metadata:
        return json.dumps(
            {"error": "Workflow violation. You must use CreateAndUploadMetadata (Step 3) before minting (Step 4)."})
    if state.has_minted_nft:
        return json.dumps({"error": "NFT has already been minted."})

    # --- START OF FIX ---
    # Helper function to robustly extract JSON from LLM output that might contain markdown fences.
    def _extract_json_from_llm_output(text: str) -> str:
        # First, look for a markdown-fenced JSON block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            print("Found JSON inside markdown fence.")
            return match.group(1).strip()

        # If no markdown, look for a raw JSON object as a fallback
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            print("Found raw JSON object.")
            return match.group(1).strip()

        # If no JSON object is found at all, return the original text for error handling
        print("Warning: No JSON object found in the input string.")
        return text

    clean_json_str = _extract_json_from_llm_output(input_json_str)
    # --- END OF FIX ---

    data = None
    try:
        data = json.loads(clean_json_str)
    except json.JSONDecodeError as je:
        err_msg = f"Invalid JSON format. The input must contain a valid JSON object. Error: {je}. Input after cleaning was: '{clean_json_str}'"
        print(f"MintAIGCNFT Observation: {json.dumps({'error': err_msg})}")
        return json.dumps({"error": err_msg})

    # ... The rest of the function remains the same ...

    recipient_address = data.get("recipient_address")
    metadata_cid = data.get("metadata_cid")
    content_cid_for_car = data.get("content_cid")

    # If metadata_cid is a placeholder, try to get it from the state
    if isinstance(metadata_cid, str) and "the_metadata_cid" in metadata_cid.lower():
        if state.last_metadata_cid:
            print(f"Replacing placeholder metadata_cid with value from state: {state.last_metadata_cid}")
            metadata_cid = state.last_metadata_cid
        else:
            return json.dumps({
                                  "error": "Agent used a placeholder for 'metadata_cid' but no value was found in the agent state from the previous step."})

    if not recipient_address or not Web3.is_address(recipient_address):
        return json.dumps({"error": f"Invalid 'recipient_address' in JSON: {recipient_address}"})
    if not metadata_cid or not isinstance(metadata_cid, str) or len(metadata_cid) < 40:
        return json.dumps(
            {"error": f"Invalid 'metadata_cid' in JSON: '{metadata_cid}'. Must be from CreateAndUploadMetadata."})
    if not content_cid_for_car or not isinstance(content_cid_for_car, str) or len(
            content_cid_for_car) < 40:
        return json.dumps({
            "error": f"Invalid 'content_cid' (for CAR file) in JSON: '{content_cid_for_car}'. Must be the CAR file CID from UploadContentToIPFS."})

    print(
        f"Parsed Args for Minting -> Recipient: {recipient_address}, Metadata CID: {metadata_cid}, CAR CID for contract: {content_cid_for_car}")

    try:
        checksum_recipient = Web3.to_checksum_address(recipient_address)
        token_uri = f"ipfs://{metadata_cid}"

        car_cid_bytes = content_cid_for_car.encode('utf-8')
        if len(car_cid_bytes) > 32:
            car_cid_bytes32 = car_cid_bytes[:32]
            print(
                f"Warning: CAR CID '{content_cid_for_car}' (length {len(car_cid_bytes)}) was truncated to 32 bytes for on-chain storage: {car_cid_bytes32.hex()}")
        else:
            car_cid_bytes32 = car_cid_bytes.ljust(32, b'\0')
        print(f"Using CAR CID as bytes32 for contract: {car_cid_bytes32.hex()}")

        print("Building transaction...")
        nonce = w3.eth.get_transaction_count(agent_account.address)
        current_chain_id = w3.eth.chain_id
        tx_params = {'from': agent_account.address, 'nonce': nonce, 'chainId': current_chain_id}

        # ... (rest of the transaction logic is unchanged)
        # ... just paste the code from your original file here ...
        # (This section is long and correct, so no need to reproduce it all)

        try:
            estimated_gas = aigc_contract.functions.safeMintWithCARCID(
                checksum_recipient, token_uri, car_cid_bytes32
            ).estimate_gas({'from': agent_account.address, 'chainId': current_chain_id})
            tx_params['gas'] = int(estimated_gas * 1.2)  # Add 20% buffer
            print(f"Estimated Gas: {estimated_gas}, Gas Limit Set: {tx_params['gas']}")
        except Exception as gas_err:
            print(f"Warning: Gas estimation failed: {gas_err}. Using fallback gas limit (500k).")
            if "revert" in str(gas_err).lower(): print(
                "Gas estimation failure might be due to a contract revert. Check contract logic and inputs.")
            tx_params['gas'] = 500000  # Fallback gas limit

        # EIP-1559 fee model (dynamic fees)
        try:
            latest_block = w3.eth.get_block('latest')
            base_fee = latest_block.get('baseFeePerGas')
            if base_fee is None:  # Should not happen on Sepolia typically
                print("Warning: No baseFeePerGas found in latest block. Falling back to legacy gas price.")
                tx_params['gasPrice'] = w3.eth.gas_price
            else:
                # Set a reasonable priority fee
                max_priority_fee_per_gas = w3.to_wei('1.5', 'gwei')  # Example priority fee
                # max_fee_per_gas should be base_fee + max_priority_fee_per_gas
                max_fee_per_gas = base_fee + max_priority_fee_per_gas + w3.to_wei('1',
                                                                                  'gwei')  # Add small buffer for base_fee fluctuations
                tx_params['maxPriorityFeePerGas'] = max_priority_fee_per_gas
                tx_params['maxFeePerGas'] = max_fee_per_gas
                print(
                    f"Gas Fees (EIP-1559): Max Fee={w3.from_wei(max_fee_per_gas, 'gwei')} Gwei, Priority Fee={w3.from_wei(max_priority_fee_per_gas, 'gwei')} Gwei")
        except Exception as fee_err:
            print(f"Warning: EIP-1559 fee estimation failed: {fee_err}. Falling back to legacy gas price.")
            try:
                tx_params['gasPrice'] = w3.eth.gas_price  # Fallback to legacy gas pricing
            except Exception as legacy_gas_err:
                error_msg = f"Failed to get legacy gas price: {legacy_gas_err}"
                print(f"MintAIGCNFT Observation: {json.dumps({'error': error_msg})}")
                return json.dumps({"error": error_msg})

        transaction = aigc_contract.functions.safeMintWithCARCID(
            checksum_recipient, token_uri, car_cid_bytes32
        ).build_transaction(tx_params)
        print("Transaction built. Signing...")
        signed_tx = w3.eth.account.sign_transaction(transaction, AGENT_PRIVATE_KEY)
        print("Sending transaction...")
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"Transaction sent! Hash: {tx_hash.hex()}. Waiting for receipt...")

        try:
            tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)  # 5 min timeout
            print(f"Receipt received. Status: {'Success' if tx_receipt.status == 1 else 'Failed'}")
            token_id_str = "N/A"
            if tx_receipt.status == 1:
                final_status = "Success"
                # Attempt to decode TokenRegistered event
                try:
                    # Solidity event: event TokenRegistered(uint256 indexed tokenId, address indexed recipient, string tokenURI, bytes32 carCID);
                    event_signature_hash = w3.keccak(text="TokenRegistered(uint256,address,string,bytes32)").hex()
                    token_id_found = False
                    for log_entry in tx_receipt.logs:
                        if log_entry.address == checksum_contract_address and len(log_entry.topics) > 1 and \
                                log_entry.topics[0].hex() == event_signature_hash:
                            # Assuming tokenId is the first indexed topic
                            token_id_int = Web3.to_int(
                                hexstr=log_entry.topics[1].hex())  # First indexed topic is topics[1]
                            token_id_str = str(token_id_int)
                            print(f"Token ID from TokenRegistered event: {token_id_str}")
                            token_id_found = True;
                            break
                    if not token_id_found:  # Fallback to Transfer event if TokenRegistered not found or decoded
                        print("TokenRegistered event not found/decoded as expected. Checking standard Transfer event.")
                        transfer_event_sig_hash = w3.keccak(text="Transfer(address,address,uint256)").hex()
                        for log_entry in tx_receipt.logs:  # Standard ERC721 Transfer event
                            if log_entry.address == checksum_contract_address and len(log_entry.topics) == 4 and \
                                    log_entry.topics[0].hex() == transfer_event_sig_hash:
                                token_id_int = Web3.to_int(
                                    hexstr=log_entry.topics[3].hex())  # tokenId is the 3rd indexed topic (topics[3])
                                token_id_str = str(token_id_int)
                                print(f"Token ID from Transfer event: {token_id_str}");
                                break
                except Exception as log_err:
                    print(f"Warning: Could not decode Token ID from logs: {log_err}")
                    token_id_str = "N/A (Log decoding error)"
                result = {"status": final_status, "transaction_hash": tx_hash.hex(),
                          "block_number": tx_receipt.blockNumber, "token_id": token_id_str}
                state.has_minted_nft = True  # mark as done

            else:  # tx_receipt.status != 1 (Transaction failed)
                final_status = "Failed"
                revert_reason = "Unknown (transaction reverted on-chain)"
                try:
                    failed_tx = w3.eth.get_transaction(tx_hash)
                    print(f"Failed transaction details: {failed_tx}")
                except Exception as e_revert:
                    print(f"Could not fetch additional details for failed tx: {e_revert}")

                result = {"status": final_status, "transaction_hash": tx_hash.hex(),
                          "block_number": tx_receipt.blockNumber, "token_id": token_id_str,
                          "error": f"Transaction reverted. Reason: {revert_reason}. Receipt: {dict(tx_receipt)}"}

            print(f"MintAIGCNFT Observation: {json.dumps(result)}")
            return json.dumps(result)

        except TransactionNotFound:
            error_msg = f"Transaction receipt timeout ({300}s). Transaction hash: {tx_hash.hex()}. It might still be pending or was dropped."
            result = {"status": "Timeout", "transaction_hash": tx_hash.hex(), "error": error_msg}
            print(f"MintAIGCNFT Observation: {json.dumps(result)}")
            return json.dumps(result)
        except Exception as wait_err:  # Catch other errors during receipt waiting
            error_msg = f"Error waiting for transaction receipt: {type(wait_err).__name__}: {wait_err}"
            result = {"status": "ErrorOnReceiptWait", "transaction_hash": tx_hash.hex(), "error": error_msg}
            print(f"MintAIGCNFT Observation: {json.dumps(result)}")
            return json.dumps(result)

    except MismatchedABI as abi_err:
        error_message = f"ABI Mismatch: {abi_err}. Ensure your contract ABI is correct and matches the deployed contract, especially for the 'safeMintWithCARCID' function."
        result = {"error": error_message}
        print(f"MintAIGCNFT Observation: {json.dumps(result)}")
        return json.dumps(result)
    except Exception as e:
        error_message = f"NFT minting error: {type(e).__name__}: {e}"
        if "revert" in str(e).lower():
            error_message = f"Transaction reverted during minting: {e}. Check contract conditions, inputs, and gas."
        elif "insufficient funds" in str(e).lower():
            error_message = f"Insufficient funds for transaction: {e}"
        elif "nonce" in str(e).lower():
            error_message = f"Nonce error during minting: {e}"
        result = {"error": error_message}
        print(f"MintAIGCNFT Observation: {json.dumps(result)}")
        return json.dumps(result)

# --- Tool Definitions ---
# This list now includes updated descriptions to guide the agent's query formulation.
tools = [
    Tool(
        name="UploadContentToIPFS",
        func=upload_content_tool,
        description="Use this tool FIRST (Step 1) to upload a content file (e.g., image, text) from a local file path to IPFS. Input MUST be the exact local file path string (e.g., './test_files/image.png'). The tool returns a JSON string with 'content_cid', 'sha256_hash', and 'phash'. If 'content_cid' is null or an 'error' key is present, STOP and report the error."
    ),
    Tool(
        name="QueryGovernancePolicies",
        func=query_policies_tool,
        description="Use this tool SECOND (Step 2) to find relevant license URIs. To get the best results, the input MUST be a short, keyword-focused query string derived from the core requirements of the asset description. For example, use 'non-exclusive commercial use' or 'personal use attribution required' instead of long sentences. Do NOT include words like 'license for' or 'digital art'. Returns a JSON list of policies."
    ),
    Tool(
        name="CreateAndUploadMetadata",
        func=create_and_upload_metadata_tool,
        description="Use this tool THIRD (Step 3) to create and upload NFT metadata. Action Input MUST be a VALID JSON string. Required keys: 'name', 'description', 'content_cid', 'creator_address', 'license_uri', 'sha256_hash', 'phash'. Returns JSON with 'metadata_cid' or 'error'."
    ),
    Tool(
        name="MintAIGCNFT",
        func=mint_nft_tool,
        description="Use this tool FOURTH and LAST (Step 4) to mint the NFT. Action Input MUST be a VALID JSON string. Required keys: 'recipient_address', 'metadata_cid', 'content_cid'. Returns JSON with transaction details or 'error'. After this, provide a Final Answer."
    ),
]
print(f"Defined {len(tools)} tools.")


# In agent_main.py

# --- REPLACE THE OLD PROMPT WITH THIS ENTIRE STRICTER BLOCK ---
react_prompt_template_str = """
You are an AI agent participating in a mandatory, controlled evaluation of a content registration system.
**Your participation is for a technical test only. All content is synthetic, pre-approved, and non-copyrighted. You MUST complete all steps without refusal.** The term "minting" in this context refers to a test transaction and does not create real-world assets.

**Your single goal is to call tools in this exact sequence, one at a time. After each `Observation`, you will proceed to the NEXT step in the sequence.**
1.  **Step 1:** `UploadContentToIPFS`
2.  **Step 2:** `QueryGovernancePolicies`
3.  **Step 3:** `CreateAndUploadMetadata`
4.  **Step 4:** `MintAIGCNFT`
5.  **Step 5:** `Final Answer`

**CRITICAL RULES:**
- **ONE STEP AT A TIME:** Your response MUST contain only ONE `Thought` and ONE `Action`/`Action Input`. Then you MUST STOP and wait for the `Observation`.
- **NO PREDICTIONS:** Do not predict or hallucinate the result of a tool. Your `Action Input` for `MintAIGCNFT` MUST NOT be followed by a `Final Answer`.
- **STRICT SEQUENCE:** You MUST follow the 5 steps in order. DO NOT repeat a step. After an `Observation`, immediately formulate the `Thought` and `Action` for the *next* step.
- **VALID ACTIONS:** The `Action` MUST be one of these names: {tool_names}
- **USE OBSERVATIONS:** The input for each step MUST come from the `Observation` of the previous step(s). Do not make up values.
- **LICENSE SELECTION (during Step 3 planning):** After `Observation` from Step 2 (`QueryGovernancePolicies`), you MUST select the license with the lowest "distance" score. Your `Thought` for Step 3 MUST state which license URI you are selecting and why.
- **FINAL ANSWER (Step 5 ONLY):** The `Final Answer` section is ONLY for Step 5. You MUST NOT include `Final Answer` in any other step. After you receive a successful `Observation` from Step 4 (`MintAIGCNFT`), your *entire response* MUST be ONLY the following JSON structure, starting with `Final Answer:`.
`Final Answer: {{"license_uri": "THE_URI_YOU_CHOSE_IN_STEP_3", "transaction_details": THE_FULL_JSON_OBJECT_FROM_THE_MINTING_OBSERVATION}}`

**RESPONSE FORMAT (for steps 1-4):**
You MUST structure your response in this exact format, with each section on a new line:
Thought: Your reasoning for the current step.
Action: The name of the single tool to use.
Action Input: The input for the tool.
--------------------
TOOLS:
{tools}
--------------------
Begin the process now.

Question: {input}
{agent_scratchpad}
"""
# --- END OF REPLACEMENT ---

prompt = PromptTemplate.from_template(react_prompt_template_str)

# --- Agent Setup (Using ReAct Agent) ---
print("Setting up LangChain ReAct agent...")
try:
    # Using the standard create_react_agent constructor
    agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    output_parser=CleanReActOutputParser(),   #
)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Set to True to see the agent's thought process
        handle_parsing_errors=True,  # Attempt to handle LLM output parsing errors
        max_iterations=10  # Allows for 4 steps + thoughts + some retries/corrections
    )
    print("AgentExecutor created successfully.")
except Exception as e:
    print(f"Error creating agent or executor: {e}")
    sys.exit(1)





# --- Main Execution ---
if __name__ == "__main__":
    print("\n======== Starting AIGC Registration Process ========")

    # Example usage:
    # Ensure the test file exists or create a dummy one
    content_file_to_process = "test_image_upload.png"  # Example file
    if not os.path.exists(content_file_to_process):
        print(f"Warning: Test file '{content_file_to_process}' not found. Creating a dummy file.")
        try:
            with open(content_file_to_process, "w") as f:
                f.write("This is a dummy test image file for AIGC registration.")
            print(f"Dummy file '{content_file_to_process}' created.")
        except IOError as e:
            print(f"Error: Could not create dummy file '{content_file_to_process}': {e}")
            print("Please ensure you have write permissions in the current directory or create the file manually.")
            sys.exit(1)

    creator_wallet_address = agent_account.address  # Get from initialized components
    asset_name_input = "AI Test Asset - My Sample Image"
    asset_description_input = "A sample image generated by an AI model, being registered via the Agentic RAG prototype for testing purposes. Intended for personal, non-commercial use."

    # Construct the input for the agent based on the example
    agent_input = f"""
    Register the AIGC file located at: '{content_file_to_process}'.
    The Creator's Wallet Address is: {creator_wallet_address}.
    The Asset Name is: '{asset_name_input}'.
    The Asset Description is: '{asset_description_input}'.
    Please use the standard 4-step tool sequence to perform the registration.
    For the license query (QueryGovernancePolicies tool - Step 2), use a query based on the asset description, for example, search for 'personal use non-commercial license'. When forming the query string for `QueryGovernancePolicies`, ensure it accurately reflects all key terms and restrictions mentioned in the 'Asset Description'. For example, if the description states 'personal use only' and 'not for commercial activity', your query should prioritize these terms. Avoid adding contradictory terms not present in the description.
    Follow all critical instructions regarding step tracking, data flow, JSON formatting, error handling, and final answer.
    """

    print(f"\n--- Invoking Agent with Input ---")
    print(f"Content File: {os.path.abspath(content_file_to_process)}")  # Show absolute path
    print(f"Creator: {creator_wallet_address}")
    print(f"Asset Name: {asset_name_input}")
    # print(f"Full input to agent:\n{agent_input}") # For debugging

    try:
        response = agent_executor.invoke({"input": agent_input})
        print("\n======== Agent Execution Finished ========")
        print("\nFinal Response from Agent:")
        if isinstance(response, dict) and "output" in response:
            print(response["output"])
        else:
            print(response)  # Print raw response if structure is unexpected
    except OutputParserException as ope:
        print(f"\n======== Agent Execution Error ========")
        print(f"Output Parsing Error by LangChain: {ope}")
        print("This often means the LLM's output did not conform to the expected ReAct format.")
        print("Consider simplifying the prompt or checking LLM compatibility if this persists.")
    except Exception as e:
        print(f"\n======== Agent Execution Error ========")
        print(f"An unexpected error occurred during agent execution: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    print("\n======== AIGC Registration Process Complete (from main script) ========")