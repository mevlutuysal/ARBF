# agent_policy.py

import json
import re
from langchain_community.llms import Ollama
from integration.vector_db_manager import VectorDBManager
from shared_utils import PolicyData


def run_policy_selection(
        asset_description: str,
        vector_db: VectorDBManager,
        llm: Ollama
) -> PolicyData:
    """
    Performs the policy selection stage using a structured RAG approach.
    """
    print("\n[Policy Agent] Starting license selection...")

    # This prompt instructs the LLM to extract key terms into a JSON format.
    prompt = f"""
**Task:** Analyze the provided asset description and extract the key license terms into a structured JSON format.

**Asset Description:**
{asset_description}

**Instructions:**
- Identify the core components of the license requirements from the description.
- Extract the following fields:
  - "asset_type": (e.g., "3D model", "photograph", "source code", "article")
  - "intended_use": A list of strings. (e.g., ["commercial", "non-commercial", "personal", "editorial"])
  - "permissions": A list of what is allowed (e.g., ["derivatives", "distribution", "modification"]).
  - "restrictions": A list of what is required or forbidden (e.g., ["attribution", "share-alike", "no-resale"]).
- If a field is not mentioned, provide an empty list.
- The output MUST be a single, valid JSON object and nothing else. Do not add explanations.

**JSON Output:**
"""

    # Generate the structured data with the LLM
    llm_output = llm.invoke(prompt).strip()
    print(f"[Policy Agent] Generated structured data: {llm_output}")

    # This new sanitization logic is more robust. It finds the JSON block
    # even if it's surrounded by markdown fences or other text.
    json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
    if not json_match:
        raise ValueError("Policy Agent Error: LLM did not return a valid JSON object.")
    json_str = json_match.group(0)

    try:
        license_terms = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Policy Agent Error: Failed to decode JSON from LLM output. Error: {e}\nContent: {json_str}")

    # This revised query construction logic correctly handles list values.
    query_parts = []

    # Add asset type if it's a string
    asset_type = license_terms.get("asset_type")
    if isinstance(asset_type, str):
        query_parts.append(asset_type)

    # A helper function to process lists from the JSON
    def process_list(terms, suffix=""):
        if isinstance(terms, list):
            for term in terms:
                if isinstance(term, str):
                    query_parts.append(f"{term}{suffix}")

    # Process each list with an optional suffix for clarity
    process_list(license_terms.get("intended_use"), " use")
    process_list(license_terms.get("permissions"))
    process_list(license_terms.get("restrictions"))

    # Use dict.fromkeys to remove duplicates while preserving order, then join.
    # Fallback to the original asset description if the generated query is empty.
    query = " ".join(list(dict.fromkeys(filter(None, query_parts)))) or asset_description

    print(f"[Policy Agent] Generated search query: '{query}'")

    # Query the vector database
    results = vector_db.query_policies(query_text=query, n_results=5)

    if not results or not results.get('ids', [[]])[0]:
        raise ValueError("Policy Agent Error: No license policies found for the given query.")

    distances = results.get('distances', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]

    best_index = distances.index(min(distances))
    selected_license = metadatas[best_index]
    selected_license_uri = selected_license.get('url')

    if not selected_license_uri:
        raise ValueError(
            f"Policy Agent Error: Selected policy (ID: {results['ids'][0][best_index]}) has no 'uri' in its metadata.")

    print(
        f"[Policy Agent] Best match found with distance {distances[best_index]:.4f}. License URI: {selected_license_uri}")

    return PolicyData(selected_license_uri=selected_license_uri)