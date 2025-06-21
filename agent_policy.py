# agent_policy.py

import json
import re
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
import numpy as np
from shared_utils import PolicyData


def parse_llm_json_output(llm_output: str) -> dict:
    """
    Parses the potentially messy output from an LLM to extract a clean JSON object.
    (This function remains unchanged)
    """
    match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', llm_output, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        match = re.search(r'\{[\s\S]*\}', llm_output)
        if match:
            json_str = match.group(0)
        else:
            raise json.JSONDecodeError("No JSON object found in the LLM response.", llm_output, 0)

    try:
        json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Failed to decode cleaned JSON string. Error: {e.msg}", json_str, e.pos)


def find_semantic_candidates(scenario_description: str, policies: list, model: SentenceTransformer,
                             top_k: int = 10) -> list:
    """
    [AGENT STEP 1 - RETRIEVE] Finds the most semantically similar policies using vector search.
    (This function remains unchanged)
    """
    policy_descriptions = []
    for p in policies:
        profile_parts = [f"{key.replace('_', ' ')} is {val}" for key, val in p['profile'].items()]
        full_desc = f"License ID: {p['id']}. Type: {p['metadata']['type']}. Description: {p['text']} Key terms: {', '.join(profile_parts)}."
        policy_descriptions.append(full_desc)

    policy_embeddings = model.encode(policy_descriptions, convert_to_tensor=False)
    scenario_embedding = model.encode(scenario_description, convert_to_tensor=False)

    scenario_embedding = scenario_embedding.reshape(1, -1)
    policy_embeddings_norm = policy_embeddings / np.linalg.norm(policy_embeddings, axis=1, keepdims=True)
    scenario_embedding_norm = scenario_embedding / np.linalg.norm(scenario_embedding)
    similarities = np.dot(policy_embeddings_norm, scenario_embedding_norm.T).flatten()

    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    return [policies[i] for i in top_k_indices]


# --- FUNCTION REMOVED ---
# extract_constraints_from_scenario() is no longer needed in a Pure RAG model.

# --- FUNCTION REMOVED ---
# filter_policies_by_constraints() is no longer needed in a Pure RAG model.


def build_final_selection_prompt(scenario_description: str, candidates: list) -> str:
    """
    [AGENT STEP 2 - SELECT] Builds the prompt for the final, nuanced selection.
    --- PROMPT MODIFIED --- This prompt is now more powerful, asking the LLM to act as the expert judge
    on a list of semantically similar candidates, not a pre-filtered one.
    """
    candidate_info = [
        {"id": p["id"], "text": p["text"], "profile": p.get("profile", {})}
        for p in candidates
    ]
    candidates_str = json.dumps(candidate_info, indent=2)

    prompt = f"""
[SYSTEM]
You are a meticulous and highly precise IP counsel expert. You will be given a user's scenario and a list of potentially relevant licenses found via semantic search. Your task is to act as the final expert judge: analyze all the information and select the single best license.

**CRITICAL INSTRUCTIONS:**
1.  **Analyze User Intent**: Read the user's scenario carefully to understand its core requirements, including commercial use, modification rights, attribution, exclusivity, and the primary goal (e.g., sharing fan art, selling a unique NFT, releasing open-source software).
2.  **Thoroughly Compare All Candidates**: Evaluate the user's scenario against EACH candidate's full text and profile. Do not assume the first candidate is the best. Weigh the pros and cons of each.
3.  **Prioritize Specificity**: A license specifically designed for the user's asset type (`scope`) is almost always the best choice. For example:
    - For software, a license with `scope: "software"` (like GPL or MIT) is superior to a general one (like CC-BY-SA).
    - For a unique NFT, `scope: "nft"` is superior to a generic commercial license.
    - For a personal photo, `scope: "personal"` is superior to other non-commercial licenses.
4.  **Output Format**: Your entire response MUST be a single, syntactically perfect JSON object as shown in the example.

**USER SCENARIO:**
`SCENARIO_DESCRIPTION`: "{scenario_description}"

**CANDIDATE POLICIES (from semantic search):**
{candidates_str}

**PERFECT RESPONSE EXAMPLE:**
{{
  "analysis": {{
    "thought_process": [
      "The user's goal is to release a 3D model kit for a game that can be used commercially but requires derivatives to be shared under the same terms.",
      "The candidates include 'gpl-3.0' and 'cc-by-sa-4.0'.",
      "Both have a 'share_alike' (copyleft) requirement.",
      "However, the asset is a '3D kit', not software code. 'gpl-3.0' is explicitly for software.",
      "'cc-by-sa-4.0' has a 'general' scope, making it perfectly suited for non-software assets like 3D models.",
      "Therefore, 'cc-by-sa-4.0' is the most appropriate and specific choice."
    ],
    "selected_license_id": "cc-by-sa-4.0"
  }},
  "justification": "The 'cc-by-sa-4.0' license is the best fit. While both it and GPL-3.0 have the required 'ShareAlike' provision, CC-BY-SA is designed for general creative works like 3D models, whereas GPL-3.0 is specifically tailored for software, making CC-BY-SA the more appropriate choice for this asset type."
}}

Now, perform the final analysis on the provided scenario and candidates, and provide the single JSON object as your response.
"""
    return prompt


def run_policy_selection(
        asset_description: str,
        all_policies: list,
        llm: Ollama,
        embedding_model: SentenceTransformer
) -> PolicyData:
    """
    Performs policy selection using a Pure RAG (Retrieve and Select) process.
    --- WORKFLOW SIMPLIFIED --- This function no longer extracts or filters.
    It retrieves candidates and sends them directly to the LLM for the final selection.
    """
    print("\n[Policy Agent v5 - Pure RAG] Starting license selection...")

    try:
        # STEP 1: Retrieve semantically similar candidates using vector search
        print("[Policy Agent v5] -> Step 1: Retrieving semantic candidates...")
        final_candidates = find_semantic_candidates(asset_description, all_policies, embedding_model, top_k=10)

        if not final_candidates:
            raise ValueError("Semantic search returned no candidate policies.")

        print(
            f"[Policy Agent v5] -> Found {len(final_candidates)} candidates: {[p['id'] for p in final_candidates]}")

        # STEP 2: Use LLM for final, nuanced selection from the high-quality candidate list
        print("[Policy Agent v5] -> Step 2: Sending candidates to LLM for final selection...")
        final_prompt = build_final_selection_prompt(asset_description, final_candidates)
        llm_output = llm.invoke(final_prompt).strip()
        result = parse_llm_json_output(llm_output)

        selected_id = result.get("analysis", {}).get("selected_license_id")
        justification = result.get("justification")

        if not selected_id or not justification:
            raise KeyError(
                "The required keys 'selected_license_id' or 'justification' were not found in the final LLM output.")

    except (json.JSONDecodeError, KeyError, AttributeError, ValueError) as e:
        error_message = f"Policy Agent Error: Failed during the selection process. Error: {e}"
        print(error_message)
        raise ValueError(error_message) from e

    print(f"[Policy Agent v5] -> LLM Justification: {justification}")
    selected_policy = next((p for p in all_policies if p.get("id") == selected_id), None)

    if not selected_policy:
        error_message = f"Policy Agent Error: The LLM selected a license ID ('{selected_id}') that does not exist."
        print(error_message)
        raise ValueError(error_message)

    selected_license_uri = selected_policy.get('metadata', {}).get('url')
    if not selected_license_uri:
        error_message = f"Policy Agent Error: Selected policy (ID: {selected_id}) has no 'url' field."
        print(error_message)
        raise ValueError(error_message)

    print(f"[Policy Agent v5] Best match found. License ID: {selected_id}, License URI: {selected_license_uri}")

    return PolicyData(selected_license_uri=selected_license_uri)