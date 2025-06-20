# agent_policy.py

import json
import re
from langchain_community.llms import Ollama
from shared_utils import PolicyData


def parse_llm_json_output(llm_output: str) -> dict:
    """
    Parses the potentially messy output from an LLM to extract a clean JSON object.

    This function handles cases where the LLM wraps the JSON in markdown,
    adds leading/trailing text, or uses problematic escape characters.
    """
    # 1. Search for a JSON object within markdown code blocks (```json ... ```)
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_output, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 2. If no markdown block is found, search for the first and last curly brace
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            # 3. If no JSON structure is found at all, raise an error
            raise json.JSONDecodeError("No JSON object found in the LLM response.", llm_output, 0)

    # 4. Clean up common issues like escaped single quotes that can break parsing
    # and ensure the structure is what we expect before loading.
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Raise a new error with the cleaned string for better debugging
        raise json.JSONDecodeError(f"Failed to decode cleaned JSON string. Error: {e.msg}", json_str, e.pos)


def build_prompt(all_policies: list, scenario_description: str) -> str:
    """
    Builds a highly structured, robust prompt with a one-shot example to guide the LLM.

    This final version includes explicit instructions to prevent hallucination and self-contradiction.
    """
    policies_str = json.dumps(all_policies, indent=2)

    prompt = f"""
[SYSTEM]
You are a meticulous and highly precise IP counsel expert. Your task is to select the single most appropriate license from a list. You must follow all instructions exactly.

**CRITICAL INSTRUCTIONS:**
1.  **Analyze Requirements**: Scrutinize the user's scenario to identify every constraint.
2.  **Compare Rigorously**: Compare these constraints against the `text` of each policy in `AVAILABLE_POLICIES`.
3.  **Prioritize Specificity**: If multiple licenses seem to fit, you MUST choose the MOST SPECIFIC one (e.g., 'editorial-use-only-v1' is better than 'cc-by-nc-4.0' for news articles).
4.  **ONLY USE PROVIDED IDs**: You MUST ONLY select an `id` from the `AVAILABLE_POLICIES` list. Do not invent, assume, or hallucinate license IDs. If the perfect license (e.g., 'cc-by-nd-4.0') is not in the list, you must choose the best available alternative from the list provided.
5.  **Common Pitfalls to Avoid**:
    * **CRITICAL CONTRADICTION**: Never select a Non-Commercial ('NC') license for a scenario that allows or requires commercial use. This is a fundamental error.
    * **No Derivatives vs. Share Alike**: Do NOT confuse 'No Derivatives' (derivatives are forbidden) with 'Share Alike' (derivatives are allowed but must use the same license).
    * **Attribution vs. Public Domain**: Do NOT confuse `cc-by-4.0` (which requires attribution) with `cc0-1.0` (which has no attribution requirement).

**OUTPUT FORMATTING (ABSOLUTE REQUIREMENT):**
- Your entire response MUST be a single, syntactically perfect JSON object and NOTHING else.
- Do NOT include any text, explanation, or markdown before or after the JSON object.
- The JSON object must be 100% machine-parsable. Pay close attention to commas, quotes, and brackets.

**EXAMPLE OF PERFECT OUTPUT:**
---
[USER EXAMPLE]
Here are the available policies: ...
Here is the user's scenario:
`SCENARIO_DESCRIPTION`: "A photograph for my blog. I don't want people using it for profit, and they must give me credit."

[YOUR PERFECT RESPONSE EXAMPLE]
{{
  "analysis": {{
    "thought_process": [
      "The user needs a license for a photograph for their blog.",
      "Constraint 1: Not for profit. This means the license must be non-commercial.",
      "Constraint 2: They must give me credit. This means the license must require attribution.",
      "Searching the policies, 'cc-by-nc-4.0' requires both Attribution (BY) and is NonCommercial (NC).",
      "This is a perfect match for all constraints."
    ],
    "selected_license_id": "cc-by-nc-4.0"
  }},
  "justification": "The 'cc-by-nc-4.0' license is the correct choice because it perfectly matches the user's two requirements: it strictly forbids commercial use ('NonCommercial') and requires that credit be given to the creator ('Attribution')."
}}
---

[USER]
Here are the available policies:
`AVAILABLE_POLICIES`:
{policies_str}

Here is the user's scenario:
`SCENARIO_DESCRIPTION`: "{scenario_description}"

Now, perform the analysis and provide the single, syntactically perfect JSON object as your response.
"""
    return prompt


def run_policy_selection(
        asset_description: str,
        all_policies: list,
        llm: Ollama
) -> PolicyData:
    """
    Performs policy selection using an LLM with a robust prompt and parser.
    """
    print("\n[Policy Agent] Starting LLM-powered license selection...")
    prompt = build_prompt(all_policies, asset_description)

    print("[Policy Agent] Sending comprehensive prompt to LLM for analysis...")
    llm_output = llm.invoke(prompt).strip()
    print(f"[Policy Agent] Received LLM response.")

    try:
        # Use the robust parser to extract clean JSON
        result = parse_llm_json_output(llm_output)

        analysis = result.get("analysis")
        if not isinstance(analysis, dict):
            raise KeyError("The 'analysis' key must be an object, not a list or other type.")

        selected_id = analysis.get("selected_license_id")
        justification = result.get("justification")

        if not selected_id or not justification:
            raise KeyError(
                "The required keys 'selected_license_id' or 'justification' were not found in the parsed JSON.")

    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        error_message = f"Policy Agent Error: Failed to parse LLM response. Error: {e}"
        print(f"{error_message}\nLLM Raw Output:\n{llm_output}")
        raise ValueError(error_message) from e

    # Ensure justification is a clean string for logging
    if isinstance(justification, list):
        justification = " ".join(justification)

    print(f"[Policy Agent] LLM Justification: {justification}")
    selected_policy = next((p for p in all_policies if p.get("id") == selected_id), None)

    if not selected_policy:
        error_message = f"Policy Agent Error: The LLM selected a license ID ('{selected_id}') that does not exist in the provided policies list."
        print(error_message)
        raise ValueError(error_message)

    selected_license_uri = selected_policy.get('metadata', {}).get('url') or selected_policy.get('metadata', {}).get(
        'uri')

    if not selected_license_uri:
        error_message = f"Policy Agent Error: Selected policy (ID: {selected_id}) has no 'url' or 'uri' field in its metadata."
        print(error_message)
        raise ValueError(error_message)

    print(f"[Policy Agent] Best match found. License ID: {selected_id}, License URI: {selected_license_uri}")
    return PolicyData(selected_license_uri=selected_license_uri)