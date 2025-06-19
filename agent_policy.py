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
    Builds a highly structured, robust prompt to guide the LLM's reasoning.

    This revised prompt is more forceful about the output format and provides
    more detailed instructions to improve the accuracy of license selection.
    """
    policies_str = json.dumps(all_policies, indent=2)

    prompt = f"""
[SYSTEM]
You are a meticulous and highly precise IP counsel expert. Your task is to select the single most appropriate license from a list. You must follow all instructions exactly.

**CRITICAL INSTRUCTIONS:**
1.  **Analyze Requirements**: Scrutinize the user's scenario to identify every constraint. Pay extremely close attention to key phrases like "non-commercial," "no derivatives," "share alike," "editorial use," "software," and "public domain."
2.  **Compare Rigorously**: Compare these constraints against the `text` of each policy in the `AVAILABLE_POLICIES` list. Your primary goal is to find the policy that satisfies ALL requirements.
3.  **Prioritize Specificity**: If multiple licenses seem to fit, you MUST choose the MOST SPECIFIC one. For example, if a scenario describes editorial use, select 'editorial-use-only-v1' over a more general 'cc-by-nc-4.0'. If it describes software, 'gpl-3.0' or 'MIT' are likely better than a general Creative Commons license.
4.  **Double-Check Your Work**: Before finalizing, re-read the scenario and your chosen policy's text to confirm they are a perfect match. Acknowledge and correctly interpret negative constraints (e.g., 'not for profit' means a Non-Commercial license is required).

**OUTPUT FORMATTING (ABSOLUTE REQUIREMENT):**
- Your entire response MUST be a single, valid JSON object and NOTHING else.
- Do NOT include any text, explanation, markdown, or formatting before or after the JSON object.
- The JSON object MUST have two top-level keys: "analysis" and "justification".
- The "analysis" value MUST be an OBJECT containing:
    - `thought_process`: A list of strings detailing your step-by-step reasoning.
    - `selected_license_id`: A string with the ID of your final chosen license.
- The "justification" value MUST be a SINGLE STRING explaining why your choice is correct.

[USER]
Here are the available policies:
`AVAILABLE_POLICIES`:
{policies_str}

Here is the user's scenario:
`SCENARIO_DESCRIPTION`: "{scenario_description}"

Now, perform the analysis and provide the single JSON object as your response.
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
        # Use the new robust parser to extract clean JSON
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