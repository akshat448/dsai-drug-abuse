"""
build_prompt.py

Constructs an optimized prompt for Gemini LLM.
Maps Q&A logic into the user's specific JSON schema structure.
"""

import json
import logging
from utils.schema import STANDARD_SCHEMA
from utils.text_utils import normalize_whitespace

logger = logging.getLogger(__name__)


def build_prompt(input_text: str, source_file: str, language: str = "English") -> str:
    """
    Builds the extraction prompt.
    
    KEY MAPPING INSTRUCTIONS:
    - Bot Response -> 'generalized_text'
    - User Question -> 'metadata.suggested_query'
    - Risk Assessment -> 'safety'
    """
    
    # We serialize the schema to show the LLM exactly what we want
    schema_json = json.dumps(STANDARD_SCHEMA, indent=2)
    instructions = f"""
    <role>
    You are Gemini 3, a precise transformation engine for addiction-recovery interview transcripts.
    You must strictly follow all constraints below.
    </role>

    <constraints>
    - Output EXACTLY one JSON object in the specified schema.
    - DO NOT add text outside the JSON.
    - DO NOT modify or redact the original_text field. It must contain the exact text as provided in the input, unchanged.
    - Apply redaction ONLY inside generalized_text, segment_type, tags, safety, and metadata — NEVER in original_text.
    - No assumptions, no clinical claims, no invented facts.
    - Generalized text must be 2–4 short sentences.
    - Segment length (original_text) may be long; but generalized_text must reflect the segment concisely.
    - Each segment must cover a coherent theme, but you do NOT need to limit original_text to 200 tokens.
    - Chunking rule applies ONLY to segments (number of segments), NOT to original_text token length.
    - Use only the allowed placeholders, tags, and segment types listed.
    </constraints>

    <allowed_placeholders>
    [AGE_GROUP], [LOCATION], [JOB_ROLE], [PERSON], [FAMILY_MEMBER],
    [PEER], [PEERS], [OUTSIDERS], [SOCIAL_CIRCLE],
    [YEAR], [YEARS_DURATION], [MONTHS_DURATION], [TIME]
    </allowed_placeholders>

    <allowed_segment_types>
    background, onset, progression, treatment_path, coping, cravings, triggers,
    family_support, emotional_support, stigma, technology_use, environment_control,
    peer_influence, relapse_prevention, barriers_to_care
    </allowed_segment_types>

    <allowed_tags>
    background, triggers, cravings, withdrawal, coping,
    family_support, stigma, relapse_prevention, environment_control,
    peer_influence, emotional_support, medication_tapering,
    routine, technology_use, barriers_to_care, treatment_path
    </allowed_tags>

    <safety_rules>
    - "safe": neutral/recovery narrative
    - "sensitive": cravings, stigma, emotional struggle
    - "unsafe": unprescribed medication or dangerous behavior
    - "red_flag": overdose risk, self-harm, explicit dangerous actions
    </safety_rules>

    <json_schema>
    {schema_json}
    </json_schema>

    <context>
    {normalize_whitespace(input_text)}
    </context>

    <task>
    Based on the information above:

    1. Preserve the original_text EXACTLY as in the transcript. Do NOT redact it.
    2. Segment the text into meaningful thematic segments.
    3. In each segment:  
    - Leave original_text unchanged.  
    - Create a generalized_text version using placeholders.  
    - Assign EXACTLY one segment_type from the allowed list.  
    - Assign 2–5 tags from the allowed tags.  
    - Assign a safety label based on the rules.  
    - Keep generalized_text short and universal (2–4 sentences).
    4. Produce ONLY the JSON object matching the schema.
    </task>

    <final_instruction>
    Return only the JSON. Nothing else.
    </final_instruction>
    """

    logger.debug("Prompt built with schema-mapped Q&A instructions.")
    return instructions.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Gemini prompt for document.")
    parser.add_argument("--file", required=True, help="Path to input text file")
    parser.add_argument("--source_file", required=True, help="Original source filename")
    args = parser.parse_args()

    # Fallback to empty string if file read fails (for safety)
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        text = ""

    prompt = build_prompt(text, args.source_file)
    print(prompt)