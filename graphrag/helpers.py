import re
import logging
from typing import Union, Optional, Dict, List, Any
from fix_busted_json import repair_json


logger = logging.getLogger(__name__)

def extract_json(text: Union[str, dict], skip_repair: Optional[bool] = False) -> str:
    """
    Extracts JSON from a string or dictionary, optionally skipping JSON repair.
    
    Args:
        text (Union[str, dict]): The input text or dictionary.
        skip_repair (Optional[bool]): Flag to skip JSON repair. Defaults to False.
        
    Returns:
        str: The extracted JSON as a string.
    """
    if not isinstance(text, str):
        text = str(text)
    regex = r"(?:```)?(?:json)?([^`]*)(?:\\n)?(?:```)?"
    matches = re.findall(regex, text, re.DOTALL)

    try:
        return repair_json("".join(matches)) if not skip_repair else "".join(matches)
    except Exception as e:
        logger.error(f"Failed to repair JSON: {e} - {text}")
        return "".join(matches)


def map_dict_to_cypher_properties(d: dict) -> str:
    """
    Maps a dictionary to Cypher query properties with proper UTF-8 encoding.
    
    Args:
        d (dict): The dictionary to map.
        
    Returns:
        str: A Cypher-formatted string of properties.
    """
    import json
    
    cypher = "{"
    if isinstance(d, list):
        if len(d) == 0:
            return "{}"
        for i, item in enumerate(d):
            # Use json.dumps for proper encoding
            item_str = json.dumps(item, ensure_ascii=False) if isinstance(item, str) else str(item)
            cypher += f"{i}: {item_str}, "
        cypher = (cypher[:-2] if len(cypher) > 1 else cypher) + "}"
        return cypher
    for key, value in d.items():
        # Escape property names with special characters using backticks
        escaped_key = f"`{key}`" if any(c in key for c in "äöüßÄÖÜ") else key
        
        # Check value type
        if isinstance(value, str):
            # Use json.dumps with ensure_ascii=False to handle German characters properly
            value = json.dumps(value, ensure_ascii=False) if f"{value}" != "None" else '""'
        else:
            value = json.dumps(value, ensure_ascii=False) if f"{value}" != "None" else '""'
        cypher += f"{escaped_key}: {value}, "
    cypher = (cypher[:-2] if len(cypher) > 1 else cypher) + "}"
    return cypher


def stringify_falkordb_response(response: Union[list, str]) -> str:
    """
    Converts FalkorDB response to a string.
    
    Args:
        response (Union[list, str]): The response to stringify.
        
    Returns:
        str: The stringified response.
    """
    if not isinstance(response, list) or len(response) == 0:
        data = str(response).strip()
    elif not isinstance(response[0], list):
        data = str(response).strip()
    else:
        for l, _ in enumerate(response):
            if not isinstance(response[l], list):
                response[l] = str(response[l])
            else:
                for i, __ in enumerate(response[l]):
                    response[l][i] = str(response[l][i])
        data = str(response).strip()

    return data


def extract_block(text: str, block_type: str = "cypher") -> str:
    """
    Extract code blocks of a specific type from text

    Args:
        text (str): The text to extract from
        block_type (str): The type of code block to extract (default: "cypher")

    Returns:
        str: The extracted code block or empty string if no match
    """
    # Try to match blocks with specified language
    pattern = rf"```{block_type}(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # If no language-specific block found, try any code block
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # If still no match, try to extract something that looks like Cypher
    if block_type.lower() == "cypher":
        # Look for common Cypher patterns
        pattern = r"(?:MATCH|MERGE|CREATE|RETURN|WHERE|WITH)\s+.*?(?:\n\n|\Z)"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return " ".join(matches).strip()
    
    return ""



def parse_ndjson(text: str) -> list:
    """
    Parses NDJSON (newline-delimited JSON) from a string.
    Handles various formats including markdown code blocks and malformed JSON.
    
    Args:
        text (str): The input text containing NDJSON data
        
    Returns:
        list: List of parsed JSON objects
        
    Raises:
        ValueError: If the input is empty or completely invalid
    """
    import json
    import re
    
    if not isinstance(text, str):
        raise ValueError(f"Expected string input, got {type(text)}")
        
     # Remove all code block markers (case-insensitive, anywhere in text)
    text = re.sub(r'```(?:json|ndjson)?\s*', '', text, flags=re.IGNORECASE)
    text = text.replace('```', '')
    text = text.strip()

    # Fix the invalid relation format
    text = re.sub(r'{"type":"relation":"label":', '{"type":"relation","label":', text)
    
    results = []
    lines = text.splitlines()


    if not lines:
        logger.warning("Empty input provided to parse_ndjson")
        return results

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # Fix malformed relation entries
        if '"type":"relation"' in line:
            # Fix the case where "type":"relation":"LABEL" appears
            line = re.sub(r'"type":"relation":"([^"]+)"', r'"type":"relation","label":"\1"', line)
            # Fix the case where "type":"relation":"label" appears
            line = re.sub(r'"type":"relation":"label":', r'"type":"relation","label":', line)
            # Catch any case where label is jammed after type
            line = re.sub(r'"type":"relation"([^,{])+"', r'"type":"relation","label":', line)
        
        # Fix common JSON syntax errors
        # Fix extra commas before closing braces/brackets
        line = re.sub(r',\s*([}\]])', r'\1', line)
        # Fix missing colons in key-value pairs
        line = re.sub(r'"([^"]+)"\s*"([^"]+)"', r'"\1": "\2"', line)
        # Fix duplicated content (common LLM error)
        line = re.sub(r'(\{[^}]*\})\s*\1', r'\1', line)
        # Fix malformed entity definitions with extra spaces
        line = re.sub(r'"type":"entity_definition"\s*,\s*"label"', r'"type":"entity_definition","label"', line)
        # Fix malformed relation definitions with extra spaces  
        line = re.sub(r'"type":"relation_definition"\s*,\s*"label"', r'"type":"relation_definition","label"', line)

        try:
            logger.info(f"Parsing line: {str(line)}")
            obj = json.loads(line)
            results.append(obj)
        except json.JSONDecodeError as e:
            # Try to fix common German text issues in JSON
            try:
                # Fix German quotes and special characters
                fixed_line = line.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
                # Fix potential encoding issues
                fixed_line = fixed_line.encode('utf-8').decode('utf-8')
                obj = json.loads(fixed_line)
                results.append(obj)
                logger.info(f"Fixed German text JSON parsing for line {i}")
            except json.JSONDecodeError as e2:
                # Try using fix_busted_json library for more robust repair
                try:
                    fixed_line = repair_json(line)
                    obj = json.loads(fixed_line)
                    results.append(obj)
                    logger.info(f"Fixed JSON using repair_json for line {i}")
                except Exception as e3:
                    logger.warning(f"Skipping invalid NDJSON line {i}: {line[:100]}... Error: {str(e)}")
                    continue

    if not results:
        logger.warning("No valid JSON objects found in input")

        # Try to parse the *entire* text as one JSON document – this covers
        # cases where the model produced a pretty-printed multi-line object.
        try:
            full_obj = json.loads(text)
            if isinstance(full_obj, dict) and ("entities" in full_obj or "relations" in full_obj):
                logger.info("Successfully parsed full-text JSON wrapper containing entities/relations array")
                results.append(full_obj)
        except Exception:
            pass  # still empty – will be handled later

    # Some models (e.g. Anthropic Claude) wrap the *real* NDJSON lines in a JSON
    # object under the key "additionalProperties".  Each such object looks like:
    #   { "additionalProperties": "<newline-delimited json string>" }
    # We detect this pattern, parse the inner string recursively, and replace the
    # wrapper with its contents so that downstream code sees proper objects with
    # a "type" field.
    expanded_results: list = []
    for item in results:
        # If it's not a dict or already a valid NDJSON object, keep as is
        if not isinstance(item, dict):
            expanded_results.append(item)
            continue

        # Directly valid entity / relation -> keep
        if item.get("type") in {"entity", "relation"}:
            expanded_results.append(item)
            continue

        # Anthropic / other wrapper pattern
        payload = item.get("additionalProperties") or item.get("additional_properties")
        if isinstance(payload, str):
            try:
                nested_objects = parse_ndjson(payload)
                expanded_results.extend(nested_objects)
                continue  # Skip adding the wrapper itself
            except Exception as e:
                logger.warning(f"Failed to parse additionalProperties payload: {e}")
                # fall through and keep wrapper
        expanded_results.append(item)

    results = expanded_results
    # ---------------------------------------------------------------------------

    # Existing wrapper-object flattening (entities / relations arrays)
    if (
        len(results) == 1
        and isinstance(results[0], dict)
        and ("entities" in results[0] or "relations" in results[0])
    ):
        root = results.pop(0)
        entities = root.get("entities", [])
        if isinstance(entities, list):
            for ent in entities:
                if isinstance(ent, dict):
                    ent.setdefault("type", "entity")
                    results.append(ent)
        relations = root.get("relations", [])
        if isinstance(relations, list):
            for rel in relations:
                if isinstance(rel, dict):
                    rel.setdefault("type", "relation")
                    results.append(rel)

    return results