import json
import logging
import os
import re

import yaml

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts")


def dict_to_markdown(d, level=1):
    markdown = ""
    for key, value in d.items():
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, dict) and not any(value.values()):
            continue
        markdown += f"{'#' * level} {key.replace('_', ' ').title()}\n"
        if isinstance(value, dict):
            markdown += dict_to_markdown(value, level + 1)
        elif isinstance(value, str) and value.strip():
            markdown += f"- {value.strip()}\n"
        elif isinstance(value, list):
            for item in value:
                if item.strip():
                    markdown += f"- {item.strip()}\n"
        else:
            markdown += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        markdown += "\n"
    return markdown


def extract_python_code(input_string, file_path=None, save=False):
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, input_string, re.DOTALL)
    if match:
        python_code = match.group(1)
        if save:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(python_code)
            logging.info(f"Python code has been written to {file_path}")
        return python_code
    else:
        logging.info("No Python code block found.")
        return ""


def read_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.info(f"Error: The file {file_path} was not found.")
        return ""
    except Exception as e:
        logging.info(f"An error occurred: {e}")
        return ""


def save_code(code, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(code)
    logging.info(f"Python code has been written to {file_path}")


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.info(f"Error: The file {path} was not found.")
        return {}
    except json.JSONDecodeError:
        logging.info(f"Error: The file {path} is not a valid JSON.")
        return {}
    except Exception as e:
        logging.info(f"An error occurred: {e}")
        return {}


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"JSON has been written to {path}")


def extract_json(input_string, file_path=None, save=False):
    pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern, input_string)
    json_code = match.group(1).strip() if match else input_string.strip()
    try:
        data = json.loads(json_code)
        if save and file_path is not None:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.info(f"JSON has been written to {file_path}")
        return data
    except Exception as e:
        logging.error(f"JSON parsing error: {e}")
        return None


def extract_json_list(input_string, file_path=None, save=False):
    pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(pattern, input_string)
    if match:
        json_code = match.group(1).strip()
    else:
        json_code = input_string.strip()

    try:
        data = json.loads(json_code)

    except json.JSONDecodeError as e:
        logging.warning(
            f"Initial JSON parsing failed: {e}. Attempting to fix and retry..."
        )

        if "Invalid \\escape" in str(e):
            corrected_json_code = json_code.replace("\\", "\\\\")
            logging.info("Applied backslash correction. Retrying parse...")

            try:
                data = json.loads(corrected_json_code)
                logging.info("JSON parsed successfully after correction.")
            except json.JSONDecodeError as e2:
                logging.error(f"JSON parsing failed AGAIN after correction: {e2}")
                logging.debug(
                    f"String that failed final parsing attempt:\n{corrected_json_code}"
                )
                return []
        else:
            logging.error(f"JSON parsing failed with an unhandled error type: {e}")
            return []

    if isinstance(data, list):
        if save and file_path:
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logging.info(f"JSON list has been written to {file_path}")
            except Exception as write_error:
                logging.error(f"Failed to write JSON to file: {write_error}")
        return data
    else:
        logging.warning(f"Extracted data is not a list, but {type(data)}.")
        return []


def parse_json_list_from_string(input_string: str):
    json_code = ""
    match_block = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", input_string, re.DOTALL)
    if match_block:
        json_code = match_block.group(1).strip()
    else:
        match_list = re.search(r"(\[[\s\S]*\])", input_string, re.DOTALL)
        if match_list:
            json_code = match_list.group(0).strip()
        else:
            logging.warning("No JSON list `[...]` found in response string.")
            return None

    sanitized_code = re.sub(r",\s*\]", "]", json_code)

    try:
        data = json.loads(sanitized_code)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list):
                    logging.warning(
                        "LLM returned an object; extracting first list found in its values."
                    )
                    return val
        return None
    except json.JSONDecodeError as e:
        logging.error(
            f"JSON list parsing error: {e}\nProblematic string: {sanitized_code}"
        )
        return None


def extract_json_object(input_string: str) -> dict | None:
    json_code = ""
    match_block = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", input_string, re.DOTALL)
    if match_block:
        json_code = match_block.group(1).strip()
    else:
        match_obj = re.search(r"(\{[\s\S]*\})", input_string, re.DOTALL)
        if match_obj:
            json_code = match_obj.group(0).strip()
        else:
            logging.warning(f"No JSON object `{input_string}` found in response.")
            return None

    logging.debug(f"Raw JSON object found: {repr(json_code)}")
    sanitized_json_code = re.sub(r",\s*\}", "}", json_code)
    sanitized_json_code = re.sub(r"\\([a-zA-Z]+)", r"\\\\\\1", sanitized_json_code)

    try:
        data = json.loads(sanitized_json_code)
        if isinstance(data, dict):
            return data
        return None
    except json.JSONDecodeError as e:
        logging.error(
            f"JSON object parsing error: {e}\nProblematic string: {sanitized_json_code}"
        )
        return None


def extract_yaml(input_string: str, file_path: str = None, save: bool = False) -> str:
    pattern = r"```yaml\s*([\s\S]*?)\s*```"
    match = re.search(pattern, input_string, re.DOTALL)
    if not match:
        logging.warning("No YAML block found in the LLM response.")
        return ""
    yaml_string = match.group(1).strip()
    logging.debug(f"--- Raw YAML Extracted ---\n{yaml_string}\n--------------------")

    try:
        parsed_yaml = yaml.safe_load(yaml_string)
        clean_yaml_string = yaml.dump(parsed_yaml, allow_unicode=True, sort_keys=False)
        logging.info("YAML parsed successfully on the first attempt.")
        if save and file_path:
            save_yaml(clean_yaml_string, file_path)
        return clean_yaml_string
    except yaml.YAMLError as e:
        logging.warning(
            f"Initial YAML parsing failed: {e}. Attempting automated fixes..."
        )

    sanitized_yaml = re.sub(r",\s*([\}\]])", r"\1", yaml_string)
    try:
        parsed_yaml = yaml.safe_load(sanitized_yaml)
        clean_yaml_string = yaml.dump(parsed_yaml, allow_unicode=True, sort_keys=False)
        logging.info("YAML parsed successfully after fixing trailing commas.")
        if save and file_path:
            save_yaml(clean_yaml_string, file_path)
        return clean_yaml_string
    except yaml.YAMLError as e:
        logging.warning(
            f"Parsing failed after fixing trailing commas: {e}. Attempting backslash sanitization..."
        )

    aggressively_sanitized_yaml = re.sub(r'\\(?!["\\/bfnrt])', r"\\\\", sanitized_yaml)
    try:
        parsed_yaml = yaml.safe_load(aggressively_sanitized_yaml)
        clean_yaml_string = yaml.dump(parsed_yaml, allow_unicode=True, sort_keys=False)
        logging.info(
            "YAML parsed successfully after aggressive backslash sanitization."
        )
        if save and file_path:
            save_yaml(clean_yaml_string, file_path)
        return clean_yaml_string
    except yaml.YAMLError as e:
        logging.error("FATAL: All YAML parsing and fixing attempts failed.")
        logging.error(f"Final error: {e}")
        logging.debug(
            f"--- Final Problematic YAML String ---\n{aggressively_sanitized_yaml}\n-----------------------------"
        )
        return ""


def save_yaml(content: str, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        logging.info(f"Validated YAML has been written to {file_path}")
    except IOError as e:
        logging.error(f"Failed to write YAML to file {file_path}: {e}")


def extract_yaml_from_config_tags(
    input_string: str, file_path: str = None, save: bool = True
) -> str:
    """
    Extracts a raw YAML block from <config> tags, programmatically fixes common LLM
    errors (like unescaped backslashes), and then validates/saves it.
    """
    pattern = r"<config>(.*?)</config>"
    match = re.search(pattern, input_string, re.DOTALL)

    if match:
        yaml_config = match.group(1).strip()
        logging.info("Successfully extracted content from <config> tags.")

        fixed_yaml_config = yaml_config.replace("\\", "\\\\")

        if fixed_yaml_config != yaml_config:
            logging.info("Applied programmatic fix for unescaped backslashes.")

        if save and file_path:
            try:
                yaml.safe_load(fixed_yaml_config)

                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(fixed_yaml_config)
                logging.info(
                    f"Valid YAML configuration has been written to {file_path}"
                )
            except yaml.YAMLError as e:
                logging.error(
                    f"Content is not valid YAML even after fixing. Error: {e}"
                )
                return ""
        return fixed_yaml_config
    else:
        logging.warning("No <config>...</config> block found.")
        return ""


def extract_markdown(input_string, file_path=None, save=True):
    pattern = r"```markdown\n(.*)```"
    match = re.search(pattern, input_string, re.DOTALL)
    if match:
        markdown_content = match.group(1)
        if save:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(markdown_content)
            logging.info(f"Markdown content has been written to {file_path}")
        return markdown_content
    else:
        logging.info("No Markdown code block found.")
        return ""


def load_prompt(prompt_name: str, **kwargs) -> str:
    prompt_path = os.path.join(PROMPTS_DIR, f"{prompt_name}.txt")

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt a_template file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    return prompt_template.format(**kwargs)


def parse_dmte_summary(summary_text: str) -> dict:
    sections = ["Data", "Model", "Training", "Evaluation"]

    parsed_content = {section: "" for section in sections}

    pattern = r"## (Data|Model|Training|Evaluation)\s*(.*?)(?=\n## (?:Data|Model|Training|Evaluation)|$)"

    matches = re.findall(pattern, summary_text, re.DOTALL)

    for match in matches:
        section_title = match[0].strip()
        section_content = match[1].strip()
        if section_title in parsed_content:
            parsed_content[section_title] = section_content

    return parsed_content


def parse_hierarchical_guide(guide_content: str) -> list:
    results = []
    path_stack = []  # Stores tuples of (indent_level, key)

    if not guide_content:
        return []

    for line in guide_content.splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line == "---":
            continue

        if stripped_line.startswith("## "):
            current_h2 = stripped_line.strip("# ").strip()
            path_stack = [(-1, current_h2)]
            continue

        if stripped_line.startswith("- "):
            indent = len(line) - len(line.lstrip(" "))
            while path_stack and path_stack[-1][0] >= indent:
                path_stack.pop()

            if ":" in stripped_line:
                parts = stripped_line.split(":", 1)
                key = parts[0].lstrip("- ").strip()
                value_str = parts[1].strip().strip("'")
                if value_str:
                    full_path = [p[1] for p in path_stack] + [key]
                    results.append({"fact_path": full_path, "fact_sentence": value_str})
                path_stack.append((indent, key))
            else:
                fact_sentence = stripped_line.lstrip("- ").strip().strip("'")
                full_path = [p[1] for p in path_stack]
                results.append({"fact_path": full_path, "fact_sentence": fact_sentence})

    return results


def _flatten_config(data, current_path=None, result=None):
    """A private helper to recursively extract all key-value pairs and their paths."""
    if current_path is None:
        current_path = []
    if result is None:
        result = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = current_path + [key]
            if isinstance(value, (dict, list)):
                _flatten_config(value, new_path, result)
            else:
                result.append((current_path, key, value))
    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_path = current_path + [f"[{index}]"]
            if isinstance(item, (dict, list)):
                _flatten_config(item, new_path, result)
            else:
                result.append([current_path, index, item])
            pass
    return result


def parse_verbatim_config_guide(config_content: str) -> list:
    """
    Parses a verbatim YAML config guide and transforms it into the standard 'fact' format.

    Args:
        config_data (str): guide file.

    Returns:
        list: A list of fact dictionaries, ready for the retrieval process.
    """
    try:
        # This logic comes from your original script's main block
        config_dict = yaml.safe_load(config_content)
        config_items = _flatten_config(config_dict)

        atomic_facts = []
        for path, key, value in config_items:
            # We only create facts for values that are strings (verbatim sentences)
            if isinstance(value, str):
                atomic_facts.append({"fact_path": path + [key], "fact_sentence": value})
        return atomic_facts

    except Exception as e:
        logging.error(f"Failed to parse config guide YAML: {e}")
        return []


def sanitize_code(content: str) -> str:
    lines = content.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1])
    return content
