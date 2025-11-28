import ast
import logging
import re

from src.utils.helper import save_code


def analyze_ast(code):
    try:
        tree = ast.parse(code)
        return tree
    except SyntaxError as e:
        logging.error(f"Error parsing code: {e}")
        return None


def extract_imports(code):
    import_statements = []
    import_pattern = re.compile(
        r"^\s*(?:import\s+[a-zA-Z0-9_.,\s]+\s*(?:as\s+[a-zA-Z0-9_]+)?"
        r"|from\s+[a-zA-Z0-9_.]+\s+import\s+[a-zA-Z0-9_.,\s\*]+\s*(?:as\s+[a-zA-Z0-9_]+)?)\s*(?:#.*)?$"
    )
    lines = code.splitlines()
    for line in lines:
        if import_pattern.match(line):
            import_statements.append(line.strip())
    return import_statements


def extract_definitions_in_order(code):
    tree = analyze_ast(code)
    if tree is None:
        return []

    lines = code.splitlines(
        True
    )  # Keep newlines for accurate source segment extraction

    definitions = []

    import_statements = extract_imports(code)
    if import_statements:
        definitions.append({"type": "import", "code": import_statements})

    # Helper function: Recursively process nodes and categorize them
    def process_nodes(nodes, target_list):
        for node in nodes:
            node_code = ast.get_source_segment(code, node)
            if node_code is None:
                start_lineno = node.lineno
                end_lineno = (
                    node.end_lineno if hasattr(node, "end_lineno") else start_lineno + 1
                )  # Basic fallback
                node_code = "".join(lines[start_lineno - 1 : end_lineno]).rstrip()

            if isinstance(node, ast.FunctionDef):
                target_list.append(
                    {
                        "type": "function",
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "code": node_code,
                    }
                )
            elif isinstance(node, ast.ClassDef):
                class_item = {
                    "type": "class",
                    "name": node.name,
                    "code": node_code,
                    "methods": [],
                    "nested_classes": [],
                }
                for subnode in node.body:
                    subnode_code = ast.get_source_segment(code, subnode)
                    if subnode_code is None:  # Fallback
                        start_lineno = subnode.lineno
                        end_lineno = (
                            subnode.end_lineno
                            if hasattr(subnode, "end_lineno")
                            else start_lineno + 1
                        )
                        subnode_code = "".join(
                            lines[start_lineno - 1 : end_lineno]
                        ).rstrip()

                    if isinstance(subnode, ast.FunctionDef):
                        class_item["methods"].append(
                            {
                                "type": "function",
                                "name": subnode.name,
                                "args": [arg.arg for arg in subnode.args.args],
                                "code": subnode_code,
                            }
                        )
                    elif isinstance(subnode, ast.ClassDef):
                        process_nodes([subnode], class_item["nested_classes"])
                target_list.append(class_item)
            elif isinstance(node, ast.If):
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Eq)
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value == "__main__"
                ):
                    target_list.append({"type": "main", "code": node_code})
            else:
                pass

    process_nodes(tree.body, definitions)

    return definitions


def extract_comment_steps_from_code(code):
    """
    Extract all # comments inside a function/method code block.
    """
    lines = code.splitlines()
    comments = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            comment_content = stripped.lstrip("#").strip()
            if comment_content:
                comments.append(comment_content)
    return comments


def restore_and_save_py_file(
    definitions, save_path=None, save=False, add_supplement=True
):
    """
    Restore full Python source code from definitions and save it as a .py file.

    Args:
        definitions (List[Dict]): List of code blocks with optional comment_supplements.
        save_name (str): Filename prefix to save the restored .py file.
    """
    restored_lines = []

    # Extract import statements first and add them at the top of the file
    import_lines = []
    import_statements = next(
        (item["code"] for item in definitions if item["type"] == "import"), []
    )
    for import_statement in import_statements:
        import_lines.append(import_statement)
    restored_lines.append("\n".join(import_lines))

    # Process other definitions (functions, classes, and main block)
    for item in definitions:
        if item["type"] == "import":
            continue  # Skip imports since they've been processed already

        # Handle main block (if __name__ == '__main__':)
        if item["type"] == "main":
            restored_lines.append(item["code"])  # Add the main block code
            continue  # Skip processing the main block further
        code_lines = item["code"].splitlines()
        if add_supplement:
            supplements = item.get("comment_supplements", {})
        new_lines = []
        for line in code_lines:
            new_lines.append(line)
            stripped = line.strip()
            if stripped.startswith("#"):
                comment_text = stripped.lstrip("#").strip()
                if add_supplement:
                    supplement = supplements.get(comment_text, "").strip()
                    if supplement and supplement != "<NO_SUPPLEMENT>":
                        indent = line[: len(line) - len(line.lstrip())]
                        supplement_lines = supplement.splitlines()
                        concatenated_supplement = ", ".join(supplement_lines)
                        new_lines.append(f"{indent}# (paper) {concatenated_supplement}")
        restored_lines.append("\n".join(new_lines))
    code_framework = "\n\n".join(restored_lines) + "\n"
    if save:
        save_code(code_framework, save_path)

    return code_framework
