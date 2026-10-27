import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


def extracted_modified_entities(
    source_code: str,
    modified_lines: set[int]
):
    tree = parser.parse(bytes(source_code, "utf-8"))
    extracted_entities = []

    def visitor(node: Node):
        if node.type == "function_definition":
            start_line = node.start_point.row + 1
            end_line = node.end_point.row + 1

            if any(
                line in modified_lines
                for line in range(start_line, end_line + 1)
            ):
                extracted_entities.append({
                    "name": (
                        node.child_by_field_name("name").text.decode("utf-8")
                        if node.child_by_field_name("name")
                        else "anonymous"
                    ),
                    "type": node.type,
                    "code_snippet": node.text.decode("utf-8"),
                    "start_line": start_line,
                    "end_line": end_line
                })
                return

        elif node.type == "class_definition":
            start_line = node.start_point.row + 1
            end_line = node.end_point.row + 1

            if any(
                line in modified_lines
                for line in range(start_line, end_line + 1)
            ):
                # First visit children so specific modified methods get extracted
                count_before = len(extracted_entities)
                for child in node.children:
                    visitor(child)
                # If no child method matched, the modification was at class-level
                if len(extracted_entities) == count_before:
                    extracted_entities.append({
                        "name": (
                            node.child_by_field_name("name").text.decode("utf-8")
                            if node.child_by_field_name("name")
                            else "anonymous"
                        ),
                        "type": node.type,
                        "code_snippet": node.text.decode("utf-8"),
                        "start_line": start_line,
                        "end_line": end_line
                    })
                return

        for child in node.children:
            visitor(child)

    visitor(tree.root_node)

    return extracted_entities