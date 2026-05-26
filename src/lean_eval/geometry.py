from __future__ import annotations

# Helpers for labeling rows or whole proofs as geometry examples.

from typing import Any


# Detect whether a row belongs to Mathlib geometry based on its module path or imports.
# row: Dataset row to inspect.
def row_is_mathlib_geometry(row: dict[str, Any]) -> bool:
    original_id = str(row.get("original_id", ""))
    module = original_id.split("::", 1)[0]
    if module == "Mathlib.Geometry" or module.startswith("Mathlib.Geometry."):
        return True

    context = str(row.get("formal_context_with_sorry", ""))
    for line in context.splitlines():
        stripped = line.strip()
        if stripped == "import Mathlib.Geometry" or stripped.startswith("import Mathlib.Geometry."):
            return True
    return False


# Decide whether a row is geometry under the requested labeling rule.
# row: Dataset row to classify.
# rule: Geometry-labeling rule to apply.
def row_is_geometry(row: dict[str, Any], *, rule: str) -> bool:
    field_value = row.get("is_geometry") is True
    module_value = row_is_mathlib_geometry(row)
    if rule == "field":
        return field_value
    if rule == "mathlib_geometry":
        return module_value
    if rule == "field_or_mathlib_geometry":
        return field_value or module_value
    raise ValueError(f"unknown geometry rule {rule!r}")


# Decide whether any row in a proof group should mark the whole proof as geometry.
# rows: Rows belonging to a single proof.
# geometry_rule: Geometry-labeling rule to apply to each row.
def proof_domain(rows: list[dict[str, Any]], *, geometry_rule: str) -> bool:
    return any(row_is_geometry(row, rule=geometry_rule) for row in rows)
