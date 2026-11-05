"""Reuse the frozen analysis with explicit v11 labels and prior test exposure."""

import report_frozen_evaluation as core

render_original = core.report_markdown


def report_markdown(report):
    text = render_original(report).replace("v2", "v11")
    original_selection = (
        "The checkpoint was selected before opening the test split. Weights, inference settings, "
        "seeds, and endpoints were fixed in protocol.json."
    )
    original_scope = (
        "Development results informed selection. The earlier validation split also informed "
        "prior editing. Only the 104-system test split was previously unused."
    )
    if text.count(original_selection) != 1 or text.count(original_scope) != 1:
        raise ValueError("The inherited report scope changed")
    text = text.replace(original_selection, (
        "The v11 checkpoint, inference settings, seeds, and endpoints were frozen before its "
        "validation and test predictions. No weights were edited during this evaluation."
    ))
    return text.replace(original_scope, (
        "The same 55/50/104 development/validation/test split is retained. Development results "
        "informed the choice of v11, and the earlier validation split informed prior editing. "
        "The 104-system test split was already evaluated with v2 before this follow-up, so "
        "these results are not presented as a new untouched-test confirmation."
    ))


if __name__ == "__main__":
    core.report_markdown = report_markdown
    core.main()
