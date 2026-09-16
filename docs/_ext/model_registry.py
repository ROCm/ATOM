# SPDX-License-Identifier: MIT
"""Render the native model registry without importing ATOM or GPU libraries."""

import ast
from pathlib import Path

from docutils import nodes
from sphinx.util.docutils import SphinxDirective


class ModelRegistry(SphinxDirective):
    def run(self):
        root = Path(self.env.srcdir).parent
        source = root / "atom/model_engine/model_runner.py"
        self.env.note_dependency(str(source))
        tree = ast.parse(source.read_text())
        registry = next(
            ast.literal_eval(statement.value)
            for statement in tree.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "support_model_arch_dict"
                for target in statement.targets
            )
        )
        table = nodes.table()
        group = nodes.tgroup(cols=2)
        table += group
        for _ in range(2):
            group += nodes.colspec(colwidth=50)
        head = nodes.thead()
        group += head
        row = nodes.row()
        head += row
        for label in ("Hugging Face architecture", "ATOM implementation"):
            entry = nodes.entry()
            entry += nodes.paragraph(text=label)
            row += entry
        body = nodes.tbody()
        group += body
        for architecture, implementation in sorted(registry.items()):
            module, _ = implementation.rsplit(".", 1)
            implementation_path = root / (module.replace(".", "/") + ".py")
            if not implementation_path.is_file():
                raise self.error(f"Missing model implementation: {implementation}")
            row = nodes.row()
            body += row
            for value in (architecture, implementation):
                entry = nodes.entry()
                paragraph = nodes.paragraph()
                paragraph += nodes.literal(text=value)
                entry += paragraph
                row += entry
        return [table]


def setup(app):
    app.add_directive("atom-model-registry", ModelRegistry)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
