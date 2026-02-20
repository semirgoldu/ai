"""
Data utilities for processing Odoo source code and documentation
into training-ready datasets.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import track

console = Console()

# Odoo file extensions to include
CODE_EXTENSIONS = {".py", ".js", ".xml", ".csv", ".scss", ".css"}
DOC_EXTENSIONS = {".rst", ".md", ".txt", ".html"}

# Directories/files to skip
SKIP_DIRS = {
    "__pycache__",
    ".git",
    "node_modules",
    ".tox",
    "egg-info",
    ".eggs",
    "static/lib",
    "static/src/lib",
}
SKIP_FILES = {".pyc", ".pyo", ".so", ".dll", ".exe", ".bin"}


class OdooDataFormatter:
    """Formats Odoo code and docs into training examples."""

    # Chat/instruction template for Qwen
    SYSTEM_PROMPT = (
        "You are an expert Odoo developer with deep knowledge of the Odoo "
        "framework, its ORM, views, controllers, and module architecture. "
        "You write clean, efficient, and well-documented Odoo code following "
        "official best practices."
    )

    def __init__(self, max_length: int = 4096):
        self.max_length = max_length

    def format_code_completion(
        self, file_path: str, code: str, module_name: str = ""
    ) -> list[dict]:
        """Create code completion training examples from a source file."""
        examples = []

        # Full file completion
        rel_path = file_path.replace("\\", "/")
        ext = Path(file_path).suffix

        if ext == ".py":
            examples.extend(
                self._format_python_examples(rel_path, code, module_name)
            )
        elif ext == ".xml":
            examples.extend(
                self._format_xml_examples(rel_path, code, module_name)
            )
        elif ext == ".js":
            examples.extend(
                self._format_js_examples(rel_path, code, module_name)
            )

        return examples

    def _format_python_examples(
        self, file_path: str, code: str, module: str
    ) -> list[dict]:
        examples = []

        # Extract classes and methods
        classes = re.findall(
            r"(class\s+\w+\(.*?\):.*?)(?=\nclass\s|\Z)",
            code,
            re.DOTALL,
        )

        for cls_code in classes:
            cls_match = re.match(r"class\s+(\w+)\((.*?)\):", cls_code)
            if not cls_match:
                continue
            cls_name = cls_match.group(1)
            parents = cls_match.group(2)

            if len(cls_code) > 200:
                prompt = (
                    f"Write an Odoo model class `{cls_name}` "
                    f"inheriting from `{parents}` "
                    f"in the `{module}` module."
                )
                if "_name" in cls_code:
                    name_match = re.search(
                        r"_name\s*=\s*['\"](.+?)['\"]", cls_code
                    )
                    if name_match:
                        prompt += (
                            f" The model name should be "
                            f"'{name_match.group(1)}'."
                        )

                examples.append(
                    self._make_chat_example(prompt, cls_code.strip())
                )

            # Individual methods
            methods = re.findall(
                r"(    def\s+\w+\(.*?\):.*?)(?=\n    def\s|\n\nclass\s|\Z)",
                cls_code,
                re.DOTALL,
            )
            for method_code in methods:
                method_match = re.match(
                    r"    def\s+(\w+)\((.*?)\):", method_code
                )
                if method_match and len(method_code) > 100:
                    method_name = method_match.group(1)
                    prompt = (
                        f"Write the `{method_name}` method for the "
                        f"Odoo model `{cls_name}` in `{module}`."
                    )
                    examples.append(
                        self._make_chat_example(
                            prompt, method_code.strip()
                        )
                    )

        # Full file as context
        if 200 < len(code) <= self.max_length * 4:
            prompt = (
                f"Write the complete Odoo Python file for "
                f"`{file_path}` in the `{module}` module."
            )
            examples.append(self._make_chat_example(prompt, code.strip()))

        return examples

    def _format_xml_examples(
        self, file_path: str, code: str, module: str
    ) -> list[dict]:
        examples = []

        # Extract individual records/views
        records = re.findall(
            r"(<record\s.*?</record>)", code, re.DOTALL
        )
        for record in records:
            model_match = re.search(r'model="(.*?)"', record)
            id_match = re.search(r'id="(.*?)"', record)
            if model_match and id_match and len(record) > 100:
                model = model_match.group(1)
                rec_id = id_match.group(1)
                prompt = (
                    f"Write an Odoo XML record for model `{model}` "
                    f"with id `{rec_id}` in the `{module}` module."
                )
                examples.append(
                    self._make_chat_example(prompt, record.strip())
                )

        # Full XML file
        if 200 < len(code) <= self.max_length * 4:
            file_type = "view" if "view" in file_path.lower() else "data"
            prompt = (
                f"Write the complete Odoo XML {file_type} file "
                f"`{file_path}` for the `{module}` module."
            )
            examples.append(self._make_chat_example(prompt, code.strip()))

        return examples

    def _format_js_examples(
        self, file_path: str, code: str, module: str
    ) -> list[dict]:
        examples = []

        if 200 < len(code) <= self.max_length * 4:
            prompt = (
                f"Write the Odoo JavaScript file `{file_path}` "
                f"for the `{module}` module."
            )
            examples.append(self._make_chat_example(prompt, code.strip()))

        return examples

    def format_documentation(
        self, title: str, content: str, url: str = ""
    ) -> list[dict]:
        """Create Q&A training examples from documentation."""
        examples = []

        if len(content) < 50:
            return examples

        # Create a general explanation example
        prompt = f"Explain the Odoo concept: {title}"
        examples.append(self._make_chat_example(prompt, content.strip()))

        # Split into sections for more granular examples
        sections = re.split(r"\n#{1,3}\s+", content)
        for section in sections:
            lines = section.strip().split("\n")
            if len(lines) >= 3 and len(section) > 200:
                section_title = lines[0].strip().rstrip(":")
                section_body = "\n".join(lines[1:]).strip()
                if section_body:
                    prompt = (
                        f"Explain the Odoo documentation topic: "
                        f"{section_title}"
                    )
                    examples.append(
                        self._make_chat_example(prompt, section_body)
                    )

        return examples

    def _make_chat_example(
        self, user_msg: str, assistant_msg: str
    ) -> dict:
        return {
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }


def tokenize_dataset(
    dataset,
    tokenizer,
    max_length: int = 4096,
    field: str = "messages",
):
    """Tokenize a chat-format dataset for training."""

    def _tokenize(example):
        messages = example[field]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(
        _tokenize,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        num_proc=os.cpu_count(),
    )


def scan_odoo_modules(source_dir: str) -> list[dict]:
    """Scan an Odoo source directory and catalog all modules."""
    source_path = Path(source_dir)
    modules = []

    for manifest in source_path.rglob("__manifest__.py"):
        module_dir = manifest.parent
        module_name = module_dir.name

        # Count files by type
        file_counts = {}
        total_lines = 0
        for ext in CODE_EXTENSIONS:
            files = list(module_dir.rglob(f"*{ext}"))
            valid_files = [
                f
                for f in files
                if not any(skip in str(f) for skip in SKIP_DIRS)
            ]
            if valid_files:
                file_counts[ext] = len(valid_files)
                for f in valid_files:
                    try:
                        total_lines += sum(
                            1 for _ in open(f, encoding="utf-8", errors="ignore")
                        )
                    except Exception:
                        pass

        modules.append(
            {
                "name": module_name,
                "path": str(module_dir),
                "file_counts": file_counts,
                "total_lines": total_lines,
            }
        )

    return sorted(modules, key=lambda m: m["total_lines"], reverse=True)


def collect_source_files(
    source_dir: str,
    extensions: Optional[set] = None,
    max_file_size: int = 100_000,
) -> list[dict]:
    """Collect all relevant source files from an Odoo directory."""
    if extensions is None:
        extensions = CODE_EXTENSIONS

    source_path = Path(source_dir)
    files = []

    for path in source_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in extensions:
            continue
        if any(skip in str(path) for skip in SKIP_DIRS):
            continue
        if path.suffix in SKIP_FILES:
            continue
        if path.stat().st_size > max_file_size:
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Determine the module name
        module_name = ""
        for parent in path.parents:
            if (parent / "__manifest__.py").exists():
                module_name = parent.name
                break

        rel_path = str(path.relative_to(source_path))
        files.append(
            {
                "path": rel_path,
                "module": module_name,
                "content": content,
                "size": len(content),
                "hash": hashlib.md5(content.encode()).hexdigest(),
            }
        )

    return files


def deduplicate_examples(
    examples: list[dict], key: str = "messages"
) -> list[dict]:
    """Remove duplicate training examples based on content hash."""
    seen = set()
    unique = []
    for ex in examples:
        content = json.dumps(ex[key], sort_keys=True)
        h = hashlib.md5(content.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    return unique
