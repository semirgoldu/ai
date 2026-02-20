#!/usr/bin/env python3
"""
Step 2: Collect and Process Odoo Documentation
Converts RST/MD documentation into structured training examples.
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.progress import track

from utils.data_utils import OdooDataFormatter

console = Console()


def parse_rst_file(file_path: Path) -> list[dict]:
    """Parse an RST file into sections with title and content."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    sections = []
    # RST section markers
    markers = set("=-~^\"'`:.+_#*")

    lines = content.split("\n")
    current_title = file_path.stem.replace("-", " ").replace("_", " ").title()
    current_content = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if next line is an underline (RST section header)
        if (
            i + 1 < len(lines)
            and len(lines[i + 1]) >= len(line.rstrip())
            and len(line.strip()) > 0
            and lines[i + 1].strip()
            and all(c in markers for c in lines[i + 1].strip())
        ):
            # Save previous section
            if current_content:
                text = "\n".join(current_content).strip()
                if len(text) > 50:
                    sections.append(
                        {"title": current_title, "content": text}
                    )
            current_title = line.strip()
            current_content = []
            i += 2  # Skip title and underline
            continue

        # Skip RST directives but keep their content
        if line.strip().startswith(".. "):
            # Process directive content (indented block)
            directive_content = []
            i += 1
            while i < len(lines) and (
                lines[i].startswith("   ") or lines[i].strip() == ""
            ):
                if lines[i].strip():
                    directive_content.append(lines[i].strip())
                i += 1

            # Include code examples
            if directive_content:
                current_content.append(
                    "\n".join(directive_content)
                )
            continue

        current_content.append(line)
        i += 1

    # Last section
    if current_content:
        text = "\n".join(current_content).strip()
        if len(text) > 50:
            sections.append({"title": current_title, "content": text})

    return sections


def parse_md_file(file_path: Path) -> list[dict]:
    """Parse a Markdown file into sections."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    sections = []
    current_title = file_path.stem.replace("-", " ").replace("_", " ").title()
    current_content = []

    for line in content.split("\n"):
        if line.startswith("#"):
            # Save previous section
            if current_content:
                text = "\n".join(current_content).strip()
                if len(text) > 50:
                    sections.append(
                        {"title": current_title, "content": text}
                    )
            current_title = line.lstrip("#").strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        text = "\n".join(current_content).strip()
        if len(text) > 50:
            sections.append({"title": current_title, "content": text})

    return sections


def process_documentation(docs_dir: Path) -> list[dict]:
    """Process all documentation files in a directory."""
    all_sections = []

    # Find all RST and MD files
    rst_files = list(docs_dir.rglob("*.rst"))
    md_files = list(docs_dir.rglob("*.md"))

    console.print(
        f"  Found [bold]{len(rst_files)}[/] RST files and "
        f"[bold]{len(md_files)}[/] MD files"
    )

    for f in track(rst_files, description="Processing RST files"):
        sections = parse_rst_file(f)
        for s in sections:
            s["source_file"] = str(f.relative_to(docs_dir))
            s["format"] = "rst"
        all_sections.extend(sections)

    for f in track(md_files, description="Processing MD files"):
        sections = parse_md_file(f)
        for s in sections:
            s["source_file"] = str(f.relative_to(docs_dir))
            s["format"] = "md"
        all_sections.extend(sections)

    return all_sections


def create_odoo_knowledge_base() -> list[dict]:
    """Create training examples from built-in Odoo knowledge."""
    knowledge = [
        {
            "title": "Odoo ORM Fields",
            "content": (
                "Odoo provides various field types for models:\n\n"
                "- fields.Char: String field\n"
                "- fields.Text: Long text field\n"
                "- fields.Integer: Integer number field\n"
                "- fields.Float: Floating point field with optional digits precision\n"
                "- fields.Boolean: True/False field\n"
                "- fields.Date: Date field\n"
                "- fields.Datetime: Date and time field\n"
                "- fields.Selection: Dropdown selection field\n"
                "- fields.Many2one: Many-to-one relationship\n"
                "- fields.One2many: One-to-many relationship (virtual)\n"
                "- fields.Many2many: Many-to-many relationship\n"
                "- fields.Binary: Binary/file field\n"
                "- fields.Html: HTML content field\n"
                "- fields.Monetary: Currency amount field\n"
                "- fields.Image: Image field with optional size limits\n\n"
                "Field attributes include: string, required, readonly, "
                "default, help, index, copy, groups, states, compute, "
                "inverse, search, store, related, domain."
            ),
        },
        {
            "title": "Odoo Model Inheritance",
            "content": (
                "Odoo supports three types of inheritance:\n\n"
                "1. Classical Inheritance (_inherit without _name):\n"
                "   Extends an existing model in-place, adding fields "
                "   and methods.\n"
                "   class ResPartner(models.Model):\n"
                "       _inherit = 'res.partner'\n"
                "       loyalty_points = fields.Integer()\n\n"
                "2. Prototype Inheritance (_inherit with _name):\n"
                "   Creates a new model copying all fields/methods from parent.\n"
                "   class CustomPartner(models.Model):\n"
                "       _name = 'custom.partner'\n"
                "       _inherit = 'res.partner'\n\n"
                "3. Delegation Inheritance (_inherits):\n"
                "   Creates a link to parent model, auto-creates many2one.\n"
                "   class LibraryBook(models.Model):\n"
                "       _name = 'library.book'\n"
                "       _inherits = {'product.product': 'product_id'}"
            ),
        },
        {
            "title": "Odoo Views Architecture",
            "content": (
                "Odoo supports various view types:\n\n"
                "- Form View: Detail view for a single record with "
                "  groups, notebooks, pages, buttons, and widgets.\n"
                "- Tree/List View: Tabular view of multiple records "
                "  with optional editable inline editing.\n"
                "- Kanban View: Card-based view, often used for stages.\n"
                "- Search View: Defines search filters, group by options.\n"
                "- Pivot View: Pivot table for data analysis.\n"
                "- Graph View: Charts and graphs (bar, line, pie).\n"
                "- Calendar View: Calendar-based display of records.\n"
                "- Activity View: Activity scheduling and tracking.\n"
                "- Cohort View: Cohort analysis (Enterprise).\n"
                "- Dashboard View: Combining multiple sub-views.\n\n"
                "Views are defined in XML and can be inherited using "
                "xpath expressions to modify existing views."
            ),
        },
        {
            "title": "Odoo Security and Access Control",
            "content": (
                "Odoo implements security at multiple levels:\n\n"
                "1. Access Control Lists (ir.model.access.csv):\n"
                "   Defines CRUD permissions per model per group.\n"
                "   Format: id,name,model_id:id,group_id:id,perm_read,"
                "   perm_write,perm_create,perm_unlink\n\n"
                "2. Record Rules (ir.rule):\n"
                "   Row-level security with domain filters.\n"
                "   <record model='ir.rule' id='rule_id'>\n"
                "     <field name='model_id' ref='model_name'/>\n"
                "     <field name='domain_force'>"
                "[('user_id','=',user.id)]</field>\n"
                "     <field name='groups' eval=\"[(4, ref('group'))]\" />\n"
                "   </record>\n\n"
                "3. Field-level access (groups attribute):\n"
                "   name = fields.Char(groups='base.group_system')\n\n"
                "4. Menu and action visibility via groups."
            ),
        },
        {
            "title": "Odoo Module Structure",
            "content": (
                "A standard Odoo module has this structure:\n\n"
                "my_module/\n"
                "  __init__.py           # Python package init\n"
                "  __manifest__.py       # Module metadata\n"
                "  models/\n"
                "    __init__.py\n"
                "    my_model.py         # Model definitions\n"
                "  views/\n"
                "    my_model_views.xml  # View definitions\n"
                "    menu.xml            # Menu items\n"
                "  security/\n"
                "    ir.model.access.csv # Access control\n"
                "    security.xml        # Groups & record rules\n"
                "  data/\n"
                "    data.xml            # Default data\n"
                "    demo.xml            # Demo data\n"
                "  controllers/\n"
                "    main.py             # HTTP controllers\n"
                "  wizard/\n"
                "    my_wizard.py        # Transient models\n"
                "  static/\n"
                "    src/\n"
                "      js/               # JavaScript\n"
                "      xml/              # QWeb templates\n"
                "      scss/             # Stylesheets\n"
                "  i18n/\n"
                "    module.pot          # Translation template\n"
                "  report/\n"
                "    report_templates.xml # QWeb reports\n\n"
                "The __manifest__.py must contain: name, version, "
                "depends, data, and optionally: category, summary, "
                "description, author, website, license, installable, "
                "application, auto_install."
            ),
        },
        {
            "title": "Odoo Controllers and Routing",
            "content": (
                "Odoo web controllers use the http module:\n\n"
                "from odoo import http\n"
                "from odoo.http import request\n\n"
                "class MyController(http.Controller):\n\n"
                "    @http.route('/my/page', type='http', auth='public', "
                "website=True)\n"
                "    def my_page(self, **kwargs):\n"
                "        return request.render('module.template', {})\n\n"
                "    @http.route('/api/data', type='json', auth='user')\n"
                "    def get_data(self, **kwargs):\n"
                "        records = request.env['my.model'].search([])\n"
                "        return records.read(['name', 'value'])\n\n"
                "Route types: 'http' for web pages, 'json' for JSON-RPC.\n"
                "Auth: 'public' (anyone), 'user' (logged in), 'none'.\n"
                "The request object provides env, session, params, etc."
            ),
        },
    ]
    return knowledge


def main():
    parser = argparse.ArgumentParser(
        description="Collect and process Odoo documentation"
    )
    parser.add_argument(
        "--docs-dir",
        default=str(
            PROJECT_ROOT / "data" / "raw" / "odoo_source" / "documentation"
        ),
        help="Path to Odoo documentation repo",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Output directory for processed docs",
    )
    parser.add_argument(
        "--include-knowledge-base",
        action="store_true",
        default=True,
        help="Include built-in Odoo knowledge examples",
    )
    args = parser.parse_args()

    console.print(
        Panel(
            "[bold]Processing Odoo Documentation[/]",
            title="Phase 1b: Documentation Processing",
            border_style="cyan",
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formatter = OdooDataFormatter()
    all_examples = []

    # Process cloned documentation
    docs_dir = Path(args.docs_dir)
    if docs_dir.exists():
        console.print(f"\n[cyan]Processing documentation from {docs_dir}[/]")
        sections = process_documentation(docs_dir)
        console.print(f"  Extracted [bold]{len(sections)}[/] sections")

        for section in track(
            sections, description="Creating training examples"
        ):
            examples = formatter.format_documentation(
                title=section["title"],
                content=section["content"],
            )
            all_examples.extend(examples)
    else:
        console.print(
            f"[yellow]Documentation directory not found: {docs_dir}[/]\n"
            f"Run 01_collect_odoo_source.py first to clone the docs repo."
        )

    # Add built-in knowledge base
    if args.include_knowledge_base:
        console.print("\n[cyan]Adding Odoo knowledge base examples...[/]")
        knowledge = create_odoo_knowledge_base()
        for item in knowledge:
            examples = formatter.format_documentation(
                title=item["title"],
                content=item["content"],
            )
            all_examples.extend(examples)
        console.print(
            f"  Added [bold]{len(knowledge)}[/] knowledge base topics"
        )

    # Save
    docs_file = output_dir / "documentation_examples.jsonl"
    with open(docs_file, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    console.print(
        Panel(
            f"[bold green]Documentation Processing Complete[/]\n\n"
            f"Training examples: {len(all_examples)}\n"
            f"Saved to: {docs_file}",
            title="Documentation Summary",
            border_style="green",
        )
    )

    return all_examples


if __name__ == "__main__":
    main()
