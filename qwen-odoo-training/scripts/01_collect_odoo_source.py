#!/usr/bin/env python3
"""
Step 1: Collect Odoo Source Code
Clones Odoo community and optionally enterprise repos,
then catalogs all modules and source files.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel

from utils.progress_tracker import PhaseTracker
from utils.data_utils import scan_odoo_modules, collect_source_files

console = Console()

# Odoo repositories
ODOO_REPOS = {
    "community": {
        "url": "https://github.com/odoo/odoo.git",
        "branch": "17.0",  # Change to target version
        "description": "Odoo Community Edition",
    },
    "enterprise": {
        "url": "https://github.com/odoo/enterprise.git",
        "branch": "17.0",
        "description": "Odoo Enterprise (requires access)",
    },
    "documentation": {
        "url": "https://github.com/odoo/documentation.git",
        "branch": "17.0",
        "description": "Odoo Official Documentation",
    },
}

# Additional community repos for more training data
EXTRA_REPOS = {
    "oca_server_tools": {
        "url": "https://github.com/OCA/server-tools.git",
        "branch": "17.0",
        "description": "OCA Server Tools",
    },
    "oca_web": {
        "url": "https://github.com/OCA/web.git",
        "branch": "17.0",
        "description": "OCA Web Addons",
    },
    "oca_account": {
        "url": "https://github.com/OCA/account-financial-tools.git",
        "branch": "17.0",
        "description": "OCA Account Financial Tools",
    },
}


def clone_repo(
    name: str,
    url: str,
    branch: str,
    target_dir: Path,
    depth: int = 1,
) -> bool:
    """Clone a git repository. Returns True on success."""
    if target_dir.exists() and any(target_dir.iterdir()):
        console.print(
            f"  [yellow]Repo '{name}' already exists at {target_dir}, "
            f"pulling latest...[/]"
        )
        try:
            subprocess.run(
                ["git", "-C", str(target_dir), "pull", "origin", branch],
                check=True,
                capture_output=True,
                text=True,
            )
            console.print(f"  [green]Updated {name}[/]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(
                f"  [yellow]Pull failed for {name}: {e.stderr}. "
                f"Using existing code.[/]"
            )
            return True

    console.print(
        f"  Cloning [bold]{name}[/] (branch: {branch})..."
    )
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            "git",
            "clone",
            "--branch",
            branch,
            "--depth",
            str(depth),
            "--single-branch",
            url,
            str(target_dir),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            console.print(f"  [green]Cloned {name} successfully[/]")
            return True
        else:
            console.print(
                f"  [red]Failed to clone {name}: {result.stderr}[/]"
            )
            return False
    except subprocess.TimeoutExpired:
        console.print(f"  [red]Timeout cloning {name}[/]")
        return False
    except FileNotFoundError:
        console.print(
            "  [red]git not found. Please install git first.[/]"
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Collect Odoo source code for training"
    )
    parser.add_argument(
        "--odoo-version",
        default="17.0",
        help="Odoo version branch (default: 17.0)",
    )
    parser.add_argument(
        "--include-enterprise",
        action="store_true",
        help="Include Odoo Enterprise repo (requires access)",
    )
    parser.add_argument(
        "--include-oca",
        action="store_true",
        default=True,
        help="Include OCA community repos (default: True)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "data" / "raw" / "odoo_source"),
        help="Target directory for source code",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="Git clone depth (default: 1 for shallow clone)",
    )
    args = parser.parse_args()

    # Update branches
    for repo in ODOO_REPOS.values():
        repo["branch"] = args.odoo_version
    for repo in EXTRA_REPOS.values():
        repo["branch"] = args.odoo_version

    phase_tracker = PhaseTracker(
        log_dir=str(PROJECT_ROOT / "outputs" / "logs")
    )
    phase_tracker.start_phase("data_collection")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold]Collecting Odoo {args.odoo_version} Source Code[/]\n"
            f"Target: {data_dir}",
            title="Phase 1: Data Collection",
            border_style="cyan",
        )
    )

    # ---- Clone repos ----
    repos_to_clone = {
        "community": ODOO_REPOS["community"],
        "documentation": ODOO_REPOS["documentation"],
    }
    if args.include_enterprise:
        repos_to_clone["enterprise"] = ODOO_REPOS["enterprise"]
    if args.include_oca:
        repos_to_clone.update(EXTRA_REPOS)

    cloned = {}
    for name, info in repos_to_clone.items():
        target = data_dir / name
        console.print(
            f"\n[cyan]>>> {info['description']}[/] ({info['url']})"
        )
        success = clone_repo(
            name, info["url"], info["branch"], target, args.depth
        )
        cloned[name] = success

    # ---- Catalog modules ----
    console.print(
        "\n[bold cyan]Cataloging Odoo modules...[/]"
    )

    all_modules = []
    for name, success in cloned.items():
        if not success:
            continue
        repo_path = data_dir / name
        if not repo_path.exists():
            continue

        # Odoo community has addons in odoo/addons and addons/
        search_dirs = [repo_path]
        if name == "community":
            for sub in ["addons", "odoo/addons"]:
                sub_path = repo_path / sub
                if sub_path.exists():
                    search_dirs.append(sub_path)

        for search_dir in search_dirs:
            modules = scan_odoo_modules(str(search_dir))
            for m in modules:
                m["source_repo"] = name
            all_modules.extend(modules)

    console.print(
        f"  Found [bold]{len(all_modules)}[/] Odoo modules"
    )

    # ---- Collect source files ----
    console.print("\n[bold cyan]Collecting source files...[/]")

    all_files = []
    for name, success in cloned.items():
        if not success:
            continue
        repo_path = data_dir / name
        if repo_path.exists():
            files = collect_source_files(str(repo_path))
            for f in files:
                f["source_repo"] = name
            all_files.extend(files)
            console.print(
                f"  {name}: [bold]{len(files)}[/] files collected"
            )

    console.print(
        f"\n  Total: [bold]{len(all_files)}[/] source files"
    )

    # ---- Save catalog ----
    catalog_dir = PROJECT_ROOT / "data" / "processed"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    modules_file = catalog_dir / "modules_catalog.json"
    with open(modules_file, "w") as f:
        json.dump(all_modules, f, indent=2)
    console.print(f"  Module catalog saved to {modules_file}")

    files_manifest = catalog_dir / "files_manifest.json"
    manifest_data = [
        {k: v for k, v in f.items() if k != "content"}
        for f in all_files
    ]
    with open(files_manifest, "w") as f:
        json.dump(manifest_data, f, indent=2)
    console.print(f"  Files manifest saved to {files_manifest}")

    # Save collection stats
    total_lines = sum(
        f["content"].count("\n") for f in all_files
    )
    total_size_mb = sum(f["size"] for f in all_files) / (1024 * 1024)

    stats = {
        "odoo_version": args.odoo_version,
        "repos_cloned": {k: v for k, v in cloned.items()},
        "total_modules": len(all_modules),
        "total_files": len(all_files),
        "total_lines": total_lines,
        "total_size_mb": round(total_size_mb, 2),
        "files_by_extension": {},
    }
    for f in all_files:
        ext = Path(f["path"]).suffix
        stats["files_by_extension"][ext] = (
            stats["files_by_extension"].get(ext, 0) + 1
        )

    stats_file = catalog_dir / "collection_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    console.print(
        Panel(
            f"[bold green]Data Collection Complete[/]\n\n"
            f"Modules:    {len(all_modules)}\n"
            f"Files:      {len(all_files)}\n"
            f"Lines:      {total_lines:,}\n"
            f"Size:       {total_size_mb:.1f} MB\n"
            f"Stats:      {stats_file}",
            title="Collection Summary",
            border_style="green",
        )
    )

    phase_tracker.end_phase("data_collection")
    return all_files


if __name__ == "__main__":
    main()
