#!/usr/bin/env python3
"""Export a Graphify graph.json into an Obsidian-compatible vault."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _safe_name(value: str) -> str:
    """Return a filesystem-safe markdown basename."""
    cleaned = re.sub(r'[\\/*?:"<>|#^\[\]]+', "_", value.strip())
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip(" ._")
    return cleaned or "node"


def _community_folder(community: Any) -> str:
    """Return a stable folder name for a node community."""
    if community is None:
        return "community_unassigned"
    return f"community_{int(community):03d}"


def _first_nonempty(node: dict[str, Any], keys: list[str]) -> str | None:
    """Return the first non-empty string value from a node dict."""
    for key in keys:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def export_graph_to_obsidian(graph_path: Path, vault_dir: Path) -> Path:
    """Write an Obsidian vault from a Graphify graph.json file."""
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes: list[dict[str, Any]] = graph.get("nodes", [])
    links: list[dict[str, Any]] = graph.get("links", [])

    vault_dir.mkdir(parents=True, exist_ok=True)
    (vault_dir / ".obsidian").mkdir(parents=True, exist_ok=True)
    (vault_dir / ".obsidian" / "app.json").write_text("{}\n", encoding="utf-8")

    node_lookup = {str(node["id"]): node for node in nodes}
    path_map: dict[str, Path] = {}
    used_paths: set[Path] = set()

    for node in nodes:
        node_id = str(node["id"])
        folder = vault_dir / _community_folder(node.get("community"))
        folder.mkdir(parents=True, exist_ok=True)
        basename = _safe_name(node_id)
        candidate = folder / f"{basename}.md"
        if candidate in used_paths:
            suffix = 2
            while True:
                candidate = folder / f"{basename}__{suffix}.md"
                if candidate not in used_paths:
                    break
                suffix += 1
        path_map[node_id] = candidate
        used_paths.add(candidate)

    neighbors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in links:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source in node_lookup and target in node_lookup:
            neighbors[source].append({"other": target, "direction": "outgoing", "edge": edge})
            neighbors[target].append({"other": source, "direction": "incoming", "edge": edge})

    for node_id, node in node_lookup.items():
        title = node.get("label", node_id)
        note_lines = [
            f"# {title}",
            "",
            f"- Node ID: `{node_id}`",
        ]

        node_type = _first_nonempty(node, ["file_type", "type"])
        if node_type:
            note_lines.append(f"- Type: `{node_type}`")

        community = node.get("community")
        if community is not None:
            note_lines.append(f"- Community: `{community}`")

        source_file = _first_nonempty(node, ["source_file"])
        if source_file:
            note_lines.append(f"- Source file: `{source_file}`")

        source_location = _first_nonempty(node, ["source_location"])
        if source_location:
            note_lines.append(f"- Source location: `{source_location}`")

        summary = _first_nonempty(node, ["summary", "description", "note"])
        if summary:
            note_lines.extend(["", "## Summary", "", summary])

        related = neighbors.get(node_id, [])
        unique_links: set[str] = set()
        edge_lines: list[str] = []
        for item in sorted(
            related,
            key=lambda entry: (
                str(entry["other"]),
                str(entry["edge"].get("relation", "")),
                entry["direction"],
            ),
        ):
            other_id = str(item["other"])
            other_node = node_lookup[other_id]
            rel_path = path_map[other_id].relative_to(vault_dir).with_suffix("")
            link_target = str(rel_path).replace("\\", "/")
            alias = other_node.get("label", other_id)
            relation = item["edge"].get("relation", "related_to")
            confidence = item["edge"].get("confidence")
            key = f"{item['direction']}|{relation}|{other_id}|{confidence}"
            if key in unique_links:
                continue
            unique_links.add(key)
            suffix = f" [{confidence}]" if confidence else ""
            edge_lines.append(
                f"- {item['direction']}: `{relation}` -> [[{link_target}|{alias}]]{suffix}"
            )

        note_lines.extend(["", "## Relationships", ""])
        if edge_lines:
            note_lines.extend(edge_lines)
        else:
            note_lines.append("- None")

        path_map[node_id].write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    readme_lines = [
        "# Graphify Obsidian Vault",
        "",
        "This vault was generated from `graphify-out/graph.json`.",
        "",
        "## Structure",
        "",
        "- One markdown note per graph node",
        "- Notes are grouped into `community_###/` folders using the Graphify community id",
        "- Filenames are based on sanitized node ids",
        "- Relationships are preserved with Obsidian `[[wikilinks]]`",
        "",
        "## Open In Obsidian",
        "",
        "1. Open Obsidian.",
        f"2. Choose `Open folder as vault`.",
        f"3. Select `{vault_dir}`.",
        "",
        f"Total notes: {len(nodes)}",
    ]
    (vault_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    return vault_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Graphify graph.json to an Obsidian vault.")
    parser.add_argument(
        "--graph",
        default="graphify-out/graph.json",
        help="Path to Graphify graph.json",
    )
    parser.add_argument(
        "--vault_dir",
        default="graphify-out/obsidian-vault",
        help="Destination Obsidian vault directory",
    )
    args = parser.parse_args()

    vault_path = export_graph_to_obsidian(
        graph_path=Path(args.graph).expanduser().resolve(),
        vault_dir=Path(args.vault_dir).expanduser().resolve(),
    )
    print(f"Created Obsidian vault at {vault_path}")


if __name__ == "__main__":
    main()
