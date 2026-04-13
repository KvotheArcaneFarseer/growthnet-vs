#!/usr/bin/env python3
"""Export GrowthNet Graphify output as an architecture-oriented Obsidian vault."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GROUPS: list[dict[str, str]] = [
    {
        "slug": "orchestration-pipeline",
        "folder": "01 Pipeline Orchestration",
        "layer": "pipeline",
        "description": "Top-level workflow entrypoints, end-to-end embedding orchestration, and batch runners.",
    },
    {
        "slug": "data-ingestion-loading",
        "folder": "02 Data Loading",
        "layer": "data",
        "description": "Temporal loading, sampling, and dataset assembly for MRI sequences.",
    },
    {
        "slug": "preprocessing-transforms",
        "folder": "03 Preprocessing and Transforms",
        "layer": "data",
        "description": "MRI preprocessing, registration, cropping, and train/eval transform pipelines.",
    },
    {
        "slug": "synthetic-generation",
        "folder": "04 Synthetic Tumor Generation",
        "layer": "synthetic",
        "description": "Synthetic VS geometry, growth, and morphology generation logic.",
    },
    {
        "slug": "model-training-inference",
        "folder": "05 Models Training and Inference",
        "layer": "model",
        "description": "Networks, loss/metrics, training operations, pretraining, and inference.",
    },
    {
        "slug": "evaluation-qc-visualization",
        "folder": "06 Evaluation QC Visualization",
        "layer": "evaluation",
        "description": "QC overlays, viewers, animations, and visual inspection assets.",
    },
    {
        "slug": "infrastructure-utils-helpers",
        "folder": "07 Infrastructure Utils Helpers",
        "layer": "support",
        "description": "Run logging, config loading, shared helpers, and framework glue.",
    },
    {
        "slug": "scripts-configs-metadata",
        "folder": "08 Scripts Configs Metadata",
        "layer": "support",
        "description": "Operational scripts, configuration, and project metadata that support the pipeline.",
    },
    {
        "slug": "docs-notes-assets",
        "folder": "09 Docs Notes Assets",
        "layer": "knowledge",
        "description": "READMEs, architecture notes, guides, and non-executable knowledge assets.",
    },
    {
        "slug": "experiments-ad-hoc-analysis",
        "folder": "10 Experiments and Ad Hoc Analysis",
        "layer": "exploration",
        "description": "Experiments, exploratory training scripts, notebooks, and one-off analyses.",
    },
]

GROUP_BY_SLUG = {item["slug"]: item for item in GROUPS}


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".", " ") else "_" for ch in value)
    safe = "_".join(part for part in safe.replace("\r", " ").replace("\n", " ").split())
    safe = safe.strip("._")
    return safe or "node"


def _yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _group_note_name(group_slug: str) -> str:
    return f"_group_{group_slug}.md"


def _classify_from_source(node: dict[str, Any]) -> tuple[str | None, list[str]]:
    """Primary source-based classifier with optional secondary groups."""
    source = str(node.get("source_file") or "")
    file_type = str(node.get("file_type") or "")
    label = str(node.get("label") or "")
    secondaries: list[str] = []

    if source == "embed_tumor.py":
        return "orchestration-pipeline", secondaries
    if source == "scripts/run_batch_embedding.py":
        secondaries.append("scripts-configs-metadata")
        return "orchestration-pipeline", secondaries
    if source.startswith("projects/vivit/src/data/temporal_loader.py") or source.startswith("projects/vivit/src/data/samplers.py"):
        return "data-ingestion-loading", secondaries
    if source.startswith("projects/vivit/src/data/transforms.py") or source.startswith("projects/vivit/src/data/utils.py"):
        secondaries.append("data-ingestion-loading")
        return "preprocessing-transforms", secondaries
    if source.startswith("projects/mri_registration/src/") or source.startswith("projects/mri_registration/scripts/"):
        secondaries.append("scripts-configs-metadata")
        return "preprocessing-transforms", secondaries
    if source.startswith("projects/vivit/src/data/synthetic.py"):
        return "synthetic-generation", secondaries
    if source in {"make_lollipop_animation.py", "make_lollipop_napari.py"}:
        secondaries.append("synthetic-generation")
        return "evaluation-qc-visualization", secondaries
    if source.startswith("projects/vivit/src/networks/") or source.startswith("projects/vivit/src/train/") or source.startswith("projects/vivit/src/inference/"):
        return "model-training-inference", secondaries
    if source.startswith("projects/vivit/experiments/"):
        secondaries.append("model-training-inference")
        return "experiments-ad-hoc-analysis", secondaries
    if source.endswith(".ipynb") or "/notebooks/" in source:
        return "experiments-ad-hoc-analysis", secondaries
    if source.startswith("shared/") or source.startswith("projects/_shared/"):
        secondaries.append("scripts-configs-metadata")
        return "infrastructure-utils-helpers", secondaries
    if source.startswith("embedding_outputs/") or source == "view_napari.py":
        return "evaluation-qc-visualization", secondaries
    if source.startswith("animation_outputs/"):
        secondaries.append("synthetic-generation")
        return "evaluation-qc-visualization", secondaries
    if source.startswith("tmp_seed_validation/") or source.startswith("tmp_batch_outputs/"):
        return "evaluation-qc-visualization", secondaries
    if source in {"README.md", "AGENTS.md", "CLAUDE.md"} or source.startswith("docs/") or source.endswith("README.md"):
        return "docs-notes-assets", secondaries
    if source.endswith(".yaml") or "/configs/" in source:
        return "scripts-configs-metadata", secondaries
    if file_type == "document":
        return "docs-notes-assets", secondaries
    if file_type == "image":
        return "evaluation-qc-visualization", secondaries
    if "experiment" in label.lower() or "notebook" in label.lower():
        return "experiments-ad-hoc-analysis", secondaries
    return None, secondaries


def _classify_nodes(nodes: list[dict[str, Any]], links: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Assign architecture groups, propagating unresolved nodes from neighbors."""
    node_lookup = {str(node["id"]): node for node in nodes}
    neighbors: dict[str, list[str]] = defaultdict(list)
    for edge in links:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source in node_lookup and target in node_lookup:
            neighbors[source].append(target)
            neighbors[target].append(source)

    primary: dict[str, str] = {}
    secondary: dict[str, list[str]] = defaultdict(list)

    for node_id, node in node_lookup.items():
        group, secondary_groups = _classify_from_source(node)
        if group is not None:
            primary[node_id] = group
        if secondary_groups:
            secondary[node_id].extend(secondary_groups)

    changed = True
    while changed:
        changed = False
        for node_id in node_lookup:
            if node_id in primary:
                continue
            counter = Counter(primary[nbr] for nbr in neighbors.get(node_id, []) if nbr in primary)
            if counter:
                group = counter.most_common(1)[0][0]
                primary[node_id] = group
                changed = True

    for node_id in node_lookup:
        if node_id not in primary:
            primary[node_id] = "infrastructure-utils-helpers"

    # Secondary neighbor-derived hints for concept nodes without a source file.
    for node_id, node in node_lookup.items():
        if node.get("source_file"):
            continue
        counter = Counter(primary[nbr] for nbr in neighbors.get(node_id, []) if nbr in primary and primary[nbr] != primary[node_id])
        for group, _ in counter.most_common(2):
            if group not in secondary[node_id]:
                secondary[node_id].append(group)

    return primary, secondary


def export_architecture_vault(graph_path: Path, vault_dir: Path) -> Path:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes: list[dict[str, Any]] = graph.get("nodes", [])
    links: list[dict[str, Any]] = graph.get("links", [])
    primary_groups, secondary_groups = _classify_nodes(nodes, links)
    node_lookup = {str(node["id"]): node for node in nodes}

    if vault_dir.exists():
        shutil.rmtree(vault_dir)
    vault_dir.mkdir(parents=True, exist_ok=True)
    obsidian_dir = vault_dir / ".obsidian"
    obsidian_dir.mkdir(parents=True, exist_ok=True)
    (obsidian_dir / "app.json").write_text("{}\n", encoding="utf-8")
    (obsidian_dir / "appearance.json").write_text("{}\n", encoding="utf-8")
    (obsidian_dir / "core-plugins.json").write_text(
        json.dumps(
            {
                "file-explorer": True,
                "global-search": True,
                "graph": True,
                "backlink": True,
                "outgoing-link": True,
                "tag-pane": True,
                "properties": True,
                "page-preview": True,
                "outline": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    path_map: dict[str, Path] = {}
    for node_id, node in node_lookup.items():
        group = GROUP_BY_SLUG[primary_groups[node_id]]
        folder = vault_dir / group["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        note_path = folder / f"{_safe_name(node_id)}.md"
        path_map[node_id] = note_path

    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in links:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source in node_lookup and target in node_lookup:
            outgoing[source].append(edge)
            incoming[target].append(edge)

    # Root architecture map
    map_lines = [
        "---",
        'title: "GrowthNet Architecture Map"',
        'type: "architecture-index"',
        "tags:",
        '  - "architecture/index"',
        "---",
        "",
        "# GrowthNet Architecture Map",
        "",
        "This vault reorganizes `graphify-out/graph.json` into architecture-facing layers.",
        "",
        "## Layers",
        "",
    ]
    for group in GROUPS:
        group_note = f"{group['folder']}/{_group_note_name(group['slug']).removesuffix('.md')}"
        map_lines.append(f"- [[{group_note}|{group['folder']}]]")
        map_lines.append(f"  - {group['description']}")
    (vault_dir / "00 Architecture Map.md").write_text("\n".join(map_lines) + "\n", encoding="utf-8")

    # Group index notes
    group_members: dict[str, list[str]] = defaultdict(list)
    for node_id, group_slug in primary_groups.items():
        group_members[group_slug].append(node_id)

    for group in GROUPS:
        folder = vault_dir / group["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        members = sorted(group_members.get(group["slug"], []))
        lines = [
            "---",
            f"title: {_yaml_quote(group['folder'])}",
            'type: "architecture-group"',
            f"architecture_group: {_yaml_quote(group['slug'])}",
            f"layer: {_yaml_quote(group['layer'])}",
            "tags:",
            f'  - "architecture/{group["slug"]}"',
            f'  - "layer/{group["layer"]}"',
            "---",
            "",
            f"# {group['folder']}",
            "",
            group["description"],
            "",
            f"## Notes ({len(members)})",
            "",
        ]
        for node_id in members[:200]:
            node = node_lookup[node_id]
            note_path = path_map[node_id].relative_to(vault_dir).with_suffix("")
            lines.append(f"- [[{str(note_path).replace(chr(92), '/') }|{node.get('label', node_id)}]]")
        (folder / _group_note_name(group["slug"])).write_text("\n".join(lines) + "\n", encoding="utf-8")

    for node_id, node in node_lookup.items():
        group = GROUP_BY_SLUG[primary_groups[node_id]]
        secondaries = secondary_groups.get(node_id, [])
        label = str(node.get("label", node_id))
        file_type = str(node.get("file_type", "unknown"))
        frontmatter = [
            "---",
            f"title: {_yaml_quote(label)}",
            f"graphify_id: {_yaml_quote(node_id)}",
            f"architecture_group: {_yaml_quote(group['slug'])}",
            f"layer: {_yaml_quote(group['layer'])}",
            f"file_type: {_yaml_quote(file_type)}",
        ]
        if node.get("community") is not None:
            frontmatter.append(f"community: {int(node['community'])}")
        if node.get("source_file"):
            frontmatter.append(f"source_file: {_yaml_quote(str(node['source_file']))}")
        if node.get("source_location"):
            frontmatter.append(f"source_location: {_yaml_quote(str(node['source_location']))}")
        if secondaries:
            frontmatter.append("secondary_groups:")
            for item in sorted(dict.fromkeys(secondaries)):
                frontmatter.append(f"  - {_yaml_quote(item)}")
        frontmatter.append("tags:")
        frontmatter.append(f'  - "architecture/{group["slug"]}"')
        frontmatter.append(f'  - "layer/{group["layer"]}"')
        frontmatter.append(f'  - "type/{file_type}"')
        for item in sorted(dict.fromkeys(secondaries)):
            frontmatter.append(f'  - "secondary/{item}"')
        frontmatter.append("---")

        summary = None
        for key in ["summary", "description", "note"]:
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                summary = value.strip()
                break

        lines = frontmatter + [
            "",
            f"# {label}",
            "",
            f"- Node ID: `{node_id}`",
            f"- Primary group: `{group['slug']}`",
        ]
        if secondaries:
            lines.append("- Secondary groups: " + ", ".join(f"`{item}`" for item in sorted(dict.fromkeys(secondaries))))
        if summary:
            lines.extend(["", "## Summary", "", summary])

        lines.extend(["", "## Outgoing Links", ""])
        out_edges = sorted(
            outgoing.get(node_id, []),
            key=lambda edge: (str(edge.get("relation", "")), str(edge.get("target", ""))),
        )
        if out_edges:
            for edge in out_edges:
                target_id = str(edge["target"])
                target_node = node_lookup[target_id]
                target_link = str(path_map[target_id].relative_to(vault_dir).with_suffix("")).replace("\\", "/")
                confidence = edge.get("confidence")
                suffix = f" [{confidence}]" if confidence else ""
                lines.append(f"- `{edge.get('relation', 'related_to')}` -> [[{target_link}|{target_node.get('label', target_id)}]]{suffix}")
        else:
            lines.append("- None")

        lines.extend(["", "## Incoming Links", ""])
        in_edges = sorted(
            incoming.get(node_id, []),
            key=lambda edge: (str(edge.get("relation", "")), str(edge.get("source", ""))),
        )
        if in_edges:
            for edge in in_edges:
                source_id = str(edge["source"])
                source_node = node_lookup[source_id]
                source_link = str(path_map[source_id].relative_to(vault_dir).with_suffix("")).replace("\\", "/")
                confidence = edge.get("confidence")
                suffix = f" [{confidence}]" if confidence else ""
                lines.append(f"- [[{source_link}|{source_node.get('label', source_id)}]] -> `{edge.get('relation', 'related_to')}`{suffix}")
        else:
            lines.append("- None")

        path_map[node_id].write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Vault README with manual graph group filters.
    readme_lines = [
        "# GrowthNet Architecture Vault",
        "",
        "This vault is an architecture-oriented view of `graphify-out/graph.json`.",
        "",
        "## Group Filters",
        "",
        "Use either folder-based filters or tag-based filters in Obsidian Graph View.",
        "",
    ]
    for group in GROUPS:
        folder_rule = f'path:"{group["folder"]}"'
        tag_rule = f'tag:#architecture/{group["slug"]}"'
        readme_lines.append(f"- **{group['folder']}**")
        readme_lines.append(f"  - Folder rule: `{folder_rule}`")
        readme_lines.append(f"  - Tag rule: `tag:#architecture/{group['slug']}`")
    (vault_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    # Helpful machine-readable assignment manifest.
    manifest = {
        "graph_source": str(graph_path),
        "group_counts": {slug: len(ids) for slug, ids in sorted(group_members.items())},
        "group_definitions": GROUPS,
        "node_assignments": {
            node_id: {
                "label": node_lookup[node_id].get("label", node_id),
                "primary_group": primary_groups[node_id],
                "secondary_groups": secondary_groups.get(node_id, []),
                "file_type": node_lookup[node_id].get("file_type"),
                "source_file": node_lookup[node_id].get("source_file"),
            }
            for node_id in sorted(node_lookup)
        },
    }
    (vault_dir / "architecture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return vault_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GrowthNet Graphify output as an architecture-oriented Obsidian vault.")
    parser.add_argument("--graph", default="graphify-out/graph.json", help="Path to graphify graph.json")
    parser.add_argument("--vault_dir", default="graphify-out/obsidian-architecture-vault", help="Destination vault directory")
    args = parser.parse_args()

    vault_dir = export_architecture_vault(
        graph_path=Path(args.graph).expanduser().resolve(),
        vault_dir=Path(args.vault_dir).expanduser().resolve(),
    )
    print(f"Created architecture vault at {vault_dir}")


if __name__ == "__main__":
    main()
