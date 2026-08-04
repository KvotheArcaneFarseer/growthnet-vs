# Mask Generation Evidence Inventory

Generated: 2026-07-18T13:06:03.353130+00:00

Scope: local repository only. No SSH, Rivanna access, remote filesystem, generator tuning, or mask overwrites.

## Evidence

| evidence | classification | finding |
| --- | --- | --- |
| Pulled masks | AVAILABLE_LOCAL | 261 masks in `rivanna_pull/analysis/synthetic_lollipop_v1/masks` |
| Pulled manifest | AVAILABLE_LOCAL | 261 rows; includes target/realized volume, final scale, rotations, seed |
| Pulled generator script | AVAILABLE_LOCAL | `rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py` hash `6efdc3f3b4e161e96f30ad2d0f843aaf786b8bcbc421b7b19b920059c84e5ab9` |
| Current generator script | AVAILABLE_LOCAL | `scripts/generate_synthetic_lollipop_cohort.py` hash `716070b9fa4cfb8ad01377ed88ab0e0b75c5fa87310ace79a8ebe0d29f094efa` |
| Current synthetic.py | AVAILABLE_LOCAL | `projects/vivit/src/data/synthetic.py` hash `bfc5e49871d3d92e28bdfa46b977562afd30569654bd16e2983d5b575b78f566` |
| Git history | AVAILABLE_LOCAL | `generate_synthetic_lollipop_cohort.py` first appears in commit `0457540`; lollipop prototype appears in `383961f` |
| Patch/status files | AVAILABLE_LOCAL | `CLAUDE_TO_CODEX_HANDOFF.md`, `growthnet_status_20260625.txt`, `growthnet_uncommitted_diff_20260625.patch` |
| Exact runtime commit for pulled masks | MISSING | No provenance JSON, command log, package lock, or code hash was stored with the pulled masks |
| Per-voxel compartment labels | MISSING | No saved stem/transition/bulb labels exist for the authoritative masks |

## Local File Dates And Hashes

- Pulled generator script mtime is local copy metadata only and should not be treated as authoritative generation time.
- Current git commit: `b7db1cc55f708ed71f4a9b11da1aef5cc27e3f5e`.
- Manifest hash: `08809ebfce66aad6d76a35672936e85849404c183553249700f6ea4689d85e93`.

## Likely Generation Window

Local evidence points to a generation path after the lollipop prototype was introduced (`383961f`) and before/around the pulled script copy under `rivanna_pull/scripts/`. The precise execution date, runtime commit, and dependency versions were not saved with the masks.
