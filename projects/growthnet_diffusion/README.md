# <PROJECT NAME>
Version: v0.0  
Last updated: 2026-03-12  
Primary owner: <NAME>

## Purpose
One to three sentences describing what this project does and what it produces.

## Inputs
1. Dataset id
   Example: dataset_curated/uva_vs_notreat/uva_vs_v0003_qc1_registered
2. Optional pipeline dependencies
   Example: dataset_pipeline/uva_vs_notreat/mri_registration/reg_v002/...

## Outputs
All outputs must be written on Rivanna under:
`code_runs/<project>/<experiment>/<runid>/`

This project produces:
1. <CHECKPOINTS OR ARTIFACTS>
2. <METRICS>
3. <OPTIONAL PLOTS OR TABLES>

## Directory layout
src contains reusable code imported by scripts  
configs contains YAML configs for training and inference  
scripts contains runnable entry points that consume configs  
tests contains lightweight tests and smoke tests  
docs contains notes and troubleshooting

## Quickstart
### 1. Smoke test
Command:
<INSERT COMMAND>

Expected result:
One sentence describing what success looks like.

### 2. Standard run on Rivanna
Command:
<INSERT COMMAND>

Where outputs go:
code_runs/<project>/<experiment>/<runid>/

## Reproducibility checklist
A run is reproducible only if the run folder contains:
config.yaml  
git_commit.txt  
dataset_id.txt  
seed.txt if applicable  
env.txt or pip_freeze.txt  
logs  
metrics  
checkpoints if training

## Maintainers and handoff notes
Primary owner: <NAME>  
Secondary: <NAME OR NONE>  
Notes:
One to three bullets about current priorities and what a successor should look at first.