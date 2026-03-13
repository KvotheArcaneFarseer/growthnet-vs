# UVA NSGY AI Lab Code Repository
Version: v1.0  
Last updated: 2026-03-12  
Owner: George Maragkos

## Purpose
This repository is the canonical source of structure for all UVA NSGY AI Lab code. The goal is to make our work reproducible, easy to onboard to, and resilient to lab member turnover.

This repo is organized as a multi project monorepo. Each major effort lives under `projects/` with a consistent internal layout so that anyone can find:
1. Source code
2. Configuration files
3. Runnable scripts and entry points
4. Tests
5. Project notes
6. A minimal README that explains how to run the project and where outputs go on Rivanna

## High level structure
```text 
├── projects/
│   ├── vs_segmentation/
│   ├── mri_registration/
│   ├── vivit/
│   ├── growthnet_diffusion/
│   ├── toy_dataset/
│   ├── radiomics_growth/
│   └── prototypes/
└── README.md
```

## Each project follows the same internal structure:
```text 
├── projects/
│   ├── <project>/
│   │   ├── src/
│   │   ├── configs/
│   │   ├── scripts/
│   │   ├── tests/
│   │   ├── docs/
│   │   └── README.md
...
```

## What goes where when you are coding
### src/
Purpose  
Reusable implementation code: models, datasets, transforms, metrics, utilities, training loops.

Rules  
1. Put code here if you expect it to be imported by multiple scripts.
2. Avoid hard coded file paths. All paths should be passed through config or environment variables.
3. Keep IO logic centralized (for example a dataset loader that reads a manifest).

Examples  
Model definitions, dataset classes, preprocessing functions, embedding extractors.

### configs/
Purpose  
Run definitions. Anything that meaningfully changes the behavior of training or inference should live in a config.

Rules  
1. Prefer YAML for configs.
2. Each experiment should have a named config file, not ad hoc command line flags.
3. Config must include: dataset id, seed, and output run directory template.

Examples  
`train_t1c_t2.yaml`, `infer_embeddings.yaml`, `register_qc1_600.yaml`.

## Environment recording policy
Each project maintains its own versioned environment freeze in GitHub, and each Rivanna run folder stores the exact environment snapshot used at runtime.

GitHub location
1. projects/<project>/configs/env/ contains the latest environment freeze file(s) with versioned modules (only need what works for your latest code)
2. projects/<project>/docs/README.md has to contain an explanation for how to recreate the environment and point to the freeze file(s)

Freeze filename convention
<project>_env_v###_YYYY_MM_DD_<method>.<ext>
Example: vivit_env_v001_2026_03_12_conda.yml

Allowed methods
The script owner may choose one or more of:
1. conda environment export
2. pip requirements or pip freeze
3. HPC module list plus any pip installs

Rivanna requirement
Every run under code_runs/<project>/<experiment>/<runid>/ must include an environment freeze file, generated automatically by the script at runtime when possible.

### scripts/
Purpose  
Human runnable entry points. These call into src and consume configs.

Rules  
1. Every script should accept a config path and optional overrides.
2. Scripts should write outputs to standardized run folders on Rivanna.
3. Scripts should log the git commit, dataset id, seed, and environment.

#### NOTE: SLURM scripts for running on the Afton/Rivanna HPC cluster should be placed here and be clearly documented.

Examples  
`train.py`, `eval.py`, `extract_embeddings.py`, `register.py`, `make_toy_dataset.py`.

### tests/
Purpose  
Small tests that prevent breakage during refactors.

Rules  
1. Keep tests lightweight and runnable on a laptop when possible.
2. At minimum, include a smoke test that imports the project and runs a tiny synthetic example.

Examples  
`test_imports.py`, `test_manifest_loader.py`, `test_forward_pass.py`.

### docs/
Purpose  
Project specific notes that are not appropriate for the README. Keep this short and practical.

Examples  
Data format notes, explanation of model variants, troubleshooting, command examples.

### README.md (within each project)
Purpose  
The minimum operational manual for the project.

Required content  
1. What the project does and the intended outputs.
2. Required inputs (dataset and any pipeline dependencies).
3. One command to run a smoke test.
4. One command to run a real training run on Rivanna.
5. Where outputs are written under the Rivanna/OpenOnDemand `code_runs/` structure.
6. How to reproduce a result (config, commit, dataset id, seed).

#### NOTE: Including separate run methodologies for running locally and on Rivanna with a SLURM script is highly recommended.

## Prototypes policy
`projects/prototypes/` is the sandbox for early work. It is allowed to be messy, but it must still follow two rules:
1. Keep code readable and minimally documented.
2. The moment a prototype becomes shared or used in a real experiment, promote it into the appropriate project and add a proper config and script.

## Reproducibility expectations
Any result referenced in a slide, abstract, manuscript, or group decision must be reproducible from:
1. A config file in `configs/`
2. A specific git commit hash
3. A dataset id (pointing to `dataset_curated` on Rivanna)
4. A seed
5. A run folder produced on Rivanna with logs, metrics, and checkpoints

## Rivanna linkage
This repo does not store large data. Datasets and run outputs live on Rivanna under the lab folder structure:
1. `data_raw/`
2. `dataset_pipeline/`
3. `dataset_curated/`
4. `code_runs/`

Project scripts should write outputs to:
`code_runs/<project>/<experiment>/<runid>/`

## .gitignore policy
This repo uses a single root `.gitignore` for all projects.

What is ignored
1. Large or reproducible artifacts that belong on Rivanna, for example run outputs, checkpoints, and tracking folders.
2. Raw and curated datasets and pipeline products, which must not live in GitHub.
3. Local development files such as virtual environments, caches, and IDE settings.

What should be tracked
1. Source code under projects/*/src and projects/*/scripts
2. Config files under projects/*/configs
3. Documentation under docs and projects/*/docs
4. Small test fixtures required for unit tests

If a file does not show up in `git status`, check whether it is ignored:
git check-ignore -v <path_to_file>

## Contributing and migration
1. Create a branch for non trivial changes.
2. Keep commit messages descriptive.
3. When migrating old scripts, aim for functional equivalence first, then refactor.

If you are unsure where something belongs, default to:
1. Put reusable logic in `src/`
2. Put experiment settings in `configs/`
3. Put runnable entry points in `scripts/`
4. Put quick notes in `docs/`
5. Put early work in `prototypes/`