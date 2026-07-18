# GrowthNet Lab Meeting Status
**Prepared:** 2026-04-24  
**Meeting:** Friday 2026-04-25, 3pm  
**Audience:** Ethan, Sofia, George

---

## Deliverable: DONE

```
lab_meeting_exports/GrowthNet_LabMeeting_2026-04-25.pptx
```

14MB PowerPoint, 8 slides, ready to open and share.

---

## What the Deck Contains

| Slide | Content |
|-------|---------|
| 1 | Title slide — GrowthNet, date, audience |
| 2 | Pipeline overview — what QC panels show, patient info, sample count |
| 3 | Sample A — auto seed α (early + late QC panel) |
| 4 | Sample B — auto seed β (early + late QC panel) |
| 5 | Sample C — alternative seed (early + late QC panel) |
| 6 | Sample D — fixed seed, run 1 (early + late QC panel) |
| 7 | Sample E — direct single-case run (early + late QC panel) |
| 8 | 3D napari renders — two camera angles of synthetic VS mask |

---

## What Each QC Panel Shows (2D)

Each case slide shows **two 3×3 grids** side by side:
- **Left panel**: t=0 (smallest/earliest synthetic tumour)
- **Right panel**: t=4 (late/elongated tumour)

Each 3×3 grid rows:
- Axial / Coronal / Sagittal slice

Each 3×3 grid columns:
- Original MRI (grey) | Real segmentation overlay (red) | Embedded synthetic tumour (cyan)

---

## Honest Assessment

### Patient Data
- **Only one patient** is in the pipeline: case 147_0_0
- All 5 samples are different synthetic tumour generations of the same patient
- They show varied size, orientation, and placement — this is by design (stochastic seeds)
- This is appropriate for showing "how variable the generation is" but not multi-patient coverage

### 3D Screenshots
- **Not freshly rendered** — extracted from existing napari MP4 files (already in repo)
- Frame quality: ~2K resolution, good visual, but these are napari screenshots of the lollipop geometry, not volumetric renders
- No pyvista/VTK 3D rendering capability confirmed (pyvista not installed)

### What Is NOT Available
- Multiple real patients (only 147_0_0 in Downloads)
- Fresh napari 3D renders (napari requires an interactive display; headless rendering not set up)
- nilearn / pyvista volumetric renders
- Growth animation embedded in PowerPoint (GIF/MP4 files are in animation_outputs/ but not in deck)

---

## To Add More Cases

If more MRI+segmentation NIfTI pairs become available, add rows to `tmp_batch_cases.csv` and run:

```bash
python3 scripts/run_batch_embedding.py \
  --input tmp_batch_cases.csv \
  --out-dir tmp_batch_outputs
```

Then re-run:
```bash
python3 scripts/build_lab_meeting_deck.py
```

The deck builder `CASES` list at the top of the script is the only thing to update.

---

## To Regenerate the Deck

```bash
cd /Users/kvothearcane/Personal/Coding\ Projects/GrowthNet

# Step 1: Extract 3D frames (only needed once)
python3 scripts/export_3d_frames.py

# Step 2: Build deck
python3 scripts/build_lab_meeting_deck.py

# Output: lab_meeting_exports/GrowthNet_LabMeeting_2026-04-25.pptx
```

---

## Files Created This Session

| File | Purpose |
|------|---------|
| `scripts/export_3d_frames.py` | Extracts PNG frames from napari MP4s |
| `scripts/build_lab_meeting_deck.py` | Assembles the PowerPoint deck |
| `lab_meeting_exports/GrowthNet_LabMeeting_2026-04-25.pptx` | **The deck** |
| `lab_meeting_exports/3d_frames/3d_angle1.png` | 3D napari frame, angle 1 |
| `lab_meeting_exports/3d_frames/3d_angle2.png` | 3D napari frame, angle 2 |
| `docs/GROWTHNET_LAB_MEETING_STATUS.md` | This file |

No core pipeline code was modified.
