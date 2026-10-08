# Data Guide — what goes where

Rule of thumb: **media files live in a module's folder; every media file gets exactly one row in that
module's CSV.** Filenames in the CSV must match the file on disk exactly. Run
`python src/check_dataset.py` after every collection session to catch mismatches.

## Layout

```
data/
├── pets.csv                       # one row per pet (shared by all modules)
├── sessions.csv                   # one row per recording session (shared)
├── gait/
│   ├── videos/healthy|limping|uncertain/
│   └── gait_labels.csv
├── posture_activity/
│   ├── videos/
│   └── posture_labels.csv
├── body_condition/
│   ├── images/
│   └── body_condition_labels.csv
├── eye_face/
│   ├── images/
│   └── eye_labels.csv
├── skin_coat/
│   ├── images/
│   └── skin_labels.csv
├── symptoms/
│   └── symptom_log.csv            # owner questionnaire answers, no media
├── injury_triage/
│   ├── videos/                    # external injury signs (non-weight-bearing, swelling ...)
│   ├── photos/
│   ├── xray_images/               # radiographs (vet-provided or public), NOT phone shots of fur
│   └── injury_labels.csv
└── external/                      # anything downloaded from public sources, see SOURCES.md
```

## Naming convention

`<pet_id>_<session>_<what>_<view>_<nn>.<ext>` — for example `dog01_s01_trot_side_01.mp4`,
`cat02_s03_bcs_side_01.jpg`. Use the same `pet_id` everywhere; the personal-baseline and
longitudinal features depend on it.

## Shared files (fill these first)

| File | One row per | Columns |
|---|---|---|
| `pets.csv` | pet | pet_id, species (dog/cat), breed, age_years, sex, weight_kg, known_conditions, notes |
| `sessions.csv` | recording session | session_id, pet_id, date (YYYY-MM-DD), recorded_by, notes |

## Per-module files

| Module | Media goes in | Label file | Allowed values / notes |
|---|---|---|---|
| **Gait** | `gait/videos/<healthy\|limping\|uncertain>/` | `gait/gait_labels.csv` | gait_type: walk, trot. camera_angle: side, front, rear, other. label: healthy, mild, significant, uncertain. suspected_leg: none, front_left, front_right, back_left, back_right. The subfolder must agree with the label (healthy folder = healthy label; limping folder = mild or significant). |
| **Posture and activity** | `posture_activity/videos/` | `posture_activity/posture_labels.csv` | activity: walking, standing, lying, sitting, standing_up, climbing_stairs. behavior_flag: normal, difficulty_rising, weight_shifting, guarding. Longer clips (30-120 s) than gait. |
| **Body condition** | `body_condition/images/` | `body_condition/body_condition_labels.csv` | view: side, top. bcs_9pt: integer 1-9 (the common 9-point body condition scale). scored_by: vet or owner. Ground truth should come from a vet. |
| **Eye / face** | `eye_face/images/` | `eye_face/eye_labels.csv` | eye: left, right, both. finding: normal, redness, discharge, foreign_matter, cloudiness, other. Describes what is visible, not a diagnosis. |
| **Skin and coat** | `skin_coat/images/` | `skin_coat/skin_labels.csv` | body_region: free text (e.g. back, ear, belly). finding: normal, hair_loss, redness, lesion, other. |
| **Owner symptoms** | (no media) | `symptoms/symptom_log.csv` | appetite: normal, reduced, increased. water_intake: normal, reduced, increased. energy: normal, low, high. vocalization: none, whining, yelping. avoids_stairs / limping_noticed: yes, no. |
| **Injury triage** | `injury_triage/videos/`, `photos/`, `xray_images/` | `injury_triage/injury_labels.csv` | media_type: video, photo, xray. injury_signs: none, non_weight_bearing, swelling, abnormal_limb_angle, guarding. fracture_confirmed: yes, no, unknown (only a vet or a radiograph report can say yes). xray_view: free text (e.g. lateral, AP). |

## Columns that appear in most label files

- `label_confirmed_by` / `scored_by`: `owner` or `vet`. Keep this honest. It is what lets you report
  results on vet-confirmed labels separately from owner-labelled ones.
- `source`: `own` (your footage) or `external` (downloaded). If `external`, fill `source_url`.
  Never mix external data into your own folders without this flag. Public downloads go in
  `data/external/` and are listed in `SOURCES.md`.

## Filming rules per module

- **Gait:** side-on, full body in frame, steady camera, hard flat floor. Film the trot as well as the
  walk (a veterinary filming checklist based on ACVS guidelines calls trot the most diagnostic gait and
  suggests 4-6 strides per angle). 5-15 s clips.
- **Posture and activity:** fixed camera, whole animal visible, 30-120 s, include lying down and
  getting up.
- **Body condition:** side and top views, animal standing squarely, coat dry and not fluffed.
- **Eye / face:** close, sharp, even daylight, no flash glare.
- **Skin and coat:** photo of the affected area with a plain background, one image with the whole
  animal for context.
- **Injury triage:** only film or photograph if the pet is already hurt, and never at the cost of
  getting it to a vet. X-ray images must be actual radiograph files or clear photos of a radiograph,
  provided by a vet.

## Minimum amounts to aim for (planning guesses, not validated thresholds)

| Module | Enough for a working demo | Enough for a real evaluation |
|---|---|---|
| Gait | 6 healthy + 4 limping clips | 30+ clips across several pets |
| Posture and activity | 10 clips | 40+ |
| Body condition | 30 images with vet scores | 150+ (spread across scores) |
| Eye / face | use a public set to start | 300+ |
| Skin and coat | use a public set to start | 300+ |
| Symptoms | 20 log rows | 100+ |
| Injury triage (X-ray) | use public/vet radiographs | 500+ |

More detail on public sources and their caveats is in `external/SOURCES.md`.
