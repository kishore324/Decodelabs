# Project 3: AI Recommendation Logic - Tech Stack Recommender

DecodeLabs AI Industrial Training Kit (Batch 2026)

A content-based recommender that maps a user's skills to the best-fitting tech job roles,
using TF-IDF weighting and cosine similarity implemented from scratch (no ML libraries).

## Run

```bash
python recommender.py                                       # interactive (asks for 3 skills)
python recommender.py Python "Cloud Computing" Automation   # one-shot
```

Requires Python 3.8+ and nothing else. Keep `raw_skills.csv` next to the script.

## How it follows the project brief

| Brief requirement | Implementation |
|---|---|
| Take user input (min. 3) | `collect_inputs()` / CLI args; refuses fewer than 3 |
| Vector mapping, shared vocabulary | `build_vocabulary()` from the unique skills in `raw_skills.csv`; `ALIASES` maps variants (e.g. "cloud" -> "cloud computing") |
| TF-IDF weighting | `compute_idf()` = log(N / df), `tfidf_vector()` TF = count / total terms |
| Cosine similarity | `cosine_similarity()` = A.B / (‖A‖‖B‖), range 0 to 1 |
| 4-step pipeline | `recommend()`: Ingestion -> Scoring -> Sorting -> Filtering (Top 3) |
| Cold start | Unrecognised input falls back to trending roles (by `demand`) |
| Display recommendations | `show()` prints ranked roles, % match and matched skills |

## Dataset

`raw_skills.csv` has columns `role`, `demand`, `skills` (semicolon-separated). Job roles are the
"items". Add rows or skills freely; the vocabulary rebuilds automatically.

## Sample output

```
$ python recommender.py Python "Cloud Computing" Automation
  1. Cloud Engineer                  44.6% match
  2. Cloud Architect                 32.8% match
  3. AI Engineer                     31.0% match
```

## Ideas to extend (per the Conclusion slide)

- Let users rate skills (1-5) instead of just listing them, and use the rating as the TF weight
- Show a role-to-role similarity score
- Add a Streamlit or Flask front end
