# Business Entity Resolution — Solution Package

Matches every Source-1 business entity to its corresponding records across Source 2 and Source 3.

## Pipeline Architecture
1. **Multi-Lingual Normalisation & Transliteration**:
   - Phonetic Brahmic transliteration across all 9 Indian scripts with Hindi schwa deletion (`भारत` $\to$ `bharat`).
   - Normalises US, Indian, and French legal forms, street types, domains, handles, and address anchors.
2. **High-Recall Sparse Inverted Blocking**:
   - IDF-weighted token bags (`n:<word>`, `s:<skeleton>`, `ns:<nospace>`, `a:<word>`, `b:<w1>_<w2>`).
   - Fast C++ top-K cosine similarity retrieval via `sparse_dot_topn.sp_matmul_topn`.
3. **Pairwise Feature Engineering**:
   - 40 vectorized features computed via multi-threaded `rapidfuzz` and sparse overlap matrices.
   - Distinctive house number conflict penalties and candidate shortlist gap features.
4. **LightGBM Matching & One-to-One Decision**:
   - LightGBM gradient boosted decision trees.
   - **`best_per_query` constraint**: assigns each Source 2/3 query to at most one Source 1 entity, heavily boosting precision for the macro $F_{0.5}$ metric.

---

## Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── er/                 # Production C++ & Sparse ML Engine
│   │   ├── blocking.py     # Candidate generation (sp_matmul_topn)
│   │   ├── features.py     # 40 vectorized pair features
│   │   ├── model.py        # LightGBM classifier & best_per_query decision
│   │   ├── indic.py        # 9-script Brahmic phonetic transliterator
│   │   ├── normalize.py    # Name/address normaliser
│   │   ├── evaluate.py     # Macro F_0.5 evaluator
│   │   ├── prepare.py      # Parquet caching engine
│   │   ├── sample.py       # Development sampling
│   │   ├── io.py           # Clean TSV parsing & writing
│   │   └── run.py          # train & predict CLI entrypoints
│   ├── normalize.py        # Standalone normaliser
│   ├── metric.py           # Competition F_0.5 metric implementation
│   ├── split.py            # Canonical 80/20 train/val splitter
│   ├── baseline.py         # Exact match baseline
│   ├── blocking.py         # Multi-key inverted index blocking
│   ├── features.py         # 14 similarity/conflict features
│   ├── matching.py         # Hybrid LightGBM / SGD matcher
│   └── pipeline.py         # Fast local & cloud orchestrator
├── scripts/
│   ├── py                  # Python runner with OpenMP resolution
│   ├── setup_vm.sh         # Automated Ubuntu VM setup & idle auto-shutdown
│   ├── idle_shutdown.sh    # Protects AWS credits by powering down after 30m idle
│   └── run_full.sh         # One-command end-to-end full run script
├── README.md               # This documentation
└── requirements.txt        # Pinned dependencies
```

---

## How to Reproduce

### 1. VM One-Time Setup (AWS Ubuntu)
```bash
bash code/business_entity_resolution/scripts/setup_vm.sh
```
*Installs dependencies, builds virtual environment, and installs auto-shutdown to protect AWS credits.*

### 2. End-to-End Full Pipeline (Train + Test Inference + Validation)
```bash
bash code/business_entity_resolution/scripts/run_full.sh
```
*Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv` and validates against `utils/validate_submission.py`.*

### 3. Fast Local Validation Mode (Zero-C++ Dependency on Local Mac)
```bash
PYTHONPATH=code/business_entity_resolution python3 code/business_entity_resolution/src/pipeline.py --mode sample
```

### 4. Official Submission Output Validation
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
