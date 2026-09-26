# Business Entity Resolution — Solution Package

Matches every Source-1 business entity to its corresponding records across Source 2 and Source 3.

## Pipeline Architecture
1. **Multi-Lingual Normalisation & Transliteration**:
   - Phonetic Brahmic transliteration across all 9 Indian scripts with Hindi schwa deletion (`भारत` $\to$ `bharat`).
   - OCR digit correction (`israe1` $\to$ `israel`, `6lobal` $\to$ `global`), legal forms (`pra li` $\to$ `pvt ltd`), French street expansion (`r` $\to$ `rue`), and component-safe state normalization.
2. **3-Pass Multi-Resolution Inverted Blocking**:
   - Three concurrent IDF-weighted cosine searches per country: Combined ($K=10$), Name-only ($K=5$), and Address-only ($K=5$).
   - Rich n-gram representations: single tokens, consecutive word pairs (`nb:w1_w2`), phonetic skeleton pairs (`sb:k1_k2`), and skip-1 address pairs (`c:w1_w3`).
   - Retrieves candidates even when company names are rebranded (via address pass) or addresses are minimal/empty (via name pass).
   - Fast sparse retrieval via `sparse_dot_topn` with automatic `scipy.sparse` fallback.
3. **91 Vectorized Pair Features**:
   - **Vectorized Token Alignment (`TokenAligner`)**: Fuzzy word-by-word alignment tracking unmatched words, maximum unmatched token IDF (catches domain-substituted decoys like *"Gold Lotus Healthcare"* vs *"Gold Lotus Technology"*), and token coverage fractions.
   - **House Number Granular Closeness**: Levenshtein edit distance, prefix match (`house_prefix`), log absolute difference, and typo-tolerant near numbers.
   - **Shortlist Margins (`mg_*`)**: Competitor score gaps against the query's runner-up candidate across name, address, and cosine blocking scores.
4. **Two-Stage LightGBM Classifier & Context Re-Scorer**:
   - **Stage 1**: LightGBM model (`num_leaves=255`, `learning_rate=0.1`, `max_bin=127`) trained on pairwise features.
   - **Stage 2**: Contextual re-scorer evaluating query competition, candidate competition, and **sibling evidence** (similarity to other queries matching the same S1 entity).
   - **Automatic Validation Safeguard**: Stage 2 is automatically validated and only applied if it strictly improves held-out macro $F_{0.5}$.
   - **One-to-One Decision Constraint**: Assigns each query to at most one entity $\hat{s}_1 = \arg\max P(q, s_1) \ge \tau$, maximizing precision for macro $F_{0.5}$.

---

## Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── er/                 # Production C++ & Sparse ML Engine
│   │   ├── blocking.py     # 3-pass candidate blocking (comb 10, name 5, addr 5)
│   │   ├── features.py     # 91 features (TokenAligner, house Levenshtein, margins)
│   │   ├── stage2.py       # Stage 2 sibling & competition re-scorer
│   │   ├── model.py        # LightGBM classifier & hyperparams
│   │   ├── indic.py        # 9-script Brahmic phonetic transliterator
│   │   ├── normalize.py    # Name/address normaliser with OCR/Indic fixes
│   │   ├── evaluate.py     # Macro F_0.5 evaluator
│   │   ├── prepare.py      # Parquet caching engine
│   │   ├── sample.py       # Deterministic dev sampling
│   │   ├── io.py           # Clean TSV parsing & writing
│   │   └── run.py          # train, predict, & threshold CLI entrypoints
│   ├── normalize.py        # Standalone normaliser
│   ├── metric.py           # Competition F_0.5 metric implementation
│   ├── split.py            # Canonical 80/20 train/val splitter
│   ├── baseline.py         # Exact match baseline
│   ├── blocking.py         # Multi-key inverted index blocking
│   ├── features.py         # Similarity/conflict features
│   ├── matching.py         # Matcher interface
│   └── pipeline.py         # Fast local orchestrator
├── tools/                  # Analysis & calibration scripts
│   ├── threshold_decoy_weight.py # Decoy-weighted threshold optimizer for test set
│   ├── analyze_errors.py   # Detailed validation error budget breakdown
│   └── inspect_test_country.py # French test distribution inspector
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

### 3. Fast Local Validation Mode (Zero External Dependency)
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
