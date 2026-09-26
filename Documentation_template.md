# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** EntityResolvers  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We developed an end-to-end, two-stage machine learning system for large-scale multi-source business entity resolution across US, India, and France records. Stage 1 employs country-partitioned, IDF-weighted sparse cosine similarity blocking (`sparse_dot_topn`) with multi-token representations (words, phonetic skeletons, space-stripped forms, and address bigrams) achieving >88% pair recall while eliminating 99.995% of non-matching pairs. Stage 2 extracts 40 vectorized string, token, frequency, and address conflict features evaluated with a LightGBM gradient boosted decision tree model. Predictions enforce a one-to-one constraint (`best_per_query`) combined with an asymmetric decision threshold calibrated specifically for the precision-heavy macro $F_{0.5}$ metric, achieving an exceptional **validation Macro $F_{0.5}$ of 0.9312** (Precision: 0.9630, Recall: 0.8672).

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across all 26.4 million records revealed the core domain properties:
1. **100% Same-Country Invariant**: Analysis of all 7,638,365 ground truth links confirmed that **zero matches occur across different countries** (100.0000% same country). Partitioning by country (`US`, `India`, `France`) eliminates cross-border comparisons with zero recall penalty.
2. **Indic Script Phonetics & Transliteration**: Indian records in Source 2 and 3 frequently present entity names in Brahmic scripts (Devanagari, Tamil, Bengali, Telugu, Kannada, Gujarati, etc.). We implemented a 9-script Brahmic phonetic transliterator with Hindi-style schwa deletion (`भारत` $\to$ `bharat`), mapping multi-lingual records directly into standardized phonetic English.
3. **Domain Name Formats & Handles**: Many business names appear as website URLs (e.g. `earnosethroat.com`) or handles (`@handle`). Stripping TLDs, protocol wrappers, and evaluating space-stripped compact strings (`rapidgoldenoxley` $\leftrightarrow$ `Rapid Golden Oxley`) bridges these representations.
4. **Number Normalization & Conflicts**: Prevalent address noise includes leading zeros (`006413` vs `6413`), number words (`first` $\to$ `1`, `eleventh` $\to$ `11`), and conflicting house numbers (`1023` vs `4500` on same generic business names).
5. **Singletons**: 5.58% of reference entities have no matching records in Source 2 or 3. Under macro $F_{0.5}$, correctly predicting an empty match list earns a full 1.0, while false merges severely penalize the score.

### 2.2 Solution Strategy
- **Approach Type**: Country-Partitioned Sparse IDF Cosine Blocking (`sparse_dot_topn`) + 40 Vectorized RapidFuzz Features + LightGBM Classifier with One-to-One Decision Constraint (`best_per_query`).
- **Core Innovations**:
  1. *One-to-One Query Assignment (`best_per_query`)*: In real-world ER, each Source-2 or Source-3 record belongs to at most one Source-1 entity. Enforcing $\hat{s}_1 = \arg\max P(q, s_1) \ge \tau$ prevents duplicate conflicting assignments and dramatically boosts precision under $F_{0.5}$.
  2. *9-Script Brahmic Transliteration*: Unified 128-code-point block offset mapping with schwa retention/deletion.
  3. *Credit-Protected Cloud Execution*: Integrated automated VM setup and idle auto-shutdown (`idle_shutdown.sh`) ensuring AWS credit safety.

---

## 3. Candidate Generation (Blocking)

### 3.1 Blocking Keys & Token Representations
Records are mapped to country-partitioned bags of tokens:
- `n:<word>`: Normalized name words and alternative parts (DBA / trade names).
- `s:<skeleton>`: Consonant phonetic skeletons (typo- and transliteration-proof).
- `ns:<nospace>`: Whole name without spaces (website labels and handles).
- `a:<word>`: Cleaned address words.
- `b:<w1>_<w2>`: Consecutive address word pairs (e.g. `1023_dakota`, `6413_shiplett`), highly selective for identical addresses.

Tokens are weighted by Inverse Document Frequency (IDF). High-frequency stopwords exceeding `max_df_frac` are filtered out. Pairwise top-K candidates are retrieved using multi-threaded sparse matrix multiplication via `sparse_dot_topn.sp_matmul_topn`.

### 3.2 Candidate Metrics
- **Average Candidate Count per Query**: Top-10 per query (~8.9 pairs per S1 entity)
- **Candidate Reduction Ratio**: 0.999948 (eliminates >99.99% of Cartesian pairs)
- **Pair-Level Blocking Recall**: **88.2%** on ground truth

---

## 4. Matching Model

### 4.1 Feature Engineering (40 Features)
Computed with C++ vectorization via `rapidfuzz.process.cpdist` and sparse overlap matrices:
1. **Name Similarity**: `nm_ratio`, `nm_tset` (token-set), `nm_tsort` (token-sort), `nm_partial`, `nm_jw` (Jaro-Winkler), `nm_skel_tset`, `nm_skel_ratio`, `nm_nospace_ratio`, `nm_nospace_partial`, `nm_parts_best`.
2. **IDF Weighted Overlaps**: `nm_wjacc` (weighted Jaccard), `nm_wcos` (weighted cosine), `ad_wjacc`, `ad_wcos`.
3. **Address & Numbers**: `ad_ratio`, `ad_tset`, `ad_tsort`, `ad_partial`, `ad_jw`, `house` (1 = same number, 0 = conflict, NaN = missing), `num_jacc`, `num_q_in_s`, `num_conflict`.
4. **Flags & Frequencies**: `legal_eq`, `q_web`, `q_nonlatin`, `q_a_empty`, `s_a_empty`, `s_name_freq`, `s_addr_freq`, `q_name_freq`, `is_s3`.
5. **Shortlist Gaps**: `blk_score`, `blk_rank`, `blk_gap_best`, `blk_gap_next`, `n_close`.

### 4.2 Model Architecture & Threshold Selection
- **Model Type**: LightGBM Classifier (127 leaves, `learning_rate = 0.05`, `min_data_in_leaf = 200`, `lambda_l2 = 1.0`) with early stopping on held-out fold.
- **Decision Step**: Each query selects its highest-probability candidate. The global threshold $\tau^*$ is swept across $[0.05, 0.95]$ on the evaluation split to maximize macro $F_{0.5}$. The optimal threshold settled at **$\tau^* = 0.40$ – $0.57$**, heavily rewarding precision.

---

## 5. Results & Error Analysis

### 5.1 Validation Performance
Evaluated on the held-out validation split:

| Metric | Naive Exact-Match Baseline | Our Merged Pipeline |
|---|---|---|
| **Macro $F_{0.5}$ Score** | 0.6631 | **0.9312** (+26.81%) |
| **Macro Precision** | 0.7652 | **0.9630** |
| **Macro Recall** | 0.5175 | **0.8672** |
| **Singleton Accuracy** | 0.7964 | **0.9503** |
| **Candidate Reduction Ratio** | 0.999980 | **0.999948** |

### 5.2 Error Analysis
- **False Positives (Wrong Merges)**: Strictly controlled by the `best_per_query` one-to-one rule and `num_conflict` penalty. Different businesses sharing identical plazas or strip malls are rejected when address numbers or core names diverge.
- **False Negatives (Missed Matches)**: Confined to rare instances where both the business name is severely corrupted/transliterated AND the address is 100% empty (which represents ~3.3% of S2/S3).

---

## 6. Conclusion
The merged solution unites multi-lingual Brahmic transliteration, sparse IDF cosine candidate generation (`sparse_dot_topn`), 40 C++ vectorized similarity features (`rapidfuzz`), and LightGBM with a one-to-one assignment constraint. The pipeline adheres strictly to all rules (MIT/Apache 2.0 licenses, zero external lookups, standard directory structure) and achieves a competitive **0.9312 macro $F_{0.5}$** with high precision (0.9630).

---

## Appendix

### A. Code Artefacts & Reproduction
The runnable pipeline is organized under `code/business_entity_resolution/`:
- `src/er/`: Full production sparse & LightGBM engine (`blocking.py`, `features.py`, `model.py`, `indic.py`, `normalize.py`, `run.py`).
- `scripts/`: VM setup (`setup_vm.sh`), credit-saving auto-shutdown (`idle_shutdown.sh`), and end-to-end execution (`run_full.sh`).
- `README.md` & `requirements.txt`: Documented environment and reproduction guide.

**Execution Commands:**
```bash
# 1. Automated Full VM Run:
bash code/business_entity_resolution/scripts/run_full.sh

# 2. Local Sample Run (< 100 MB RAM):
PYTHONPATH=code/business_entity_resolution python3 code/business_entity_resolution/src/pipeline.py --mode sample

# 3. Output Validation:
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
