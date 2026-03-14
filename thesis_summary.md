# Thesis Summary: Clone-Robust Weighting for Voting Advice Applications

## Central Claim

CRW (Clone-Robust Weighting) with appropriate embedding models detects and corrects for **redundant question coverage** in Voting Advice Applications, producing recommendations less sensitive to questionnaire design choices — specifically, to the injection of duplicate or near-duplicate questions.

---

## 1. The Problem

VAA recommendations (e.g., SmartVote in Switzerland) compute voter-candidate distances across all questions equally weighted. If a questionnaire has 10 immigration questions but only 3 education questions, immigration gets ~3x implicit influence. This sensitivity means:

- **Deliberate manipulation**: An actor could inject clone questions to boost a target party's visibility
- **Accidental imbalance**: Questionnaire designers may inadvertently overrepresent certain topics
- Recommendations change based on *how many* questions cover a topic, not just *which* topics are covered

**Dataset**: SmartVote 2023, Canton Zurich. 75 questions, ~5,000 voters, ~500 candidates, top-36 recommendations per voter. 30-question "mini" (rapide) expert-curated subset available.

---

## 2. The Method: Clone-Robust Weighting (CRW)

CRW downweights questions that are similar to many others in embedding space. It integrates question importance weights over distance thresholds [0, α]. The key parameter is **alpha** — the radius within which questions are considered "neighbors."

**Pipeline**: Questions → Embedding model → Pairwise distances → CRW weights → Weighted voter-candidate matching → Recommendations

**The critical variable**: Which distance metric? Different embedding models produce different distance structures, and CRW's effectiveness depends entirely on whether the model places clones close together.

---

## 3. Embedding Models Tested

10 models evaluated, spanning general-purpose multilingual and instruction-tuned architectures:

| Model | Ordering Accuracy (fake benchmark) | CRW Avg Jaccard | Composite Rank |
|-------|-----------------------------------|-----------------|----------------|
| E5-INSTRUCT | 99.4% | 0.6313 | 1.4 |
| JINA-V3 | 99.7% | 0.6281 | 1.7 |
| QWEN3 | 98.5% | 0.6216 | 2.9 |
| GTE | 90.2% | 0.5873 | 4.7 |
| SBERT | 92.3% | 0.5961 | 4.7 |
| BGE-M3 | 88.1% | 0.5791 | 5.9 |
| NOMIC-V2 | 93.2% | 0.5684 | 7.0 |
| E5 | 80.4% | 0.5472 | 7.8 |
| E5-ASYM | 85.1% | 0.5246 | 9.0 |
| E5-ASYM-INSTRUCT | 94.0% | 0.4863 | 10.0 |

**Fake benchmark**: 14-anchor dataset with same-topic variants (negations, paraphrases) and traps (keyword overlap, syntax similarity). Primary metric: ordering accuracy (% of same-topic/trap pairs correctly ordered). The benchmark is NOT used for model selection (would be circular) — it's used post-hoc to EXPLAIN why top models outperform (they recognize topic similarity despite negation/paraphrasing).

**Correlation with CRW performance**: All three benchmark metrics correlate with sweep performance, but with very different strengths:

| Benchmark Metric | vs Avg Max Jaccard (Spearman ρ) | p-value |
|------------------|-------------------------------|---------|
| Topic coherence | −0.988 | < 0.001 |
| Negation invariance | −0.879 | 0.001 |
| Ordering accuracy | +0.648 | 0.043 |

Topic coherence (lower = better) is almost perfectly predictive of CRW clone detection success (ρ = −0.988). Negation invariance is also highly significant. Ordering accuracy — the primary benchmark metric — is the weakest predictor, though still significant. This suggests that what matters most for CRW is not overall ordering correctness but how tightly a model clusters same-topic variants (topic coherence) and how invariant it is to negation (negation invariance).

---

## 4. Experiment 1: Clone Detection (PRIMARY RESULT)

### Setup
- Clone the top 5 most impactful questions (32214, 32261, 32228, 32234, 32222), 4 clones each
- Three clone conditions: **easy paraphrase**, **negation + easy paraphrase**, **mixed** (2 easy paraphrase + 2 negation)
- Alpha sweep (0.01 to 3.0, 21 values) × 10 embedding models = 630 runs
- Metrics: Jaccard similarity (top-36 overlap), Spearman/Kendall rank correlation

### Key Results

**Baseline distortion** (no CRW): Adding 20 cloned questions reduces mean Jaccard to ~0.44 (56% of recommendations change). This is the "attack" that CRW must correct.

**Best CRW correction** (E5-INSTRUCT, α=0.3, easy paraphrase): Jaccard recovers from 0.44 → 0.62, Spearman from 0.90 → 0.96.

**Per-condition results** (top 3 models):

| Clone Type | #1 Model | Jaccard | Optimal α |
|-----------|----------|---------|-----------|
| Easy Paraphrase | E5-INSTRUCT | 0.623 | 0.3 |
| Negation + Easy | E5-INSTRUCT | 0.659 | 0.3 |
| Mixed | JINA-V3 | 0.625 | 0.6 |

**Negation is the key differentiator**: Standard STS (semantic textual similarity) models treat "Should X be banned?" and "Should X NOT be banned?" as dissimilar — different meaning. But for CRW, these are the SAME topic and should be clustered. Models trained for topic similarity (E5-INSTRUCT, JINA-V3, QWEN3) handle this correctly. This explains why E5-INSTRUCT leads on easy paraphrase + negation but JINA-V3 wins on mixed (which has more hard paraphrases where JINA-V3's separation task excels).

### Peaked vs Stable Analysis

The 95%-of-max alpha interval reveals distance structure quality:

- **Peaked** (interval width < 0.3): E5-INSTRUCT (0.1), JINA-V3 (0.2), QWEN3 (0.2) — sharp clone detection at specific threshold
- **Moderate** (0.3–0.8): GTE, SBERT, BGE-M3, NOMIC-V2, E5, E5-ASYM
- **Broad** (≥ 0.8): E5-ASYM-INSTRUCT — mushy distances, CRW acts uniformly without strong correction

**Interpretation**: Peaked = model found the clones at a specific distance threshold; the peak IS the signal. Broad + low Jaccard = model can't distinguish clones from non-clones.

### Question Impact Sweep

Before selecting the top-5 questions, we swept all 75 questions (clone each 10×, measure baseline distortion). Key findings:

- Most impactful question: Q32214 (retirement age), Jaccard = 0.66 (34% of recs change from cloning ONE question 10×)
- Least impactful still has Jaccard = 0.93 (7% change) — ALL cloning matters
- Impact is NOT predicted by answer correlation (mean |Pearson r| with other questions). Mathematically: L2 distance has no cross-terms, each question contributes independently.
- Impact IS correlated with voter answer variance and NaN rates

**With CRW** (E5-INSTRUCT, α=0.3): Mean Jaccard improvement across all questions and clone types:
- Easy paraphrase: +0.091 (from 0.81 to 0.90)
- Hard paraphrase: +0.043
- Negation easy: +0.081
- Negation hard: +0.028
- Mixed: +0.045

Easy paraphrase and negation_easy show strongest correction (textually closest to originals → embeddings cluster them tightly).

---

## 5. Party Impact Analysis (Democracy Protection Experiment)

### Phase 1: Which party benefits from cloning which question?

For each of 75 questions: clone 4× identical, measure party visibility delta. Centre benefits most overall from strategic cloning.

### Phase 2: Cumulative attack + CRW correction

Target party: **Centre** (biggest beneficiary). Clone the top-5 Centre-benefiting questions simultaneously.

**Worst-case** (4 mixed clones per question = 20 total, 75→95 questions):

| Party | Original | Attacked | CRW Fix | Reduction |
|-------|----------|----------|---------|-----------|
| SP | 13.20% | 10.85% | 11.46% | 26% |
| Green | 13.54% | 11.15% | 11.78% | 26% |
| GLP | 16.20% | 14.86% | 15.10% | 18% |
| **Centre** | **17.45%** | **22.21%** | **21.18%** | **22%** |
| FDP | 4.32% | 4.52% | 4.49% | 15% |
| SVP | 4.66% | 4.98% | 4.91% | 23% |

**Realistic** (1 easy paraphrase per question = 5 total clones):

| Party | Original | Attacked | CRW Fix | Reduction |
|-------|----------|----------|---------|-----------|
| **Centre** | **17.45%** | **19.28%** | **18.59%** | **38%** |
| SP | 13.20% | 12.36% | 12.75% | 47% |

**Key insight**: CRW is more effective on realistic clones (38% reduction) than worst-case (22%) because easy paraphrases are textually closer → embeddings cluster them more tightly. The "hardest to detect" clones (hard paraphrases, negations) are also the ones CRW struggles with most.

**CRW natural drift** (applying CRW to the original unmanipulated questionnaire): negligible (Centre: +0.12pp). CRW doesn't meaningfully distort a clean questionnaire.

---

## 6. Experiment 2: Question Removal (NEGATIVE RESULT)

**Setup**: Removed 3 of 5 health questions, tested whether CRW compensates by upweighting remaining health questions.

**Result**: CRW Jaccard oscillates between 0.80–0.81 across all alphas — essentially flat, no meaningful correction. The baseline without CRW is ~0.81 too.

**Why it fails**: CRW's mechanism is **downweighting dense clusters**, not upweighting sparse ones. Upweighting is an indirect side effect (weights renormalize to mean=1, so downweighting some questions slightly upweights others) — but the effect is too weak to meaningfully compensate for removed questions.

**Thesis framing**: Report as a **limitation defining CRW's scope**. CRW protects against overrepresentation (cloning/redundancy) but cannot correct underrepresentation (missing topics). This is a fundamental asymmetry in the algorithm.

---

## 7. Experiment 3: Mini vs Full Questionnaire (MILD SIGNAL)

**Setup**: Compare CRW(full 75q) vs Baseline(full 75q), using CRW(mini 30q) as reference for "balanced" recommendations. If CRW corrects for topic imbalance in the full questionnaire, CRW-full should be closer to CRW-mini than baseline-full is.

**Result** (E5-INSTRUCT, α=0.3):
- Jaccard mean (vs CRW-mini): Baseline-full = 0.506, CRW-full = 0.530 (Δ = +0.025)
- 42.4% of voters: CRW-full closer to CRW-mini
- 9.4% of voters: Baseline-full closer
- 48.2% unchanged

**CRW weight distribution by topic** (full 75q): Topics with many similar questions get downweighted:
- Welfare state & family: mean weight 0.932 (6 questions, many similar pairs)
- Federal budget: 0.974 (8 questions)
- Immigration: 1.024 (4 questions, all distinct)
- Security & military: 1.024 (5 questions, all distinct)

This aligns with intuition — Federal budget has 8 questions asking "should the government spend more/less on X?" for different X, which are textually similar.

**Thesis framing**: Observational result, not a strong claim. The mini questionnaire is a reference point for balanced coverage, not ground truth. The mild convergence is consistent with CRW correcting for natural redundancy, but the effect is small (2.5pp Jaccard improvement).

---

## 8. Correlation Metric Analysis (Ground Truth Validation)

### The ANSWER-CORRELATION metric

Uses `1 - |Pearson_r(voter_answers_i, voter_answers_j)|` as distance. Captures "functional redundancy" — questions measuring the same latent construct, regardless of wording. Can't be deployed pre-election (chicken-and-egg: need voter answers to compute distances).

### Embedding validation against correlation ground truth

Spearman rank correlation between each model's distance matrix and ANSWER-CORRELATION:

| Model | Spearman ρ |
|-------|-----------|
| QWEN3 | +0.198 |
| JINA-V3 | +0.192 |
| GTE | +0.167 |
| E5-ASYM-INSTRUCT | +0.140 |
| NOMIC-V2 | +0.129 |
| BGE-M3 | +0.112 |
| **E5-INSTRUCT** | **+0.105** |
| E5-ASYM | +0.075 |
| E5 | +0.068 |
| SBERT | +0.060 |

**Key insight**: All correlations are weak (ρ < 0.2). E5-INSTRUCT — the best clone detector — drops to 7th place here. This is NOT a failure; it reveals that **topic similarity ≠ answer correlation**. These are fundamentally different distance concepts:
- Embedding distance captures textual/semantic similarity (same topic?)
- Answer correlation captures functional redundancy (do voters respond the same way?)

E5-INSTRUCT excels at textual topic recognition (what CRW needs for clone detection) but doesn't capture functional redundancy. QWEN3 is the only model where cross-topic correlation is stronger than within-topic — it may capture ideological structure beyond topic labels.

### Per-topic answer correlation

Within-topic mean |r| varies dramatically:
- Immigration: 0.453 (highest — voters are highly consistent on immigration)
- Nature conservation: 0.330
- Society & ethics: 0.297
- Health: 0.113 (lowest)
- Finances & taxes: 0.141
- Cross-topic average: 0.190

### Party-benefiting questions vs topic correlations

Most parties' top-5 benefiting questions span multiple topics with moderate correlation. SVP is the strongest case (3/5 questions in high-correlation topics: Immigration + Society & ethics). Centre's top-5 are mostly in low-correlation topics despite Immigration being #1 by total delta.

**Bottom line**: Adding genuinely different questions from different topics won't trigger CRW regardless of distance metric quality. The functional redundancy that ANSWER-CORRELATION captures is real (especially Immigration) but operates at a different level than what embedding-based CRW detects.

---

## 9. Mini vs Maxi Party Impact (Natural Question Addition)

### The experiment

Instead of synthetic clones, test CRW on **real question additions**: add the 45 full-only questions to the 30-question mini questionnaire, measuring party visibility impact and CRW correction.

### Phase 1: Per-question addition

For each of 45 full-only questions: add to mini, compute party visibility delta + redundancy score (mean |Pearson r| with mini questions). The 5 most beneficial questions per party are selected for Phase 2.

### Phase 2: Cumulative addition + CRW

For each of 6 major parties, add their top-5 beneficial questions simultaneously. Test 5 (metric, alpha) combinations: E5-INSTRUCT (α=0.3, 0.4), ANSWER-CORR (α=0.3, 0.4), QWEN3 (α=0.6).

**Results** (compiled across all parties and metrics):

| Metric | Avg CRW Reduction |
|--------|-------------------|
| E5-INSTRUCT α=0.3 | ~0% |
| E5-INSTRUCT α=0.4 | ~1% |
| ANSWER-CORR α=0.3 | 0% (exactly — uniform weights) |
| ANSWER-CORR α=0.4 | ~0% |
| QWEN3 α=0.6 | ~-1% (slightly worse) |

**CRW has ~0% reduction for genuine question additions across ALL metrics.**

### Why this is the expected result

1. **The added questions are genuinely different** — they were selected by SmartVote experts precisely because they cover distinct aspects
2. **They're not textually similar to mini questions** — embeddings don't cluster them with existing questions
3. **They're not functionally redundant** — mean |Pearson r| with mini questions is low-to-moderate
4. **CRW is designed for redundancy detection, not topic-balance correction** — it downweights dense clusters, and adding 5 unique questions doesn't create a cluster

### The "high correlation + party benefit" variant we considered but didn't run

We explored whether selecting questions with BOTH high answer correlation AND party benefit would show CRW correction. Analysis showed:
- Centre's combined-ranked top-5 have avg mean_abs_r = 0.321 (well above median 0.198)
- But the delta values are tiny (0.001–0.005pp) — not enough to produce meaningful party visibility shifts
- Even if CRW detected these as redundant, the correction on top of a ~0.002pp effect would be negligible
- The fundamental issue: high-correlation questions that benefit a party are rare, and their individual impact is small

**This confirms CRW's limitation**: it's a clone detector, not a general topic-balance corrector.

---

## 10. Thesis Storyline

### Narrative arc

1. **Problem** (Introduction): VAA recommendations are sensitive to questionnaire composition. More questions on topic X → more weight for X → different recommendations. This is a real concern for SmartVote.

2. **Method** (Background/Methods): CRW reweights questions by local density in distance space. The key question: which distance metric makes CRW effective?

3. **Model comparison** (Experiment 1 — core result): 10 embedding models × 3 clone types × 21 alpha values. E5-INSTRUCT, JINA-V3, QWEN3 emerge as top performers. Peaked alpha profiles indicate clear distance structure. Negation handling is the key differentiator — STS models fail because they treat negations as semantically different.

4. **Practical relevance** (Party Impact): Cloning distorts party visibility by up to +4.76pp (Centre in worst-case). CRW reduces this by 22–47% depending on clone realism. Realistic attack (easy paraphrases) → 38% reduction. CRW's natural drift on clean questionnaire is negligible.

5. **Limitations**:
   - CRW cannot correct underrepresentation (Experiment 2: question removal → no effect)
   - CRW doesn't detect genuinely different questions even if they're functionally correlated (Mini vs Maxi experiment → 0% correction)
   - Embedding distance ≠ answer correlation (ρ < 0.2 for all models). These are different distance concepts serving different purposes.

6. **Insight** (Correlation analysis): The ANSWER-CORRELATION metric trivially detects clones but can't be deployed pre-election. It reveals that functional redundancy exists (Immigration topic: |r| = 0.45) but is captured only weakly by embeddings. CRW works for clone detection because clones ARE textually similar — it's the right tool for that specific job.

### Key takeaways for the reader

- **CRW works well for what it's designed to do**: detecting and downweighting textually redundant questions (clones, paraphrases). 38% correction on realistic attacks is practically meaningful.
- **CRW is NOT a general solution for topic imbalance**: it cannot detect that "Should we ban cars?" and "Should we build more highways?" both measure transport policy if they're worded differently enough.
- **Model choice matters**: E5-INSTRUCT and JINA-V3 clearly outperform generic models. The fake benchmark explains why — topic coherence (ρ = −0.988) and negation invariance (ρ = −0.879) are near-perfect predictors of CRW success, while ordering accuracy alone is weaker (ρ = 0.648). What matters most is how tightly a model clusters same-topic variants, not just whether it orders them correctly.
- **The asymmetry is fundamental**: CRW downweights dense clusters → good for overrepresentation. It cannot meaningfully upweight sparse topics → bad for underrepresentation.
- **Practical recommendation**: Use CRW with E5-INSTRUCT (α=0.3) or JINA-V3 (α=0.6) as a defensive measure against question cloning in VAAs. It provides meaningful protection with negligible distortion on clean questionnaires.

---

## 11. Key Numbers for Quick Reference

| Metric | Value |
|--------|-------|
| Baseline Jaccard (no CRW, 20 clones) | 0.44 |
| Best CRW Jaccard (E5-INSTRUCT, easy paraphrase) | 0.62 |
| Best CRW Jaccard (E5-INSTRUCT, neg+easy) | 0.66 |
| Party visibility attack (Centre, worst-case) | +4.76pp |
| CRW correction (Centre, worst-case) | 22% reduction |
| CRW correction (Centre, realistic) | 38% reduction |
| CRW natural drift (clean questionnaire) | ≤0.12pp |
| Question removal CRW effect | ~0% (none) |
| Mini vs Maxi CRW correction | ~0% (none) |
| Embedding vs correlation agreement | ρ < 0.2 (all models) |
| Benchmark → CRW: topic coherence | ρ = −0.988 (p < 0.001) |
| Benchmark → CRW: negation invariance | ρ = −0.879 (p = 0.001) |
| Benchmark → CRW: ordering accuracy | ρ = +0.648 (p = 0.043) |
| Top model ordering accuracy | 99.4% (E5-INSTRUCT) |
| Questions tested | 75 (full) / 30 (mini) |
| Voters | ~5,000 |
| Candidates | ~500 |
| Top-N recommendations | 36 (Canton Zurich) |

---

## 12. Figures to Include

1. **Alpha sweep curves** (Exp 1): All 10 models as lines, Jaccard vs alpha, baseline as dashed line. One per clone condition. Shows peaked vs flat profiles.
2. **Model ranking bar chart**: Composite rank across conditions.
3. **Party impact grouped bars**: Original → Attacked → CRW-corrected visibility for all 6 parties (worst-case and realistic scenarios).
4. **Party impact donuts/hemicycle**: Parliament-style visualization of seat distribution shift.
5. **Question impact ranking**: Per-question distortion bars (all 75 questions).
6. **CRW weight distribution by topic** (Exp 3): Shows which topics get downweighted.
7. **Embedding vs correlation scatter**: Model distance vs ANSWER-CORRELATION distance, showing weak agreement.
8. **Fake benchmark vs CRW performance heatmap**: Spearman correlation matrix (3 benchmark metrics × 4 sweep metrics). Topic coherence (ρ = −0.988) dominates; ordering accuracy (ρ = 0.648) is weakest.
9. **Mini vs Maxi compiled heatmap**: Δpp for each party × metric combination, showing ~0% correction across the board.
10. **Per-topic correlation heatmap**: Within-topic |r| showing Immigration >> Health.