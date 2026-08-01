# Supplementary Analysis (run 0)

- Pairs analyzed: **1916**
- True positives in set: **206**

## A. Ensemble Analysis

| Policy                                              | TP  | FP  | FN  | TN   | Precision | Recall | F1     |
| --------------------------------------------------- | --- | --- | --- | ---- | --------- | ------ | ------ |
| Single-LLM                                          | 198 | 33  | 8   | 1677 | 0.8571    | 0.9612 | 0.9062 |
| Multi-Agent                                         | 192 | 26  | 14  | 1684 | 0.8807    | 0.9320 | 0.9057 |
| Union (either MATCH)                                | 198 | 39  | 8   | 1671 | 0.8354    | 0.9612 | 0.8939 |
| Intersection (both MATCH)                           | 192 | 20  | 14  | 1690 | 0.9057    | 0.9320 | 0.9187 |
| High-confidence agreement (both MATCH, conf >= 0.9) | 122 | 3   | 84  | 1707 | 0.9760    | 0.5922 | 0.7372 |

**Interpretation.** The _union_ policy upper-bounds recall (at the price of precision) and the _intersection_ policy upper-bounds precision. If the union F1 exceeds both individual pipelines, the two pipelines have complementary errors in the strong sense.

## B. Per-agent Counterfactual

| Configuration         | TP  | FP  | FN  | TN   | Precision | Recall | F1     |
| --------------------- | --- | --- | --- | ---- | --------- | ------ | ------ |
| Syntactic agent alone | 193 | 28  | 13  | 1682 | 0.8733    | 0.9369 | 0.9040 |
| Semantic agent alone  | 175 | 33  | 31  | 1677 | 0.8413    | 0.8495 | 0.8454 |
| Orchestrated (paper)  | 192 | 26  | 14  | 1684 | 0.8807    | 0.9320 | 0.9057 |

**Interpretation.** If the syntactic agent's standalone F1 exceeds the orchestrated F1, the orchestration layer is degrading the best single agent by mixing in the weaker semantic agent's verdicts.

## C. Precision-Recall Curve (confidence threshold sweep)

For each pipeline we reinterpret confidence as a match-probability score (score = confidence if verdict = MATCH else 1 - confidence) and sweep the decision threshold over [0, 1]. The table below shows the maximum F1 each pipeline can reach by tuning its threshold, and the threshold at which that F1 occurs.

| Pipeline    | Max F1 | Argmax tau | Precision at max F1 | Recall at max F1 |
| ----------- | ------ | ---------- | ------------------- | ---------------- |
| Single-LLM  | 0.9091 | 0.79       | 0.8962              | 0.9223           |
| Multi-Agent | 0.9157 | 0.76       | 0.9091              | 0.9223           |

See `paper/figures/pr_curves.pdf` for the full curves.
