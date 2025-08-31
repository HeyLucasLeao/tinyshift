# TransactionAnalyzer

`TransactionAnalyzer` is a class for performing **association rule mining** and **transaction pattern analysis** — particularly suited for **market basket analysis**.

It processes transactional data (lists of items purchased together), encodes it using one-hot encoding, and computes **item-to-item relationships** through various association metrics.

## Features

- One-hot encoding for transactions via `TransactionEncoder`
- Multiple association metrics:
  - Lift
  - Confidence
  - Bayesian Confidence
  - Conviction
  - Zhang’s Metric
  - Yule’s Q Coefficient
  - Hypergeometric *p*-value
- Correlation matrix generation with any supported metric
- Model persistence (save/load with pickle)

## Usage

1. Fit the analyzer on a dataset of transactions:

```python
from transaction_analyzer import TransactionAnalyzer

transactions = [
    ["milk", "bread", "butter"],
    ["beer", "diapers", "bread"],
    ["milk", "beer", "diapers"],
    ["bread", "butter"]
]

analyzer = TransactionAnalyzer().fit(transactions)
## Usage

1. Fit the analyzer on a dataset of transactions:

```python
from transaction_analyzer import TransactionAnalyzer

transactions = [
    ["milk", "bread", "butter"],
    ["beer", "diapers", "bread"],
    ["milk", "beer", "diapers"],
    ["bread", "butter"]
]

analyzer = TransactionAnalyzer().fit(transactions)

```

2. Calculate associations between items:

```python
print(analyzer.confidence("milk", "bread"))
print(analyzer.zhang_metric("diapers", "beer"))
```

3. Or create a correlation matrix:
```python
matrix = analyzer.correlation_matrix(
    ["milk", "bread"],
    ["butter", "jam"],
    metric="hypergeom"
)
print(matrix)

```

## Association Metrics

Each metric measures different aspects of the relationship between items in transactions.

| Metric                  | Range  | Interpretation                                | Question You Want to Answer                                    | Recommended Usage                                             |
| ----------------------- | ------ | --------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------- |
| **Lift**                | 0 → ∞  | Correlation between antecedent and consequent | “How much more often do A and B occur together than expected?” | Good for overall correlation strength                         |
| **Confidence**          | 0 → 1  | Probability of consequent given antecedent    | “If A occurs, how likely is B?”                                | Simple baseline, but may be misleading with rare items        |
| **Bayesian Confidence** | 0 → 1  | Smoothed probability using Bayesian prior     | “If A occurs, how likely is B (with uncertainty handling)?”    | Preferable when data is sparse or sample sizes are small      |
| **Conviction**          | 1 → ∞  | Reliability of a rule beyond independence     | “How strongly does A imply B compared to chance?”              | Useful for rule filtering; robust against skewed supports     |
| **Zhang’s Metric**      | -1 → 1 | Deviation from statistical independence       | “How far is the A→B relation from being independent?”          | Balanced measure less biased by item frequency                |
| **Yule’s Q**            | -1 → 1 | Odds ratio-based association                  | “Do A and B strongly reinforce or oppose each other?”          | Best when interpreting direction and strength of association  |
| **Hypergeom p-value**   | 0 → 1  | Statistical significance of co-occurrence     | “Is the co-occurrence of A and B statistically significant?”   | Use when testing whether an association is unlikely by chance |

