# TransactionAnalyzer

`TransactionAnalyzer` is a class for performing **association rule mining** and **transaction pattern analysis** — particularly suited for **market basket analysis**.

It processes transactional data (lists of items purchased together), encodes it using one-hot encoding, and computes **item-to-item relationships** through various association metrics.

## Features

- One-hot encoding for transactions using `TransactionEncoder`
- Computes multiple association metrics:
  - Confidence
  - Conviction
  - Zhang’s metric
  - Leverage
  - Cosine similarity
  - Yule’s Q
- Correlation matrix generation for any selected metric
- Model persistence (save/load via pickle)

## Usage

1. Fit the analyzer on a dataset of transactions:

```python
analyzer = TransactionAnalyzer().fit(transactions)
```

2. Calculate associations between items:

```python
analyzer.confidence("milk", "bread")
analyzer.zhang_metric("diapers", "beer")
```

3. Or create a correlation matrix:
```python
analyzer.correlation_matrix(["milk", "bread"], ["butter", "jam"], metric="cosine")
```

## Association Metrics Explained
Each metric measures different aspects of the relationship between items in transactions.
| Metric             | Range         | Interpretation                                         | Recommended Use Case                                                   |
| ------------------ | ------------- | ------------------------------------------------------ | ---------------------------------------------------------------------- |
| **Confidence**     | 0 to 1        | Probability of consequent given antecedent             | Rule filtering: "If A occurs, how likely is B?"                        |
| **Conviction**     | 1 to ∞        | Rule reliability vs independence                       | When measuring rule strength beyond coincidence                        |
| **Zhang’s Metric** | -1 to 1       | Deviation from statistical independence                | Balanced measure, less sensitive to item support                       |
| **Leverage**       | -0.25 to 0.25 | Difference between observed and expected co-occurrence | For detecting statistically significant co-occurrences                 |
| **Cosine**         | 0 to 1        | Vector similarity between items                        | Good for symmetric similarity analysis (e.g., clustering)              |
| **Yule’s Q**       | -1 to 1       | Based on odds ratio                                    | Effective for identifying strong positive/negative binary associations |
