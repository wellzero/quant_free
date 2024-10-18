## Explainability

### Motivation

Here's the translation:

---

One of the most commonly known mistakes in financial research is to select a specific dataset, run a machine learning algorithm, conduct backtesting on the predictions, and repeat this process until satisfactory backtesting results are achieved. Academic journals are filled with such false discoveries, and even large hedge funds continue to fall into this trap. The problem persists even when backtesting is done using walk-forward out-of-sample testing.

Continuously repeating tests on the same data is likely to lead to false discoveries.

This methodological error is notorious among statisticians and is sometimes considered scientific fraud, with warnings about it included in the ethical guidelines of the American Statistical Association. Generally, discovering an investment strategy that meets a standard significance level of 5% is possible after about 20 repetitions.

In this chapter, we will explore why this approach is a waste of time and money and examine how feature importance can offer an alternative.