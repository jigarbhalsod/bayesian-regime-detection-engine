# 06. Advanced Forecasting Models

## Purpose

This document covers the two advanced forecasting approaches we decided to study:

1. **Bayesian Neural Networks (BNN)**
2. **Time-Series Foundation Models (TSFM)**

Both are mainly considered for **equity-direction forecasting**, not as the primary hidden-regime detector.

---

# 1. Bayesian Neural Network

## 1.1 What Is a BNN?

A **Bayesian Neural Network is a neural network that also represents uncertainty in its predictions**.

Simple idea:

> **BNN = Neural Network + Uncertainty**

---

## 1.2 Why Are We Considering It?

Financial markets can have complicated and nonlinear relationships.

For example:

- Volatility may affect returns differently during different regimes.
- Several features may interact in ways that are difficult for simple models to capture.
- The same market signal may have different effects under different conditions.

A BNN can potentially learn these complex relationships.

---

## 1.3 What Will BNN Do in Project 1A?

BNN will **not be our main regime detector**.

Its main role will be:

> **Predict equity direction using financial features and information from the regime layer.**

For example:

```text
Market Features
      +
HMM Regime Information
      ↓
     BNN
      ↓
Direction Prediction
      +
Uncertainty
```

---

## 1.4 What Can We Give to the BNN?

Potential inputs include:

- NIFTY returns
- Momentum
- Volatility
- Volume
- Market breadth
- India VIX
- Selected macro/market variables
- HMM regime probabilities
- BOCPD change signal
- Potentially RS-VAR outputs

We will not automatically use everything.

Features will be selected and tested later.

---

## 1.5 What Does BNN Give Us?

Potential outputs:

- Up/down direction probability
- Prediction uncertainty
- Potentially a probability distribution rather than only one prediction

Example:

```text
NIFTY Direction

Up:   72%
Down: 28%

Prediction uncertainty: Moderate
```

The exact output design will be finalized during the modelling phase.

---

## 1.6 Why Not Use BNN for Regime Detection?

Our main regime problem is:

> **The regime is hidden and needs to be inferred from observable market behaviour.**

HMM is naturally designed around this hidden-state problem.

BNN is better suited to learning complex relationships for forecasting.

Therefore:

> **HMM detects the hidden regime; BNN helps forecast direction.**

---

## 1.7 Main Advantage

**Nonlinear forecasting + uncertainty.**

BNN can potentially learn relationships that simpler statistical models may miss.

---

## 1.8 Main Concern

BNNs are more complex than basic statistical models.

Potential concerns include:

- Higher computational cost.
- More difficult training.
- More hyperparameters.
- Greater overfitting risk.
- Potential latency during inference.
- More difficult debugging and interpretation.

For our system, the additional complexity must be justified by measurable improvement.

---

## 1.9 How BNN Fits the Hybrid Architecture

We do not want BNN to control the whole system.

Instead:

```text
HMM
 ↓
Regime information
 ↓
BNN
 ↓
Direction prediction
 ↓
Ensemble
```

The BNN's influence can increase when the current situation is one where nonlinear forecasting appears useful.

Other models still provide supporting information.

---

## 1.10 BNN Decision

🟡 **TEST**

### Reason

> **BNN is promising for directional forecasting because it can model complex relationships and uncertainty, but its extra complexity and latency mean we should keep it only if experiments prove that it adds real value.**

---

# 2. Time-Series Foundation Models

## 2.1 What Is a TSFM?

A **Time-Series Foundation Model is a model that has been pre-trained on large amounts of time-series data and can then be used for forecasting.**

Simple idea:

> **TSFM = a pre-trained model for time-series forecasting.**

Examples we will consider:

- **Chronos**
- **TimesFM**

---

## 2.2 Why Are We Considering Them?

Traditional models usually need to be trained specifically on our data.

Foundation models offer another possibility:

> Start with a model that has already learned general time-series patterns, then test whether those patterns help with our financial forecasting problem.

---

## 2.3 What Will TSFMs Do?

They will mainly act as **forecasting models/benchmarks**.

They are not our primary regime detectors.

Example:

```text
Historical Market Data
       ↓
     TSFM
       ↓
Forecast
```

We can then compare that forecast against our specialized models.

---

## 2.4 Chronos

Chronos is a pre-trained time-series forecasting approach designed to work across different time-series forecasting tasks.

For Project 1A, we will test whether it provides useful information for our market forecasting problem.

We should not assume that strong general forecasting results automatically mean strong Indian-equity results.

---

## 2.5 TimesFM

TimesFM is another pre-trained time-series forecasting model.

We will treat it similarly:

> **Test whether its forecasts add useful information to our system.**

We do not need to assume that it will replace our specialized financial models.

---

## 2.6 What Can TSFMs Give Us?

Potentially:

- Directional forecasting information.
- Future-value forecasts.
- Another independent view of market behaviour.
- A benchmark against our specialized models.

The exact outputs depend on the model and how we use it.

---

## 2.7 Main Advantage

The main attraction is:

> **We can test a powerful pre-trained forecasting model without building the entire forecasting model from scratch.**

---

## 2.8 Main Concerns

Potential concerns include:

- Financial data may behave differently from the data used during pre-training.
- General forecasting strength does not guarantee financial usefulness.
- Lower interpretability.
- Additional computational requirements.
- Possible latency.
- Model/version compatibility and implementation complexity.
- It may not understand our specific Indian-market context as well as a specialized model.

---

## 2.9 How We Will Test TSFMs

We will compare them against simpler and specialized approaches.

For example:

```text
Baseline Forecast
      ↓
BNN
      ↓
Chronos
      ↓
TimesFM
      ↓
Compare
```

We will look at:

- Directional performance.
- Forecast quality.
- Robustness.
- Behaviour during different market conditions.
- Computational cost.
- Whether the model adds information that other models do not already provide.

---

## 2.10 TSFM Decision

🟡 **TEST**

### Reason

> **Chronos and TimesFM are useful forecasting candidates and benchmarks, but we will not make them core components unless they prove that they add meaningful value on our data.**

---

# 3. BNN vs TSFM

| Aspect | BNN | TSFM |
|---|---|---|
| Main purpose | Direction forecasting | Time-series forecasting |
| Learns from our data | Yes | Uses pre-training and potentially adaptation |
| Nonlinear modelling | Strong | Strong |
| Uncertainty | Strong potential | Depends on model/output |
| Financial specialization | Can be designed for our problem | Not guaranteed |
| Interpretability | Moderate/low | Lower |
| Complexity | Higher | Higher |
| Project role | Direction model | Forecasting benchmark/extra signal |
| Current decision | 🟡 Test | 🟡 Test |

---

# 4. How They Fit With the Rest of Project 1A

Our current concept is:

```text
                    Market Data
                        ↓
                 Feature Layer
                        ↓
             HMM / Bayesian HMM
                        ↓
                Regime Information
                        ↓
          ┌─────────────┼─────────────┐
          ↓             ↓             ↓
       RS-VAR          BNN           TSFM
          ↓             ↓             ↓
     Relationship   Direction      Forecast
       Signal       Forecast       Signal
          └─────────────┼─────────────┘
                        ↓
                 Hybrid Ensemble
                        ↓
                   Calibration
                        ↓
              Final Direction Signal
```

BOCPD can also provide an early-warning signal that changes how much influence different models receive.

---

# 5. Situation-Aware Use

We agreed that **no model should permanently control the system**.

Instead, model influence should depend on the current market situation.

Example:

| Situation | Possible dominant model | Supporting models |
|---|---|---|
| Stable market | HMM | BNN, RS-VAR, TSFM |
| Strong nonlinear behaviour | BNN | HMM, RS-VAR |
| Forecast-heavy situation | BNN / TSFM | HMM, RS-VAR |
| Possible transition | HMM + BOCPD | BNN, RS-VAR |
| Extreme movement | Student-t HMM | BOCPD, BNN |

These are starting hypotheses.

Experiments will determine whether these dominance rules actually work.

---

# 6. Confidence Adjustment

Each model can provide its own confidence or prediction strength.

We do not want to simply average all model outputs equally.

Instead, the system can adjust each model's influence based on:

- Current market situation.
- Model reliability in that situation.
- Model confidence.
- Recent performance.
- Early-warning signals.
- Calibration quality.

Simple idea:

> **Model confidence × situation-based influence = model contribution**

This allows the system to give more influence to a model when it is better suited to the current situation.

---

# 7. Important Rule

Neither BNN nor TSFM becomes a permanent part of the final architecture just because it is advanced.

The rule is:

> **Research → Test → Compare → Keep only if useful.**

A simpler model that performs better should win.

---

# 8. Current Decisions

| Component | Current role | Decision |
|---|---|---|
| BNN | Direction forecasting | 🟡 TEST |
| Chronos | Forecasting benchmark/extra signal | 🟡 TEST |
| TimesFM | Forecasting benchmark/extra signal | 🟡 TEST |

### BNN

> **Test it for directional forecasting because it can capture complex patterns and uncertainty, but keep it only if the benefit justifies its complexity and latency.**

### TSFM

> **Test Chronos and TimesFM as forecasting benchmarks/extra signals, not as core regime models.**

---

# 9. Final Understanding

The simplest way to remember this document:

> **HMM answers "What market state are we in?"**

> **RS-VAR answers "How are financial variables behaving in that state?"**

> **BNN answers "What direction might the market move, especially when relationships are complex?"**

> **TSFM answers "What does a general pre-trained time-series model forecast?"**

And finally:

> **The ensemble decides how much each model should influence the final prediction based on the current situation and each model's reliability.**
