# 11. Method Comparison Matrix

## Purpose

This document brings together the methods we researched in Phase 2 and compares them using the same criteria.

The goal is simple:

> **Decide which methods deserve to be used, tested further, rejected, or kept open for Project 1A.**

This is a **research decision document**, not a final implementation document.

---

## 1. Our Decision Categories

Every major method falls into one of four groups:

| Decision | Meaning |
|---|---|
| 🟢 **USE** | Strong fit and important enough to include, subject to normal validation. |
| 🟡 **TEST** | Promising, but we need experiments before committing. |
| 🔴 **REJECT** | Not useful enough or not practical for Project 1A. |
| 🔵 **KEEP OPEN** | Interesting, but not necessary to decide now. |

Important rule:

> **Even a USE candidate must still be validated experimentally.**

---

## 2. Evaluation Criteria

We compare methods using:

- Regime detection quality
- Forecasting usefulness
- Uncertainty handling
- Financial meaning
- Interpretability
- Robustness
- Early detection
- Data requirements
- Computational cost
- Real-time suitability
- Implementation complexity
- Overfitting risk
- Data leakage risk
- Practical value

### Main evaluation lens

> **Performance + Financial Meaning + Uncertainty + Robustness + Interpretability + Practicality**

---

# 3. Regime Detection Methods

| Method | Main job | Strength | Main concern | Current decision |
|---|---|---|---|---|
| **Standard HMM** | Detect hidden market regimes | Simple, interpretable, naturally fits hidden-state problem | Financial assumptions may be too simple | 🟢 **USE / Baseline** |
| **Bayesian HMM** | Detect regimes + better parameter uncertainty | Strong uncertainty handling | More complex and computationally heavier | 🟡 **TEST** |
| **Student-t HMM** | Detect regimes with better handling of extreme movements | Better suited to heavy-tailed financial data | More complex than Gaussian HMM | 🟡 **TEST** |
| **Rule-Based Regimes** | Define regimes using fixed thresholds | Very simple and transparent | Can be too rigid | 🟢 **BASELINE** |
| **Clustering** | Discover groups of similar market behaviour | Simple unsupervised alternative | Does not naturally model regime transitions | 🟢 **BASELINE** |

### Current conclusion

> **HMM is our main regime-detection candidate, but it must prove that it adds value over simpler baselines.**

---

# 4. Regime-Switching VAR

| Method | Main job | Strength | Main concern | Current decision |
|---|---|---|---|---|
| **Standard VAR** | Model relationships between financial variables | Useful baseline for variable interactions | Does not naturally adapt to changing regimes | 🟢 **BASELINE** |
| **RS-VAR** | Model variable relationships differently across regimes | Strong economic interpretation | Higher complexity and overfitting risk | 🟡 **TEST** |
| **Bayesian RS-VAR** | Add Bayesian uncertainty to RS-VAR | More flexible uncertainty modelling | Even more complex | 🔵 **KEEP OPEN** |

### Current conclusion

> **RS-VAR is a supporting candidate because it can explain how financial relationships change across regimes, but its extra complexity must be justified.**

---

# 5. Directional Forecasting Methods

| Method | Main job | Strength | Main concern | Current decision |
|---|---|---|---|---|
| **BNN** | Predict equity direction with uncertainty | Can learn nonlinear relationships | More complex and potentially higher latency | 🟡 **TEST** |
| **Chronos** | General time-series forecasting | Pre-trained and potentially useful as an independent forecast | General time-series knowledge may not transfer well to Indian equities | 🟡 **TEST** |
| **TimesFM** | General time-series forecasting | Strong pre-trained forecasting candidate | Same financial-domain transfer concern | 🟡 **TEST** |
| **Simple forecasting baseline** | Provide a basic comparison | Easy to understand and validate | Limited modelling power | 🟢 **BASELINE** |

### Current conclusion

> **BNN, Chronos, and TimesFM are forecasting candidates, not primary regime detectors.**

---

# 6. Online Detection & Transition Methods

| Method | Main job | Strength | Main concern | Current decision |
|---|---|---|---|---|
| **HMM Forward Filtering** | Update regime probabilities as new data arrives | Simple and directly fits HMM | Depends on HMM assumptions | 🟢 **USE** |
| **Particle Filtering** | Advanced online hidden-state estimation | More flexible for complex systems | More computation and implementation complexity | 🔵 **KEEP OPEN** |
| **BOCPD** | Warn about possible behaviour changes | Useful early-warning signal | Can mistake noise for a real change | 🟡 **TEST** |

### Current conclusion

> **Use HMM Forward Filtering for online regime updates and test BOCPD as an early-warning signal.**

---

# 7. Ensemble Methods

| Method | Main job | Strength | Main concern | Current decision |
|---|---|---|---|---|
| **Equal Weighting** | Give every model equal influence | Very simple baseline | Ignores model differences | 🟢 **BASELINE** |
| **Weighted Averaging** | Give models different influence | Simple and interpretable | Weights can be poorly chosen | 🟢 **USE / Baseline Ensemble** |
| **Situation-Based Weighting** | Change model influence by market situation | Matches our hybrid architecture | Can overfit if too complicated | 🟡 **TEST** |
| **Stacking** | Learn how models should be combined | Can learn complex combinations | Adds another learning layer and overfitting risk | 🟡 **TEST** |
| **Bayesian Model Averaging** | Combine models probabilistically | Strong theoretical basis | More complexity than we currently need | 🔵 **KEEP OPEN** |

---

# 8. Uncertainty & Calibration

| Method | Main job | Strength | Main concern | Current decision |
|---|---|---|---|---|
| **Regime Probabilities** | Show probability of each regime | Directly useful for decisions | Probability quality must be checked | 🟢 **USE** |
| **Model Confidence** | Show prediction strength | Useful for model weighting | Can be overconfident | 🟢 **USE** |
| **Calibration** | Check whether confidence is trustworthy | Essential for confidence-based decisions | Adds validation work | 🟢 **USE** |
| **Brier Score** | Measure probability quality | Easy to interpret | Only one view of performance | 🟢 **USE** |
| **Log Loss** | Penalize bad/overconfident probabilities | Strong probability metric | Can be sensitive to very confident mistakes | 🟢 **USE** |
| **ECE** | Measure calibration error | Useful additional calibration check | Depends on probability grouping | 🟢 **USE** |
| **Adaptive Conformal Prediction** | Give prediction ranges/sets | Useful additional uncertainty information | Adds complexity | 🟡 **TEST** |

---

# 9. Feature Approaches

| Feature group | Main purpose | Current decision |
|---|---|---|
| NIFTY returns | Market direction | 🟢 **USE** |
| Momentum / trend | Direction and persistence | 🟢 **USE** |
| Realized volatility | Market stress | 🟢 **USE** |
| India VIX | Expected volatility / fear | 🟢 **USE** |
| Volume / turnover | Market activity | 🟢 **USE** |
| Market breadth | Strength of market participation | 🟢 **USE** |
| USD/INR | Currency / external pressure | 🟢 **USE initially** |
| Crude oil | External/economic pressure | 🟢 **USE initially** |
| Global equity signals | Global risk environment | 🟢 **USE initially** |
| FPI/FII flows | Institutional activity | 🟡 **TEST** |
| DII flows | Institutional activity | 🟡 **TEST** |
| Interest rates | Monetary/financial conditions | 🟡 **TEST** |
| Inflation | Economic conditions | 🟡 **TEST** |
| GDP | Slow-moving economic context | 🔵 **KEEP OPEN** |
| Sentiment | Market/news sentiment | 🔵 **KEEP OPEN** |
| Gold | Defensive/alternative asset signal | 🔵 **KEEP OPEN** |

---

# 10. Overall Model Decision

Putting the research together:

| Component | Current role | Decision |
|---|---|---|
| **Standard HMM** | Main regime candidate | 🟢 USE / validate |
| **Bayesian HMM** | Advanced regime + uncertainty | 🟡 TEST |
| **Student-t HMM** | Extreme-market regime candidate | 🟡 TEST |
| **RS-VAR** | Regime-specific financial relationships | 🟡 TEST |
| **BNN** | Direction forecasting | 🟡 TEST |
| **Chronos** | Forecasting benchmark / extra signal | 🟡 TEST |
| **TimesFM** | Forecasting benchmark / extra signal | 🟡 TEST |
| **HMM Forward Filtering** | Real-time regime updates | 🟢 USE |
| **Particle Filtering** | Advanced online inference | 🔵 KEEP OPEN |
| **BOCPD** | Early-warning signal | 🟡 TEST |
| **Weighted Averaging** | Simple ensemble | 🟢 USE / baseline |
| **Situation-Based Weighting** | Dynamic model influence | 🟡 TEST |
| **Stacking** | Learned model combination | 🟡 TEST |
| **Bayesian Model Averaging** | Advanced ensemble | 🔵 KEEP OPEN |
| **Calibration** | Confidence validation | 🟢 USE |
| **Adaptive Conformal** | Additional uncertainty method | 🟡 TEST |

---

# 11. What We Are NOT Doing

We are **not** going to build every method just because it appears in the research.

The research produced a toolbox.

Testing will decide which tools stay.

The rule is:

> **Research → Test → Compare → Keep what works.**

---

# 12. Proposed Initial Architecture Direction

Based on the research so far:

```text
                    Market Data
                        ↓
                  Feature Layer
                        ↓
               HMM / HMM Variants
                        ↓
             Current Regime + Confidence
                        ↓
                 Situation Layer
                  ↙          ↘
             BOCPD          Regime Info
          Early Warning          ↓
                  ↘          ↙
               Model Candidates
             ↙        ↓        ↘
           BNN      RS-VAR     TSFM
             ↘        ↓        ↙
                Ensemble
                    ↓
              Calibration
                    ↓
       Final Direction + Confidence
                    ↓
             Decision Support
```

This is our **current research direction**, not the final implementation architecture.

---

# 13. Most Important Architectural Decision

Our strongest design decision from the research is:

> **No single model should permanently control the system.**

Instead:

> **The model that is most suitable for the current situation gets more influence, while the other models provide supporting evidence.**

The influence should consider:

- Current market situation.
- Model confidence.
- Model calibration.
- Historical reliability.
- Model disagreement.
- Early-warning signals.

---

# 14. What Still Needs Experimental Proof

The research cannot answer these questions by itself:

- Which K value works best: 2, 3, 4, or 5?
- Whether HMM actually beats simpler regime baselines.
- Whether Bayesian HMM improves enough to justify its complexity.
- Whether Student-t HMM improves extreme-event handling.
- Whether RS-VAR adds useful information.
- Whether BNN improves directional forecasting.
- Whether Chronos or TimesFM add value.
- Whether BOCPD produces useful early warnings.
- Whether situation-based weighting improves the ensemble.
- Whether stacking beats simple weighted averaging.
- Which exact features provide the most useful information.
- What final model combination performs best.

These are **Phase 4–11 experimental questions**.

---

# Final Decision

> **Our research has narrowed Project 1A down to a small set of serious candidates, but we will not force a final model stack before testing them.**

The next step is to turn these research decisions into a **clear experimental plan**, where each candidate gets a fair comparison using the same data, validation rules, and evaluation metrics.
