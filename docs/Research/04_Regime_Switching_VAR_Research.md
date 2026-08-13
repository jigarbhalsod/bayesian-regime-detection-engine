# 05. Regime-Switching VAR Research

## Purpose

This document explains **VAR, Regime-Switching VAR (RS-VAR), why we are considering it for Project 1A, and what role it may play in the final system.**

The goal is to understand how **multiple financial variables interact with each other and whether those relationships change across market regimes.**

---

## 1. What Is VAR?

**VAR (Vector Autoregression) is a statistical model that studies multiple time-series variables together and learns how they move and influence each other over time.**

For example, we could study:

- NIFTY returns
- India VIX
- USD/INR
- Crude oil
- Interest rates

The model can learn relationships such as:

> When VIX changes, how does NIFTY usually respond?

and:

> When USD/INR changes, how does the market tend to respond?

---

## 2. Why VAR Is Relevant to Project 1A

HMM mainly helps us answer:

> **What market regime are we currently in?**

VAR helps us answer:

> **How are important financial variables interacting with each other?**

This gives us a different type of information.

---

## 3. What Is Regime-Switching VAR?

**RS-VAR is a VAR model where the relationships between variables can change depending on the market regime.**

For example:

```text
Risk-On
NIFTY ↔ VIX relationship
       ↓
Different from
       ↓
Risk-Off
NIFTY ↔ VIX relationship
```

So instead of assuming that financial relationships always remain the same, RS-VAR allows them to change with the market state.

---

## 4. Simple Example

Imagine:

### Risk-On

- VIX is low.
- NIFTY is strong.
- USD/INR is relatively stable.

### Risk-Off

- VIX rises.
- NIFTY falls.
- USD/INR may weaken.

RS-VAR can help us study whether the **strength and direction of these relationships are actually different between the two regimes.**

---

## 5. What Are Variables?

Variables are simply the **financial measurements we give to the model**.

Examples:

- NIFTY returns
- India VIX
- USD/INR
- Crude oil price/returns
- Interest rates
- Volume
- Selected global market indicators

We should not put every available variable into the model.

We will start with a small set and expand only if testing shows that additional variables provide useful information.

---

## 6. What Does RS-VAR Add to HMM?

### HMM

Helps answer:

> **Which hidden market regime are we probably in?**

### RS-VAR

Helps answer:

> **How do financial variables behave and interact differently within that regime?**

So they are complementary.

```text
HMM
 ↓
Market Regime
 ↓
RS-VAR
 ↓
Regime-specific financial relationships
```

---

## 7. What Can RS-VAR Tell Us?

RS-VAR can potentially help us understand:

- How NIFTY responds to changes in VIX.
- How currency movements relate to equities.
- How external variables affect Indian equities.
- Whether relationships become stronger or weaker during stress.
- Whether financial relationships change between calm and stressed markets.

This makes RS-VAR useful for **economic interpretation**, not only prediction.

---

## 8. Regime Dynamics

RS-VAR allows relationships to change when the market moves between regimes.

For example:

```text
Risk-On
   ↓
Transition
   ↓
Risk-Off
```

The relationships between:

**NIFTY ↔ VIX ↔ USD/INR ↔ Crude**

may change during this sequence.

RS-VAR helps us investigate those differences.

---

## 9. Model Design Questions

Before implementing RS-VAR, we need to decide:

### Number of Variables

Start small.

A practical starting point is **3–5 important variables**.

### Number of Lags

The model needs to know how much past information to use.

We should select the lag length using appropriate statistical criteria and validation rather than choosing it arbitrarily.

### Number of Regimes

Use the same regime structures being tested for Project 1A:

**K = 2–5**

This keeps the comparison consistent with HMM.

### Data Transformation

Many financial variables need to be transformed before modelling.

For example:

- Prices → returns
- Non-stationary series → appropriate stationary representation

The exact transformation will be decided during the data and modelling phases.

---

## 10. Stationarity

A major consideration with VAR is **stationarity**.

In simple terms:

> The statistical behaviour of the series should be reasonably stable over the period being modelled.

Raw asset prices often trend over time, so using returns or other suitable transformations is generally more appropriate.

We will test stationarity rather than assume it.

---

## 11. Estimation and Inference

RS-VAR is more complicated than a normal VAR because it must estimate:

1. Relationships between variables.
2. How those relationships differ by regime.
3. Which regime is active at each point.

This introduces additional computational and estimation challenges.

Therefore, RS-VAR should not become a core component unless it proves useful on our data.

---

## 12. Bayesian RS-VAR

Bayesian methods can also be applied to regime-switching VAR models.

Potential benefits include:

- Better representation of parameter uncertainty.
- More flexible probabilistic modelling.
- Better handling of limited information in some settings.

However, this also increases complexity.

### Project 1A decision

**Do not start with Bayesian RS-VAR.**

First test a practical RS-VAR approach.

Only consider a Bayesian extension if the simpler model shows clear value and uncertainty becomes an important limitation.

---

## 13. Economic Interpretation

One of the strongest reasons to consider RS-VAR is **interpretability**.

We can ask:

> Does the relationship between two financial variables change when the market changes regime?

For example:

```text
Normal Market:
VIX ↑ → NIFTY ↓ moderately

Stress Market:
VIX ↑ → NIFTY ↓ strongly
```

This can help us understand **why market behaviour changes**, rather than only detecting that it changed.

---

## 14. Impulse Response

Impulse Response analysis can help answer:

> **What happens to one variable after another variable suddenly changes?**

For example:

> What happens to NIFTY after a sudden rise in VIX?

We can potentially compare this response across regimes.

This can provide useful economic interpretation.

---

## 15. Indian Market Suitability

Project 1A focuses on the Indian equity market.

We therefore need to check whether:

- Enough historical data is available.
- The selected variables are reliable.
- The market contains enough meaningful regime variation.
- The relationships are stable enough to estimate.
- The model remains useful when tested on unseen periods.

We should not assume that results from US or other markets will automatically transfer to India.

---

## 16. RS-VAR vs HMM

| Question | HMM | RS-VAR |
|---|---|---|
| Detect hidden regimes | **Strong** | Possible, but not its main strength |
| Model variable interactions | Limited | **Strong** |
| Understand regime-specific relationships | Limited | **Strong** |
| Regime probabilities | **Strong** | Available depending on setup |
| Economic interpretation | Good | **Very good** |
| Complexity | Lower | Higher |
| Main purpose | Regime detection | Regime-dependent dynamics |

### Simple conclusion

> **HMM tells us what state the market may be in; RS-VAR helps us understand how financial variables behave in that state.**

---

## 17. Can They Work Together?

Yes.

Our preferred concept is:

```text
Market Data
     ↓
HMM
     ↓
Regime Information
     ↓
RS-VAR
     ↓
Regime-specific relationships
```

However, the final architecture will depend on experimental results.

RS-VAR should not automatically become part of the final system just because it is theoretically useful.

---

## 18. Main Risks

### Overfitting

Too many variables or regimes can make the model fit historical noise.

### High Dimensionality

Adding many variables and lags can make the model difficult to estimate reliably.

### Non-Stationarity

Changing market behaviour can make estimated relationships unstable.

### Structural Breaks

Major changes in the market can make historical relationships less useful.

### Identification Problems

Different model specifications may produce different regime interpretations.

### Computational Cost

RS-VAR is more complex than a basic HMM or VAR.

---

## 19. Baseline Comparison

RS-VAR should be compared against simpler approaches.

At minimum:

- Standard VAR
- HMM
- Simple financial relationships/baselines

The key question is:

> **Does RS-VAR provide useful information that simpler models cannot provide?**

If the answer is no, we should not keep its extra complexity.

---

## 20. Possible Roles in Project 1A

RS-VAR could potentially become:

- A supporting regime model.
- A regime-analysis tool.
- A feature generator.
- A forecasting component.
- An experimental model.

### Our current preference

> **RS-VAR is a supporting model, mainly useful for understanding regime-specific financial relationships.**

It should not permanently dominate the system.

---

## 21. How RS-VAR Fits Our Hybrid Architecture

We want a situation-aware system where no single model controls everything.

For example:

```text
Market Data
     ↓
HMM → Current Regime
     ↓
Situation Understanding
     ↓
RS-VAR → Financial Relationship Signal
     ↓
BNN / TSFM → Direction Forecast
     ↓
Ensemble
     ↓
Final Prediction
```

The actual influence of RS-VAR should depend on whether its information is useful in the current market situation.

---

## 22. Current Decision

### RS-VAR

🟡 **TEST**

### Why?

> **RS-VAR can provide information that HMM does not naturally provide, especially about how financial variables interact differently across regimes, but its extra complexity must be justified by experiments.**

---

## 23. What We Will Test

The initial RS-VAR experiment should evaluate:

- A small set of financial variables.
- K = 2–5 regime structures.
- Appropriate lag lengths.
- Stationary/transformed data.
- Regime-specific relationships.
- Out-of-sample forecasting performance.
- Economic interpretability.
- Stability across different market periods.

---

## 24. Final RS-VAR Understanding

The simplest way to remember it:

> **VAR learns how financial variables move together over time.**

> **RS-VAR allows those relationships to change depending on the market regime.**

> **HMM finds the hidden regime; RS-VAR helps explain what is happening between financial variables inside that regime.**

---

## Final Decision

> **RS-VAR will be tested as a supporting component, not assumed to be a core component. We will keep it only if it provides useful regime-specific information that improves forecasting, interpretation, or decision support enough to justify its additional complexity.**
