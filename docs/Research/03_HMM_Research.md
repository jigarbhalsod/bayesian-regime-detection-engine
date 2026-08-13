# 04. HMM Research

## Purpose

This document explains **what an HMM is, why it fits Project 1A, how it works, what versions we may use, and what role it could play in the final system.**

We are keeping the explanation practical and simple.

---

## 1. What Is an HMM?

**HMM (Hidden Markov Model) is a statistical model that estimates a hidden state from things we can observe.**

For Project 1A:

**Market data → HMM → Hidden market regime**

The regime is hidden because we cannot directly see whether the market is currently Risk-On, Risk-Off, Transition, etc.

We only see signals such as:

- NIFTY returns
- Volatility
- Volume
- India VIX
- Market breadth
- Other financial variables

The HMM uses these observations to estimate the most likely hidden regime.

---

## 2. Why HMM Fits Project 1A

Our main problem is:

> **The market regime is hidden, but its effects can be observed through financial data.**

That is exactly the type of problem HMMs are designed to model.

An HMM also allows us to model how the market can move from one state to another over time.

For example:

```text
Risk-On → Risk-On → Transition → Risk-Off
```

---

## 3. The Three Main Parts of an HMM

### Hidden States

The possible market conditions.

Example:

- Risk-On
- Risk-Off
- Transition
- Recovery

The final number of states is not fixed yet.

We will test:

**K = 2, 3, 4, 5**

---

### Observations

The financial data we can actually see.

Examples:

- Returns
- Volatility
- Volume
- India VIX
- Breadth
- Cross-asset signals

---

### Transition Probabilities

These describe how likely the market is to move from one state to another.

Example:

> If the market is currently Risk-On, how likely is it to remain Risk-On tomorrow?

or:

> How likely is it to move from Risk-On to Transition?

---

## 4. What Is an Emission?

**Emission means what the market data usually looks like when the model believes it is in a particular state.**

Example:

```text
Risk-On
→ positive returns
→ lower volatility
→ stronger market breadth
```

Another state might look like:

```text
Risk-Off
→ negative returns
→ high volatility
→ weak breadth
```

The model learns these patterns from historical data.

---

## 5. How HMM Works

The basic idea is:

```text
Historical Market Data
        ↓
Learn hidden states
        ↓
Learn what each state looks like
        ↓
Learn how states transition
        ↓
Estimate the current regime
        ↓
Output regime probabilities
```

Example output:

```text
Risk-On       12%
Transition    18%
Risk-Off      65%
Recovery       5%
```

So the model does not have to give only one hard answer.

---

## 6. HMM Inference Methods

There are three important methods we need to understand.

### Forward

Uses information available **up to the current time** to estimate the current regime.

**Best for live regime detection.**

### Viterbi

Finds the most likely sequence of states across a historical period.

**Best for historical analysis.**

### Forward-Backward

Uses past and future observations to estimate historical states more accurately.

**Best for research and historical analysis, not live prediction.**

### Project 1A decision

> **Use Forward for live regime detection; use Viterbi/Forward-Backward for historical analysis and validation.**

---

## 7. HMM Training

HMM parameters are commonly learned using:

**Baum-Welch / Expectation-Maximization (EM)**

In simple terms, the model repeatedly:

1. Estimates which states probably occurred.
2. Updates the model parameters.
3. Repeats until the model improves or stabilizes.

A practical issue is that training can depend on initialization.

Therefore, we should use:

- Multiple initializations
- Model stability checks
- Appropriate model-selection criteria

---

## 8. Gaussian HMM

A common HMM assumes the observations in each state follow a Gaussian distribution.

This gives us a simple and interpretable starting point.

### Decision

🟢 **Use as the baseline HMM.**

---

## 9. Student-t HMM

Financial returns can contain extreme movements that are not well represented by a simple Gaussian distribution.

A Student-t distribution has heavier tails and can better represent unusual observations.

### Decision

🟡 **Test as an improved emission option.**

We should keep it only if it performs better in our own out-of-sample experiments.

---

## 10. Multivariate HMM

A multivariate HMM uses multiple financial variables together.

Instead of looking only at NIFTY returns, we can provide:

- NIFTY returns
- Volatility
- India VIX
- Volume
- Market breadth
- Selected cross-asset signals

### Why this matters

A market regime is usually not defined by one signal.

Multiple signals can give the model a better picture of the market condition.

### Decision

🟢 **Use a compact multivariate HMM.**

We will start with a small set of strong features rather than dozens of variables.

---

## 11. Bayesian HMM

A standard HMM estimates model parameters as fixed values.

A Bayesian HMM treats uncertain parameters as probability distributions.

This allows us to represent more uncertainty around:

- Hidden states
- Transition probabilities
- Emission parameters

### Simple idea

**Standard HMM:**

> "Risk-Off = 70%."

**Bayesian HMM:**

> "Risk-Off is the most likely state, and here is how uncertain we are about the model's estimate."

### Decision

🟡 **Test as the advanced HMM version.**

The standard HMM remains our baseline.

---

## 12. HMM vs Bayesian HMM

| | Standard HMM | Bayesian HMM |
|---|---|---|
| Basic regime detection | ✅ | ✅ |
| Probability of regimes | ✅ | ✅ |
| Parameter uncertainty | Limited | ✅ Better |
| Complexity | Lower | Higher |
| Speed | Faster | Slower |
| Role | Baseline/core candidate | Advanced candidate |

### Our approach

**Standard HMM → Bayesian HMM**

We first build and understand the simpler model, then test whether the Bayesian version adds enough value.

---

## 13. Choosing the Number of States

We will not assume that the market definitely has five regimes.

We will test:

**K = 2, 3, 4, 5**

Possible structures:

| K | Example |
|---|---|
| 2 | Risk-On, Risk-Off |
| 3 | Risk-On, Transition, Risk-Off |
| 4 | Risk-On, Risk-Off, Transition, Recovery |
| 5 | Risk-On, Risk-Off, Transition, Late-Cycle, Post-Shock |

These are only possible interpretations.

The final choice will consider:

- Statistical quality
- Stability
- State separation
- Economic meaning
- Forecasting usefulness

---

## 14. How We Will Interpret the States

The HMM does not automatically know that:

> State 2 = Risk-Off

It only discovers a state with certain statistical characteristics.

We will study:

- Average returns
- Volatility
- Volume
- Breadth
- VIX
- Other relevant signals

Then we give the state a meaningful financial interpretation.

### Principle

> **The model discovers the state; we interpret the state.**

---

## 15. Regime Transitions

The transition matrix helps us understand how persistent each state is.

For example:

```text
Risk-On → Risk-On       High probability
Risk-On → Transition    Medium probability
Risk-On → Risk-Off      Low probability
```

This helps us understand:

- Which regimes are persistent.
- Which transitions are common.
- Which transitions are unusual.

However, transition probabilities should not automatically be treated as guaranteed predictions.

---

## 16. Real-Time Regime Detection

For live use, we must avoid using future information.

At time `t`:

> **The model can only use information available up to time `t`.**

Therefore:

**Forward filtering → live regime probability**

is our main approach.

Historical methods that use future observations are useful for research, but should not be used as if they were available in real time.

---

## 17. HMM Failure Risks

We will specifically test for:

### Non-Stationarity

Market behaviour changes over time.

### Structural Breaks

Relationships can change significantly.

### Heavy Tails

Extreme movements may not fit simple Gaussian assumptions.

### Outliers

One unusual event can distort the model.

### State Instability

The model may discover different states after retraining.

### False Regime Changes

A temporary shock may look like a new regime.

### Look-Ahead Bias

Future information may accidentally enter historical predictions.

### Overfitting

The model may learn historical noise instead of meaningful market behaviour.

---

## 18. Baselines We Should Compare Against

HMM should not be accepted simply because it is more advanced.

We should compare it against simpler approaches such as:

- Rule-based regimes
- Rolling volatility thresholds
- Simple clustering
- Basic Markov/regime models

The question is:

> **Does HMM provide meaningful improvement over simpler methods?**

If not, we should reconsider its role.

---

## 19. HMM Role in Project 1A

Our current plan is:

> **HMM is the main candidate for hidden regime detection, but it must still prove itself through experimentation.**

The HMM can provide:

- Current regime probabilities
- State transition information
- Historical regime sequences
- Inputs for other models

Other components can then use this information.

For example:

```text
HMM
 ↓
Regime probabilities
 ↓
BNN / RS-VAR / other models
```

---

## 20. How HMM Fits Our Hybrid Architecture

We are not building a system where one model controls everything.

Instead:

```text
Market Data
     ↓
HMM
     ↓
Regime Probabilities
     ↓
Situation Understanding
     ↓
Different models get different influence
     ↓
Final Ensemble
```

The HMM is therefore an important source of regime information, but its influence can change depending on the market situation.

---

## 21. Current HMM Decisions

| Component | Decision |
|---|---|
| Standard HMM | 🟢 Use as baseline |
| Multivariate HMM | 🟢 Use as main HMM setup |
| Bayesian HMM | 🟡 Test |
| Gaussian emissions | 🟢 Baseline |
| Student-t emissions | 🟡 Test |
| K = 2–5 | 🟢 Test all |
| Forward filtering | 🟢 Live detection |
| Viterbi | 🟢 Historical analysis |
| Forward-Backward | 🟢 Historical research |
| Higher-order HMM | 🔵 Keep open |
| Duration-aware HMM | 🔵 Keep open |

---

## Final HMM Decision

> **We will start with a simple multivariate HMM, test Bayesian and Student-t improvements, and keep the version that gives the best balance of regime quality, reliability, uncertainty, and practical performance.**

The key idea is:

> **HMM is a strong candidate, not a guaranteed winner.**

It must prove its value through our experiments before becoming a permanent part of the final architecture.
