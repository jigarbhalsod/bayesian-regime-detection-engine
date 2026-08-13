# 07. Online Detection & Transition Research

## Purpose

This document covers two related topics:

1. **Sequential / Online Inference**
2. **Changepoint Detection**

Both are about what happens when **new market data keeps arriving**.

The main goal is to help Project 1A update its understanding of the market and notice when market behaviour may be changing.

---

# 1. Sequential / Online Inference

## 1.1 What Does It Mean?

Online inference simply means:

> **Update the model's belief whenever new market data arrives.**

Example:

```text
Yesterday
Risk-On = 75%

↓ New market data

Today
Risk-On = 55%
Transition = 35%
Risk-Off = 10%
```

The model does not need to wait until the entire historical dataset is available again.

---

## 1.2 Why Is It Important?

Project 1A is intended to support a system that works with ongoing market data.

Therefore, we need the system to answer:

> **"Based on everything available right now, what regime are we most likely in?"**

This makes online updating important for real-time regime detection.

---

## 1.3 HMM Already Supports This

A standard HMM can use **Forward filtering** to update the probability of each hidden state as new observations arrive.

Therefore:

```text
New Market Data
      ↓
HMM Forward Filtering
      ↓
Updated Regime Probabilities
```

This gives us a relatively simple way to perform online regime estimation.

---

## 1.4 Particle Filtering

Particle filtering is a more flexible sequential inference technique.

In simple words:

> **It keeps many possible guesses about the current hidden state and updates those guesses as new data arrives.**

It can be useful for systems that are more complex, nonlinear, or non-Gaussian.

---

## 1.5 Why Are We Considering Particle Filtering?

Particle filtering could potentially help if our final system needs:

- More flexible state estimation.
- Nonlinear dynamics.
- Non-Gaussian behaviour.
- More advanced online inference.

However, it also introduces:

- More complexity.
- More computation.
- More implementation work.

---

## 1.6 Project 1A Decision

We do **not** need Particle Filtering immediately.

Our HMM already provides a practical way to update regime probabilities.

### Decision

🔵 **KEEP OPEN**

### Reason

> **HMM can already handle real-time regime updates, so Particle Filtering will only be considered later if experiments show that we need a more advanced online method.**

---

# 2. Real-Time Data Rule

For real-time detection:

> **At time `t`, the system can only use information available up to time `t`.**

It must not use future information.

Example:

```text
Allowed:
Today's data → Today's regime estimate

Not allowed:
Tomorrow's data → Today's regime estimate
```

This is essential to avoid **look-ahead bias**.

---

# 3. Changepoint Detection

## 3.1 What Is a Changepoint?

A changepoint is a point where the underlying behaviour of a time series may have changed.

Simple example:

```text
Low volatility
Low volatility
Low volatility
      ↓
Possible change
      ↓
High volatility
High volatility
```

The important question is:

> **"Has the market behaviour actually changed, or is this just temporary noise?"**

---

## 3.2 Changepoint vs Regime Detection

These are related but different.

### Regime Detection

> **What market state are we currently in?**

### Changepoint Detection

> **Has the underlying market behaviour recently changed?**

So:

```text
HMM
↓
"What state are we in?"

BOCPD
↓
"Could the market behaviour be changing?"
```

---

# 4. BOCPD

## 4.1 What Is BOCPD?

**BOCPD = Bayesian Online Changepoint Detection.**

It estimates the probability that a new segment of behaviour has started.

In simple terms:

> **BOCPD warns us when the market may have started behaving differently.**

---

## 4.2 Why Is BOCPD Useful?

Suppose HMM says:

```text
Risk-On = 70%
```

But BOCPD detects:

```text
Possible behaviour change
```

The system can become more cautious.

It does not automatically declare:

> "The regime has changed."

Instead, it says:

> **"Something may be changing, so pay more attention."**

---

# 5. Early-Warning Role

This is the main role we want for BOCPD.

Example:

```text
Market Data
     ↓
HMM
     ↓
Risk-On
     ↓
BOCPD detects possible change
     ↓
Early Warning
     ↓
Other models receive more attention
     ↓
System reassesses the situation
```

This fits our hybrid architecture.

---

# 6. Why BOCPD Should Not Replace HMM

BOCPD and HMM answer different questions.

BOCPD is focused on:

> **"Has behaviour changed?"**

HMM is focused on:

> **"What hidden regime best explains the current behaviour?"**

Therefore, BOCPD should remain a supporting signal.

---

# 7. Noise vs Real Change

One major challenge is that financial markets are noisy.

A sudden movement could be:

- Normal market noise.
- A temporary shock.
- A short-lived volatility spike.
- A genuine regime transition.

BOCPD can provide an early warning, but it should not be treated as absolute proof of a regime change.

The final system should combine:

- HMM regime probabilities.
- BOCPD change signal.
- Other model outputs.
- Confidence and calibration.

---

# 8. How BOCPD Fits the Hybrid Architecture

We want:

> **No single model should drive the whole system.**

So BOCPD does not directly make the final prediction.

Instead:

```text
                  Market Data
                       ↓
             ┌─────────┴─────────┐
             ↓                   ↓
           HMM                 BOCPD
             ↓                   ↓
      Regime Probability    Change Warning
             └─────────┬─────────┘
                       ↓
                Situation Layer
                       ↓
          Adjust Model Influence
                       ↓
               Hybrid Ensemble
```

---

# 9. Situation-Based Model Influence

The BOCPD signal can help change how much influence different models receive.

For example:

### Stable market

```text
HMM → Higher influence
BNN → Supporting
RS-VAR → Supporting
TSFM → Supporting/benchmark
```

### Possible transition

```text
BOCPD → Strong warning
HMM → High influence
BNN → Supporting
RS-VAR → Supporting
```

### Extreme movement

```text
Student-t HMM → Higher influence
BOCPD → Strong warning
BNN → Supporting
```

These are starting hypotheses.

The actual dominance rules must be tested later.

---

# 10. Confidence Adjustment

We agreed that each model should not have a fixed influence.

Instead, the system should consider:

- Current market situation.
- Model confidence.
- Model reliability in that situation.
- Recent model performance.
- BOCPD warning.
- Calibration quality.

Simple idea:

> **Model confidence × situation-based influence = model contribution**

This allows the system to give more weight to a model when it is better suited to the current situation.

---

# 11. Sequential Inference vs Changepoint Detection

| Component | Main job |
|---|---|
| **HMM Forward Filtering** | Update the current regime as new data arrives |
| **Particle Filtering** | Advanced online state updating |
| **BOCPD** | Warn that market behaviour may be changing |

The first two are about **updating the hidden state**.

BOCPD is about **detecting possible change in behaviour**.

---

# 12. Main Risks

## False Alarms

BOCPD may react to temporary market noise.

## Delayed Detection

A genuine regime change may not be detected immediately.

## Overreaction

The system could change model weights too aggressively after a short shock.

## Look-Ahead Bias

Historical analysis must not accidentally use future information when simulating live behaviour.

## Complexity

Adding too many online components can make the system harder to understand and maintain.

---

# 13. How We Will Validate These Components

For online detection, we should test:

- How quickly a real change is detected.
- How many false warnings are generated.
- How stable the signal is.
- Behaviour during major market shocks.
- Whether early warnings improve forecasting.
- Whether the warning improves model selection/dominance.
- Computational cost.
- Performance using only information available at that time.

The important question is:

> **Does early detection actually improve the final system?**

Not simply:

> "Can BOCPD detect a changepoint?"

---

# 14. Current Decisions

| Component | Role | Decision |
|---|---|---|
| HMM Forward Filtering | Real-time regime updating | 🟢 **USE** |
| Particle Filtering | Advanced online inference | 🔵 **KEEP OPEN** |
| BOCPD | Early-warning signal | 🟡 **TEST** |

---

# 15. Final Understanding

The simplest way to remember this:

> **HMM tells us what regime we may be in.**

> **HMM Forward Filtering keeps that estimate updated as new data arrives.**

> **BOCPD warns us that the market may be changing.**

> **Particle Filtering is a more advanced option that we keep for later if HMM is not enough.**

---

# Final Decision

> **Project 1A will use HMM Forward Filtering for real-time regime updates and test BOCPD as an early-warning signal. Particle Filtering remains a future option unless experiments show that we need it.**

The goal is not to react to every market movement.

The goal is:

**Update → Detect possible change → Reassess → Adjust model influence → Forecast**
