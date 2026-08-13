# 03. Market Regime Research

## Purpose

This document explains **what a market regime means for Project 1A, how we can identify different regimes, what can cause them to change, and how we can check whether a detected regime is actually useful.**

The main focus is the **Indian equity market**, with **NIFTY 50 as our primary market target**.

---

## 1. What Is a Market Regime?

A market regime is a **period when the market behaves in a particular way for some time**.

For example:

- The market is rising steadily with low volatility.
- The market is falling with high volatility.
- The market is uncertain and changing direction.
- The market is recovering after a major shock.

We cannot directly see the regime.

We see signals such as:

**Prices → Returns → Volatility → Volume → VIX → Flows → Economic Data**

The model uses these signals to estimate the hidden market condition.

### Simple idea

**Observable market behaviour → Model → Hidden market regime**

---

## 2. How We Will Define Regimes

We will use a **data-driven approach with economic interpretation**.

This means:

1. The model discovers different market states from the data.
2. We study the behaviour of each state.
3. We give the states meaningful financial names afterward.

We will **not force the market into fixed labels before looking at the data**.

---

## 3. Number of Regimes

We will test:

- **K = 2**
- **K = 3**
- **K = 4**
- **K = 5**

Possible interpretations are:

| K | Example structure |
|---|---|
| 2 | Risk-On, Risk-Off |
| 3 | Risk-On, Transition, Risk-Off |
| 4 | Risk-On, Risk-Off, Transition, Recovery |
| 5 | Risk-On, Risk-Off, Transition, Late-Cycle, Post-Shock |

These are **examples, not fixed labels**.

The final number will be selected using:

- Statistical quality
- Regime stability
- Economic meaning
- Forecasting usefulness

### Final decision

> **Test K = 2–5 and let the evidence decide the final number of regimes.**

---

## 4. Regime Output

The system should not simply say:

> Risk-Off

Instead, it should provide probabilities.

Example:

```text
Risk-On       10%
Transition    18%
Risk-Off      67%
Recovery       5%
```

This tells us both:

- What the most likely regime is.
- How certain the system is.

### Final decision

> **Project 1A will use probabilistic regime outputs rather than only hard regime labels.**

---

## 5. What Makes Regimes Different?

We will look at several types of market behaviour.

### Returns

Is the market generally moving up, down, or sideways?

### Volatility

Is the market calm or experiencing large movements?

### Volume and Liquidity

Is market activity normal, strong, or stressed?

### Market Breadth

Is the movement supported by many stocks or only a few?

### Cross-Asset Behaviour

Are currency, crude oil, global markets, and other external factors supporting or hurting Indian equities?

### Investor Flows

Are major investors buying or selling?

### Macro Conditions

Are economic and monetary conditions supportive or stressful?

No single signal should define a regime by itself.

---

## 6. What Can Cause a Regime Change?

We will group possible drivers into four categories.

### Structural Drivers

- Economic growth
- Inflation
- Interest rates
- Liquidity
- Earnings conditions

### Market Drivers

- Volatility shocks
- Trend changes
- Liquidity changes
- Correlation changes

### External Drivers

- Global market stress
- Commodity shocks
- Currency movements
- Geopolitical events

### Event-Driven Drivers

- Major policy announcements
- Unexpected economic data
- Major market events
- Financial crises

The system does not need to predict every cause directly.

Its main job is to **detect the resulting change in market behaviour**.

---

## 7. Hidden-State Problem

The actual market regime is not directly observable.

For example, we may observe:

> NIFTY falling + VIX rising + FPI selling

But we do not directly observe:

> "The market is now Risk-Off."

This makes the regime a **hidden state**.

That is why HMM-based approaches are relevant to Project 1A.

---

## 8. Regime Transitions

Markets do not always switch cleanly from one regime to another.

A possible sequence is:

```text
Risk-On
   ↓
Uncertainty
   ↓
Transition
   ↓
Risk-Off
```

Therefore, the system should be able to show uncertainty during transitions.

We also want to identify whether a change is:

- Normal market noise
- A temporary shock
- A possible structural change
- A genuine regime transition

This is where **BOCPD will later act as an early-warning signal**.

---

## 9. Indian Market Focus

### Primary target

**NIFTY 50**

It is our main representation of the Indian equity market for Project 1A.

### Additional validation/context

We can later use:

- NIFTY 500
- Sector indices
- India VIX
- Other relevant Indian market indicators

The purpose is to check whether the detected behaviour is only specific to NIFTY 50 or also appears more broadly.

---

## 10. Core Market Signals

Our initial signal groups are:

| Signal group | What it tells us |
|---|---|
| Returns | Market direction |
| Momentum/Trend | Direction strength and persistence |
| Volatility | Market stress and uncertainty |
| India VIX | Expected near-term market volatility |
| Volume/Liquidity | Market activity and stress |
| Breadth | How widely the market movement is supported |
| USD/INR | Currency pressure |
| Crude Oil | Important external cost/risk factor |
| Global Equity Signals | Global risk environment |
| Institutional Flows | Buying/selling pressure |
| Macro Variables | Broader economic conditions |

We will not automatically use every signal.

Each feature must later prove that it adds useful information.

---

## 11. Main Challenges

### Non-Stationarity

Market behaviour changes over time.

### Structural Breaks

The relationships in the market can change permanently or for long periods.

### Look-Ahead Bias

The model must never use future information when estimating the current regime.

### Overfitting

The model could learn historical noise instead of real market behaviour.

### Rare Events

Major crashes are important but happen infrequently.

### Ambiguous Regimes

Sometimes the market does not clearly belong to one state.

---

## 12. How We Will Avoid Look-Ahead Bias

For live regime detection:

> **At time `t`, the model can only use information available up to time `t`.**

Historical methods that use future observations can still be useful for research and understanding the past, but they cannot be used as if they were available in real time.

This is especially important when validating regime detection.

---

## 13. How We Will Judge Whether a Regime Is Useful

A statistically different state is not automatically a useful market regime.

We will ask:

1. Is the regime stable?
2. Does it have clear financial characteristics?
3. Does it appear across meaningful historical periods?
4. Does it make economic sense?
5. Can the model identify it without future information?
6. Does knowing the regime improve equity-direction forecasting?
7. Does it improve decision-making?

### Core principle

> **A regime is useful only if it is both statistically meaningful and financially useful.**

---

## 14. What HMM Will Do

HMM will help us estimate the hidden market state.

Conceptually:

```text
Market Data
    ↓
HMM
    ↓
Hidden Regime
    ↓
Regime Probabilities
```

For example:

```text
Risk-On       15%
Transition    20%
Risk-Off      60%
Recovery       5%
```

The HMM does not automatically know that a state is called "Risk-Off".

We interpret the discovered state based on its behaviour.

---

## 15. Role of BOCPD

BOCPD will not replace the regime model.

Its role will be:

> **Warn the system that market behaviour may have changed.**

Example:

```text
HMM:
Risk-On = 72%

BOCPD:
Possible change detected

↓

System becomes more cautious
and allows other models to have more influence.
```

This fits our planned **situation-aware model combination**.

---

## 16. Our Hybrid Approach

We do not want one model to control the entire system.

Different models may be more useful in different situations.

For example:

| Situation | Potential dominant model | Supporting models |
|---|---|---|
| Stable market | HMM | BNN, RS-VAR, TSFM |
| Possible transition | HMM + BOCPD | RS-VAR, BNN |
| Extreme movement | Student-t HMM | BOCPD, BNN |
| Strong nonlinear behaviour | BNN | HMM, RS-VAR |
| Strong variable interaction | RS-VAR | HMM, BNN |
| Forecast-heavy situation | BNN / TSFM | HMM, RS-VAR |

These roles are **proposed, not permanently fixed**.

They must be validated experimentally.

---

## 17. Final Understanding

For Project 1A:

> **We want to observe how the Indian equity market behaves, infer the hidden regime behind that behaviour, understand when the regime may be changing, and use that information to improve equity-direction forecasting.**

Our system should:

**Observe → Detect → Estimate uncertainty → Watch for changes → Adjust model influence → Forecast → Validate**

---

## Current Decisions

- **Primary market:** NIFTY 50
- **Regime count:** Test K = 2–5
- **Regime definition:** Data-discovered + economic interpretation
- **Regime output:** Probabilities, not only labels
- **Primary regime approach:** HMM family
- **Extreme-movement candidate:** Student-t HMM
- **Change warning:** BOCPD
- **Architecture style:** Situation-aware hybrid ensemble
- **Final model roles:** Must be validated experimentally

---

## Phase 2.2 Conclusion

> **Project 1A will treat market regimes as hidden states that must be inferred from multiple observable financial signals, with NIFTY 50 as the primary target and probabilistic outputs used to represent uncertainty.**
