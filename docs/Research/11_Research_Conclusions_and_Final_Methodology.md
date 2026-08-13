# 12. Research Conclusions & Final Methodology

## Purpose

This document closes **Phase 2, Research & Literature**.

It brings together what we learned, what we currently believe is worth using, what still needs testing, and what we will carry forward into the next phases.

The most important rule is:

> **Research gives us candidates. Experiments will decide what finally stays.**

---

# 1. What We Set Out to Solve

Project 1A aims to:

> **Detect the current market regime in the Indian equity market and use that information to improve equity-direction forecasting.**

Our primary market target is:

**NIFTY 50**

The system should also understand uncertainty and recognize when market behaviour may be changing.

---

# 2. What We Learned

## Market Regimes

A market regime is a period where the market behaves in a particular way.

The regime itself is not directly visible.

We observe:

- Returns
- Volatility
- Volume
- Breadth
- India VIX
- Flows
- Macro variables
- External market signals

and use them to estimate the hidden market condition.

---

## HMM

HMM is a strong fit for the hidden-regime problem because it directly models hidden states and how they change over time.

Our baseline direction is:

> **Use a multivariate HMM as the starting regime model.**

But it still needs to prove itself against simpler baselines.

---

## Bayesian HMM

Bayesian HMM can provide a richer representation of uncertainty.

However, it is more complex.

Therefore:

> **Test it rather than automatically making it the core model.**

---

## Student-t HMM

Financial returns can contain extreme movements that simple Gaussian assumptions may not handle well.

Therefore:

> **Test Student-t emissions to see whether they improve behaviour during extreme market conditions.**

---

## RS-VAR

RS-VAR gives us something different from HMM.

HMM mainly helps answer:

> **What regime are we in?**

RS-VAR helps answer:

> **How do financial variables behave and interact differently in that regime?**

Therefore:

> **Test RS-VAR as a supporting model.**

---

## BNN

BNN is better suited to complex and nonlinear directional forecasting than to discovering hidden regimes.

Therefore:

> **Test BNN as a directional forecasting component.**

The extra complexity and latency must be justified by measurable improvement.

---

## Time-Series Foundation Models

Chronos and TimesFM are interesting because they provide pre-trained time-series forecasting capabilities.

However, general forecasting ability does not automatically mean they will work well for Indian equities.

Therefore:

> **Test them as forecasting benchmarks or additional signals, not as core regime detectors.**

---

## Sequential Inference

HMM Forward Filtering already gives us a practical way to update regime probabilities as new data arrives.

Therefore:

> **Use HMM Forward Filtering for online regime updates.**

Particle Filtering remains more advanced but also more complex.

Therefore:

> **Keep Particle Filtering open for later.**

---

## BOCPD

BOCPD answers a different question from HMM.

HMM:

> **What regime are we probably in?**

BOCPD:

> **Could the market behaviour be changing?**

Therefore:

> **Test BOCPD as an early-warning signal, not as the main regime detector.**

---

## Ensemble

We do not want one model to control the entire system.

Different models should be able to have different levels of influence depending on the current market situation.

Therefore:

> **Build a hybrid, situation-aware ensemble.**

The dominant model should depend on the situation, while other models provide supporting evidence.

---

## Uncertainty & Calibration

Confidence is useful only if it is trustworthy.

Therefore:

> **Probabilities and calibration should be core parts of the system.**

We will also test stronger uncertainty methods such as Bayesian HMM and Adaptive Conformal Prediction where useful.

---

# 3. Final Research Decisions

| Component | Current decision | Intended role |
|---|---|---|
| **Standard Multivariate HMM** | 🟢 USE / Validate | Main regime candidate |
| **Bayesian HMM** | 🟡 TEST | Advanced regime + uncertainty |
| **Student-t HMM** | 🟡 TEST | Better handling of extreme movements |
| **RS-VAR** | 🟡 TEST | Regime-specific financial relationships |
| **BNN** | 🟡 TEST | Directional forecasting |
| **Chronos** | 🟡 TEST | Forecasting benchmark / extra signal |
| **TimesFM** | 🟡 TEST | Forecasting benchmark / extra signal |
| **HMM Forward Filtering** | 🟢 USE | Online regime updates |
| **Particle Filtering** | 🔵 KEEP OPEN | Future advanced online inference |
| **BOCPD** | 🟡 TEST | Early-warning signal |
| **Weighted Averaging** | 🟢 USE / Baseline | Initial ensemble |
| **Situation-Based Weighting** | 🟡 TEST | Dynamic model influence |
| **Stacking** | 🟡 TEST | Learned model combination |
| **Bayesian Model Averaging** | 🔵 KEEP OPEN | Future advanced ensemble |
| **Calibration** | 🟢 USE | Check confidence quality |
| **Adaptive Conformal Prediction** | 🟡 TEST | Additional uncertainty information |

---

# 4. Our Proposed Methodology

The current research-backed direction is:

```text
                    Market Data
                        ↓
                  Feature Layer
                        ↓
             HMM / HMM Variants
                        ↓
              Regime + Probability
                        ↓
                 Situation Layer
                        ↓
             ┌──────────┼──────────┐
             ↓          ↓          ↓
           BNN        RS-VAR      TSFM
             ↓          ↓          ↓
        Direction   Relationship  Forecast
         Signal       Signal       Signal
             └──────────┼──────────┘
                        ↓
                Hybrid Ensemble
                        ↓
                   Calibration
                        ↓
             Final Direction Signal
                        ↓
                Decision Support
```

Alongside this:

```text
BOCPD
  ↓
Early Warning
  ↓
Influences Situation / Model Weighting
```

And:

```text
HMM Forward Filtering
  ↓
Keeps regime probabilities updated as new data arrives
```

---

# 5. The Most Important Architectural Principle

We are **not** building:

```text
HMM
 ↓
Everything else
```

We are building:

```text
Situation
    ↓
Which model is currently more useful?
    ↓
Give it more influence
    ↓
Keep other models as supporting evidence
    ↓
Combine
```

This means:

> **No model permanently drives the whole system.**

---

# 6. Model Dominance

Dominance is situation-based.

For example:

| Situation | Possible stronger influence |
|---|---|
| Stable market | HMM |
| Possible transition | HMM + BOCPD |
| Extreme movement | Student-t HMM |
| Strong nonlinear behaviour | BNN |
| Strong variable interaction | RS-VAR |
| Forecast-heavy situation | BNN / TSFM |

These are **hypotheses to test**, not permanent rules.

---

# 7. Confidence and Model Influence

Each model may provide its own confidence.

We do not want to treat that confidence as automatically trustworthy.

The system should consider:

- Model confidence.
- Calibration.
- Current market situation.
- Model reliability in that situation.
- Recent performance.
- Model disagreement.
- BOCPD warning.

Our conceptual rule is:

> **Model confidence × situation-based influence = model contribution**

The exact mathematical formula will be designed and tested later.

---

# 8. Feature Direction

We will start with a relatively small set of financially meaningful features.

Initial candidates include:

- NIFTY returns
- Momentum / trend
- Realized volatility
- India VIX
- Volume / turnover
- Market breadth
- USD/INR
- Crude oil
- Global equity signals

Additional candidates such as:

- FPI/FII flows
- DII flows
- Interest rates
- Inflation

will be tested.

Slow-moving or more optional signals such as GDP, sentiment, and gold remain open for later.

---

# 9. Regime Count

We will test:

**K = 2, 3, 4, 5**

Possible interpretations may include:

- Risk-On
- Risk-Off
- Transition
- Recovery
- Late-Cycle
- Post-Shock

These labels are examples only.

The model will first discover the states, and we will then interpret them based on their financial characteristics.

The final K will be chosen using:

- Statistical quality
- Stability
- Economic meaning
- Forecasting usefulness

---

# 10. What We Have NOT Finalized

Research alone cannot tell us:

- Which K value will perform best.
- Whether HMM beats simpler baselines.
- Whether Bayesian HMM is worth its complexity.
- Whether Student-t HMM improves extreme-event handling.
- Whether RS-VAR adds enough value.
- Whether BNN improves directional forecasting.
- Whether Chronos or TimesFM add useful information.
- Whether BOCPD gives useful early warnings.
- Whether dynamic weighting improves the ensemble.
- Whether stacking beats simple weighted averaging.
- Which exact features should remain.
- What final model combination should go into production.

These must be answered through experiments.

---

# 11. What Will Happen Next

The next phases will turn the research into evidence.

The broad progression is:

```text
Research
   ↓
Architecture
   ↓
Data
   ↓
Features
   ↓
Financial Analysis
   ↓
Baseline Models
   ↓
Advanced Models
   ↓
Ensemble + Uncertainty
   ↓
Validation
   ↓
Decision Support
   ↓
Deployment
```

Each stage will provide evidence that helps us decide which research ideas actually deserve to stay.

---

# 12. Research-to-Implementation Rule

We will follow:

> **Do not implement something permanently just because it sounds advanced.**

Instead:

```text
Research
   ↓
Candidate
   ↓
Experiment
   ↓
Compare
   ↓
Validate
   ↓
Keep / Remove
```

A simpler model that performs better should win.

---

# 13. Research Gaps

Some questions intentionally remain open:

- What is the most stable regime structure for NIFTY 50?
- How quickly can regime changes be detected?
- Which features provide the strongest regime information?
- How much does each advanced model improve forecasting?
- How reliable are model confidence scores?
- What weighting strategy works best?
- How should the system behave during extreme market shocks?

These are not research failures.

They are **experimental questions for the next phases.**

---

# 14. Final Phase 2 Conclusion

> **Our research supports building a hybrid, situation-aware market regime and forecasting system centered around HMM-based regime detection, with other models providing complementary information rather than competing equally at all times.**

The current direction is:

**HMM → Regime**

**BOCPD → Early Warning**

**RS-VAR → Financial Relationships**

**BNN → Nonlinear Direction Forecast**

**TSFM → Forecasting Benchmark / Extra Signal**

**Ensemble → Combine Model Evidence**

**Calibration → Check Confidence**

**Decision Support → Final Useful Signal**

But the final model stack will only be locked after experimentation and validation.

---

# Phase 2 Status

## 🟢 RESEARCH COMPLETE

We have:

- Defined the research strategy.
- Studied the major candidate approaches.
- Compared their strengths and weaknesses.
- Defined how uncertainty will be handled.
- Defined the feature direction.
- Created preliminary model roles.
- Established the hybrid ensemble philosophy.
- Recorded what is USE, TEST, or KEEP OPEN.
- Identified the questions that require experiments.

### Next Phase

> **Phase 3, Solution Architecture**

The purpose of Phase 3 will be to turn these research conclusions into a clear technical system design.
