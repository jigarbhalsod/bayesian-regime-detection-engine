# 02. Research Strategy

## Purpose

This document explains **how we will do research for Project 1A**.

The goal is not to collect as many papers or technologies as possible. The goal is to find the approaches that are actually useful for detecting Indian market regimes and improving equity-direction forecasting.

---

## 1. What We Want to Learn

Our research should help us understand:

- What a market regime really is.
- How different market regimes can be identified.
- What causes a regime to change.
- Which market signals can show a regime.
- How statistical and machine-learning methods can detect regimes.
- How Bayesian methods can help us measure uncertainty.
- How different financial variables can be studied together.
- How regime changes can be detected early.
- Which financial features are useful.
- How detected regimes can improve equity-direction forecasting.
- Which methods work well in financial markets and where they fail.
- Which methods are practical with our data and resources.
- Which combination of methods is most suitable for Project 1A.

---

## 2. Main Research Areas

We will study these areas:

1. **Market Regime Detection**  
   How market regimes are defined and detected.

2. **Probabilistic & Bayesian Methods**  
   HMMs, Bayesian HMMs, Bayesian inference, and Bayesian neural networks.

3. **Time-Series Modelling**  
   Regime-Switching VAR and modern time-series models.

4. **Online & Sequential Inference**  
   How the system can update its understanding as new market data arrives.

5. **Changepoint Detection**  
   How we can identify possible changes in market behaviour.

6. **Uncertainty & Calibration**  
   How we measure confidence and check whether that confidence can be trusted.

7. **Ensemble & Model Combination**  
   How different models can support each other.

8. **Financial Features & Economic Drivers**  
   Price, volatility, volume, breadth, flows, macro data, and other useful signals.

9. **Forecasting & Decision Support**  
   How regime information can improve equity-direction forecasting and decisions.

10. **Validation & Backtesting**  
    How we determine whether our results are genuinely useful and robust.

---

## 3. Where We Get Our Information

We will prefer sources in this order:

### Highest priority

- Peer-reviewed research papers.
- Original academic research.
- Official financial and government sources such as NSE, SEBI, and RBI.
- Original papers for important models and methods.

### Next priority

- Official model and framework documentation.
- Official GitHub repositories.
- Research from banks, exchanges, asset managers, and established financial institutions.

### Supporting sources

- Good technical articles and tutorials.
- Community discussions such as Reddit or Stack Overflow.

Community and blog sources can help us understand ideas or implementation problems, but they should not be the main reason for an important project decision.

### Simple rule

**Use strong evidence for important decisions, and use weaker sources mainly for understanding or exploration.**

---

## 4. How We Will Judge Each Method

Every method will be checked using these points:

- **Performance:** Does it actually improve prediction?
- **Regime quality:** Does it find meaningful market states?
- **Uncertainty:** Can it tell us when it is unsure?
- **Interpretability:** Can we understand its output?
- **Financial meaning:** Does the result make sense in the real market?
- **Robustness:** Does it continue to work across different market conditions?
- **Early detection:** Can it notice important changes quickly?
- **Data needs:** Can we get the required data reliably?
- **Speed:** Can it work fast enough for our use case?
- **Complexity:** Is it practical to build and maintain?
- **Overfitting risk:** Could it simply be learning historical noise?
- **Data leakage risk:** Could it accidentally use future information?
- **Practical value:** Does it actually help the final decision-support system?

### Our main evaluation lens

> **Performance + Financial Meaning + Uncertainty + Robustness + Interpretability + Practicality**

---

## 5. How Research Becomes a Decision

We will not automatically use something just because research says it is good.

Every major method will be placed into one of four groups:

### 🟢 USE

Strong evidence and good fit for Project 1A.

### 🟡 TEST

Looks promising, but we need our own experiments before committing.

### 🔴 REJECT

Poor fit, too risky, too complex, or not useful enough.

### 🔵 KEEP OPEN

Interesting idea that we do not need to decide on right now.

### Decision flow

**Research → Evidence → Compare → Test if needed → Decide**

---

## 6. What Counts as Good Evidence

We will look for:

- **Method evidence:** Does the method make sense for the problem?
- **Experimental evidence:** Has it worked in real experiments?
- **Financial evidence:** Has it worked on financial data?
- **Comparison evidence:** Does it offer something better than simpler methods?
- **Robustness evidence:** Does it work across different market conditions?
- **Reproducibility:** Can we reasonably reproduce or verify the approach?
- **Data compatibility:** Can we actually get the required data?
- **Operational fit:** Can we realistically run it in our system?
- **Risk evidence:** Have we considered leakage, overfitting, instability, and uncertainty?
- **Decision value:** Does it make the final system more useful?

### Evidence strength

**Strong:** Multiple credible sources + good financial evidence + robust results.

**Moderate:** Good research support, but our own testing is still needed.

**Weak:** Mainly theory, isolated results, blogs, or opinions.

---

## 7. What We Will Produce From the Research

By the end of Phase 2, we should have:

1. **Research Notes**  
   Simple notes on important methods and concepts.

2. **Literature Review**  
   Important findings from research papers and reliable sources.

3. **Method Profiles**  
   What each method does, its strengths, weaknesses, and relevance to Project 1A.

4. **Financial Feature Rationale**  
   Why we think each important feature may be useful.

5. **Method Comparison Matrix**  
   A side-by-side comparison of the methods.

6. **Evidence & Source Record**  
   Important claims, their sources, and how strong the evidence is.

7. **Research Decision Log**  
   Why we chose, tested, rejected, or kept each major approach.

8. **Recommended Methodology**  
   Our proposed combination of models and techniques.

9. **Research Gaps & Open Questions**  
   Things that still need to be proven through experiments.

10. **Final Research Conclusions**  
    What we learned, what we decided, and why.

---

## 8. Important Project Rule

**Research is not the same as implementation.**

A method can look excellent in research and still fail on our data.

Therefore:

> **Research tells us what is worth testing. Experiments tell us what deserves to stay.**

This keeps Project 1A focused on building a useful system rather than simply collecting advanced technologies.

---

## Phase 2 Research Flow

**Research Strategy → Market Regimes → HMM → RS-VAR → Advanced Forecasting → Online Detection → Changepoints → Ensembles → Uncertainty → Features → Comparison → Final Conclusions**
