# 13. Research Source Library

## Purpose

This document is the **source library for Phase 2 research**.

It records the important papers, official sources, model documentation, and other references behind our research decisions.

The goal is simple:

> **If we make an important research decision, we should be able to trace it back to a reliable source.**

---

# 1. Source Priority

We will prefer sources in this order:

1. **Peer-reviewed research / original academic papers**
2. **Official financial and government sources**
3. **Official model/framework documentation**
4. **Established financial or technical research**
5. **Blogs and tutorials**
6. **Community discussions**

For major architecture decisions, lower-level sources should not be our only evidence.

---

# 2. Source Status

We will use these labels:

| Status | Meaning |
|---|---|
| **Reviewed** | We have examined the source and used it for research. |
| **Reference** | Useful background material. |
| **To Review** | Relevant source that still needs deeper study. |
| **Implementation** | Useful mainly when we start building. |
| **Official Data** | Source for actual project data. |

---

# 3. Market Regime & HMM Research

## 3.1 Predicting Risk-Adjusted Returns Using an Asset-Independent Regime-Switching Model

**Type:** Academic research  
**Topic:** HMM / financial market regimes  
**Status:** Reviewed

This research applies a hidden Markov regime-switching approach across different financial markets and studies how regime information can support risk-adjusted return prediction.

**Why it matters to Project 1A:**

It supports the basic idea that financial markets can move between different hidden regimes and that HMM-based regime information can be useful for forecasting.

citeturn0search0

---

## 3.2 Clustering Financial Time Series: New Insights from an Extended Hidden Markov Model

**Type:** Academic research  
**Topic:** HMM / regime structures  
**Status:** Reviewed

This research studies financial time series using regime-switching models and reports different market groups and regimes.

**Why it matters:**

It is useful for understanding that different numbers and types of regimes can exist and that regime structures should be investigated rather than assumed.

citeturn0search2

---

## 3.3 Dynamic Asset Allocation for Varied Financial Markets Under Regime Switching Framework

**Type:** Academic research  
**Topic:** HMM / market regimes / decision use  
**Status:** Reviewed

This research uses HMMs to identify regimes across stock, bond, and commodity markets and studies how regime information can support portfolio decisions.

**Why it matters:**

It provides evidence that regime information can be connected to financial decision-making rather than being only a statistical classification.

citeturn0search3

---

## 3.4 A Novel Dynamic Asset Allocation System Using Feature Saliency Hidden Markov Models

**Type:** Academic research  
**Topic:** HMM / feature selection / regime detection  
**Status:** Reviewed

This research combines HMM-based regime detection with feature selection and evaluates the approach on real financial indices.

**Why it matters:**

It supports our interest in keeping the feature set controlled and testing whether selected features improve regime identification.

citeturn0search5turn0search8

---

## 3.5 Testing for the Number of States in Hidden Markov Models

**Type:** Academic research  
**Topic:** HMM / number of states  
**Status:** Reviewed

This work focuses on statistical testing related to the number of states in HMMs and discusses financial applications involving different volatility states.

**Why it matters:**

It supports our decision to **test K = 2–5 rather than assuming one fixed number of regimes**.

citeturn0search9

---

# 4. Regime-Switching & Financial Risk

## 4.1 Regime Switching Model for Financial Data: Empirical Risk Analysis

**Type:** Academic research  
**Topic:** HMM / financial risk / regime changes  
**Status:** Reviewed

This research combines HMM with extreme-value methods and uses regime classification to distinguish crisis and steady periods.

**Why it matters:**

It is relevant to our interest in handling extreme market behaviour and understanding how regime models can react to stressed periods.

citeturn0search4

---

# 5. Changepoint Detection

## 5.1 Bayesian Online Changepoint Detection

**Authors:** Ryan Prescott Adams, David J. C. MacKay  
**Year:** 2007  
**Type:** Academic research  
**Topic:** BOCPD / online change detection  
**Status:** Reviewed

This is the foundational paper for Bayesian Online Changepoint Detection.

It describes an online Bayesian method for estimating when the underlying behaviour of a time series may have changed.

**Why it matters:**

This is the main research basis for our decision to test **BOCPD as an early-warning signal**, rather than using it as the main regime detector.

citeturn0academia24

---

# 6. Time-Series Foundation Models

## 6.1 Chronos: Learning the Language of Time Series

**Authors:** Abdul Fatir Ansari et al.  
**Year:** 2024  
**Type:** Research paper  
**Topic:** Time-series foundation models  
**Status:** Reviewed

Chronos is a pretrained probabilistic time-series forecasting approach based on tokenized time-series values and transformer architectures.

**Why it matters:**

It is one of the forecasting models we will test as an additional forecasting signal/benchmark.

The important point is that strong general forecasting performance does **not** automatically prove usefulness for Indian equities, so we need our own testing.

citeturn0academia25

---

## 6.2 A Decoder-Only Foundation Model for Time-Series Forecasting

**Authors:** Abhimanyu Das, Weihao Kong, Rajat Sen, Yichen Zhou  
**Year:** 2023  
**Type:** Research paper  
**Topic:** TimesFM  
**Status:** Reviewed

This paper introduces a decoder-only time-series foundation model designed for forecasting across different datasets and forecasting settings.

**Why it matters:**

It provides the research basis for considering **TimesFM as a forecasting benchmark/extra signal**.

Again, we will test it rather than assume that a general-purpose model is automatically suitable for Indian financial data.

citeturn0academia26

---

# 7. Indian Market & Financial Sources

## 7.1 NSE / NIFTY 50

**Type:** Official market source  
**Topic:** Indian equity market / NIFTY 50  
**Status:** Official Data

NIFTY 50 is our primary market target for Project 1A.

We will use official NSE information as the reference point for index-related definitions and market data requirements.

---

## 7.2 SEBI

**Type:** Official regulator  
**Topic:** Indian securities market / regulation  
**Status:** Official Source

SEBI provides official information about India's securities market, market participants, research, investor protection, and regulatory framework.

**Why it matters:**

Useful for the BFSI context, regulatory understanding, and market terminology.

SEBI also provides research reports and statistics that can support our financial-market research. citeturn1search2turn1search6

---

## 7.3 SEBI Investor Resources

**Type:** Official source  
**Topic:** Indian securities market  
**Status:** Official Source

SEBI's investor resources explain important market concepts, including securities markets, exchanges, indices, derivatives, research analysts, and related topics.

**Why it matters:**

Useful for keeping our BFSI understanding grounded in official Indian-market terminology.

SEBI identifies NIFTY 50 and Sensex as major Indian securities-market indices and describes NIFTY 50 as representing the performance of 50 large and actively traded NSE-listed companies. citeturn1search8turn1search0

---

## 7.4 Reserve Bank of India

**Type:** Official central bank  
**Topic:** Indian macroeconomic and financial data  
**Status:** Official Data

RBI's Handbook of Statistics on the Indian Economy provides macroeconomic and financial variables across multiple frequencies.

**Why it matters:**

This is an important candidate source for:

- Interest rates
- Inflation-related data
- Banking/financial indicators
- Monetary variables
- Other macroeconomic features

The RBI describes the Handbook as a comprehensive source covering output, prices, money, banking, financial markets, public finance, foreign trade, balance of payments, and other indicators. citeturn1search12

---

# 8. Sources for Research vs Sources for Data

We should keep these two purposes separate.

### Research sources

Used to understand:

- Methods
- Algorithms
- Financial theory
- Model strengths/weaknesses
- Existing evidence

Examples:

**Academic papers → methodology evidence**

### Data sources

Used to obtain:

- NIFTY data
- Market variables
- Macro variables
- Institutional flows
- Volatility data

Examples:

**NSE / RBI / SEBI → official Indian-market information and data**

---

# 9. Source-to-Decision Mapping

| Decision | Main supporting source type |
|---|---|
| Market regimes are useful to study | Financial regime-switching research |
| HMM is suitable for hidden regimes | HMM financial research |
| Test K = 2–5 | HMM state-selection research + our own experiments |
| Student-t HMM is worth testing | Financial heavy-tail/regime research |
| RS-VAR should be tested | Time-series/regime-switching research |
| BNN should be tested for direction | Bayesian deep-learning research |
| Chronos should be tested | Chronos research |
| TimesFM should be tested | TimesFM research |
| BOCPD should be an early warning | Adams & MacKay (2007) |
| HMM should update online | HMM filtering literature |
| Confidence must be calibrated | Probabilistic forecasting/calibration literature |
| NIFTY 50 is primary target | Official NSE/SEBI information |
| Indian macro data can support features | RBI official data |

---

# 10. Sources We Still Need to Add

The library is **not finished forever**.

As we move into implementation and experimentation, we should add stronger sources for:

### Bayesian HMM

Need deeper research on:

- Bayesian parameter estimation
- Priors
- Financial applications
- Practical inference

### RS-VAR

Need deeper research on:

- Regime-switching VAR estimation
- Bayesian RS-VAR
- Financial applications
- Impulse-response analysis

### BNN

Need sources on:

- Bayesian neural-network uncertainty
- Financial forecasting applications
- Calibration
- Practical inference

### Calibration

Need stronger references for:

- Brier Score
- Log Loss
- Expected Calibration Error
- Probability calibration methods

### Conformal Prediction

Need research on:

- Financial forecasting
- Time-series conformal methods
- Adaptive conformal prediction

### Ensemble Methods

Need sources for:

- Weighted averaging
- Stacking
- Bayesian Model Averaging
- Dynamic model weighting

These can be added as we deepen the research or before the corresponding implementation phase.

---

# 11. Source Quality Rules

Before using a source for an important decision, ask:

1. **Who published it?**
2. **Is it original research or a secondary explanation?**
3. **Was it tested on financial data?**
4. **Was it tested out-of-sample?**
5. **Does the market/data resemble our problem?**
6. **Are the results reproducible?**
7. **Does another credible source support the conclusion?**
8. **What are the limitations?**

---

# 12. Important Rule for Project 1A

A source can support an idea without proving that we should use it.

For example:

> A paper can show that HMM works well on financial data.

That does **not** mean:

> HMM will definitely work best on NIFTY 50.

The second question must be answered through our own experiments.

---

# Final Research Source Principle

> **Use research to understand what is possible, use strong evidence to choose what is worth testing, and use our own experiments to decide what finally stays.**

This source library should therefore grow alongside the project, but it should remain **focused on sources that actually influence Project 1A decisions.**
