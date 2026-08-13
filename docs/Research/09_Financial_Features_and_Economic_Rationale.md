# 10. Financial Features & Economic Rationale

## Purpose

This document defines **which financial signals we may use in Project 1A, what each signal tells us, and why it could help identify market regimes or forecast equity direction.**

The main rule is:

> **We will not add a feature just because it is available. Each feature must have a clear reason for being useful and must prove its value through testing.**

---

## 1. Primary Market Target

Our primary target is:

> **NIFTY 50**

The feature layer will mainly describe the behaviour of the Indian equity market around NIFTY 50.

---

## 2. Feature Groups

We will organize features into groups instead of treating every variable separately.

### A. Price & Returns

These describe the basic direction and movement of the market.

Potential features:

- Daily returns
- Multi-day returns
- Rolling returns
- Momentum
- Trend indicators

**Why useful:** They show whether the market is moving up, down, or staying directionless.

### Decision

🟢 **USE**

---

## 3. Volatility

Potential features:

- Realized volatility
- Rolling volatility
- Volatility changes
- India VIX

**Why useful:** Volatility helps identify calm, unstable, and stressed market conditions.

### Decision

🟢 **USE**

---

## 4. Volume & Market Activity

Potential features:

- Trading volume
- Turnover
- Volume changes
- Abnormal volume

**Why useful:** Strong changes in activity can indicate increased participation, stress, or changing market behaviour.

### Decision

🟢 **USE**

---

## 5. Market Breadth

Breadth measures whether market movement is supported by many stocks or only a small number of stocks.

Potential features:

- Advance/decline ratio
- Percentage of stocks above moving averages
- Number of advancing stocks
- Number of declining stocks

**Why useful:** A broad market move is usually different from a move driven by only a few large stocks.

### Decision

🟢 **USE**

---

## 6. Institutional Flows

Potential features:

- FPI/FII flows
- DII flows

**Why useful:** Institutional buying and selling can provide information about market demand and risk appetite.

### Decision

🟡 **TEST**

Reason:

> These signals are potentially useful, but we need to verify their timing, quality, and incremental value.

---

## 7. India VIX

India VIX is a market volatility indicator.

It gives information about expected near-term market volatility.

**Why useful:** It can help identify changes in market fear, uncertainty, and stress.

### Decision

🟢 **USE**

---

## 8. USD/INR

USD/INR represents the Indian rupee's value relative to the US dollar.

**Why useful:** Large currency movements can reflect external pressure, capital flows, global risk conditions, or changing economic expectations.

### Decision

🟢 **USE initially**

We will verify whether it provides useful incremental information.

---

## 9. Crude Oil

Crude oil is particularly relevant to India because changes in oil prices can affect:

- Inflation
- Import costs
- Currency pressure
- Corporate costs
- Economic expectations

**Why useful:** Major oil movements can create broader pressure or support for Indian markets.

### Decision

🟢 **USE initially**

Again, its actual predictive contribution must be tested.

---

## 10. Global Equity Signals

Potential signals:

- Major global equity index returns
- Global volatility
- Global risk indicators

**Why useful:** Indian equities are connected to global markets, so global risk conditions can influence local market behaviour.

### Decision

🟢 **USE initially**

The exact global indicators will be selected later based on data availability and relevance.

---

## 11. Interest Rates

Potential signals:

- RBI policy rate
- Government bond yields
- Relevant short/long-term rates

**Why useful:** Interest rates influence liquidity, borrowing costs, valuations, and investor behaviour.

### Decision

🟡 **TEST**

We should first verify which rate measures are available at the right frequency and timing.

---

## 12. Inflation

Potential signals:

- CPI
- WPI
- Relevant inflation measures

**Why useful:** Inflation can influence monetary policy, interest rates, consumer behaviour, and market expectations.

### Decision

🟡 **TEST**

Important issue:

> Macro data is often released with a delay, so we must use the value that was actually available at prediction time.

---

## 13. GDP and Slow-Moving Macro Variables

Potential signals:

- GDP growth
- Other low-frequency economic indicators

**Why useful:** They describe broader economic conditions.

### Decision

🔵 **KEEP OPEN**

These variables may provide useful context but may not add much value to short-term regime detection because they change relatively slowly.

---

## 14. Sentiment

Potential signals could include:

- News sentiment
- Market sentiment indicators
- Other text-based signals

**Why useful:** Sentiment may capture information that price and traditional financial variables do not immediately capture.

### Decision

🔵 **KEEP OPEN**

We should not add sentiment until the core system is working and we have a clear data source and testing plan.

---

## 15. Gold

Gold can sometimes act as a defensive or alternative asset signal.

**Why useful:** Its relationship with equities may provide additional information about risk appetite.

### Decision

🔵 **KEEP OPEN**

It should only be added if experiments show that it provides useful information beyond our existing features.

---

# 16. Initial Feature Set

We do not want a huge feature list at the beginning.

Our starting set should be around **8–12 strong features**.

A reasonable initial group is:

| Feature | Initial role |
|---|---|
| NIFTY returns | 🟢 Use |
| Rolling returns / momentum | 🟢 Use |
| Realized volatility | 🟢 Use |
| India VIX | 🟢 Use |
| Volume / turnover | 🟢 Use |
| Market breadth | 🟢 Use |
| USD/INR | 🟢 Use |
| Crude oil | 🟢 Use |
| Global equity signal | 🟢 Use |
| FPI/FII flows | 🟡 Test |
| DII flows | 🟡 Test |
| Interest rates | 🟡 Test |

This is a **starting candidate set**, not the final feature list.

---

# 17. Why We Keep the Feature Set Small

Too many features can create problems:

- More noise
- Higher dimensionality
- More overfitting risk
- More missing data
- More computation
- Harder interpretation
- More difficult debugging

Therefore:

> **Start small, test carefully, then expand only when there is evidence.**

---

# 18. Feature Selection Rule

Every feature should answer three questions:

### 1. What does it represent?

Example:

> VIX represents expected market volatility.

### 2. Why could it help?

Example:

> Rising VIX may indicate increasing uncertainty or stress.

### 3. Does it actually help?

This must be answered through our experiments.

A feature that sounds economically useful but does not improve the model should not automatically stay.

---

# 19. Data Timing and Look-Ahead Bias

This is especially important for financial and macro features.

We must use:

> **The value that was actually available at the time of prediction.**

Example:

If an economic indicator for June is officially released in July, we cannot use the June value for a June prediction as if we already knew it.

This is a common source of look-ahead bias.

---

# 20. Different Feature Frequencies

Our data may come at different frequencies:

- Daily market data
- Weekly indicators
- Monthly macro data
- Policy announcements

We need to align them carefully.

The system should never pretend that a slow-moving indicator was updated before it was actually published.

---

# 21. Feature Quality Checks

Before using a feature, we should check:

- Data availability
- Data frequency
- Data timing
- Missing values
- Outliers
- Stability
- Relationship with the target
- Redundancy with other features
- Historical coverage
- Leakage risk

---

# 22. Feature Redundancy

Some features may contain almost the same information.

For example:

```text
NIFTY returns
       +
Short-term momentum
       +
Several similar return windows
```

Adding many highly similar features may not add useful information.

We should therefore check whether each new feature provides something different.

---

# 23. Economic Meaning vs Statistical Performance

A feature can be statistically useful without having a clear economic explanation.

A feature can also have a strong economic explanation but provide little predictive value.

We want both where possible:

> **Statistical usefulness + Financial meaning**

This is especially important for a decision-support system.

---

# 24. Features for Different Models

Not every model needs exactly the same features.

For example:

### HMM

Focus on features that clearly describe market states.

### RS-VAR

Focus on variables whose relationships we want to study.

### BNN

Can use a broader set of useful nonlinear predictors.

### TSFM

May require its own input format depending on the model.

### BOCPD

Can operate on selected time-series signals where detecting behavioural change is meaningful.

Therefore:

> **Feature selection will be model-specific when necessary.**

---

# 25. Feature Role in Our Hybrid Architecture

Features provide the common market information layer.

```text
Financial Data
      ↓
Feature Engineering
      ↓
Market Signals
      ↓
 ┌────┼────┬────┬────┐
 ↓    ↓    ↓    ↓    ↓
HMM RS-VAR BNN TSFM BOCPD
 ↓    ↓    ↓    ↓    ↓
Model-specific information
      ↓
Hybrid Ensemble
```

The same raw data can therefore support different models for different purposes.

---

# 26. Current Feature Decisions

| Feature group | Decision |
|---|---|
| NIFTY returns | 🟢 USE |
| Momentum / trend | 🟢 USE |
| Realized volatility | 🟢 USE |
| India VIX | 🟢 USE |
| Volume / turnover | 🟢 USE |
| Market breadth | 🟢 USE |
| USD/INR | 🟢 USE initially |
| Crude oil | 🟢 USE initially |
| Global equity signals | 🟢 USE initially |
| FPI/FII flows | 🟡 TEST |
| DII flows | 🟡 TEST |
| Interest rates | 🟡 TEST |
| Inflation | 🟡 TEST |
| GDP | 🔵 KEEP OPEN |
| Sentiment | 🔵 KEEP OPEN |
| Gold | 🔵 KEEP OPEN |

---

# 27. Important Rule for Phase 4–6

The above list is **not the final dataset or final feature engineering specification**.

The final features will be decided after:

**Data availability → Data quality → Timing check → Feature engineering → Testing → Validation**

So we should not lock exact columns, transformations, or lookback windows yet.

---

# Final Decision

> **Start Project 1A with a compact set of 8–12 financially meaningful features, validate their timing and quality, and add or remove features only when experiments show clear value.**

The simplest mental model is:

**Choose meaningful signals → Check when they were available → Test their value → Keep only useful features.**
