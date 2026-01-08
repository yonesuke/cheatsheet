# The Seven Sins of Quantitative Investing

> **Based on Deutsche Bank Quantitative Strategy Research**

This cheatsheet outlines the seven most common biases and mistakes ("Sins") that occur during the backtesting of quantitative investment models, along with their remedies.

---

## 1. Survivorship Bias

**The Sin:**
Considering only companies that are currently in existence, ignoring those that have gone bankrupt, merged, or been delisted. This artificially inflates historical performance because "dead" companies usually performed poorly.

**Examples:**
- **Russell 3000:** "Survivor-only" universe yields significantly higher returns.
- **Credit Risk Models:** Testing on survivors makes risky firms appear to have highest returns (7x better) because failed risky firms are excluded.

**The Remedy:**
- **Use Point-in-Time (PIT) Data:** Include "dead" companies active during the historical period.
- **Handle Missing Data:** Treat companies as part of the universe until they formally exit; don't filter out based on missing trailing data if it means excluding delistings.

---

## 2. Look-ahead Bias

**The Sin:**
Using information in a backtest that was not actually available to investors at that time.

**Examples:**
- **Earnings Data Lag:** Assuming data is available on quarter-end, when it's actually reported 1-3 months later.
- **Restated Data:** Using revised historical data instead of preliminary data originally released.
- **Split Adjustments:** Using split-adjusted prices for price-level strategies (e.g., "Buy stocks under $5").

**The Remedy:**
- **Lag Data:** Mechanically lag financial data (e.g., 3 months) if PIT timestamps are unavailable.
- **Use Unadjusted Prices:** For strategies based on absolute price levels.
- **Strict PIT Databases:** Use databases with exact public availability timestamps.

---

## 3. The Sin of Storytelling

**The Sin:**
Post-hoc rationalization. Creating a plausible "story" to explain a statistical anomaly after seeing the results.

**Examples:**
- **Leverage:** Interpreting performance as "risk premium" vs "penalizing distress" depending on the outcome.
- **Tech Bubble (1998-2000):** Inventing stories that "traditional valuation is dead" when Value failed, only for it to rebound.

**The Remedy:**
- **Long-Term Validation:** Test over long horizons (20+ years) across multiple economic cycles.
- **Skepticism:** Be wary of theories created *after* data analysis.

---

## 4. Data Mining (and Snooping)

**The Sin:**
"Torturing the data until it confesses." Running hundreds of variations and picking the best one, leading to overfitting.

**Examples:**
- **72-Factor Experiment:** Picking best factors from a large pool yields high in-sample Sharpe (0.7) but zero out-of-sample return.

**The Remedy:**
- **Out-of-Sample Testing:** Always withhold data for testing (different time period or region).
- **Economic Intuition:** Select factors based on logic *before* testing.
- **Lock the Validation Set:** Do not touch validation data until the model is finalized.

---

## 5. Signal Decay and Turnover

**The Sin:**
Ignoring transaction costs and signal decay speed. High theoretical returns can turn into losses after costs.

**Examples:**
- **Short-Term Reversals:** "Buying yesterday's losers" has high returns but requires high turnover; costs eat the profit.
- **Implementation Delay:** Trading at "Open" instead of "Close" can degrade fast-moving signals.

**The Remedy:**
- **Net-of-Cost Testing:** Simulate returns after deducting realistic commissions and impact costs.
- **Staggered Rebalancing:** Rebalance a small portion (e.g., 3-4%) monthly instead of all at once to keep turnover manageable.

---

## 6. Outliers

**The Sin:**
Mishandling extreme data points. Including errors ruins results; excluding valid extremes removes risk info.

**Examples:**
- **1994 Earnings Distortion:** One company with 300,000% earnings yield distorted the index average.
- **Ranking Momentum:** Ranking (0-100) hurts Momentum strategies which rely on the *magnitude* of the price jump (~35% power loss).

**The Remedy:**
- **Winsorization:** Cap extreme values (e.g., 1st/99th percentile) instead of deleting.
- **Context-Aware Processing:** Use ranking for noisy data; preserve magnitude for distance-based factors (Momentum).

---

## 7. The Asymmetric Pattern and Shorting Cost

**The Sin:**
Assuming Shorting is as easy/cheap as Buying.

**Examples:**
- **Borrowing Costs:** "Hard-to-borrow" costs often eat up alpha for shorting bad companies.
- **Asymmetry:** Value works best Long; Momentum often works best Short.

**The Remedy:**
- **Explicit Constraints:** Assume higher costs for shorting; prohibit shorting "hard-to-borrow" stocks.
- **Diversification:** Increase holdings (e.g., to 500+) to reduce idiosyncratic risk if shorting is constrained.
