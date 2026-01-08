# Option Theory Cheatsheet

A comprehensive guide to analytical solutions for option pricing, stochastic calculus foundations, and the Greeks.

## 1. Fundamentals

### Stochastic Differential Equation (SDE)
A general 1-dimensional SDE is described as:

$$
dX_t = \mu(t, X_t) dt + \sigma(t, X_t) dW_t
$$

Where:
*   $X_t$: Stochastic process
*   $\mu$: Drift term
*   $\sigma$: Diffusion term
*   $W_t$: Standard Brownian Motion (Wiener Process)

### Ito's Lemma
For a stochastic process $X_t$ following the above SDE, the differential of a function $f(t, X_t)$ is:

$$
df(t, X_t) = \left( \frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial x^2} \right) dt + \sigma \frac{\partial f}{\partial x} dW_t
$$

Special case $X_t = W_t$ ($\mu=0, \sigma=1$):
$$
df(t, W_t) = \left( \frac{\partial f}{\partial t} + \frac{1}{2} \frac{\partial^2 f}{\partial x^2} \right) dt + \frac{\partial f}{\partial x} dW_t
$$

### Geometric Brownian Motion (GBM)
Standard model for stock prices $S_t$:

$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$

Applying Ito's Lemma to $f(S_t) = \ln S_t$:

$$
S_T = S_t \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)(T-t) + \sigma (W_T - W_t) \right)
$$

Properties:
*   $E[S_T | S_t] = S_t e^{\mu(T-t)}$
*   $Var[\ln(S_T/S_t)] = \sigma^2(T-t)$

## 2. Pricing Framework

### Risk-Neutral Measure ($\mathbb{Q}$)
In an arbitrage-free market, there exists a measure $\mathbb{Q}$ under which the discounted asset price is a martingale.
Stock price dynamics under $\mathbb{Q}$ (with continuous dividend yield $q$):

$$
dS_t = (r - q) S_t dt + \sigma S_t dW_t^\mathbb{Q}
$$

### Risk-Neutral Valuation
The value $V_t$ of a derivative with payoff $V_T$ is the discounted expected value under $\mathbb{Q}$:

$$
V_t = e^{-r(T-t)} E^\mathbb{Q} [ V_T | \mathcal{F}_t ]
$$

### Feynman-Kac Formula
The conditional expectation $V(t, x) = E^\mathbb{Q} [ e^{-r(T-t)} \psi(S_T) | S_t = x ]$ solves the Partial Differential Equation (PDE):

$$
\frac{\partial V}{\partial t} + (r-q) x \frac{\partial V}{\partial x} + \frac{1}{2} \sigma^2 x^2 \frac{\partial^2 V}{\partial x^2} - rV = 0
$$
Boundary Condition: $V(T, x) = \psi(x)$

## 3. Black-Scholes Model

### Black-Scholes PDE
$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r-q) S \frac{\partial V}{\partial S} - rV = 0
$$

### Assumptions
*   Stock price follows GBM.
*   Risk-free rate $r$ and volatility $\sigma$ are constant.
*   No transaction costs or taxes.
*   Short selling is allowed.
*   Continuous trading is possible.

## 4. Analytical Solutions & Greeks

Notation:
*   $\tau = T - t$ (Time to maturity)
*   $N(x)$: Cumulative standard normal distribution function
*   $N'(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$: Standard normal probability density function

Standard Variables:
$$
d_1 = \frac{\ln(S/K) + (r - q + \frac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}}
$$
$$
d_2 = d_1 - \sigma\sqrt{\tau} = \frac{\ln(S/K) + (r - q - \frac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}}
$$

### European Options (Vanilla)
*   **Call Payoff**: $\max(S_T - K, 0)$
*   **Put Payoff**: $\max(K - S_T, 0)$

### Digital Options (Binary)
These are options that pay a fixed amount depending on the position of the underlying price at maturity.

*   **Cash-or-Nothing Call**: Pays amount $Q$ (assume $Q=1$ below) if $S_T > K$.
*   **Cash-or-Nothing Put**: Pays amount $Q$ (assume $Q=1$ below) if $S_T < K$.
*   **Asset-or-Nothing Call**: Pays stock shares $S_T$ if $S_T > K$.
*   **Asset-or-Nothing Put**: Pays stock shares $S_T$ if $S_T < K$.

### Summary Table of Solutions and Greeks

| Measure | European Call | European Put | Digital Call (Cash-or-Nothing) | Digital Put (Cash-or-Nothing) |
| :--- | :--- | :--- | :--- | :--- |
| **Price** ($V$) | $S e^{-q\tau} N(d_1) - K e^{-r\tau} N(d_2)$ | $K e^{-r\tau} N(-d_2) - S e^{-q\tau} N(-d_1)$ | $e^{-r\tau} N(d_2)$ | $e^{-r\tau} N(-d_2)$ |
| **Delta** ($\Delta = \frac{\partial V}{\partial S}$) | $e^{-q\tau} N(d_1)$ | $-e^{-q\tau} N(-d_1)$ | $\frac{e^{-r\tau} N'(d_2)}{S\sigma\sqrt{\tau}}$ | $-\frac{e^{-r\tau} N'(d_2)}{S\sigma\sqrt{\tau}}$ |
| **Gamma** ($\Gamma = \frac{\partial^2 V}{\partial S^2}$) | $\frac{e^{-q\tau} N'(d_1)}{S \sigma \sqrt{\tau}}$ | $\frac{e^{-q\tau} N'(d_1)}{S \sigma \sqrt{\tau}}$ | $-\frac{e^{-r\tau} d_1 N'(d_2)}{S^2 \sigma^2 \tau}$ | $\frac{e^{-r\tau} d_1 N'(d_2)}{S^2 \sigma^2 \tau}$ |
| **Vega** ($\mathcal{V} = \frac{\partial V}{\partial \sigma}$) | $S e^{-q\tau} \sqrt{\tau} N'(d_1)$ | $S e^{-q\tau} \sqrt{\tau} N'(d_1)$ | $-\frac{e^{-r\tau} d_1 N'(d_2)}{\sigma}$ | $\frac{e^{-r\tau} d_1 N'(d_2)}{\sigma}$ |
| **Theta** ($\Theta = \frac{\partial V}{\partial t}$) | $-\frac{S e^{-q\tau} N'(d_1)\sigma}{2\sqrt{\tau}} - r K e^{-r\tau} N(d_2) + q S e^{-q\tau} N(d_1)$ | $-\frac{S e^{-q\tau} N'(d_1)\sigma}{2\sqrt{\tau}} + r K e^{-r\tau} N(-d_2) - q S e^{-q\tau} N(-d_1)$ | see note* | see note* |
| **Rho** ($\rho = \frac{\partial V}{\partial r}$) | $K \tau e^{-r\tau} N(d_2)$ | $-K \tau e^{-r\tau} N(-d_2)$ | $-\tau e^{-r\tau} N(d_2) + \frac{\sqrt{\tau}}{\sigma} e^{-r\tau} N'(d_2)$ | $-\tau e^{-r\tau} N(-d_2) - \frac{\sqrt{\tau}}{\sigma} e^{-r\tau} N'(d_2)$ |

*Note on Digital Theta*:
$$
\Theta_{DigCall} = r e^{-r\tau} N(d_2) + \frac{e^{-r\tau} N'(d_2)}{\sigma \sqrt{\tau}}\left(\frac{\ln(S/K)}{\tau} - (r - q - \frac{\sigma^2}{2})\right)
$$
(Often simplified using relationships between Greeks).

### Asset-or-Nothing
Notice that a European Call is essentially:
$$
\text{Call} = (\text{Asset-or-Nothing Call}) - K \times (\text{Cash-or-Nothing Call})
$$
Thus:
*   Asset-or-Nothing Call Price: $S e^{-q\tau} N(d_1)$
*   Asset-or-Nothing Put Price: $S e^{-q\tau} N(-d_1)$

## 5. Exotic Options (Path Dependent)

### Barrier Options
Options that become active (Knock-in) or null (Knock-out) if the asset price $S_t$ crosses a barrier $H$.
Solutions provided here are for **Single Barrier Options** (Reiner-Rubinstein).

#### Parameters and Definitions
*   $\tau = T - t$
*   $\lambda = \frac{r - q + \frac{1}{2}\sigma^2}{\sigma^2}$
*   $y = \frac{\ln(H^2 / (SK))}{\sigma\sqrt{\tau}}$
*   $x_1 = \frac{\ln(S/K)}{\sigma\sqrt{\tau}} + (1 + \lambda)\sigma\sqrt{\tau}$
*   $x_2 = \frac{\ln(S/K)}{\sigma\sqrt{\tau}} + (1 - \lambda)\sigma\sqrt{\tau}$ (Same as standard $d_2$)
*   $y_1 = \frac{\ln(H^2 / (SK))}{\sigma\sqrt{\tau}} + (1 + \lambda)\sigma\sqrt{\tau}$
*   $y_2 = \frac{\ln(H^2 / (SK))}{\sigma\sqrt{\tau}} + (1 - \lambda)\sigma\sqrt{\tau}$

**Standard Terms ($A$ - $F$ common notation)**:
(Assuming payout is Vanilla Call/Put if valid, no separate rebate for simplicity)

*   $A = \phi S e^{-q\tau} N(\phi x_1) - \phi K e^{-r\tau} N(\phi x_2)$  (Vanilla Price, $\phi=1$ for Call, $-1$ for Put)
*   $B = \phi S e^{-q\tau} N(\phi x_1) - \phi K e^{-r\tau} N(\phi x_2)$  (Same as A but with barrier condition logic)
*   $C = \phi S e^{-q\tau} (H/S)^{2(\lambda+1)} N(\eta y_1) - \phi K e^{-r\tau} (H/S)^{2\lambda} N(\eta y_2)$
*   $D = \phi S e^{-q\tau} (H/S)^{2(\lambda+1)} N(\eta y_1) - \phi K e^{-r\tau} (H/S)^{2\lambda} N(\eta y_2)$

#### 1. Down Options ($S_0 > H$)
Payoffs depend on price falling to $H$.

| Option Type | Condition | Formula |
| :--- | :--- | :--- |
| **Down-and-In Call** | $K > H$ | $(H/S)^{2\lambda} (S^c(y) - S^c(x_1))$ (Often simplified: CD-like terms) |
| **Down-and-Out Call** | $S_0 > H$ | **Standard**: $A - C$ (where $A$=Vanilla Call, $C$=Reflection) <br> $C_{do} = S N(x_1) - K e^{-r\tau} N(x_2) - (H/S)^{2\lambda} [ S N(y_1) - K e^{-r\tau} N(y_2) ]$ <br> (Note: Validity strictly for $K > H$. If $K < H$, formula adjusts). |
| **Down-and-In Put** | Any | $P_{di} = S e^{-q\tau} (H/S)^{2\lambda+2} N(y_1) - K e^{-r\tau} (H/S)^{2\lambda} N(y_2)$ |
| **Down-and-Out Put** | $K > H$ | $A - B + C - D$ (Complex interactions if $K$ vs $H$ varies). <br> **Simplified**: $P_{do} = P_{vanilla} - P_{di}$ |
| **Down-and-In Call (Full)** | | $S (H/S)^{2\lambda} N(y_1) - K e^{-r\tau} (H/S)^{2\lambda - 2} N(y_2)$ |

*To be rigorous, here is the full set of 8 formulas relative to Vanilla Price ($V_{BS}$):*

Let $\eta = 1$ for In, $-1$ for Out. Use indicator functions for $K$ relative to $H$. The breakdown is complex so standard texts define terms $A, B, C, D$.

**Analytical Formula Summary (Haug representation)**:

**Call Options ($S > H$, Down)**
1.  **Down-and-In Call ($K > H$)**: $C_{di} = S e^{-q\tau} (H/S)^{2\lambda} N(y_1) - K e^{-r\tau} (H/S)^{2\lambda - 2} N(y_2)$
2.  **Down-and-Out Call ($K > H$)**: $C_{do} = C_{vanilla} - C_{di}$
3.  **Down-and-In Call ($K < H$)**: $C_{di} = C_{vanilla} - C_{do}$ (See Out formula below)
4.  **Down-and-Out Call ($K < H$)**: $C_{do} = S e^{-q\tau} N(x_1) - K e^{-r\tau} N(x_2) - S e^{-q\tau} (H/S)^{2\lambda} N(y_1) + K e^{-r\tau} (H/S)^{2\lambda - 2} N(y_2)$

**Put Options ($S > H$, Down)**
5.  **Down-and-In Put ($K > H$)**: $P_{di} = P_{vanilla} - P_{do}$
6.  **Down-and-Out Put ($K > H$)**: $P_{do} = S e^{-q\tau} N(-x_1) - K e^{-r\tau} N(-x_2) - S e^{-q\tau} (H/S)^{2\lambda} N(-y_1) + K e^{-r\tau} (H/S)^{2\lambda - 2} N(-y_2)$
7.  **Down-and-In Put ($K < H$)**: $P_{di} = P_{vanilla}$ (Knock-in is certain if ITM?) No, careful. Actually $P_{di} = A (\text{put terms})$.
    *   Correct Logic: If $K < H$ and we are Down-and-In, we must hit $H$ to activate. Since $S > H > K$, if we hit $H$, we are not consistently ITM deeper.
    *   Formula: $P_{di} = -S e^{-q\tau} (H/S)^{2\lambda} N(-y_1) + K e^{-r\tau} (H/S)^{2\lambda - 2} N(-y_2)$

**Up Options ($S < H$, Up)**
(Symmetric to Down options by swapping $S \leftrightarrow K$, etc., but specific formulas exist).
*   **Up-and-In Call**: $A (\text{reflection})$
*   **Up-and-Out Call**: Vanilla - In
*   **Up-and-In Put**: Reflexive to Down-In Call.
*   **Up-and-Out Put**: Vanilla - In.

### Lookback Options

Definitions:
*   $S_T$: Price at maturity
*   $M_T = \max_{0 \le t \le T} S_t$ (Maximum)
*   $m_T = \min_{0 \le t \le T} S_t$ (Minimum)
*   $b = r - q$ (Carry cost)

#### 1. Floating Strike Call
Payoff: $C_T = S_T - m_T$ (Buy at min, sell at spot)
$$
C = S e^{-q\tau} N(a_1) - S e^{-q\tau} \frac{\sigma^2}{2b} N(-a_1) - m_t e^{-r\tau} N(a_2) + S e^{-r\tau} \frac{\sigma^2}{2b} e^{Y} N(-a_3)
$$
Where:
*   $a_1 = \frac{\ln(S/m_t) + (b + \sigma^2/2)\tau}{\sigma\sqrt{\tau}}$
*   $a_2 = a_1 - \sigma\sqrt{\tau}$
*   $a_3 = \frac{\ln(S/m_t) + (-b + \sigma^2/2)\tau}{\sigma\sqrt{\tau}}$
*   $Y = \frac{2b}{\sigma^2} \ln(S/m_t)$

#### 2. Floating Strike Put
Payoff: $P_T = M_T - S_T$ (Sell at max, buy at spot)
$$
P = M_t e^{-r\tau} N(-b_2) - S e^{-q\tau} N(-b_1) + S e^{-r\tau} \frac{\sigma^2}{2b} [ -(M_t/S)^{2b/\sigma^2} N(b_3) + e^{b\tau} N(b_3 + \frac{2b}{\sigma}\sqrt{\tau}) ]
$$
(Formula form varies by recursive $M_t$ logic, standard form above simplifies for $t=0, M_0=S_0$).
Standard $t=0$ ($M_0=S_0$):
$$
P = S e^{-q\tau} [ \frac{\sigma^2}{2b} N(-x_1) - N(-x_2) ] + S e^{-r\tau} (1 + \frac{\sigma^2}{2b}) N(x_2 - \frac{2b}{\sigma}\sqrt{\tau})
$$

#### 3. Fixed Strike Call
Payoff: $C_T = \max(M_T - K, 0)$
$$
C = S e^{-q\tau} N(d_1) - K e^{-r\tau} N(d_2) + S e^{-r\tau} \frac{\sigma^2}{2b} [ -(S/K)^{-2b/\sigma^2} N(d_1 - \frac{2b}{\sigma}\sqrt{\tau}) + e^{b\tau} N(d_1) ]
$$
(Note: $M_t$ term omitted for $t=0$ case where $M_0=S_0 < K$). If $M_t > K$, split into certainty part + option part.

#### 4. Fixed Strike Put
Payoff: $P_T = \max(K - m_T, 0)$
$$
P = K e^{-r\tau} N(-d_2) - S e^{-q\tau} N(-d_1) + S e^{-r\tau} \frac{\sigma^2}{2b} [ (S/K)^{-2b/\sigma^2} N(-d_1 + \frac{2b}{\sigma}\sqrt{\tau}) - e^{b\tau} N(-d_1) ]
$$
(For $t=0$ case where $m_0=S_0 > K$).

### Asian Options (Geometric)

### Asian Options
Payoff depends on the average price $A_T$.

**Geometric Asian Option** (Closed Form):
Average defined as $G_T = \exp\left( \frac{1}{T} \int_0^T \ln S_t dt \right)$.
$G_T$ follows a lognormal distribution. We can use the Black-Scholes formula with adjusted parameters:
*   Volatility: $\sigma_{adj} = \sigma / \sqrt{3}$
*   Drift: $b_{adj} = \frac{1}{2}(r - q - \frac{1}{2}\sigma^2) + \frac{1}{6}\sigma^2$ (Example approximation for drift term depends on precise average definition)

**Arithmetic Asian Option**:
Approximations (e.g., Edgeworth expansion, Moment matching) or Monte Carlo are required. No exact closed form.

## 6. American Options

Exercisable at possible any time $t \le T$.

### Finite Horizon American Put
*   **No analytical closed-form solution**.
*   **Binomial Trees**: CRR model.
*   **Finite Difference**: Solve the PDE/Variational Inequality numerically.
*   **BBAW Approximation**: Splits value into European value + Early exercise premium.

### Perpetual American Put
Infinite maturity ($T \to \infty$).
The solution becomes time-independent (ODE).

$$
P(S) = \frac{K}{1 - \gamma} \left( \frac{(\gamma - 1)S}{\gamma K} \right)^\gamma
$$

Optimal Exercise Boundary:
$$
S^* = \frac{\gamma}{\gamma - 1} K
$$

where $\gamma$ is the negative root of the characteristic equation (assuming $q=0$ for simplicity):
$$
\frac{1}{2}\sigma^2 \gamma(\gamma - 1) + r \gamma - r = 0 \quad \Rightarrow \quad \gamma = \frac{-(r - \frac{\sigma^2}{2}) - \sqrt{(r - \frac{\sigma^2}{2})^2 + 2\sigma^2 r}}{\sigma^2}
$$

## 7. Delta Hedging & PnL Dynamics

### Theory: Market Completeness
In the Black-Scholes framework, risk can be completely eliminated because the market is **Complete**. This allows the creation of a **Replicating Portfolio** that perfectly mimics the option's behavior using the underlying asset $S$ and cash (bond).

### 1. PnL Decomposition
For a Delta-Hedged portfolio (Short Option + Long Stock), the Total PnL at maturity $T$ is:

$$
\text{Total PnL} = \underbrace{\text{Premium Received}}_{V_0} - \underbrace{\text{Option Payoff}}_{V_T} - \underbrace{\text{Total Rebalancing Cost}}_{\text{Hedge Cost}}
$$

If the hedge is perfect (and assumptions hold), $\text{Total PnL} \approx 0$ (ignoring interest rate for simplicity).

#### Rebalancing Cost (RC) Forms
Let $h_t = \Delta_t$ be the number of shares held. The cost can be analyzed in two ways using integration by parts ($d(hS) = h dS + S dh$):

**1. Cash Flow View ($\int S dh$)** - *Practical View*
Focuses on the cash spent/received to adjust the position size.
$$
RC = \int_0^T S_t dh_t \approx \sum_{i=1}^N S_i (h_i - h_{i-1})
$$
*   **Intuition**: If you are Short Gamma (Short Option), you buy stock when $S$ rises ($dh > 0$ at high $S$) and sell when $S$ falls ($dh < 0$ at low $S$). You consistently **Buy High, Sell Low**, incurring positive cost ($RC > 0$).

**2. Capital Gain View ($\int h dS$)** - *Theoretical View*
Focuses on the trading gains/losses from holding the stock.
$$
RC = h_T S_T - h_0 S_0 - \int_0^T h_t dS_t
$$
*   $\int h dS$: The cumulative profit from holding $h_t$ shares while price changes by $dS_t$.
*   Since Payoff ($V_T$) is replicated by Initial Wealth ($V_0$) + Trading Gains ($\int h dS$):
    $$
    V_T = V_0 + \int_0^T \Delta_t dS_t \quad (\text{if } r=0)
    $$
    This proves PnL is zero.

### 2. Hedging Simulation Table (Symbolic)
**Scenario**: Short 1 Call Option ($V$), Hedged with Stock ($S$).
**Objective**: Delta Neutral ($\Delta_{portfolio} = 0 \implies h = \Delta_{call}$).

| Time ($t$) | Stock ($S_t$) | Option ($V_t$) | Delta ($\Delta_t$) | Stock Pos ($h_t$) | Trade ($dh_t$) | Trade Cost ($S_t dh_t$) | Cash Balance ($B_t$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | $S_0$ | $V_0$ | $\Delta_0$ | $h_0 = \Delta_0$ | $+\Delta_0$ | $S_0 \Delta_0$ | $B_0 = V_0 - S_0 \Delta_0$ |
| **1** | $S_1$ | $V_1$ | $\Delta_1$ | $h_1 = \Delta_1$ | $\Delta_1 - \Delta_0$ | $S_1 (h_1 - h_0)$ | $B_1 = B_0 - S_1 dh_1$ |
| **2** | $S_2$ | $V_2$ | $\Delta_2$ | $h_2 = \Delta_2$ | $\Delta_2 - \Delta_1$ | $S_2 (h_2 - h_1)$ | $B_2 = B_1 - S_2 dh_2$ |
| ... | ... | ... | ... | ... | ... | ... | ... |
| **k** | $S_k$ | $V_k$ | $\Delta_k$ | $h_k$ | $h_k - h_{k-1}$ | $S_k dh_k$ | $B_k = B_{k-1} - S_k dh_k$ |
| ... | ... | ... | ... | ... | ... | ... | ... |
| **T** | $S_T$ | $V_T$ | $\Delta_T$ | $h_T$ | | | $B_T = B_{T-1} - S_T (h_T - h_{T-1})$ |

**Final Portfolio Value**:
$$
\Pi_T = \underbrace{B_T}_{\text{Cash}} + \underbrace{h_T S_T}_{\text{Stock liquidation}} - \underbrace{V_T}_{\text{Option Payoff}}
$$

If hedging is continuous and parameters match ($r=0$ case for simplicity):
$$
\Pi_T \approx 0
$$
This demonstrates that the initial premium $V_0$ plus trading gains/losses exactly covers the final payoff $V_T$.

### 3. Impact of Volatility
The PnL of a delta-hedged strategy depends on the difference between Realized Volatility ($\sigma_{real}$) and Implied Volatility ($\sigma_{imp}$).

$$
\text{PnL} \approx \frac{1}{2} S^2 \Gamma (\sigma_{imp}^2 - \sigma_{real}^2) dt
$$

*   **Short Gamma** (Short Option): You lose if market is more volatile than expected ($\sigma_{real} > \sigma_{imp}$). "Buy High Sell Low" costs overwhelm the Premium.
*   **Long Gamma** (Long Option): You gain if market is more volatile than expected. You "Buy Low Sell High" during rebalancing.

---
**References**
*   John C. Hull, "Options, Futures, and Other Derivatives"
*   Steven E. Shreve, "Stochastic Calculus for Finance II"
