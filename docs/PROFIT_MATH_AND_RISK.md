# PocketBot Apex — Profit Mathematics, Quantitative Expectancy & Risk Model

## 1. The Core 2R Mathematical Expectancy Engine

PocketBot Apex is built entirely on the principle of asymmetric risk-to-reward profiling. Every trade setup is engineered with a strict **2:1 Reward-to-Risk (2R)** ratio:
- **Risk ($1R$):** $0.50\%$ of account equity (e.g., $\$50.00$ on a standard $\$10,000$ account).
- **Reward ($2R$):** $1.00\%$ of account equity (e.g., $\$100.00$ on a standard $\$10,000$ account).

### 1.1 Expected Value Formula
The mathematical expectancy $E$ per trade in units of $R$ is given by:

$$E = (W \cdot 2.0) - ((1 - W) \cdot 1.0) = 3W - 1$$

where $W$ is the empirical win rate.

### 1.2 Profit Expectancy Across Win Rates

| Win Rate ($W$) | Win Multiplier | Loss Multiplier | Expectancy ($E$) per Trade | 100-Trade Return ($0.5\%$ Risk) |
| :---: | :---: | :---: | :---: | :---: |
| **33.3%** | $0.333 \times 2.0 = 0.667$ | $0.667 \times 1.0 = 0.667$ | **$0.00R$ (Break-Even)** | $\$0.00$ ($0.0\%$) |
| **38.0%** | $0.380 \times 2.0 = 0.760$ | $0.620 \times 1.0 = 0.620$ | **$+0.14R$ (Profitable)** | $+\$700.00$ ($+7.0\%$) |
| **42.0%** | $0.420 \times 2.0 = 0.840$ | $0.580 \times 1.0 = 0.580$ | **$+0.26R$** | $+\$1,300.00$ ($+13.0\%$) |
| **45.0%** | $0.450 \times 2.0 = 0.900$ | $0.550 \times 1.0 = 0.550$ | **$+0.35R$ (Baseline)** | $+\$1,750.00$ ($+17.5\%$) |
| **50.0%** | $0.500 \times 2.0 = 1.000$ | $0.500 \times 1.0 = 0.500$ | **$+0.50R$** | $+\$2,500.00$ ($+25.0\%$) |
| **55.0%** | $0.550 \times 2.0 = 1.100$ | $0.450 \times 1.0 = 0.450$ | **$+0.65R$** | $+\$3,250.00$ ($+32.5\%$) |

> **Key Insight:** Even with an adverse market environment where the win rate drops to **$38\%$**, the system remains mathematically profitable due to the $2:1$ payoff asymmetry.

---

## 2. Friction Control: The 15% Cost Ceiling Rule

A major flaw in retail trading robots is ignoring broker friction (spread, commission, and slippage). If friction consumes too large a portion of the target reward, mathematical edge vanishes.

PocketBot Apex enforces the **$15\%$ Cost Ceiling Rule**:

$$\text{Friction Ratio} = \frac{\text{Spread Pips} + \text{Commission Pips} + \text{VWAP Slippage Pips}}{\text{Stop-Loss Distance Pips}} \le 0.15$$

### Example Calculation:
- Pair: `EURUSD`
- Entry: `1.10000`, Stop-Loss: `1.09900` ($10.0$ pips distance = $1R$).
- Quoted Spread: $0.6$ pips.
- Round-turn commission: $\$4.50/\text{lot}$ ($\approx 0.45$ pips).
- Estimated DOM VWAP Slippage: $0.15$ pips.
- Total Friction = $0.6 + 0.45 + 0.15 = 1.20$ pips.
- Friction Ratio = $1.20 / 10.0 = 12.0\% \le 15.0\%$ $\to$ **PASSED**.

If spread widens during economic news to $2.0$ pips, Friction Ratio jumps to $26\% > 15\%$, and the order router automatically rejects the setup.

---

## 3. Position Sizing & Fractional Kelly Criterion

Position sizing is computed dynamically based on the distance between Entry and Stop-Loss, contract specifications, and account equity:

$$\text{Lots} = \frac{\text{Account Equity} \times \text{Risk Percentage}}{\text{SL Distance in Pips} \times \text{Pip Value per Lot}}$$

### Fractional Kelly Optimization
The theoretical optimal Kelly fraction for a $2:1$ payoff with probability $W$:

$$f^* = \frac{b \cdot W - (1 - W)}{b} = \frac{2W - (1 - W)}{2} = \frac{3W - 1}{2}$$

For our baseline $W = 0.45$:

$$f^* = \frac{3(0.45) - 1}{2} = 0.175 \quad (17.5\%)$$

Because full Kelly generates unacceptable drawdown volatility for prop firm rules, PocketBot Apex dampens the allocation using a **$\frac{1}{35}\text{th}$ Fractional Kelly Shield**, enforcing a maximum risk ceiling of **$0.50\%$ per trade**.

---

## 4. Multi-Asset Portfolio Protection: Correlation Shield

Trading multiple pairs simultaneously can inadvertently multiply factor exposure (e.g., buying `EURUSD`, `GBPUSD`, and `AUDUSD` all at once constitutes a $3\times$ short USD bet).

### 4.1 Pearson & Kendall Tau Matrix
The `CorrelationShield` computes rolling cross-asset correlations on 60-period return vectors:

$$r_{xy} = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sqrt{\sum (x - \bar{x})^2 \sum (y - \bar{y})^2}}$$

$$\tau = \frac{(\text{concordant pairs}) - (\text{discordant pairs})}{\frac{1}{2} n (n - 1)}$$

### 4.2 Factor Exposure Limit
- Maximum Concurrent Trades across the entire portfolio: **2 positions**.
- If Symbol A and Symbol B exhibit correlation $|r| \ge 0.70$, simultaneous entry in the same directional factor is vetoed.
- Total portfolio factor exposure is hard-capped at **$1.0\%$ total equity at risk**.

---

## 5. Daily Drawdown Kill-Switch (-2.0%)

To protect accounts against intraday market shocks or black-swan events, the execution engine tracks high-water equity and daily cumulative P/L:

$$\text{Daily P/L} = \text{Closed P/L Today} + \text{Floating Unrealized P/L}$$

If $\text{Daily P/L} \le -2.0\%$ of starting daily equity:
1. The engine enters **KILL-SWITCH ACTIVE** state.
2. All pending orders are immediately cancelled.
3. Order routing is locked for the remainder of the 24-hour UTC session.
4. An emergency alert is dispatched to Telegram cockpit.

---

## 6. Time Machine v2: 10,000-Path Block-Bootstrap Simulation

Before approving any symbol for live execution, PocketBot Apex runs a 10,000-path Monte Carlo block-bootstrap simulation:
- Resamples historical returns with block-length preservation to model dependency structures.
- Inflicts a simulated $1.5\text{-pip}$ adverse execution slippage and standard commission on every simulated trade.
- Derives the empirical $95\%$ Value-at-Risk ($\text{VaR}_{95}$):

$$\text{VaR}_{95} = \text{Quantile}_{0.05}(\text{Simulated Path Returns})$$

If $\text{VaR}_{95}$ exceeds the pre-set threshold or maximum consecutive loss paths exceed 6, the symbol's risk tier is automatically reduced.
