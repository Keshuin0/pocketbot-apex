"""Currency & Factor Exposure Guard (`apex/quant/correlation_shield.py`).

Decomposes every proposed and open position into underlying institutional factor exposures:
  - ``USD``
  - ``EUR``
  - ``GBP``
  - ``JPY``
  - ``Gold``
  - ``US_Tech_Beta``
  - ``Crypto_Beta``
  (plus any minor G10 currency leg such as ``CHF``, ``AUD``, ``CAD``, ``NZD``)

Blocks hidden concentration traps (e.g., going Long ``EURUSD`` + Long ``GBPUSD`` + Long ``XAUUSD``
simultaneously = 3x Short ``USD``!). Caps net single-currency / factor exposure strictly at ``1.0%``
total equity at risk across all open trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from apex.feeds.alias_mapper import SmartAliasMapper
from apex.feeds.models import AssetClass, Side

CANONICAL_FACTORS: tuple[str, ...] = (
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "AUD",
    "CAD",
    "CHF",
    "NZD",
    "Gold",
    "Silver",
    "Metals_Beta",
    "Energy_Beta",
    "Commodities_Beta",
    "US_Tech_Beta",
    "US_Broad_Market_Beta",
    "Global_Equity_Beta",
    "Crypto_Beta",
)

TECH_STOCKS: set[str] = {
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "GOOG",
    "AMD", "NFLX", "COIN", "INTC", "CSCO", "ORCL", "CRM", "UBER",
    "SHOP", "PLTR", "TSM", "PYPL", "SQ", "HOOD", "SNOW", "ABNB",
    "SPOT", "QCOM", "AVGO", "TXN", "MU", "AMAT", "ADBE", "NOW",
    "INTU", "EBAY", "ETSY", "ROKU", "TWLO", "DOCU", "PINS", "SNAP",
    "LYFT", "BIDU", "JD", "PDD", "NTES", "BABA", "ASML", "NVIDIA",
    "TESLA", "APPLE", "MICROSOFT", "AMAZON", "ALPHABET-A", "ALPHABET-C",
}

# Base factor decomposition for a LONG (BUY) position of +1 unit of risk.
# A SHORT (SELL) position negates every factor weight.
_SYMBOL_LONG_FACTOR_MAP: dict[str, dict[str, float]] = {
    "EURUSD": {"EUR": +1.0, "USD": -1.0},
    "GBPUSD": {"GBP": +1.0, "USD": -1.0},
    "AUDUSD": {"AUD": +1.0, "USD": -1.0},
    "NZDUSD": {"NZD": +1.0, "USD": -1.0},
    "USDJPY": {"USD": +1.0, "JPY": -1.0},
    "USDCHF": {"USD": +1.0, "CHF": -1.0},
    "USDCAD": {"USD": +1.0, "CAD": -1.0},
    "EURGBP": {"EUR": +1.0, "GBP": -1.0},
    "EURJPY": {"EUR": +1.0, "JPY": -1.0},
    "GBPJPY": {"GBP": +1.0, "JPY": -1.0},
    "XAUUSD": {"Gold": +1.0, "USD": -1.0},
    "GOLD": {"Gold": +1.0, "USD": -1.0},
    "BTCUSD": {"Crypto_Beta": +1.0, "USD": -1.0},
    "ETHUSD": {"Crypto_Beta": +1.0, "USD": -1.0},
    "SOLUSD": {"Crypto_Beta": +1.0, "USD": -1.0},
    "US500": {"US_Broad_Market_Beta": +1.0, "USD": -0.5},
    "SPX": {"US_Broad_Market_Beta": +1.0, "USD": -0.5},
    "SPX500": {"US_Broad_Market_Beta": +1.0, "USD": -0.5},
    "NAS100": {"US_Tech_Beta": +1.0, "USD": -0.5},
    "NQ": {"US_Tech_Beta": +1.0, "USD": -0.5},
    "USTEC": {"US_Tech_Beta": +1.0, "USD": -0.5},
    "TSLA": {"US_Tech_Beta": +1.0, "Stock_TSLA": +1.0},
    "NVDA": {"US_Tech_Beta": +1.0, "Stock_NVDA": +1.0},
    "AAPL": {"US_Tech_Beta": +1.0, "Stock_AAPL": +1.0},
    "MSFT": {"US_Tech_Beta": +1.0, "Stock_MSFT": +1.0},
    "AMD": {"US_Tech_Beta": +1.0, "Stock_AMD": +1.0},
    "META": {"US_Tech_Beta": +1.0, "Stock_META": +1.0},
    "AMZN": {"US_Tech_Beta": +1.0, "Stock_AMZN": +1.0},
    "GOOGL": {"US_Tech_Beta": +1.0, "Stock_GOOGL": +1.0},
}


def decompose_symbol_factors(symbol: str, side: Side | str) -> dict[str, float]:
    """Decompose any of the 251 universe symbols into signed institutional factor weights.

    Handles:
      - All 86 Forex pairs across G10 and exotic currencies (e.g. EURTRY -> +EUR, -TRY).
      - Metals: Gold (XAUUSD/XAUEUR), Silver (XAGUSD/XAGEUR), and industrial metals.
      - Energy: WTI (XTIUSD), Brent (XBRUSD), Natural Gas (XNGUSD).
      - Commodities: Softs & Agriculture (COFARA, CORN, COTTON, OJ, SOYBEAN, etc.).
      - Crypto: BTC, ETH, SOL, LTC, DOGE, ADA, DOT, AVAX, LNK, XLM, etc. -> Crypto_Beta & USD.
      - Indices: US Tech (NAS100), Broad Market (US500, US30, US2000), and Global Indices (GER40, UK100, JPN225).
      - US Equities: All 109 single stocks mapped to US_Tech_Beta or US_Broad_Market_Beta + individual stock factor.
    """
    clean = symbol.upper().strip().lstrip("#")
    base_sym = clean.split(".")[0].split("_")[0]
    if len(base_sym) == 7 and base_sym.endswith("M") and base_sym[:6] in _SYMBOL_LONG_FACTOR_MAP:
        base_sym = base_sym[:6]

    norm_side = Side(side.upper()) if isinstance(side, str) else side
    sign = +1.0 if norm_side == Side.BUY else -1.0

    if base_sym in _SYMBOL_LONG_FACTOR_MAP:
        return {factor: sign * weight for factor, weight in _SYMBOL_LONG_FACTOR_MAP[base_sym].items()}

    mapper = SmartAliasMapper()
    canon = mapper.canonicalize(clean)
    asset_class = mapper.classify_asset_class(clean)

    # 1. Forex: Any 6-letter currency pair (G10 or exotics)
    if asset_class == AssetClass.FOREX and len(canon) == 6 and canon.isalpha():
        base_ccy = canon[:3]
        quote_ccy = canon[3:]
        return {base_ccy: +1.0 * sign, quote_ccy: -1.0 * sign}

    # 2. Metals
    if asset_class == AssetClass.METALS:
        if canon.startswith("XAU") or canon == "GOLD":
            quote = canon[3:] if len(canon) >= 6 else "USD"
            return {"Gold": +1.0 * sign, quote: -1.0 * sign}
        if canon.startswith("XAG") or canon == "SILVER":
            quote = canon[3:] if len(canon) >= 6 else "USD"
            return {"Silver": +1.0 * sign, quote: -1.0 * sign}
        quote = canon[3:] if len(canon) >= 6 else "USD"
        return {"Metals_Beta": +1.0 * sign, quote: -1.0 * sign}

    # 3. Energy
    if asset_class == AssetClass.ENERGY:
        return {"Energy_Beta": +1.0 * sign, "USD": -1.0 * sign}

    # 4. Commodities
    if asset_class == AssetClass.COMMODITIES:
        return {"Commodities_Beta": +1.0 * sign, "USD": -1.0 * sign}

    # 5. Crypto
    if asset_class == AssetClass.CRYPTO:
        quote = canon[-3:] if canon.endswith(("USD", "EUR")) else "USD"
        return {"Crypto_Beta": +1.0 * sign, quote: -1.0 * sign}

    # 6. Indices
    if asset_class == AssetClass.INDICES:
        if canon in ("NAS100", "USTEC", "NQ"):
            return {"US_Tech_Beta": +1.0 * sign, "USD": -0.5 * sign}
        if canon in ("US500", "US30", "US2000", "SPX"):
            return {"US_Broad_Market_Beta": +1.0 * sign, "USD": -0.5 * sign}
        ccy = "EUR" if canon in ("GER40", "FRA40", "EUSTX50", "E35") else ("GBP" if canon == "UK100" else ("JPY" if canon == "JPN225" else "USD"))
        return {"Global_Equity_Beta": +1.0 * sign, ccy: -0.5 * sign}

    # 7. US Equities (all 109 single-stock CFDs)
    if asset_class == AssetClass.US_EQUITIES:
        ticker = mapper.canonicalize(clean)
        is_tech = ticker in TECH_STOCKS or clean in TECH_STOCKS
        beta_factor = "US_Tech_Beta" if is_tech else "US_Broad_Market_Beta"
        return {beta_factor: +1.0 * sign, f"Stock_{ticker}": +1.0 * sign}

    # 8. Generic 6-letter fallback
    if len(base_sym) == 6 and base_sym.isalpha():
        return {base_sym[:3]: +1.0 * sign, base_sym[3:]: -1.0 * sign}

    return {"US_Tech_Beta": +1.0 * sign}


@dataclass(frozen=True, slots=True)
class PositionRiskSpec:
    """Open or proposed position specification with equity risk percentage (or USD risk)."""

    symbol: str
    side: Side
    risk_pct: float
    risk_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class FactorExposureVerdict:
    """Result of evaluating a proposed trade against the Portfolio Factor Exposure Guard."""

    symbol: str
    side: Side
    proposed_risk_pct: float
    allowed: bool
    max_factor_risk_pct_cap: float
    factor_exposures_before_pct: Mapping[str, float]
    factor_exposures_after_pct: Mapping[str, float]
    breached_factors: tuple[str, ...] = field(default_factory=tuple)
    reason: str = "ok"


def compute_ledoit_wolf_and_tail_correlation(
    returns_matrix: np.ndarray | None,
    tail_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (1) Ledoit-Wolf Shrinkage correlation matrix (rho_lw) and
    (2) Lower-Tail Crash Co-Movement matrix (rho_tail conditioned on <= -1.5 sigma stress bars).
    """
    if returns_matrix is None:
        eye = np.eye(1, dtype=np.float64)
        return eye, eye.copy()
    x = np.asarray(returns_matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 5 or x.shape[1] < 2:
        m = x.shape[1] if x.ndim == 2 else 1
        eye = np.eye(m, dtype=np.float64)
        return eye, eye.copy()

    t_len, m_cols = x.shape
    mean = np.mean(x, axis=0, keepdims=True)
    xc = x - mean
    sample_cov = (xc.T @ xc) / float(t_len)

    # Ledoit-Wolf analytical shrinkage toward scaled identity target
    mu = np.trace(sample_cov) / float(m_cols)
    target = mu * np.eye(m_cols, dtype=np.float64)
    gamma = float(np.sum((sample_cov - target) ** 2))
    if gamma <= 1e-18:
        shrunk_cov = sample_cov
    else:
        # Sum of squared Frobenius deviations of rank-1 outer products
        sq_norms = np.sum(xc**2, axis=1)
        phi = float(np.sum(sq_norms**2) / float(t_len) - np.sum(sample_cov**2)) / float(t_len)
        delta = max(0.0, min(1.0, phi / gamma))
        shrunk_cov = (1.0 - delta) * sample_cov + delta * target

    stds_lw = np.sqrt(np.maximum(np.diag(shrunk_cov), 1e-18))
    rho_lw = shrunk_cov / np.outer(stds_lw, stds_lw)
    rho_lw = np.clip(rho_lw, -1.0, 1.0)
    np.fill_diagonal(rho_lw, 1.0)

    # Lower-Tail Crash Co-Movement (<= -tail_sigma joint stress bars)
    stds = np.std(xc, axis=0)
    stds = np.where(stds > 1e-12, stds, 1e-12)
    z = xc / stds
    rho_tail = np.eye(m_cols, dtype=np.float64)
    for i in range(m_cols):
        for j in range(i + 1, m_cols):
            stress_mask = (z[:, i] <= -tail_sigma) | (z[:, j] <= -tail_sigma)
            if int(np.sum(stress_mask)) >= 4:
                zi = z[stress_mask, i]
                zj = z[stress_mask, j]
                zi_c = zi - np.mean(zi)
                zj_c = zj - np.mean(zj)
                denom = float(np.sqrt(np.sum(zi_c**2) * np.sum(zj_c**2)))
                val = float(np.sum(zi_c * zj_c) / denom) if denom > 1e-12 else 0.0
                val = max(-1.0, min(1.0, val))
                rho_tail[i, j] = val
                rho_tail[j, i] = val
            else:
                rho_tail[i, j] = rho_lw[i, j]
                rho_tail[j, i] = rho_lw[j, i]

    return rho_lw, rho_tail


class CorrelationShield:
    """Currency, Factor & Empirical Tail-Crash Exposure Guard (`apex/quant/correlation_shield.py`).

    Respects `enabled` (`config["filters"]["correlation_cap"]["enabled"]`) for walk-forward A/B testing.
    Caps net single-currency / factor exposure at `max_factor_risk_pct` (default `1.0%` total equity at risk),
    and blocks hidden crash-correlated trades where max(|rho_lw|, |rho_tail|) >= `max_empirical_corr` (0.72).
    """

    def __init__(
        self,
        enabled: bool = True,
        max_factor_risk_pct: float = 1.0,
        account_equity_usd: float = 10_000.0,
        max_empirical_corr: float = 0.72,
        tail_sigma: float = 1.5,
    ) -> None:
        if max_factor_risk_pct <= 0.0:
            raise ValueError(f"max_factor_risk_pct must be > 0, got {max_factor_risk_pct}")
        self.enabled: bool = bool(enabled)
        self.max_factor_risk_pct: float = float(max_factor_risk_pct)
        self.account_equity_usd: float = float(account_equity_usd)
        self.max_empirical_corr: float = float(max_empirical_corr)
        self.tail_sigma: float = float(tail_sigma)
        self._return_histories: dict[str, np.ndarray] = {}

    def register_return_series(self, symbol: str, returns: Sequence[float] | None) -> None:
        """Cache trailing return series for empirical Ledoit-Wolf & tail-crash correlation checks."""
        if returns is None:
            return
        arr = np.asarray(returns, dtype=np.float64)
        if arr.ndim == 1 and len(arr) >= 5:
            self._return_histories[symbol.upper().strip()] = arr

    def _resolve_risk_pct(self, pos: PositionRiskSpec) -> float:
        if pos.risk_pct > 0.0:
            return float(pos.risk_pct)
        if pos.risk_usd > 0.0 and self.account_equity_usd > 0.0:
            return float((pos.risk_usd / self.account_equity_usd) * 100.0)
        return 0.0

    def compute_portfolio_factor_exposures(
        self,
        positions: Sequence[PositionRiskSpec] | None,
    ) -> dict[str, float]:
        """Compute net signed factor exposures (in % of total equity at risk) across ``positions``."""
        exposures: dict[str, float] = {f: 0.0 for f in CANONICAL_FACTORS}
        for pos in positions or ():
            r_pct = self._resolve_risk_pct(pos)
            weights = decompose_symbol_factors(pos.symbol, pos.side)
            for factor, w in weights.items():
                exposures[factor] = round(exposures.get(factor, 0.0) + r_pct * w, 6)
        return exposures

    def evaluate_proposed_trade(
        self,
        proposed: PositionRiskSpec,
        open_positions: Sequence[PositionRiskSpec] | None,
        return_histories: Mapping[str, Sequence[float]] | None = None,
    ) -> FactorExposureVerdict:
        """Check whether adding ``proposed`` to ``open_positions`` breaches the 1.0% factor cap or 0.72 tail-crash correlation."""
        open_list = list(open_positions or ())
        before = self.compute_portfolio_factor_exposures(open_list)
        after = self.compute_portfolio_factor_exposures([*open_list, proposed])
        prop_risk_pct = self._resolve_risk_pct(proposed)

        proposed_factors = decompose_symbol_factors(proposed.symbol, proposed.side)
        breached: list[str] = []
        breach_details: list[str] = []

        for factor in proposed_factors:
            net_after = after.get(factor, 0.0)
            net_before = before.get(factor, 0.0)
            # Flag if absolute factor exposure after adding this trade exceeds max_factor_risk_pct
            # AND this trade increases the magnitude of that factor exposure
            if abs(net_after) > self.max_factor_risk_pct + 1e-6 and abs(net_after) > abs(net_before) + 1e-6:
                breached.append(factor)
                direction = "Long" if net_after > 0 else "Short"
                breach_details.append(
                    f"{factor} net exposure would reach {net_after:+.2f}% ({direction} {factor} "
                    f"{abs(net_after):.2f}% > {self.max_factor_risk_pct:.2f}% equity risk cap)"
                )

        # Empirical Ledoit-Wolf + Lower-Tail Crash Co-Movement check
        hist_map: dict[str, np.ndarray] = dict(self._return_histories)
        if return_histories:
            for k, v in return_histories.items():
                if v is None:
                    continue
                hist_map[k.upper().strip()] = np.asarray(v, dtype=np.float64)

        prop_sym_u = proposed.symbol.upper().strip()
        if prop_sym_u in hist_map and open_list:
            prop_ret = hist_map[prop_sym_u]
            sign_prop = 1.0 if proposed.side == Side.BUY else -1.0
            for op in open_list:
                op_sym_u = op.symbol.upper().strip()
                if op_sym_u not in hist_map or op_sym_u == prop_sym_u:
                    continue
                op_ret = hist_map[op_sym_u]
                n_common = min(len(prop_ret), len(op_ret))
                if n_common < 10:
                    continue
                mat = np.column_stack([prop_ret[-n_common:], op_ret[-n_common:]])
                rho_lw_mat, rho_tail_mat = compute_ledoit_wolf_and_tail_correlation(mat, tail_sigma=self.tail_sigma)
                rho_lw = float(rho_lw_mat[0, 1])
                rho_tail = float(rho_tail_mat[0, 1])
                sign_op = 1.0 if op.side == Side.BUY else -1.0
                # Effective directional correlation: positive means stacking same-direction risk
                dom_rho = rho_tail if abs(rho_tail) >= abs(rho_lw) else rho_lw
                eff_corr = sign_prop * sign_op * dom_rho
                if eff_corr >= self.max_empirical_corr:
                    tag = f"TAIL_CORR_{op_sym_u}"
                    breached.append(tag)
                    breach_details.append(
                        f"Empirical/Tail crash correlation {proposed.symbol}({proposed.side.value}) vs "
                        f"{op.symbol}({op.side.value}): rho_lw={rho_lw:+.2f}, rho_tail={rho_tail:+.2f} "
                        f">= {self.max_empirical_corr:.2f}"
                    )

        if not self.enabled:
            return FactorExposureVerdict(
                symbol=proposed.symbol.upper(),
                side=proposed.side,
                proposed_risk_pct=prop_risk_pct,
                allowed=True,
                max_factor_risk_pct_cap=self.max_factor_risk_pct,
                factor_exposures_before_pct=before,
                factor_exposures_after_pct=after,
                breached_factors=tuple(breached),
                reason="[correlation_cap_disabled] " + ("; ".join(breach_details) if breach_details else "ok"),
            )

        allowed = len(breached) == 0
        reason = (
            "CORRELATION_SHIELD_BLOCK: " + "; ".join(breach_details)
            if breach_details
            else f"ok (all factor exposures within {self.max_factor_risk_pct:.2f}% cap)"
        )

        return FactorExposureVerdict(
            symbol=proposed.symbol.upper(),
            side=proposed.side,
            proposed_risk_pct=prop_risk_pct,
            allowed=allowed,
            max_factor_risk_pct_cap=self.max_factor_risk_pct,
            factor_exposures_before_pct=before,
            factor_exposures_after_pct=after,
            breached_factors=tuple(breached),
            reason=reason,
        )
