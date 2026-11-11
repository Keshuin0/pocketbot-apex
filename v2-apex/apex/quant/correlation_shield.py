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

from apex.feeds.models import Side

CANONICAL_FACTORS: tuple[str, ...] = (
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "Gold",
    "US_Tech_Beta",
    "Crypto_Beta",
)

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
    "US500": {"US_Tech_Beta": +1.0, "USD": -0.5},
    "SPX": {"US_Tech_Beta": +1.0, "USD": -0.5},
    "SPX500": {"US_Tech_Beta": +1.0, "USD": -0.5},
    "NAS100": {"US_Tech_Beta": +1.0, "USD": -0.5},
    "NQ": {"US_Tech_Beta": +1.0, "USD": -0.5},
    "USTEC": {"US_Tech_Beta": +1.0, "USD": -0.5},
    "TSLA": {"US_Tech_Beta": +1.0},
    "NVDA": {"US_Tech_Beta": +1.0},
    "AAPL": {"US_Tech_Beta": +1.0},
    "MSFT": {"US_Tech_Beta": +1.0},
    "AMD": {"US_Tech_Beta": +1.0},
    "META": {"US_Tech_Beta": +1.0},
    "AMZN": {"US_Tech_Beta": +1.0},
    "GOOGL": {"US_Tech_Beta": +1.0},
}


def decompose_symbol_factors(symbol: str, side: Side | str) -> dict[str, float]:
    """Decompose a symbol + side into signed underlying factor weights.

    Examples:
      - ``decompose_symbol_factors("EURUSD", Side.BUY)`` -> ``{"EUR": +1.0, "USD": -1.0}``
      - ``decompose_symbol_factors("USDJPY", Side.SELL)`` -> ``{"USD": -1.0, "JPY": +1.0}``
      - ``decompose_symbol_factors("XAUUSD", Side.BUY)`` -> ``{"Gold": +1.0, "USD": -1.0}``
    """
    clean = symbol.upper().strip().lstrip("#")
    # Strip broker suffixes such as EURUSDm, EURUSD.pro, EURUSD_i
    base_sym = clean.split(".")[0].split("_")[0]
    if len(base_sym) == 7 and base_sym.endswith("M") and base_sym[:6] in _SYMBOL_LONG_FACTOR_MAP:
        base_sym = base_sym[:6]

    norm_side = Side(side.upper()) if isinstance(side, str) else side
    sign = +1.0 if norm_side == Side.BUY else -1.0

    if base_sym in _SYMBOL_LONG_FACTOR_MAP:
        return {factor: sign * weight for factor, weight in _SYMBOL_LONG_FACTOR_MAP[base_sym].items()}

    # Generic 6-character FX pair decomposition (e.g., AUDJPY -> +AUD, -JPY)
    if len(base_sym) == 6 and base_sym.isalpha():
        base_ccy = base_sym[:3]
        quote_ccy = base_sym[3:]
        if base_ccy == "XAU":
            base_ccy = "Gold"
        elif base_ccy in ("BTC", "ETH", "SOL"):
            base_ccy = "Crypto_Beta"
        return {base_ccy: +1.0 * sign, quote_ccy: -1.0 * sign}

    # Default single-stock equity -> US_Tech_Beta
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


class CorrelationShield:
    """Currency & Factor Exposure Guard (`apex/quant/correlation_shield.py`).

    Respects `enabled` (`config["filters"]["correlation_cap"]["enabled"]`) for walk-forward A/B testing.
    Caps net single-currency / factor exposure at `max_factor_risk_pct` (default `1.0%` total equity at risk).
    """

    def __init__(
        self,
        enabled: bool = True,
        max_factor_risk_pct: float = 1.0,
        account_equity_usd: float = 10_000.0,
    ) -> None:
        if max_factor_risk_pct <= 0.0:
            raise ValueError(f"max_factor_risk_pct must be > 0, got {max_factor_risk_pct}")
        self.enabled: bool = bool(enabled)
        self.max_factor_risk_pct: float = float(max_factor_risk_pct)
        self.account_equity_usd: float = float(account_equity_usd)

    def _resolve_risk_pct(self, pos: PositionRiskSpec) -> float:
        if pos.risk_pct > 0.0:
            return float(pos.risk_pct)
        if pos.risk_usd > 0.0 and self.account_equity_usd > 0.0:
            return float((pos.risk_usd / self.account_equity_usd) * 100.0)
        return 0.0

    def compute_portfolio_factor_exposures(
        self,
        positions: Sequence[PositionRiskSpec],
    ) -> dict[str, float]:
        """Compute net signed factor exposures (in % of total equity at risk) across ``positions``."""
        exposures: dict[str, float] = {f: 0.0 for f in CANONICAL_FACTORS}
        for pos in positions:
            r_pct = self._resolve_risk_pct(pos)
            weights = decompose_symbol_factors(pos.symbol, pos.side)
            for factor, w in weights.items():
                exposures[factor] = round(exposures.get(factor, 0.0) + r_pct * w, 6)
        return exposures

    def evaluate_proposed_trade(
        self,
        proposed: PositionRiskSpec,
        open_positions: Sequence[PositionRiskSpec],
    ) -> FactorExposureVerdict:
        """Check whether adding ``proposed`` to ``open_positions`` breaches the 1.0% factor exposure cap."""
        before = self.compute_portfolio_factor_exposures(open_positions)
        after = self.compute_portfolio_factor_exposures([*open_positions, proposed])
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
