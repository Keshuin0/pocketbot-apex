"""Smart Alias Mapper & Universal Asset Class Classifier for PocketBot Apex.

Handles bidirectional translation between canonical tickers, industry symbols,
and broker-specific identifiers across MetaTrader 5 and Spotware cTrader (e.g.
Fusion Markets, Pepperstone, IC Markets).
"""

from __future__ import annotations

import re

from apex.feeds.models import AssetClass

# ---------------------------------------------------------------------------
# Canonical <-> Broker Alias Dictionaries
# ---------------------------------------------------------------------------

INDEX_ALIASES: dict[str, str] = {
    # US Indices
    "SPX": "US500",
    "SPX500": "US500",
    "S&P500": "US500",
    "US500": "US500",
    "NAS100": "NAS100",
    "USTEC": "NAS100",
    "NQ": "NAS100",
    "NASDAQ": "NAS100",
    "NDX": "NAS100",
    "US30": "US30",
    "DJ30": "US30",
    "YM": "US30",
    "DOW": "US30",
    "DJI": "US30",
    "US2000": "US2000",
    "RUSSELL": "US2000",
    "RTY": "US2000",
    "USDX": "USDX",
    "DXY": "USDX",
    # European & Asian Indices
    "GER40": "GER40",
    "DAX": "GER40",
    "DAX40": "GER40",
    "DE40": "GER40",
    "UK100": "UK100",
    "FTSE": "UK100",
    "FTSE100": "UK100",
    "JPN225": "JPN225",
    "NIKKEI": "JPN225",
    "N225": "JPN225",
    "AUS200": "AUS200",
    "ASX200": "AUS200",
    "EUSTX50": "EUSTX50",
    "STOXX50": "EUSTX50",
    "FRA40": "FRA40",
    "CAC40": "FRA40",
    "E35": "E35",
    "IBEX35": "E35",
    "CA60": "CA60",
    "TSX60": "CA60",
    "HK50": "HK50",
    "HSI": "HK50",
    "NETH25": "NETH25",
    "AEX": "NETH25",
    "CHINAH": "CHINAH",
}

ENERGY_ALIASES: dict[str, str] = {
    "XTIUSD": "XTIUSD",
    "USOUSD": "XTIUSD",
    "WTI": "XTIUSD",
    "CRUDE": "XTIUSD",
    "OIL": "XTIUSD",
    "CL": "XTIUSD",
    "XBRUSD": "XBRUSD",
    "UKOUSD": "XBRUSD",
    "BRENT": "XBRUSD",
    "XNGUSD": "XNGUSD",
    "NATGAS": "XNGUSD",
    "NG": "XNGUSD",
}

METAL_ALIASES: dict[str, str] = {
    "XAUUSD": "XAUUSD",
    "GOLD": "XAUUSD",
    "XAU": "XAUUSD",
    "XAUEUR": "XAUEUR",
    "XAUAUD": "XAUAUD",
    "XAGUSD": "XAGUSD",
    "SILVER": "XAGUSD",
    "XAG": "XAGUSD",
    "XAGEUR": "XAGEUR",
    "XPTUSD": "XPTUSD",
    "PLATINUM": "XPTUSD",
    "XPT": "XPTUSD",
    "XPDUSD": "XPDUSD",
    "PALLADIUM": "XPDUSD",
    "XPD": "XPDUSD",
    "XALUSD": "XALUSD",
    "ALUMINUM": "XALUSD",
    "XCUUSD": "XCUUSD",
    "COPPER": "XCUUSD",
    "XZNUSD": "XZNUSD",
    "ZINC": "XZNUSD",
    "XPBUSD": "XPBUSD",
    "LEAD": "XPBUSD",
    "XNIUSD": "XNIUSD",
    "NICKEL": "XNIUSD",
}

COMMODITY_ALIASES: dict[str, str] = {
    "COFARA": "COFARA",
    "COFFEE": "COFARA",
    "COFROB": "COFROB",
    "CORN": "CORN",
    "COTTON": "COTTON",
    "OJ": "OJ",
    "ORANGEJUICE": "OJ",
    "SOYBEAN": "SOYBEAN",
    "SUGAR": "SUGAR",
    "WHEAT": "WHEAT",
    "COCOA": "COCOA",
}

# Standard Ticker <-> Fusion Markets MT5 Equities Name
EQUITY_TICKER_TO_FUSION: dict[str, str] = {
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "META": "Meta",
    "GOOGL": "Alphabet-A",
    "GOOG": "Alphabet-C",
    "AMD": "AMD",
    "NFLX": "Netflix",
    "COIN": "Coinbase",
    "DIS": "WaltDisney",
    "JPM": "JPMorganChase",
    "BAC": "BankofAmericaCorp",
    "BA": "Boeing",
    "INTC": "Intel",
    "CSCO": "Cisco",
    "ORCL": "Oracle",
    "CRM": "Salesforce",
    "UBER": "Uber",
    "SHOP": "Shopify",
    "PLTR": "Palantir",
    "BABA": "AlibabaGroup",
    "TSM": "TSMC",
    "NKE": "Nike",
    "SBUX": "Starbucks",
    "WMT": "Walmart",
    "KO": "Coca-Cola",
    "PEP": "Pepsi",
    "MCD": "McDonalds",
    "V": "Visa",
    "MA": "Mastercard",
    "PYPL": "PayPal",
    "SQ": "BlockInc",
    "HOOD": "RobinhoodMarkets",
    "SNOW": "Snowflake",
    "ABNB": "Airbnb",
    "SPOT": "Spotify",
    "PFE": "Pfizer",
    "MRK": "Merck",
    "JNJ": "Johnson&Johnson",
    "XOM": "ExxonMobil",
    "CVX": "Chevron",
    "GE": "GeneralElectric",
    "GM": "GeneralMotors",
    "F": "Ford",
    "IBM": "IBM",
    "COST": "CostcoWholesale",
    "HD": "HomeDepot",
    "T": "AT&T",
    "VZ": "Verizon",
    "QCOM": "Qualcomm",
    "AVGO": "Broadcom",
    "TXN": "TexasInstruments",
    "MU": "MicronTechnology",
    "AMAT": "AppliedMaterials",
    "ADBE": "Adobe",
    "NOW": "ServiceNow",
    "INTU": "Intuit",
    "BKNG": "Booking",
    "CAT": "Caterpillar",
    "GS": "GoldmanSachs",
    "MS": "MorganStanley",
    "C": "Citigroup",
    "WFC": "WellsFargo",
    "BLK": "BlackRock",
    "BRK.B": "Berkshire",
    "BRKB": "Berkshire",
    "UNH": "UnitedHealth",
    "UPS": "UnitedParcelService",
    "RTX": "Raytheon",
    "HON": "Honeywell",
    "LMT": "LockheedMartin",
    "BMY": "BristolMyersSquibb",
    "ABT": "AbbottLaboratories",
    "MDT": "Medtronic",
    "CVS": "CVSHealth",
    "TMO": "ThermoFisherScientific",
    "GILD": "GileadSciences",
    "AMGN": "Amgen",
    "ISRG": "IntuitiveSurgical",
    "DHR": "Danaher",
    "NEE": "NextEraEnergy",
    "PM": "PhilipMorrisInternational",
    "MO": "Altria",
    "MDLZ": "Mondelez",
    "CL": "ColgatePalmolive",
    "PG": "Procter&Gamble",
    "LIN": "Linde",
    "APD": "AirProducts",
    "ECL": "Ecolab",
    "SHW": "SherwinWilliams",
    "FCX": "FreeportMcMoRan",
    "NEM": "Newmont",
    "NUE": "Nucor",
    "DD": "DuPont",
    "DOW": "Dow",
    "UNP": "UnionPacific",
    "FDX": "FedEx",
    "NSC": "NorfolkSouthern",
    "CSX": "CSXCorp",
    "DE": "Deere&Co",
    "EMR": "EmersonElectric",
    "ETN": "EatonCorp",
    "ITW": "IllinoisToolWorks",
    "PH": "ParkerHannifin",
    "ROK": "RockwellAutomation",
    "TT": "TraneTechnologies",
    "WM": "WasteManagement",
    "RSG": "RepublicServices",
    "CTAS": "Cintas",
    "PCAR": "PACCAR",
    "FAST": "Fastenal",
    "ODFL": "OldDominion",
    "EXPD": "Expeditors",
    "CHRW": "CHRobinson",
    "JBHT": "JBHunt",
    "KNX": "KnightSwift",
    "LUV": "SouthwestAirlines",
    "DAL": "DeltaAirLines",
    "UAL": "UnitedAirlines",
    "AAL": "AmericanAirlines",
    "MAR": "Marriott",
    "HLT": "Hilton",
    "YUM": "YumBrands",
    "CMG": "Chipotle",
    "DRI": "DardenRestaurants",
    "EBAY": "eBay",
    "ETSY": "Etsy",
    "ROKU": "Roku",
    "TWLO": "Twilio",
    "DOCU": "DocuSign",
    "PINS": "Pinterest",
    "SNAP": "Snapchat",
    "LYFT": "Lyft",
    "LCID": "LucidGroup",
    "QS": "QuantumScape",
    "RIVN": "Rivian",
    "BYND": "BeyondMeat",
    "EA": "ElectronicArts",
    "TTWO": "TakeTwo",
    "WBD": "WarnerBrosDiscovery",
    "PARA": "Paramount",
    "FOX": "FoxCorp",
    "FOXA": "FoxCorp-A",
    "CMCSA": "Comcast",
    "CHTR": "Charter",
    "TMUS": "TMobile",
    "BIDU": "Baidu",
    "JD": "JDcom",
    "PDD": "PDDHoldings",
    "NTES": "NetEase",
    "TCEHY": "Tencent",
    "NIO": "NIO",
    "XPEV": "XPeng",
    "LI": "LiAuto",
    "TM": "Toyota",
    "HMC": "Honda",
    "SONY": "Sony",
    "ASML": "ASML",
    "SAP": "SAP",
    "AZN": "AstraZeneca",
    "NVO": "NovoNordisk",
    "SNY": "Sanofi",
    "GSK": "GSK",
    "BTI": "BritishAmericanTobacco",
    "DEO": "Diageo",
    "BUD": "AnheuserBusch",
    "RELX": "RELX",
    "UL": "Unilever",
    "BP": "BP",
    "SHEL": "Shell",
    "TTE": "TotalEnergies",
    "EQNR": "Equinor",
    "ENB": "Enbridge",
    "CNQ": "CanadianNaturalResources",
    "SU": "Suncor",
    "TRP": "TCEnergy",
    "BMO": "BankofMontreal",
    "BNS": "BankofNovaScotia",
    "RY": "RoyalBankofCanada",
    "TD": "TorontoDominionBank",
    "BBD": "BancoBradesco",
    "ITUB": "ItauUnibanco",
    "VALE": "Vale",
    "PBR": "Petrobras",
    "PBR.A": "Petrobras-Pref",
    "HDB": "HDFCBank",
    "IBN": "ICICIBank",
    "INFY": "Infosys",
    "WIT": "Wipro",
}

# Reverse lookup: Fusion MT5 Name -> Standard Ticker
FUSION_TO_EQUITY_TICKER: dict[str, str] = {
    v.upper(): k for k, v in EQUITY_TICKER_TO_FUSION.items()
}
FUSION_TO_EQUITY_TICKER["JPMORGAN"] = "JPM"

CRYPTO_BASES: set[str] = {
    "BTC", "ETH", "SOL", "LTC", "DOGE", "ADA", "XRP", "DOT",
    "AVAX", "LINK", "MATIC", "BCH", "BNB", "UNI", "ATOM",
}

FOREX_CURRENCIES: set[str] = {
    "USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF",
    "NOK", "SEK", "DKK", "SGD", "HKD", "PLN", "HUF", "CZK",
    "ZAR", "TRY", "MXN", "CNH", "THB", "ILS",
}


class SmartAliasMapper:
    """Bidirectional alias resolver and asset class classifier."""

    def __init__(self) -> None:
        self._custom_to_canonical: dict[str, str] = {}
        self._custom_to_mt5: dict[str, str] = {}
        self._custom_to_ctrader: dict[str, str] = {}
        self._discovered_symbols: set[str] = set()

    def register_symbol(
        self,
        broker_symbol: str,
        canonical_name: str | None = None,
        broker: str = "mt5",
    ) -> None:
        """Register a broker symbol into dynamic cache."""
        norm_b = broker_symbol.strip()
        canon = canonical_name.strip().upper() if canonical_name else self.canonicalize(norm_b)
        self._discovered_symbols.add(norm_b)
        self._custom_to_canonical[norm_b.upper()] = canon

        if broker.lower() == "mt5":
            self._custom_to_mt5[canon] = norm_b
        elif broker.lower() == "ctrader":
            self._custom_to_ctrader[canon] = norm_b

    def canonicalize(self, symbol: str) -> str:
        """Resolve any ticker/symbol to its canonical standard representation."""
        clean = self.strip_broker_suffixes(symbol.strip())
        upper_clean = clean.upper()

        if upper_clean in self._custom_to_canonical:
            return self._custom_to_canonical[upper_clean]

        # 1. Check Equities
        if upper_clean in FUSION_TO_EQUITY_TICKER:
            return FUSION_TO_EQUITY_TICKER[upper_clean]
        if upper_clean in EQUITY_TICKER_TO_FUSION:
            return upper_clean

        # 2. Check Indices
        if upper_clean in INDEX_ALIASES:
            return INDEX_ALIASES[upper_clean]

        # 3. Check Energy
        if upper_clean in ENERGY_ALIASES:
            return ENERGY_ALIASES[upper_clean]

        # 4. Check Metals
        if upper_clean in METAL_ALIASES:
            return METAL_ALIASES[upper_clean]

        # 5. Check Commodities
        if upper_clean in COMMODITY_ALIASES:
            return COMMODITY_ALIASES[upper_clean]

        return upper_clean

    def resolve_to_broker(self, symbol: str, broker: str = "mt5") -> str:
        """Resolve a canonical or alternate symbol to a broker's exact symbol name."""
        clean = symbol.strip()
        upper_clean = clean.upper()

        if broker.lower() == "mt5":
            if upper_clean in self._custom_to_mt5:
                return self._custom_to_mt5[upper_clean]
            # If standard stock ticker like NVDA -> return NVIDIA for Fusion Markets MT5
            if upper_clean in EQUITY_TICKER_TO_FUSION:
                return EQUITY_TICKER_TO_FUSION[upper_clean]
            if upper_clean in INDEX_ALIASES:
                return INDEX_ALIASES[upper_clean]
            if upper_clean in ENERGY_ALIASES:
                return ENERGY_ALIASES[upper_clean]
            if upper_clean in METAL_ALIASES:
                return METAL_ALIASES[upper_clean]
            if upper_clean in COMMODITY_ALIASES:
                return COMMODITY_ALIASES[upper_clean]
            return clean

        if broker.lower() == "ctrader":
            if upper_clean in self._custom_to_ctrader:
                return self._custom_to_ctrader[upper_clean]
            # Standard cTrader names commonly use canonical symbols
            if upper_clean in INDEX_ALIASES:
                return INDEX_ALIASES[upper_clean]
            if upper_clean in ENERGY_ALIASES:
                return ENERGY_ALIASES[upper_clean]
            if upper_clean in METAL_ALIASES:
                return METAL_ALIASES[upper_clean]
            return clean

        return clean

    @staticmethod
    def strip_broker_suffixes(symbol: str) -> str:
        """Strip broker-specific suffixes such as .OQ, .NYSE, .cash, _i, .pro, m, etc."""
        s = symbol.strip()
        # Common suffixes
        s = re.sub(r"\.(OQ|NYSE|NASDAQ|cash|pro|raw|ecn|std)$", "", s, flags=re.IGNORECASE)
        # Trailing 'm' or '_i' for micro/institution accounts when symbol has 6+ chars
        if len(s) == 7 and s.endswith("m") and s[:6].isalpha():
            s = s[:6]
        elif s.endswith("_i"):
            s = s[:-2]
        return s

    @classmethod
    def classify_asset_class(
        cls,
        symbol: str,
        path: str = "",
        description: str = "",
    ) -> AssetClass:
        """Classify a symbol into an exact AssetClass using path, description, and ticker patterns."""
        p_low = path.lower() if path else ""
        d_low = description.lower() if description else ""
        norm_sym = symbol.strip().upper()
        clean_sym = cls.strip_broker_suffixes(norm_sym)

        # 1. Path-based classification (authoritative when broker provides MT5 hierarchy)
        if "forex" in p_low:
            return AssetClass.FOREX
        if "crypto" in p_low:
            return AssetClass.CRYPTO
        if "indices" in p_low or "index" in p_low:
            return AssetClass.INDICES
        if "energy" in p_low:
            return AssetClass.ENERGY
        if "soft commodit" in p_low or "agricultural" in p_low:
            return AssetClass.COMMODITIES
        if "equit" in p_low or "shares" in p_low or "stock" in p_low:
            return AssetClass.US_EQUITIES
        if "commodit" in p_low:
            # Note: in Fusion Markets MT5, 'Commodities' contains Metals (XAU, XAG, XPT, XPD, XCU)
            if any(clean_sym.startswith(prefix) for prefix in ("XAU", "XAG", "XPT", "XPD", "XAL", "XCU", "XZN", "XPB", "XNI")):
                return AssetClass.METALS
            return AssetClass.COMMODITIES
        if "metal" in p_low:
            return AssetClass.METALS

        # 2. Description-based clues
        if "index" in d_low or "indices" in d_low:
            return AssetClass.INDICES
        if "crypto" in d_low or "bitcoin" in d_low or "ethereum" in d_low:
            return AssetClass.CRYPTO
        if any(w in d_low for w in ("gold", "silver", "platinum", "palladium", "copper")):
            return AssetClass.METALS
        if any(w in d_low for w in ("crude", "brent", "gas", "oil")):
            return AssetClass.ENERGY
        if any(w in d_low for w in ("coffee", "corn", "cotton", "wheat", "sugar", "soybean", "cocoa")):
            return AssetClass.COMMODITIES

        # 3. Symbol pattern matching
        # Metals
        if any(clean_sym.startswith(prefix) for prefix in ("XAU", "XAG", "XPT", "XPD", "XAL", "XCU", "XZN", "XPB", "XNI")):
            return AssetClass.METALS
        if clean_sym in METAL_ALIASES:
            return AssetClass.METALS

        # Energy
        if any(clean_sym.startswith(prefix) for prefix in ("XTI", "XBR", "XNG")) or clean_sym in ENERGY_ALIASES:
            return AssetClass.ENERGY

        # Crypto
        for c in CRYPTO_BASES:
            if clean_sym.startswith(c) and (clean_sym.endswith("USD") or clean_sym.endswith("EUR") or clean_sym.endswith("USDT")):
                return AssetClass.CRYPTO

        # Indices
        if clean_sym in INDEX_ALIASES or clean_sym in {"US500", "NAS100", "US30", "GER40", "UK100", "JPN225", "AUS200", "EUSTX50", "FRA40", "E35", "CA60", "HK50", "NETH25", "US2000", "USDX"}:
            return AssetClass.INDICES

        # Commodities
        if clean_sym in COMMODITY_ALIASES:
            return AssetClass.COMMODITIES

        # US Equities
        if clean_sym in FUSION_TO_EQUITY_TICKER or clean_sym in EQUITY_TICKER_TO_FUSION:
            return AssetClass.US_EQUITIES

        # Forex: 6-letter currency pairs or exotics
        if len(clean_sym) == 6:
            base, quote = clean_sym[:3], clean_sym[3:]
            if base in FOREX_CURRENCIES and quote in FOREX_CURRENCIES:
                return AssetClass.FOREX

        return AssetClass.OTHER
