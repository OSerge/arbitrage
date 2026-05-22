from __future__ import annotations

from datetime import UTC, datetime

SUMMARY_D00014 = {
    "buyingPower": 100000,
    "buyingPowerAtMorning": 100000,
    "buyingPowerByCurrency": [{"buyingPower": 100000, "currency": "RUB"}],
    "commission": 0,
    "correctedMargin": 0,
    "derivativesDebt": None,
    "initialMargin": 0,
    "portfolioEvaluation": 100000,
    "portfolioLiquidationValue": 100000,
    "profit": 0,
    "profitRate": 0,
    "riskBeforeForcePositionClosing": 100000,
}

POSITION_APTK = {
    "volume": 549.198,
    "currentVolume": 555.24,
    "symbol": "APTK",
    "brokerSymbol": "MOEX:APTK",
    "portfolio": "D00013",
    "exchange": "MOEX",
    "avgPrice": 9.1533,
    "qtyUnits": 60,
    "openUnits": 60,
    "lotSize": 10,
    "shortName": "Аптеки36и6",
    "qty": 6,
    "unrealisedPl": 7.962,
    "isCurrency": False,
    "existing": True,
}

ORDER_WORKING = {
    "id": "2006665023417483347",
    "symbol": "ALRS-3.24",
    "brokerSymbol": "MOEX:ALRS-3.24",
    "portfolio": "D39004",
    "exchange": "MOEX",
    "type": "limit",
    "side": "buy",
    "status": "working",
    "transTime": "2024-01-08T15:57:12.2911671Z",
    "updateTime": "2024-01-08T15:57:12.2911671Z",
    "qtyUnits": 1,
    "qtyBatch": 1,
    "qty": 1,
    "filledQtyUnits": 0,
    "filled": 0,
    "price": 6400.0,
    "existing": True,
}

TRADE_GAZP = {
    "id": "4421925358",
    "orderno": "42030831469",
    "symbol": "GAZP",
    "brokerSymbol": "MOEX:GAZP",
    "exchange": "MOEX",
    "date": "2023-12-29T12:35:06.0000000Z",
    "board": "TQBR",
    "qtyUnits": 10,
    "qtyBatch": 1,
    "qty": 1,
    "price": 170.21,
    "side": "buy",
    "existing": True,
    "commission": 1.36105,
}

MAPPING_CAPTURED_AT = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)
