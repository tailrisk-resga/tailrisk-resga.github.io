from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add ticker, company names, and industry labels to GitHub Pages data.")
    parser.add_argument("--predictions-root", default="docs/data/predictions")
    parser.add_argument("--usa-id-csv", required=True)
    parser.add_argument("--industry-codes", default="docs/data/Siccodes49.txt")
    parser.add_argument("--crsp-csv", default="data/metadata/crsp.csv")
    parser.add_argument("--compustat-csv", default="data/metadata/compustat.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_root = Path(args.predictions_root)
    stocks_path = predictions_root / "stocks.json"
    stocks = json.loads(stocks_path.read_text(encoding="utf-8"))
    stock_ids = {int(row["id"]) for row in stocks}

    # TODO: Confirm public redistribution rights for ticker/name mappings before final release.
    id_panel = load_id_panel(Path(args.usa_id_csv), stock_ids)
    industry_map = load_industry_codes(Path(args.industry_codes))
    metadata = build_metadata(id_panel, Path(args.crsp_csv), Path(args.compustat_csv), industry_map)

    update_stock_index(stocks_path, stocks, metadata)
    update_prediction_files(predictions_root, metadata)
    update_stock_series(predictions_root / "stocks", metadata)

    matched_names = sum(1 for row in metadata.values() if row.get("ticker") or row.get("name"))
    matched_industries = sum(1 for row in metadata.values() if row.get("industry"))
    print(f"Updated {len(stock_ids):,} stocks.")
    print(f"Matched ticker/name for {matched_names:,} stocks.")
    print(f"Matched industry for {matched_industries:,} stocks.")


def load_id_panel(path: Path, stock_ids: set[int]) -> pd.DataFrame:
    usecols = ["id", "eom", "gvkey", "permno", "source_crsp", "ff49"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype={"gvkey": "string"}, chunksize=500_000):
        chunk = chunk[pd.to_numeric(chunk["id"], errors="coerce").isin(stock_ids)].copy()
        if chunk.empty:
            continue
        chunk["id"] = pd.to_numeric(chunk["id"], errors="coerce").astype("int64")
        chunk["eom"] = pd.to_datetime(chunk["eom"], errors="coerce")
        chunk["permno"] = pd.to_numeric(chunk["permno"], errors="coerce").astype("Int64")
        chunk["source_crsp"] = pd.to_numeric(chunk["source_crsp"], errors="coerce").fillna(0).astype("int64")
        chunk["ff49"] = pd.to_numeric(chunk["ff49"], errors="coerce").astype("Int64")
        chunk["gvkey"] = chunk["gvkey"].astype("string").str.zfill(6)
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True).dropna(subset=["id", "eom"])


def load_industry_codes(path: Path) -> dict[int, dict[str, str]]:
    industry_map: dict[int, dict[str, str]] = {}
    header_pattern = re.compile(r"^\s*(\d{1,2})\s+([A-Za-z0-9]+)\s+(.+?)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = header_pattern.match(line)
        if not match:
            continue
        code = int(match.group(1))
        if code < 1 or code > 49:
            continue
        industry_map[code] = {
            "industry_code": code,
            "industry_short": match.group(2).strip(),
            "industry": match.group(3).strip(),
        }
    return industry_map


def build_metadata(
    id_panel: pd.DataFrame,
    crsp_path: Path,
    compustat_path: Path,
    industry_map: dict[int, dict[str, str]],
) -> dict[int, dict[str, str]]:
    metadata: dict[int, dict[str, str]] = {}
    metadata.update(build_crsp_metadata(id_panel, crsp_path))
    metadata.update(build_compustat_metadata(id_panel, compustat_path))
    merge_industry_metadata(metadata, id_panel, industry_map)
    return metadata


def merge_industry_metadata(
    metadata: dict[int, dict[str, str]],
    id_panel: pd.DataFrame,
    industry_map: dict[int, dict[str, str]],
) -> None:
    latest = (
        id_panel.dropna(subset=["ff49"])
        .sort_values(["id", "eom"])
        .drop_duplicates("id", keep="last")
    )
    for row in latest.itertuples(index=False):
        stock_id = int(row.id)
        info = industry_map.get(int(row.ff49))
        if not info:
            continue
        metadata.setdefault(stock_id, {}).update(info)


def build_crsp_metadata(id_panel: pd.DataFrame, crsp_path: Path) -> dict[int, dict[str, str]]:
    crsp_ids = {
        int(value)
        for value in id_panel.loc[id_panel["source_crsp"].eq(1), "permno"].dropna().unique()
        if pd.notna(value)
    }
    crsp_ids.update(int(value) for value in id_panel.loc[id_panel["id"].lt(100000), "id"].dropna().unique())
    if not crsp_ids:
        return {}

    crsp = pd.read_csv(
        crsp_path,
        usecols=["PERMNO", "SecInfoEndDt", "Ticker", "IssuerNm", "TradingSymbol"],
        dtype={"Ticker": "string", "IssuerNm": "string", "TradingSymbol": "string"},
    )
    crsp = crsp[pd.to_numeric(crsp["PERMNO"], errors="coerce").isin(crsp_ids)].copy()
    if crsp.empty:
        return {}

    crsp["PERMNO"] = pd.to_numeric(crsp["PERMNO"], errors="coerce").astype("int64")
    crsp["SecInfoEndDt"] = pd.to_datetime(crsp["SecInfoEndDt"], errors="coerce").fillna(pd.Timestamp.max)
    crsp = crsp.sort_values(["PERMNO", "SecInfoEndDt"])
    metadata = {}
    for permno, group in crsp.groupby("PERMNO", sort=False):
        latest = group.iloc[-1]
        ticker = latest_nonempty(group["TradingSymbol"]) or latest_nonempty(group["Ticker"])
        metadata[int(permno)] = {
            "ticker": ticker,
            "name": clean_company_name(latest["IssuerNm"]),
        }
    return metadata


def build_compustat_metadata(id_panel: pd.DataFrame, compustat_path: Path) -> dict[int, dict[str, str]]:
    comp_ids = {int(value) for value in id_panel.loc[id_panel["source_crsp"].ne(1), "id"].dropna().unique() if int(value) >= 100000}
    comp_keys = {decode_compustat_id(stock_id) for stock_id in comp_ids}
    comp_keys.discard(None)
    if not comp_keys:
        return {}

    wanted_gvkeys = {key[1] for key in comp_keys}
    comp = pd.read_csv(
        compustat_path,
        usecols=["gvkey", "datadate", "iid", "tic", "conm"],
        dtype={"gvkey": "string", "iid": "string", "tic": "string", "conm": "string"},
    )
    comp["gvkey"] = comp["gvkey"].astype("string").str.zfill(6)
    comp["iid"] = comp["iid"].astype("string").str.zfill(2)
    comp = comp[comp["gvkey"].isin(wanted_gvkeys)].copy()
    if comp.empty:
        return {}

    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
    comp["key"] = list(zip(comp["gvkey"], comp["iid"]))
    comp = comp[comp["key"].isin({(key[1], key[2]) for key in comp_keys})]
    comp = comp.sort_values(["gvkey", "iid", "datadate"]).drop_duplicates(["gvkey", "iid"], keep="last")
    by_key = {
        (str(row.gvkey), str(row.iid)): {
            "ticker": clean_ticker(row.tic),
            "name": clean_company_name(row.conm),
        }
        for row in comp.itertuples(index=False)
    }

    metadata = {}
    for stock_id in comp_ids:
        decoded = decode_compustat_id(stock_id)
        if decoded is None:
            continue
        _, gvkey, iid = decoded
        if (gvkey, iid) in by_key:
            metadata[stock_id] = by_key[(gvkey, iid)]
    return metadata


def decode_compustat_id(stock_id: int) -> tuple[str, str, str] | None:
    value = str(int(stock_id))
    if len(value) < 9:
        return None
    return value[0], value[1:7], value[7:9]


def update_stock_index(path: Path, stocks: list[dict], metadata: dict[int, dict[str, str]]) -> None:
    for row in stocks:
        info = metadata.get(int(row["id"]), {})
        row["ticker"] = info.get("ticker", row.get("ticker") or "")
        row["name"] = info.get("name", row.get("name") or "")
        row["industry_code"] = info.get("industry_code", row.get("industry_code") or "")
        row["industry_short"] = info.get("industry_short", row.get("industry_short") or "")
        row["industry"] = info.get("industry", row.get("industry") or "")
    path.write_text(json.dumps(stocks, separators=(",", ":")), encoding="utf-8")


def update_prediction_files(predictions_root: Path, metadata: dict[int, dict[str, str]]) -> None:
    files = [predictions_root / "latest.json"]
    files.extend((predictions_root / "monthly").glob("*.json"))
    files.extend((predictions_root / "annual").glob("*.json"))
    for path in files:
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            info = metadata.get(int(row["id"]), {})
            row["ticker"] = info.get("ticker", row.get("ticker") or "")
            row["name"] = info.get("name", row.get("name") or "")
            row["industry_code"] = info.get("industry_code", row.get("industry_code") or "")
            row["industry_short"] = info.get("industry_short", row.get("industry_short") or "")
            row["industry"] = info.get("industry", row.get("industry") or "")
        path.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")


def update_stock_series(stocks_dir: Path, metadata: dict[int, dict[str, str]]) -> None:
    for path in stocks_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        stock_id = int(payload["id"])
        info = metadata.get(stock_id, {})
        payload["ticker"] = info.get("ticker", payload.get("ticker") or "")
        payload["name"] = info.get("name", payload.get("name") or "")
        payload["industry_code"] = info.get("industry_code", payload.get("industry_code") or "")
        payload["industry_short"] = info.get("industry_short", payload.get("industry_short") or "")
        payload["industry"] = info.get("industry", payload.get("industry") or "")
        payload.pop("price", None)
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def clean_ticker(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text.upper()


def clean_company_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    text = re.sub(r"\s*\(?\s*last\s+known\s*\)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return smart_title(text)


def latest_nonempty(values: pd.Series) -> str:
    for value in reversed(values.tolist()):
        text = clean_ticker(value)
        if text:
            return text
    return ""


def smart_title(text: str) -> str:
    suffixes = {
        "INC": "Inc",
        "LTD": "Ltd",
        "CORP": "Corp",
        "CO": "Co",
        "COS": "Cos",
        "PLC": "PLC",
        "LLC": "LLC",
        "LP": "LP",
        "NV": "NV",
        "SA": "SA",
        "AG": "AG",
        "ADR": "ADR",
    }
    words = []
    for word in text.split(" "):
        if not word:
            continue
        upper = word.upper()
        if upper in suffixes:
            words.append(suffixes[upper])
        elif any(char.isdigit() for char in word) or "&" in word:
            words.append(word.upper())
        else:
            words.append(word[:1].upper() + word[1:].lower())
    return " ".join(words)


if __name__ == "__main__":
    main()
