#!/usr/bin/env python3
"""
XML一括インデックス生成ツール

- 指定ディレクトリ内の国土地理院DEM XMLを走査
- 各XMLの緯度経度範囲を抽出しCSV化
- 任意でPNG化（既存の convert_xml_to_png を使用）
- HTMLサムネイル一覧を生成（クリックで拡大）

使い方:
  python batch_xml_index.py --input-dir "path/to/xmls" --output-dir static/results/batch --make-png --target-lat 36.123 --target-lon 139.567

引数:
  --input-dir   処理するXMLディレクトリ（省略時: uploads）
  --output-dir  出力ベースディレクトリ（省略時: static/results/batch）
  --make-png    PNGを生成（省略時: 生成しない、CSVとHTMLのみ生成）
  --target-lat  目的地緯度（省略可）
  --target-lon  目的地経度（省略可）

出力:
  output_dir/index.csv
  output_dir/index.html
  output_dir/png/*.png (--make-png 指定時)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from my_utils import parse_latlon_range
from xml_to_png import convert_xml_to_png


@dataclasses.dataclass
class XmlTileInfo:
    xml_path: Path
    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float
    lat_center: float
    lon_center: float
    distance_km: Optional[float] = None
    png_rel_path: Optional[str] = None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return radius_km * c


def find_xml_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.xml") if p.is_file()])


def extract_bounds(xml_path: Path) -> Tuple[float, float, float, float]:
    lat0, lon0, lat1, lon1 = parse_latlon_range(str(xml_path))
    # 正規化（min/max順）
    lat_min, lat_max = min(lat0, lat1), max(lat0, lat1)
    lon_min, lon_max = min(lon0, lon1), max(lon0, lon1)
    return lat_min, lon_min, lat_max, lon_max


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(rows: Iterable[XmlTileInfo], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "xml",
                "lat_min",
                "lon_min",
                "lat_max",
                "lon_max",
                "lat_center",
                "lon_center",
                "distance_km",
                "png",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    str(r.xml_path),
                    f"{r.lat_min:.6f}",
                    f"{r.lon_min:.6f}",
                    f"{r.lat_max:.6f}",
                    f"{r.lon_max:.6f}",
                    f"{r.lat_center:.6f}",
                    f"{r.lon_center:.6f}",
                    f"{r.distance_km:.3f}" if r.distance_km is not None else "",
                    r.png_rel_path or "",
                ]
            )


def write_html(rows: List[XmlTileInfo], out_html: Path, title: str = "XML Tiles Index") -> None:
    ensure_dir(out_html.parent)
    # シンプルなグリッドUI
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang=\"ja\"><head><meta charset=\"UTF-8\">")
    html.append(f"<title>{title}</title>")
    html.append(
        "<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:16px;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;}"
        ".card{border:1px solid #ddd;border-radius:8px;overflow:hidden;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,0.06);}"
        ".thumb{width:100%;height:180px;object-fit:cover;background:#f5f5f5;}"
        ".meta{padding:8px 10px;font-size:12px;line-height:1.4}"
        ".meta b{font-size:12px}"
        ".head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}"
        ".small{color:#666}"
        "table{width:100%;border-collapse:collapse;margin-top:16px}th,td{border:1px solid #ddd;padding:6px;font-size:12px}th{background:#fafafa}"
        "</style>"
    )
    html.append("</head><body>")
    html.append(f"<h2>{title}</h2>")
    html.append("<div class=\"grid\">")

    for r in rows:
        img_tag = (
            f"<a href=\"{r.png_rel_path}\" target=\"_blank\"><img class=\"thumb\" src=\"{r.png_rel_path}\"/></a>"
            if r.png_rel_path
            else "<div class=\"thumb\"></div>"
        )
        html.append("<div class=\"card\">")
        html.append(img_tag)
        html.append("<div class=\"meta\">")
        html.append(
            f"<div class=\"head\"><b>{r.xml_path.name}</b>"
            f"<span class=\"small\">{r.distance_km:.2f} km</span>" if r.distance_km is not None else ""
        )
        html.append("</div>")
        html.append(
            f"<div>lat:[{r.lat_min:.5f}, {r.lat_max:.5f}] lon:[{r.lon_min:.5f}, {r.lon_max:.5f}]</div>"
        )
        html.append(
            f"<div>center: ({r.lat_center:.5f}, {r.lon_center:.5f})</div>"
        )
        html.append("</div></div>")

    html.append("</div>")

    # テーブル（CSV相当）
    html.append("<h3>一覧 (CSV相当)</h3>")
    html.append("<table><thead><tr><th>xml</th><th>lat_min</th><th>lon_min</th><th>lat_max</th><th>lon_max</th><th>lat_center</th><th>lon_center</th><th>distance_km</th><th>png</th></tr></thead><tbody>")
    for r in rows:
        html.append(
            "<tr>"
            f"<td>{r.xml_path.name}</td>"
            f"<td>{r.lat_min:.6f}</td>"
            f"<td>{r.lon_min:.6f}</td>"
            f"<td>{r.lat_max:.6f}</td>"
            f"<td>{r.lon_max:.6f}</td>"
            f"<td>{r.lat_center:.6f}</td>"
            f"<td>{r.lon_center:.6f}</td>"
            f"<td>{r.distance_km:.3f}</td>" if r.distance_km is not None else "<td></td>"
            f"<td><a href=\"{r.png_rel_path}\" target=\"_blank\">png</a></td>" if r.png_rel_path else "<td></td>"
            "</tr>"
        )
    html.append("</tbody></table>")

    html.append("</body></html>")
    out_html.write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index GSI DEM XML tiles and optionally render to PNG.")
    parser.add_argument("--input-dir", type=str, default="uploads", help="Directory containing XML files")
    parser.add_argument(
        "--output-dir", type=str, default=str(Path("static") / "results" / "batch"), help="Output base directory"
    )
    parser.add_argument("--make-png", action="store_true", help="Render PNG thumbnails using xml_to_png")
    parser.add_argument("--target-lat", type=float, default=None, help="Target latitude to compute distance (km)")
    parser.add_argument("--target-lon", type=float, default=None, help="Target longitude to compute distance (km)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    png_dir = output_dir / "png"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    xml_files = find_xml_files(input_dir)
    if not xml_files:
        print(f"No XML files found under: {input_dir}")
        return

    ensure_dir(output_dir)
    if args.make_png:
        ensure_dir(png_dir)

    rows: List[XmlTileInfo] = []
    for idx, xml_path in enumerate(xml_files, start=1):
        try:
            lat_min, lon_min, lat_max, lon_max = extract_bounds(xml_path)
            lat_center = (lat_min + lat_max) / 2.0
            lon_center = (lon_min + lon_max) / 2.0
            distance_km: Optional[float] = None
            if args.target_lat is not None and args.target_lon is not None:
                distance_km = haversine_km(args.target_lat, args.target_lon, lat_center, lon_center)

            png_rel_path: Optional[str] = None
            if args.make_png:
                out_name = xml_path.stem + ".png"
                out_png = png_dir / out_name
                try:
                    convert_xml_to_png(str(xml_path), str(out_png))
                    # HTML で使う相対パス（output_dir からの相対）
                    png_rel_path = f"png/{out_name}"
                except Exception as e:
                    print(f"[WARN] PNG conversion failed for {xml_path.name}: {e}")

            rows.append(
                XmlTileInfo(
                    xml_path=xml_path,
                    lat_min=lat_min,
                    lon_min=lon_min,
                    lat_max=lat_max,
                    lon_max=lon_max,
                    lat_center=lat_center,
                    lon_center=lon_center,
                    distance_km=distance_km,
                    png_rel_path=png_rel_path,
                )
            )
        except Exception as e:
            print(f"[WARN] Skipped {xml_path.name}: {e}")

    # 距離指定があれば距離昇順で並べ替え
    if any(r.distance_km is not None for r in rows):
        rows.sort(key=lambda r: (1e9 if r.distance_km is None else r.distance_km))

    # CSV/HTML 出力
    csv_path = output_dir / "index.csv"
    write_csv(rows, csv_path)
    html_path = output_dir / "index.html"
    title = "XML Tiles Index" if args.target_lat is None else f"XML Tiles Index (target: {args.target_lat:.5f},{args.target_lon:.5f})"
    write_html(rows, html_path, title=title)

    print(f"✅ Indexed {len(rows)} XML files")
    print(f"📄 CSV: {csv_path}")
    print(f"🌐 HTML: {html_path}")
    if args.make_png:
        print(f"🖼️ PNG dir: {png_dir}")


if __name__ == "__main__":
    main()

