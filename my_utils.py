# utils.py
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import os

# XMLファイルから標高データ配列を取得（reshape対応）
def parse_dem_xml(xml_path):
    ns = {'gml': 'http://www.opengis.net/gml/3.2'}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # tupleList から標高値を抽出
    tuple_list_elem = root.find('.//gml:tupleList', ns)
    if tuple_list_elem is None:
        raise ValueError("gml:tupleList が見つかりません")
    rows = tuple_list_elem.text.strip().split('\n')
    elevation_values = [float(row.split(',')[1]) for row in rows if ',' in row]

    # 高さ・幅を gml:high から取得
    high_elem = root.find('.//gml:high', ns)
    if high_elem is None:
        raise ValueError("gml:high が見つかりません")
    high_vals = list(map(int, high_elem.text.strip().split()))
    cols, rows_ = high_vals[0] + 1, high_vals[1] + 1

    # reshape
    arr = np.array(elevation_values)
    if len(arr) != rows_ * cols:
        raise ValueError(f"reshape 失敗: 値数={len(arr)}, 期待={rows_}x{cols}")
    return arr.reshape((rows_, cols))

# PNG画像保存（グレースケール）
def save_dem_as_png(array, output_path):
    norm_array = 255 * (array - np.min(array)) / (np.ptp(array))  # 0-255 に正規化
    img = Image.fromarray(norm_array.astype(np.uint8))
    img.save(output_path)

# YOLOラベル読み込み
def read_yolo_labels(txt_path):
    detections = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id, x_center, y_center, width, height = map(float, parts)
                detections.append({
                    'class_id': int(cls_id),
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
    return detections

# 緯度・経度への変換
def bbox_to_latlon(detection, lat0, lon0, lat1, lon1):
    x_center = float(detection['x_center'])
    y_center = float(detection['y_center'])

    lat = lat0 + (lat1 - lat0) * y_center
    lon = lon0 + (lon1 - lon0) * x_center

    return round(lat, 6), round(lon, 6)

# XMLから緯度経度範囲を取得
def parse_latlon_range(xml_path):
    ns = {'gml': 'http://www.opengis.net/gml/3.2'}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lower_corner = root.find('.//gml:lowerCorner', ns)
    upper_corner = root.find('.//gml:upperCorner', ns)

    if lower_corner is None or upper_corner is None:
        raise ValueError("緯度経度範囲がXMLに見つかりません")

    lat0, lon0 = map(float, lower_corner.text.strip().split())
    lat1, lon1 = map(float, upper_corner.text.strip().split())

    return lat0, lon0, lat1, lon1
