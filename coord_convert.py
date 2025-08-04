# coord_convert.py

import xml.etree.ElementTree as ET

def parse_bounds(xml_path):
    """GML/XMLから緯度経度の範囲（lowerCorner / upperCorner）を取得"""
    ns = {'gml': 'http://www.opengis.net/gml/3.2'}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lower = root.find('.//gml:lowerCorner', ns).text.strip().split()
    upper = root.find('.//gml:upperCorner', ns).text.strip().split()

    min_lat, min_lon = map(float, lower)
    max_lat, max_lon = map(float, upper)

    return min_lat, min_lon, max_lat, max_lon

def pixel_to_latlon(x, y, img_width, img_height, xml_path):
    """画像上のピクセル座標 (x, y) を緯度経度に変換"""
    min_lat, min_lon, max_lat, max_lon = parse_bounds(xml_path)

    lon = min_lon + (x / img_width) * (max_lon - min_lon)
    lat = max_lat - (y / img_height) * (max_lat - min_lat)  # 上が北（y軸は下向き）

    return round(lat, 6), round(lon, 6)

# 使用例
if __name__ == '__main__':
    # 推論で得たピクセル位置（例：中心点）
    x_center = 112
    y_center = 75
    image_width = 225
    image_height = 150
    xml_path = 'inputs/FG-GML-4930-64-08-DEM5A-20250620.xml'

    lat, lon = pixel_to_latlon(x_center, y_center, image_width, image_height, xml_path)
    print(f"[INFO] 緯度経度: ({lat}, {lon})")
