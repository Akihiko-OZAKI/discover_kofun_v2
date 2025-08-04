# generate_image.py

import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import os

def parse_gml(xml_path):
    """GML/XMLから標高データを抽出して2次元配列にする"""
    ns = {
        'gml': 'http://www.opengis.net/gml/3.2',
        'fgd': 'http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema'
    }

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Gridサイズの取得
    low = root.find('.//gml:low', ns).text.split()
    high = root.find('.//gml:high', ns).text.split()
    width = int(high[0]) - int(low[0]) + 1
    height = int(high[1]) - int(low[1]) + 1

    # 標高値の抽出
    tuples = root.find('.//gml:tupleList', ns).text.strip().split()
    elevations = [float(t.replace('地表面,', '')) for t in tuples if '地表面,' in t]

    # numpy配列に変換（行：y, 列：x）
    arr = np.array(elevations, dtype=np.float32).reshape((height, width))
    return arr

def normalize_to_image(arr):
    """標高値を0-255に正規化し、uint8画像に変換"""
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    norm = (arr - min_val) / (max_val - min_val + 1e-5)  # 0-1
    img = (norm * 255).astype(np.uint8)
    return img

def save_image(arr, output_path):
    """画像として保存"""
    img = normalize_to_image(arr)
    Image.fromarray(img).save(output_path)
    print(f"[INFO] 標高画像を保存: {output_path}")

def main(xml_path='inputs/FG-GML-4930-64-08-DEM5A-20250620.xml',
         output_path='outputs/dem_image.png'):
    arr = parse_gml(xml_path)
    save_image(arr, output_path)

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    main()
