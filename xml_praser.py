# xml_parser.py
import xml.etree.ElementTree as ET
import numpy as np

def parse_gml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {
        'gml': 'http://www.opengis.net/gml/3.2',
        'fgd': 'http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema'
    }

    # 緯度経度の範囲を取得
    lower_corner = root.find('.//gml:Envelope/gml:lowerCorner', ns).text.strip().split()
    upper_corner = root.find('.//gml:Envelope/gml:upperCorner', ns).text.strip().split()

    min_lat, min_lon = map(float, lower_corner)
    max_lat, max_lon = map(float, upper_corner)

    # グリッドサイズを取得
    low = root.find('.//gml:GridEnvelope/gml:low', ns).text.strip().split()
    high = root.find('.//gml:GridEnvelope/gml:high', ns).text.strip().split()
    width = int(high[0]) + 1  # 例: 224 + 1 = 225
    height = int(high[1]) + 1  # 例: 149 + 1 = 150

    # 標高データの読み取り
    tuples = root.find('.//gml:tupleList', ns).text.strip().split(' ')
    elevations = [float(t.split(',')[-1]) for t in zip(tuples[::2], tuples[1::2]) if t[1].replace('.', '', 1).isdigit()]
    elevation_array = np.array(elevations).reshape((height, width))

    return elevation_array, (min_lat, min_lon), (max_lat, max_lon)
