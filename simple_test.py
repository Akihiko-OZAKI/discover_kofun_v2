#!/usr/bin/env python3
"""
シンプルなテストスクリプト - XML→PNG変換と座標解析
"""

import os
from xml_to_png import convert_xml_to_png
from my_utils import parse_latlon_range

def simple_test(xml_file_path):
    """
    基本的なXML→PNG変換と座標解析をテスト
    """
    print(f"🔍 Simple test with: {xml_file_path}")
    
    # 出力ディレクトリの準備
    output_dir = "simple_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # XML → PNG 変換
    png_path = os.path.join(output_dir, "test_converted.png")
    print("📊 Converting XML to PNG...")
    try:
        convert_xml_to_png(xml_file_path, png_path)
        print(f"✅ PNG conversion successful: {png_path}")
    except Exception as e:
        print(f"❌ PNG conversion failed: {e}")
        return
    
    # 座標範囲の解析
    print("📍 Analyzing coordinate range...")
    try:
        lat0, lon0, lat1, lon1 = parse_latlon_range(xml_file_path)
        print(f"✅ Coordinate range: Lat {lat0:.6f}-{lat1:.6f}, Lon {lon0:.6f}-{lon1:.6f}")
        
        # 地域の特定
        if 33.0 <= lat0 <= 34.0 and 130.0 <= lon0 <= 131.0:
            print("📍 Location: Kyushu region (likely Fukuoka/Saga area)")
        elif 35.0 <= lat0 <= 36.0 and 139.0 <= lon0 <= 140.0:
            print("📍 Location: Kanto region (likely Tokyo area)")
        else:
            print(f"📍 Location: Unknown region (Lat: {lat0:.2f}, Lon: {lon0:.2f})")
            
    except Exception as e:
        print(f"❌ Coordinate analysis failed: {e}")
    
    # ファイルサイズの確認
    if os.path.exists(png_path):
        size = os.path.getsize(png_path)
        print(f"📁 Generated PNG size: {size:,} bytes ({size/1024:.1f} KB)")
    
    print(f"📁 Results saved in: {output_dir}")
    
    # 次のステップの提案
    print("\n🎯 Next steps:")
    print("1. Check the generated PNG image")
    print("2. If image looks good, proceed with YOLOv5 inference")
    print("3. Adjust confidence threshold if needed")

if __name__ == "__main__":
    # テスト用XMLファイルのパス
    test_xml = "../テスト用xml/FG-GML-4930-64-23-DEM5A-20250620.xml"
    
    if os.path.exists(test_xml):
        simple_test(test_xml)
    else:
        print(f"❌ Test XML file not found: {test_xml}")
        print("Available files:")
        test_dir = "../テスト用xml"
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.endswith('.xml'):
                    print(f"  - {file}") 