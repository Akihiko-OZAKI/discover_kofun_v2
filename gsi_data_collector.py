#!/usr/bin/env python3
"""
国土地理院からDEM5Aデータを自動収集するスクリプト
"""

import requests
import os
import time
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
import zipfile
import shutil

class GSIDataCollector:
    def __init__(self, output_dir="collected_data"):
        self.base_url = "https://fgd.gsi.go.jp/download/API2/contents/XML/"
        self.output_dir = output_dir
        self.dem_dir = os.path.join(output_dir, "dem5a")
        os.makedirs(self.dem_dir, exist_ok=True)
        
    def get_dem5a_list(self, lat_min, lat_max, lon_min, lon_max):
        """
        指定範囲のDEM5Aファイルリストを取得
        """
        # 国土地理院のタイル情報を取得
        tile_list_url = "https://fgd.gsi.go.jp/download/API2/contents/XML/dem5a.xml"
        
        try:
            response = requests.get(tile_list_url)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            tiles = []
            
            for tile in root.findall(".//tile"):
                lat = float(tile.find("lat").text)
                lon = float(tile.find("lon").text)
                
                # 指定範囲内のタイルを抽出
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    tile_info = {
                        'lat': lat,
                        'lon': lon,
                        'filename': tile.find("filename").text,
                        'url': tile.find("url").text
                    }
                    tiles.append(tile_info)
            
            return tiles
            
        except Exception as e:
            print(f"タイルリスト取得エラー: {e}")
            return []
    
    def download_dem5a(self, tile_info):
        """
        個別のDEM5Aファイルをダウンロード
        """
        filename = tile_info['filename']
        url = tile_info['url']
        output_path = os.path.join(self.dem_dir, filename)
        
        if os.path.exists(output_path):
            print(f"既に存在: {filename}")
            return True
        
        try:
            print(f"ダウンロード中: {filename}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ 完了: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ ダウンロード失敗: {filename} - {e}")
            return False
    
    def download_range(self, lat_min, lat_max, lon_min, lon_max, delay=1):
        """
        指定範囲のDEM5Aデータを一括ダウンロード
        """
        print(f"範囲: 緯度 {lat_min}-{lat_max}, 経度 {lon_min}-{lon_max}")
        
        tiles = self.get_dem5a_list(lat_min, lat_max, lon_min, lon_max)
        print(f"対象タイル数: {len(tiles)}")
        
        success_count = 0
        for i, tile in enumerate(tiles, 1):
            print(f"[{i}/{len(tiles)}] 処理中...")
            
            if self.download_dem5a(tile):
                success_count += 1
            
            # サーバー負荷軽減のため待機
            time.sleep(delay)
        
        print(f"ダウンロード完了: {success_count}/{len(tiles)} ファイル")
        return success_count

def main():
    # 使用例：奈良県周辺のデータ収集
    collector = GSIDataCollector()
    
    # 奈良県の座標範囲（概算）
    lat_min, lat_max = 34.0, 35.0  # 緯度
    lon_min, lon_max = 135.5, 136.5  # 経度
    
    print("🗺️ 国土地理院DEM5Aデータ収集開始")
    collector.download_range(lat_min, lat_max, lon_min, lon_max)

if __name__ == "__main__":
    main() 