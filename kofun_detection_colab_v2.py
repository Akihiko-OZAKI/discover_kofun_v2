#!/usr/bin/env python3
"""
古墳検出モデル学習 - Google Colab V2最適化版
国土地理院のDEM5Aデータを使用して古墳検出モデルを学習する統合スクリプト
"""

import os
import requests
import zipfile
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time
import concurrent.futures
from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_bounds

# 古墳座標データ（サンプル）
kofun_coordinates = [
    [1, 34.698046, 135.809032],  # 仁徳天皇陵
    [2, 34.697919, 135.804807],  # 履中天皇陵
    [3, 34.701343, 135.802867],  # 反正天皇陵
    [4, 34.694167, 135.792778],  # 応神天皇陵
    [5, 34.691667, 135.790278],  # 仲哀天皇陵
]

class GSIDataCollectorV2:
    def __init__(self, output_dir="collected_data_v2"):
        self.base_url = "https://fgd.gsi.go.jp/download/API2/contents/XML/"
        self.output_dir = output_dir
        self.dem_dir = os.path.join(output_dir, "dem5a")
        self.processed_dir = os.path.join(output_dir, "processed")
        
        os.makedirs(self.dem_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def get_dem5a_list(self, lat_min, lat_max, lon_min, lon_max):
        """指定範囲のDEM5Aファイルリストを取得"""
        tile_list_url = "https://fgd.gsi.go.jp/download/API2/contents/XML/dem5a.xml"
        
        try:
            response = requests.get(tile_list_url)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            tiles = []
            
            for tile in root.findall(".//tile"):
                lat = float(tile.find("lat").text)
                lon = float(tile.find("lon").text)
                
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
        """個別のDEM5Aファイルをダウンロード"""
        filename = tile_info['filename']
        url = tile_info['url']
        output_path = os.path.join(self.dem_dir, filename)
        
        if os.path.exists(output_path):
            return True
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"❌ ダウンロード失敗: {filename} - {e}")
            return False
    
    def download_range(self, lat_min, lat_max, lon_min, lon_max, delay=1):
        """指定範囲のDEM5Aデータを一括ダウンロード"""
        print(f"🌍 範囲: 緯度 {lat_min}-{lat_max}, 経度 {lon_min}-{lon_max}")
        
        tiles = self.get_dem5a_list(lat_min, lat_max, lon_min, lon_max)
        print(f"📊 対象タイル数: {len(tiles)}")
        
        success_count = 0
        for i, tile in enumerate(tqdm(tiles, desc="ダウンロード中")):
            if self.download_dem5a(tile):
                success_count += 1
            
            time.sleep(delay)
        
        print(f"✅ ダウンロード完了: {success_count}/{len(tiles)} ファイル")
        return success_count

class DataPreprocessorV2:
    def __init__(self, kofun_coordinates):
        self.kofun_coordinates = kofun_coordinates
        
    def xml_to_image(self, xml_path, output_path):
        """XMLファイルを画像に変換"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            bounds = root.find('.//{http://www.opengis.net/gml}Envelope')
            if bounds is None:
                return False
                
            lower_corner = bounds.find('.//{http://www.opengis.net/gml}lowerCorner').text.split()
            upper_corner = bounds.find('.//{http://www.opengis.net/gml}upperCorner').text.split()
            
            lon_min, lat_min = map(float, lower_corner)
            lon_max, lat_max = map(float, upper_corner)
            
            grid_data = root.find('.//{http://www.opengis.net/gml}tupleList')
            if grid_data is None:
                return False
                
            lines = grid_data.text.strip().split('\n')
            elevation_data = []
            
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 3:
                        elevation = float(parts[2])
                        elevation_data.append(elevation)
            
            size = int(len(elevation_data) ** 0.5)
            if size * size != len(elevation_data):
                size = int(len(elevation_data) ** 0.5) + 1
            
            grid = np.array(elevation_data[:size*size]).reshape(size, size)
            grid_normalized = ((grid - grid.min()) / (grid.max() - grid.min()) * 255).astype(np.uint8)
            
            cv2.imwrite(output_path, grid_normalized)
            return True
            
        except Exception as e:
            print(f"XML変換エラー: {e}")
            return False
    
    def generate_labels(self, xml_path, image_path, label_path):
        """古墳座標データからラベルを生成"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            bounds = root.find('.//{http://www.opengis.net/gml}Envelope')
            lower_corner = bounds.find('.//{http://www.opengis.net/gml}lowerCorner').text.split()
            upper_corner = bounds.find('.//{http://www.opengis.net/gml}upperCorner').text.split()
            
            lon_min, lat_min = map(float, lower_corner)
            lon_max, lat_max = map(float, upper_corner)
            
            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]
            
            labels = []
            for kofun in self.kofun_coordinates:
                kofun_lat, kofun_lon = kofun[1], kofun[2]
                
                if lat_min <= kofun_lat <= lat_max and lon_min <= kofun_lon <= lon_max:
                    x = (kofun_lon - lon_min) / (lon_max - lon_min)
                    y = 1 - (kofun_lat - lat_min) / (lat_max - lat_min)
                    
                    x_center = x
                    y_center = y
                    width = 0.05
                    height = 0.05
                    
                    labels.append(f"0 {x_center} {y_center} {width} {height}")
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
            
            return len(labels)
            
        except Exception as e:
            print(f"ラベル生成エラー: {e}")
            return 0

def create_training_dataset_v2(collected_data_dir, kofun_coordinates):
    """V2最適化版の学習用データセットを作成"""
    print("📊 V2最適化版データセットを作成中...")
    
    dataset_dir = "dataset_v2"
    os.makedirs(f"{dataset_dir}/images/train", exist_ok=True)
    os.makedirs(f"{dataset_dir}/images/val", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labels/val", exist_ok=True)
    
    preprocessor = DataPreprocessorV2(kofun_coordinates)
    
    xml_files = list(Path(collected_data_dir).glob("**/*.xml"))
    print(f"📁 処理対象ファイル数: {len(xml_files)}")
    
    train_count = 0
    val_count = 0
    
    for i, xml_file in enumerate(tqdm(xml_files, desc="データセット作成中")):
        try:
            image_path = f"{dataset_dir}/images/temp_{i}.png"
            if preprocessor.xml_to_image(str(xml_file), image_path):
                
                label_path = f"{dataset_dir}/labels/temp_{i}.txt"
                label_count = preprocessor.generate_labels(str(xml_file), image_path, label_path)
                
                if label_count > 0:
                    if np.random.random() < 0.8:
                        final_image_path = f"{dataset_dir}/images/train/train_{train_count:04d}.png"
                        final_label_path = f"{dataset_dir}/labels/train/train_{train_count:04d}.txt"
                        train_count += 1
                    else:
                        final_image_path = f"{dataset_dir}/images/val/val_{val_count:04d}.png"
                        final_label_path = f"{dataset_dir}/labels/val/val_{val_count:04d}.txt"
                        val_count += 1
                    
                    os.rename(image_path, final_image_path)
                    os.rename(label_path, final_label_path)
                else:
                    os.remove(image_path)
                    if os.path.exists(label_path):
                        os.remove(label_path)
            
        except Exception as e:
            print(f"ファイル処理エラー {xml_file}: {e}")
            continue
    
    dataset_info = {
        'train_count': train_count,
        'val_count': val_count,
        'total_count': train_count + val_count
    }
    
    print(f"✅ データセット作成完了: 訓練 {train_count}件, 検証 {val_count}件")
    return dataset_info

def main():
    """メイン実行関数"""
    print("🚀 古墳検出モデル学習 - Google Colab V2最適化版")
    print("=" * 60)
    
    # 1. データ収集
    print("\n📦 ステップ1: 国土地理院データ収集")
    collector = GSIDataCollectorV2()
    
    areas = [
        (34.4, 35.0, 135.4, 136.0),  # 大阪・奈良周辺
        (34.8, 35.2, 135.6, 135.9),  # 京都周辺
        (34.6, 34.8, 135.7, 135.9),  # 堺市周辺
    ]
    
    total_collected = 0
    for i, area in enumerate(areas, 1):
        print(f"\n📦 地域 {i}: {area}")
        collected = collector.download_range(*area, delay=0.5)
        total_collected += collected
    
    print(f"\n🎉 総収集データ数: {total_collected}件")
    
    # 2. データセット作成
    print("\n📊 ステップ2: データセット作成")
    dataset_info = create_training_dataset_v2("collected_data_v2/dem5a", kofun_coordinates)
    
    # 3. データセット設定ファイル作成
    print("\n⚙️ ステップ3: 設定ファイル作成")
    dataset_config_v2 = f"""
# Kofun Detection Dataset V2 Configuration
path: ./dataset_v2  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
nc: 1  # number of classes
names: ['kofun']  # class names

# V2 Optimization Settings
optimization:
  ensemble_enabled: true
  augmentation_enabled: true
  validation_enhanced: true
"""
    
    with open('kofun_dataset_v2.yaml', 'w') as f:
        f.write(dataset_config_v2)
    
    print("✅ V2最適化版データセット設定ファイルを作成しました")
    print(f"📊 データセット情報: {dataset_info}")
    
    # 4. 学習コマンドの生成
    print("\n🤖 ステップ4: 学習コマンド")
    print("以下のコマンドをYOLOv5ディレクトリで実行してください:")
    print("=" * 50)
    print("cd yolov5")
    print("python train.py --img 640 --batch 16 --epochs 100 --data ../kofun_dataset_v2.yaml --weights yolov5s.pt --cache --patience 20 --save-period 10 --project runs/train --name kofun_v2_optimized --exist-ok")
    print("=" * 50)
    
    print("\n🎉 V2最適化版データ収集・学習準備完了！")
    print("\n📋 次のステップ:")
    print("1. YOLOv5ディレクトリに移動")
    print("2. 上記の学習コマンドを実行")
    print("3. 学習完了後、best.ptをダウンロード")
    print("4. プロジェクトのyolov5/weights/に配置")

if __name__ == "__main__":
    main() 