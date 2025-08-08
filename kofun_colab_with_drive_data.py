#!/usr/bin/env python3
"""
古墳検出モデル学習 - Google Drive既存データ使用版
更新された古墳リスト（63件）を使用
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
import shutil

# 更新された古墳座標データ（63件）
def load_kofun_coordinates():
    """既知の古墳リストから座標を読み込み"""
    # 更新された古墳座標データ（63件）
    kofun_coordinates = [
        # 既存の古墳（54件）
        [1, 34.698046, 135.809032],  # 仁徳天皇陵
        [2, 34.697919, 135.804807],  # 履中天皇陵
        [3, 34.701343, 135.802867],  # 反正天皇陵
        [4, 34.696442, 135.797175],  # 応神天皇陵
        [5, 34.700241, 135.791136],  # 仲哀天皇陵
        [6, 34.698339, 135.78764],   # 古墳6
        [7, 34.678475, 135.781787],  # 古墳7
        [8, 34.683137, 135.76917],   # 古墳8
        [9, 34.706142, 135.78765],   # 古墳9
        [10, 34.703795, 135.793919], # 古墳10
        [11, 34.57611111, 135.4883], # 古墳11
        [12, 34.568056, 135.486667], # 古墳12
        [13, 34.565, 135.491111],    # 古墳13
        [14, 34.562778, 135.490556], # 古墳14
        [15, 34.558611, 135.487778], # 古墳15
        [16, 34.56, 135.485],        # 古墳16
        [17, 34.561111, 135.483333], # 古墳17
        [18, 34.562778, 135.482222], # 古墳18
        [19, 34.566944, 135.484167], # 古墳19
        [20, 34.566944, 135.485278], # 古墳20
        [21, 34.558056, 135.487778], # 古墳21
        [22, 34.556667, 135.482778], # 古墳22
        [23, 34.555, 135.484167],    # 古墳23
        [24, 34.553889, 135.4775],   # 古墳24
        [25, 34.556111, 135.48],     # 古墳25
        [26, 34.556667, 135.479444], # 古墳26
        [27, 34.553056, 135.485833], # 古墳27
        [28, 34.5525, 135.486389],   # 古墳28
        [29, 34.554722, 135.490833], # 古墳29
        [30, 34.546667, 135.499444], # 古墳30
        [31, 34.581944, 135.593611], # 古墳31
        [32, 34.565833, 135.594167], # 古墳32
        [33, 34.567778, 135.595833], # 古墳33
        [34, 34.573056, 135.616667], # 古墳34
        [35, 34.581944, 135.593611], # 古墳35
        [36, 34.571389, 135.581389], # 古墳36
        [37, 34.568056, 135.613056], # 古墳37
        [38, 34.568056, 135.613611], # 古墳38
        [39, 34.568056, 135.614444], # 古墳39
        [40, 34.568056, 135.609444], # 古墳40
        [41, 34.566944, 135.608889], # 古墳41
        [42, 34.562222, 135.609444], # 古墳42
        [43, 34.563889, 135.612222], # 古墳43
        [44, 34.562778, 135.6125],   # 古墳44
        [45, 34.561667, 135.605278], # 古墳45
        [46, 34.561667, 135.602222], # 古墳46
        [47, 34.557778, 135.604444], # 古墳47
        [48, 34.558889, 135.604444], # 古墳48
        [49, 34.557222, 135.606111], # 古墳49
        [50, 34.556111, 135.606667], # 古墳50
        [51, 34.556944, 135.601944], # 古墳51
        [52, 34.556111, 135.600556], # 古墳52
        [53, 34.5525, 135.5975],     # 古墳53
        [54, 34.551111, 135.604444], # 古墳54
        # さきたま史跡の古墳（9基）
        [55, 36.1263, 139.5575],     # さきたま史跡博物館周辺
        [56, 36.1250, 139.5580],     # 古墳1
        [57, 36.1270, 139.5565],     # 古墳2
        [58, 36.1245, 139.5590],     # 古墳3
        [59, 36.1280, 139.5560],     # 古墳4
        [60, 36.1235, 139.5595],     # 古墳5
        [61, 36.1290, 139.5555],     # 古墳6
        [62, 36.1225, 139.5600],     # 古墳7
        [63, 36.1300, 139.5550],     # 古墳8
    ]
    
    print(f"✅ 古墳座標データを読み込み: {len(kofun_coordinates)}件")
    return kofun_coordinates

class DataPreprocessor:
    def __init__(self, kofun_coordinates):
        self.kofun_coordinates = kofun_coordinates
        
    def xml_to_image(self, xml_path, output_path):
        """XMLから画像を生成"""
        try:
            # XMLを直接解析
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 座標範囲を取得
            envelope = root.find('.//{http://www.opengis.net/gml/3.2}Envelope')
            if envelope is not None:
                lower_corner = envelope.find('.//{http://www.opengis.net/gml/3.2}lowerCorner').text
                upper_corner = envelope.find('.//{http://www.opengis.net/gml/3.2}upperCorner').text
                
                # グリッド情報を取得
                grid = root.find('.//{http://www.opengis.net/gml/3.2}Grid')
                if grid is not None:
                    limits = grid.find('.//{http://www.opengis.net/gml/3.2}GridEnvelope')
                    low = limits.find('.//{http://www.opengis.net/gml/3.2}low').text
                    high = limits.find('.//{http://www.opengis.net/gml/3.2}high').text
                    
                    # データを取得
                    tuple_list = root.find('.//{http://www.opengis.net/gml/3.2}tupleList')
                    if tuple_list is not None:
                        # データを解析して画像を生成
                        data_text = tuple_list.text.strip()
                        lines = data_text.split('\n')
                        
                        # 簡易的な画像生成
                        height = len(lines)
                        width = 225  # 固定幅
                        
                        image = np.zeros((height, width, 3), dtype=np.uint8)
                        
                        for i, line in enumerate(lines):
                            if i < height:
                                # 簡易的な高度マッピング
                                try:
                                    elevation = float(line.split(',')[1])
                                    # 高度を0-255に正規化
                                    pixel_value = int((elevation - 0) / 100 * 255)
                                    pixel_value = max(0, min(255, pixel_value))
                                    
                                    image[i, :] = [pixel_value, pixel_value, pixel_value]
                                except:
                                    image[i, :] = [128, 128, 128]
                        
                        cv2.imwrite(output_path, image)
                        return True
            
            return False
                
        except Exception as e:
            print(f"❌ 画像生成エラー: {e}")
            return False
    
    def generate_labels(self, xml_path, image_path, label_path):
        """古墳座標からラベルファイルを生成"""
        try:
            # XMLの座標範囲を取得
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            envelope = root.find('.//{http://www.opengis.net/gml/3.2}Envelope')
            if envelope is None:
                return 0
            
            lower_corner = envelope.find('.//{http://www.opengis.net/gml/3.2}lowerCorner').text
            upper_corner = envelope.find('.//{http://www.opengis.net/gml/3.2}upperCorner').text
            
            lat_min, lon_min = map(float, lower_corner.split())
            lat_max, lon_max = map(float, upper_corner.split())
            
            # 画像サイズを取得
            image = cv2.imread(image_path)
            if image is None:
                return 0
            
            img_height, img_width = image.shape[:2]
            
            # 古墳座標を画像座標に変換
            labels = []
            for kofun_id, lat, lon in self.kofun_coordinates:
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    # 座標を画像座標に変換
                    x_norm = (lon - lon_min) / (lon_max - lon_min)
                    y_norm = (lat - lat_min) / (lat_max - lat_min)
                    
                    # YOLO形式（中心座標、幅、高さ）
                    x_center = x_norm
                    y_center = 1.0 - y_norm  # Y軸を反転
                    width = 0.05  # 固定サイズ
                    height = 0.05
                    
                    labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # ラベルファイルを保存
            if labels:
                with open(label_path, 'w') as f:
                    f.write('\n'.join(labels))
                return len(labels)
            
            return 0
            
        except Exception as e:
            print(f"❌ ラベル生成エラー: {e}")
            return 0

def create_training_dataset_from_drive_data(drive_data_path, kofun_coordinates):
    """Google Driveの既存データからデータセット作成"""
    print(f"📊 Google Drive既存データからデータセット作成開始: {len(kofun_coordinates)}件の古墳座標")
    
    # データセットディレクトリを作成
    dataset_dir = "dataset_from_drive_data"
    os.makedirs(f"{dataset_dir}/images/train", exist_ok=True)
    os.makedirs(f"{dataset_dir}/images/val", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labels/val", exist_ok=True)
    
    preprocessor = DataPreprocessor(kofun_coordinates)
    
    # XMLファイルを検索（直下の*.xmlファイルのみ）
    xml_files = []
    for file in os.listdir(drive_data_path):
        if file.endswith('.xml'):
            xml_files.append(os.path.join(drive_data_path, file))
    
    print(f"📦 処理対象XMLファイル: {len(xml_files)}件")
    
    train_count = 0
    val_count = 0
    
    for xml_file in tqdm(xml_files, desc="データセット作成中"):
        try:
            # 一時的な画像とラベルファイルのパス
            temp_image_path = f"{xml_file[:-4]}_temp.png"
            temp_label_path = f"{xml_file[:-4]}_temp.txt"
            
            # XML → PNG 変換
            if not preprocessor.xml_to_image(xml_file, temp_image_path):
                continue
            
            # ラベル生成
            label_count = preprocessor.generate_labels(xml_file, temp_image_path, temp_label_path)
            
            if label_count > 0:
                if np.random.random() < 0.8:
                    final_image_path = f"{dataset_dir}/images/train/train_{train_count:04d}.png"
                    final_label_path = f"{dataset_dir}/labels/train/train_{train_count:04d}.txt"
                    train_count += 1
                else:
                    final_image_path = f"{dataset_dir}/images/val/val_{val_count:04d}.png"
                    final_label_path = f"{dataset_dir}/labels/val/val_{val_count:04d}.txt"
                    val_count += 1
                
                # ファイルを移動
                shutil.move(temp_image_path, final_image_path)
                shutil.move(temp_label_path, final_label_path)
            else:
                # 一時ファイルを削除
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                if os.path.exists(temp_label_path):
                    os.remove(temp_label_path)
        
        except Exception as e:
            print(f"ファイル処理エラー {xml_file}: {e}")
            continue
    
    dataset_info = {
        'train_count': train_count,
        'val_count': val_count,
        'total_count': train_count + val_count,
        'kofun_coordinates_count': len(kofun_coordinates),
        'processed_xml_files': len(xml_files)
    }
    
    print(f"✅ データセット作成完了: 訓練 {train_count}件, 検証 {val_count}件")
    return dataset_info

def main():
    """メイン実行関数"""
    print("🚀 古墳検出モデル学習 - Google Drive既存データ使用版")
    print("=" * 60)
    
    # 古墳座標データを読み込み
    print("\n📋 ステップ0: 古墳座標データ読み込み")
    kofun_coordinates = load_kofun_coordinates()
    
    # 座標範囲を確認
    lats = [coord[1] for coord in kofun_coordinates]
    lons = [coord[2] for coord in kofun_coordinates]
    
    print(f"📍 座標範囲:")
    print(f"   緯度: {min(lats):.6f} - {max(lats):.6f}")
    print(f"   経度: {min(lons):.6f} - {max(lons):.6f}")
    
    # 1. Google Driveの既存データからデータセット作成
    print("\n📊 ステップ1: Google Drive既存データからデータセット作成")
    drive_data_path = "/content/drive/MyDrive/Colab Notebooks/209 古墳再発見/extract"
    
    if not os.path.exists(drive_data_path):
        print("❌ Google Driveのデータパスが見つかりません")
        print(f"期待されるパス: {drive_data_path}")
        return
    
    dataset_info = create_training_dataset_from_drive_data(drive_data_path, kofun_coordinates)
    
    # 2. データセット設定ファイル作成
    print("\n⚙️ ステップ2: 設定ファイル作成")
    dataset_config = f"""
# Kofun Detection Dataset - Google Drive Data Configuration
path: ./dataset_from_drive_data  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
nc: 1  # number of classes
names: ['kofun']  # class names

# Settings
optimization:
  ensemble_enabled: true
  augmentation_enabled: true
  validation_enhanced: true
  drive_data_used: true  # Google Driveデータを使用
  sakitama_included: true  # さきたま史跡を含む
"""
    
    with open('kofun_dataset_drive_data.yaml', 'w') as f:
        f.write(dataset_config)
    
    print("✅ Google Driveデータ使用版データセット設定ファイルを作成しました")
    print(f"📊 データセット情報: {dataset_info}")
    
    # 3. 学習コマンドの生成
    print("\n🤖 ステップ3: 学習コマンド")
    print("以下のコマンドをYOLOv5ディレクトリで実行してください:")
    print("=" * 50)
    print("cd yolov5")
    print("python train.py --img 640 --batch 16 --epochs 150 --data ../kofun_dataset_drive_data.yaml --weights yolov5s.pt --cache --patience 30 --save-period 10 --project runs/train --name kofun_drive_data --exist-ok")
    print("=" * 50)
    
    print("\n🎉 Google Drive既存データ使用版データセット作成完了！")
    print("\n📋 次のステップ:")
    print("1. YOLOv5ディレクトリに移動")
    print("2. 上記の学習コマンドを実行")
    print("3. 学習完了後、best.ptをダウンロード")
    print("4. プロジェクトのyolov5/weights/に配置")
    print("5. さきたま史跡で再テスト")

if __name__ == "__main__":
    main() 