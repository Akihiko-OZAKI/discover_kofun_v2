#!/usr/bin/env python3
"""
古墳検出システムのテストスクリプト
"""

import os
import sys
sys.path.insert(0, os.path.abspath('yolov5'))

from xml_to_png import convert_xml_to_png
from my_utils import parse_latlon_range, bbox_to_latlon, read_yolo_labels
import torch
import cv2
import numpy as np
from pathlib import Path

def test_detection(xml_file_path):
    """
    指定されたXMLファイルで推論をテスト
    """
    print(f"🔍 Testing detection with: {xml_file_path}")
    
    # 出力ディレクトリの準備
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # XML → PNG 変換
    png_path = os.path.join(output_dir, "test_converted.png")
    print("📊 Converting XML to PNG...")
    convert_xml_to_png(xml_file_path, png_path)
    
    # YOLOv5 推論実行（簡素化版）
    print("🤖 Running YOLOv5 inference...")
    try:
        # モデルを直接読み込み
        model_path = 'yolov5/weights/best.pt'
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        
        # 画像読み込み
        image = cv2.imread(png_path)
        if image is None:
            print(f"❌ Could not read image: {png_path}")
            return
        
        # 画像サイズ調整
        img_size = 640
        img = cv2.resize(image, (img_size, img_size))
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]
        
        # 推論実行（CPU使用）
        device = torch.device('cpu')
        img = img.to(device)
        
        # モデル読み込みと推論（PyTorch 2.6対応）
        model = torch.load(model_path, map_location=device, weights_only=False)
        if 'model' in model:
            model = model['model']
        
        # モデルをfloat32に変換
        model = model.float()
        model.eval()
        
        with torch.no_grad():
            pred = model(img)
        
        # 結果処理
        conf_threshold = 0.1
        detections = []
        
        if len(pred) > 0:
            pred = pred[0]  # 最初の画像の結果
            print(f"推論結果の形状: {pred.shape}")
            
            # 結果を適切に処理
            for i in range(pred.shape[0]):
                detection = pred[i]  # 25200個の検出結果
                
                # 各検出結果を処理
                for j in range(detection.shape[0]):
                    single_detection = detection[j]
                    x1 = single_detection[0].item()
                    y1 = single_detection[1].item()
                    x2 = single_detection[2].item()
                    y2 = single_detection[3].item()
                    conf = single_detection[4].item()
                    cls = single_detection[5].item()
                    
                    if conf > conf_threshold:
                        # YOLO座標を中心座標に変換
                        x_center = (x1 + x2) / 2 / img_size
                        y_center = (y1 + y2) / 2 / img_size
                        width = (x2 - x1) / img_size
                        height = (y2 - y1) / img_size
                        
                        detections.append({
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height,
                            'confidence': conf,
                            'class': cls
                        })
        
        print(f"🎯 Found {len(detections)} detection(s)")
        
        # 結果の解析
        print("📈 Analyzing results...")
        try:
            # 座標範囲を取得
            lat0, lon0, lat1, lon1 = parse_latlon_range(xml_file_path)
            print(f"📍 Coordinate range: Lat {lat0:.6f}-{lat1:.6f}, Lon {lon0:.6f}-{lon1:.6f}")
            
            if detections:
                print("✅ KOFUN DETECTED!")
                for i, detection in enumerate(detections):
                    lat, lon = bbox_to_latlon(detection, lat0, lon0, lat1, lon1)
                    conf = detection.get('confidence', 0.0)
                    print(f"   Detection #{i+1}: Lat {lat:.6f}, Lon {lon:.6f}, Confidence {conf:.3f}")
            else:
                print("❌ No kofun detected")
                
        except Exception as e:
            print(f"❌ Error analyzing results: {e}")
        
        # 結果をファイルに保存
        result_file = os.path.join(output_dir, "detection_results.txt")
        with open(result_file, 'w') as f:
            f.write(f"Detection Results for {xml_file_path}\n")
            f.write(f"Total detections: {len(detections)}\n")
            for i, detection in enumerate(detections):
                lat, lon = bbox_to_latlon(detection, lat0, lon0, lat1, lon1)
                conf = detection.get('confidence', 0.0)
                f.write(f"Detection #{i+1}: Lat {lat:.6f}, Lon {lon:.6f}, Confidence {conf:.3f}\n")
        
        print(f"📁 Results saved to: {result_file}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # テスト用XMLファイルのパス（ローカルファイルを使用）
    test_xml = "uploads/FG-GML-4930-64-23-DEM5A-20250620.xml"
    
    if os.path.exists(test_xml):
        test_detection(test_xml)
    else:
        print(f"❌ Test XML file not found: {test_xml}")
        print("Available XML files:")
        uploads_dir = "uploads"
        if os.path.exists(uploads_dir):
            for file in os.listdir(uploads_dir):
                if file.endswith('.xml'):
                    print(f"  - {uploads_dir}/{file}")
        else:
            print("  No uploads directory found") 