#!/usr/bin/env python3
"""
古墳検出モデルの精度向上のための最適化スクリプト
"""

import torch
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Replace yolov5 with ultralytics
from ultralytics import YOLO

class KofunDetectionOptimizer:
    def __init__(self, weights_path='weights/best.pt'):
        self.weights_path = weights_path
        # Remove device selection - ultralytics handles this automatically
        self.model = None
        self.load_model()
        
    def load_model(self):
        """モデルの読み込みと最適化"""
        print("🔄 Loading and optimizing model...")
        
        # モデル読み込み（ultralytics YOLO）
        self.model = YOLO(self.weights_path)
        
        print("✅ Model loaded successfully")
    
    def optimize_inference_parameters(self, test_image_path: str) -> Dict:
        """推論パラメータの最適化"""
        print("🔧 Optimizing inference parameters...")
        
        # テスト画像の読み込み
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"❌ Could not load test image: {test_image_path}")
            return {}
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # パラメータの組み合わせをテスト
        conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        best_params = {
            'conf_threshold': 0.3,
            'iou_threshold': 0.5,
            'max_detections': 0,
            'avg_confidence': 0.0
        }
        
        for conf_thresh in conf_thresholds:
            for iou_thresh in iou_thresholds:
                detections = self.run_inference(img, conf_thresh, iou_thresh)
                
                if len(detections) > 0:
                    avg_conf = np.mean([d['confidence'] for d in detections])
                    
                    if avg_conf > best_params['avg_confidence']:
                        best_params = {
                            'conf_threshold': conf_thresh,
                            'iou_threshold': iou_thresh,
                            'max_detections': len(detections),
                            'avg_confidence': avg_conf
                        }
        
        print(f"✅ Best parameters found: {best_params}")
        return best_params
    
    def run_inference(self, img: np.ndarray, conf_threshold: float = 0.3, 
                     iou_threshold: float = 0.5) -> List[Dict]:
        """推論実行"""
        # 推論（ultralytics YOLO）
        results = self.model(img, conf=conf_threshold, iou=iou_threshold)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    
                    detections.append({
                        'x_center': (x1 + x2) / 2,
                        'y_center': (y1 + y2) / 2,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'confidence': conf,
                        'class': int(cls)
                    })
        
        return detections
    
    def apply_ensemble_detection(self, img: np.ndarray, num_models: int = 3) -> List[Dict]:
        """アンサンブル検出による精度向上"""
        print("🎯 Applying ensemble detection...")
        
        # 複数のモデルサイズで検出
        model_sizes = [(640, 640), (512, 512), (768, 768)]
        all_detections = []
        
        for size in model_sizes[:num_models]:
            # 画像をリサイズ
            img_resized = cv2.resize(img, size)
            
            # 推論（ultralytics YOLO）
            results = self.model(img_resized, conf=0.25, iou=0.45)
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        
                        # 元の画像サイズにスケールバック
                        scale_x = img.shape[1] / size[0]
                        scale_y = img.shape[0] / size[1]
                        
                        all_detections.append({
                            'x_center': (x1 + x2) / 2 * scale_x,
                            'y_center': (y1 + y2) / 2 * scale_y,
                            'width': (x2 - x1) * scale_x,
                            'height': (y2 - y1) * scale_y,
                            'confidence': conf,
                            'class': int(cls)
                        })
        
        # 重複検出の統合
        merged_detections = self.merge_ensemble_detections(all_detections)
        return merged_detections
    
    def merge_ensemble_detections(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """アンサンブル検出結果の統合"""
        if not detections:
            return []
        
        # 信頼度でソート
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        merged = []
        
        for det in detections:
            is_duplicate = False
            for existing in merged:
                iou = self.calculate_iou(det, existing)
                if iou > iou_threshold:
                    # 重複の場合、信頼度の高い方を保持
                    if det['confidence'] > existing['confidence']:
                        merged.remove(existing)
                        merged.append(det)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(det)
        
        return merged
    
    def calculate_iou(self, det1: Dict, det2: Dict) -> float:
        """IoU計算"""
        # バウンディングボックスを計算
        def det_to_bbox(det):
            x1 = det['x_center'] - det['width'] / 2
            y1 = det['y_center'] - det['height'] / 2
            x2 = det['x_center'] + det['width'] / 2
            y2 = det['y_center'] + det['height'] / 2
            return x1, y1, x2, y2
        
        bbox1 = det_to_bbox(det1)
        bbox2 = det_to_bbox(det2)
        
        # 交差部分を計算
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def create_optimized_model_config(self) -> Dict:
        """最適化されたモデル設定を作成"""
        return {
            'ensemble_enabled': True,
            'num_ensemble_models': 3,
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'max_detections': 100,
            'post_processing': {
                'merge_overlapping': True,
                'confidence_boost': True,
                'size_filtering': True
            }
        }
    
    def save_optimization_results(self, results: Dict, output_path: str = 'optimization_results.json'):
        """最適化結果の保存"""
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Optimization results saved to {output_path}")

def main():
    """メイン実行関数"""
    print("🚀 Starting Kofun Detection Model Optimization...")
    
    # 最適化器の初期化
    optimizer = KofunDetectionOptimizer()
    
    # テスト画像のパス（存在する場合）
    test_image_path = 'static/results/converted.png'
    
    if os.path.exists(test_image_path):
        # パラメータ最適化
        best_params = optimizer.optimize_inference_parameters(test_image_path)
        
        # 最適化された設定を作成
        optimized_config = optimizer.create_optimized_model_config()
        optimized_config.update(best_params)
        
        # 結果を保存
        optimizer.save_optimization_results(optimized_config)
        
        print("🎉 Model optimization completed!")
        print(f"📊 Optimized parameters: {optimized_config}")
    else:
        print(f"⚠️ Test image not found: {test_image_path}")
        print("💡 Please run inference first to generate test images")

if __name__ == "__main__":
    main() 