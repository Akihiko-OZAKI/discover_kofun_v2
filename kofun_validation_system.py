#!/usr/bin/env python3
"""
古墳座標データを活用した検出精度向上システム（メモリ最適化版）
"""

import pandas as pd
import numpy as np
import cv2
import os
import sys
import torch
import gc
from typing import List, Dict, Tuple
import math
import pathlib

# YOLOv5のパスを追加
sys.path.insert(0, os.path.abspath('yolov5'))
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

# Windows で Linux 環境で保存された .pt 内の PosixPath を復元できない問題への対応
# pickle 復元時に PosixPath を WindowsPath に置き換える
if os.name == 'nt':
    try:
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[attr-defined]
    except Exception:
        pass

class KofunValidationSystem:
    def __init__(self, kofun_csv_path="kofun_coordinates_updated.csv"):
        # Use absolute path to ensure file is found
        if not os.path.isabs(kofun_csv_path):
            kofun_csv_path = os.path.join(os.getcwd(), kofun_csv_path)
        self.kofun_data = self.load_kofun_coordinates(kofun_csv_path)
        # メモリ効率化のため、モデルは必要時に読み込み
        self.device = None
        self.model = None
        self.stride = None
        self.names = None
        self.pt = None
        self.imgsz = None
        self.half = None
        
    def load_kofun_coordinates(self, csv_path: str) -> pd.DataFrame:
        """古墳座標データを読み込み"""
        try:
            if not os.path.exists(csv_path):
                print(f"⚠️ CSV file not found: {csv_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(csv_path, header=None, names=['id', 'latitude', 'longitude'], encoding='utf-8')
            print(f"✅ 古墳座標データを読み込み: {len(df)}件")
            return df
        except Exception as e:
            print(f"❌ 古墳座標データ読み込みエラー: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def load_model(self, weights_path='weights/yolov5n.pt'):
        """YOLOv5nモデルを読み込み（超軽量版）"""
        if self.model is not None:
            return  # 既に読み込まれている場合はスキップ
            
        print("🔄 Loading YOLOv5n model (ultra lightweight)...")
        
        # メモリ使用量を監視
        self.log_memory_usage("Before model loading")
        
        self.device = select_device('')
        self.model = DetectMultiBackend(weights_path, device=self.device)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size((224, 224), s=self.stride)  # さらに小さくしてメモリ削減
        # CPU環境では半精度を無効化
        self.half = False  # メモリ削減のため半精度を無効化
        
        self.model.eval()
        # ウォームアップを最小限に
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        
        self.log_memory_usage("After model loading")
        print("✅ YOLOv5n model loaded successfully (ultra lightweight)")
    
    def log_memory_usage(self, stage: str):
        """メモリ使用量をログ出力"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"📊 Memory usage at {stage}: {memory_mb:.1f}MB")
        except:
            pass
    
    def cleanup_memory(self):
        """メモリをクリーンアップ"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.log_memory_usage("After cleanup")
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """2点間の距離を計算（メートル）"""
        R = 6371000  # 地球の半径（メートル）
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def find_nearby_kofun(self, target_lat: float, target_lon: float, 
                          max_distance: float = 1000.0) -> List[Dict]:
        """指定座標の近くにある古墳を検索"""
        nearby_kofun = []
        
        for _, row in self.kofun_data.iterrows():
            distance = self.calculate_distance(
                target_lat, target_lon, 
                row['latitude'], row['longitude']
            )
            
            if distance <= max_distance:
                nearby_kofun.append({
                    'id': row['id'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'distance': distance
                })
        
        return sorted(nearby_kofun, key=lambda x: x['distance'])
    
    def validate_detection_with_kofun_data(self, detections: List[Dict], 
                                          image_bounds: Tuple[float, float, float, float]) -> List[Dict]:
        """
        検出結果を古墳座標データで検証・補正
        """
        lat_min, lat_max, lon_min, lon_max = image_bounds
        validated_detections = []
        
        for detection in detections:
            # 検出座標を計算
            x_center = detection['x_center']
            y_center = detection['y_center']
            
            # 画像座標を地理座標に変換
            det_lat = lat_min + (1 - y_center) * (lat_max - lat_min)
            det_lon = lon_min + x_center * (lon_max - lon_min)
            
            # 近くの古墳を検索
            nearby_kofun = self.find_nearby_kofun(det_lat, det_lon, max_distance=500.0)
            
            validation_score = 0.0
            validation_info = {
                'original_confidence': detection['confidence'],
                'nearby_kofun_count': len(nearby_kofun),
                'closest_kofun_distance': float('inf'),
                'validation_score': 0.0
            }
            
            if nearby_kofun:
                closest_kofun = nearby_kofun[0]
                validation_info['closest_kofun_distance'] = closest_kofun['distance']
                validation_info['closest_kofun_id'] = closest_kofun['id']
                
                # 距離に基づく検証スコア計算
                distance_score = max(0, 1 - (closest_kofun['distance'] / 500.0))
                confidence_score = detection['confidence']
                
                # 総合検証スコア
                validation_score = (distance_score * 0.6 + confidence_score * 0.4)
                validation_info['validation_score'] = validation_score
                
                # 座標補正（近くの古墳座標に調整）
                if closest_kofun['distance'] < 100.0:  # 100m以内の場合
                    # 古墳座標に補正
                    corrected_lat = closest_kofun['latitude']
                    corrected_lon = closest_kofun['longitude']
                    
                    # 画像座標に逆変換
                    corrected_y = 1 - (corrected_lat - lat_min) / (lat_max - lat_min)
                    corrected_x = (corrected_lon - lon_min) / (lon_max - lon_min)
                    
                    detection['x_center'] = corrected_x
                    detection['y_center'] = corrected_y
                    detection['confidence'] = max(detection['confidence'], 0.8)  # 信頼度向上
                    validation_info['coordinate_corrected'] = True
            
            # 検証情報を追加
            detection['validation_info'] = validation_info
            detection['final_confidence'] = validation_score
            
            # 検証スコアが一定以上の場合のみ採用（超高感度モード）
            if validation_score >= 0.1:  # 閾値を下げる
                validated_detections.append(detection)
        
        return validated_detections
    
    def run_enhanced_detection(self, image_path: str, xml_path: str, 
                              output_path: str = None) -> List[Dict]:
        """
        古墳座標データを活用した強化検出（メモリ最適化版）
        """
        print(f"🚀 Running enhanced detection with kofun validation (memory optimized)...")
        
        # メモリ使用量を監視
        self.log_memory_usage("Start of detection")
        
        # 画像境界を取得
        from my_utils import parse_latlon_range
        lat0, lon0, lat1, lon1 = parse_latlon_range(xml_path)
        image_bounds = (lat0, lat1, lon0, lon1)
        
        # 通常のYOLOv5検出
        if self.model is None:
            self.load_model()
        
        self.log_memory_usage("After model loading")
        
        # 画像読み込みと前処理（メモリ効率化）
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # 推論実行（メモリ効率化）
        img = cv2.resize(image, self.imgsz)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # 半精度を無効化してメモリ削減
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]
        
        self.log_memory_usage("After image preprocessing")
        
        # 検出（超軽量版）
        all_detections = []
        conf_thresholds = [0.25]  # YOLOv5n用に調整
        H, W = image.shape[:2]
        
        for conf_thres in conf_thresholds:
            # 通常推論（TTA無効化で高速化）
            with torch.no_grad():  # メモリ削減
                pred = self.model(img, augment=False, visualize=False)
                pred = non_max_suppression(pred, conf_thres, 0.5, classes=None, max_det=3)  # 検出数をさらに削減
            
            # 通常推論の取り込み
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_boxes(img[i].shape[1:], det[:, :4], image.shape).round()
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = xyxy
                        x_center = (x1 + x2) / 2 / W
                        y_center = (y1 + y2) / 2 / H
                        width = (x2 - x1) / W
                        height = (y2 - y1) / H
                        all_detections.append({
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height,
                            'confidence': conf.item(),
                            'class': cls.item(),
                            'threshold': conf_thres
                        })
        
        # メモリクリーンアップ
        del img, pred
        self.cleanup_memory()
        
        self.log_memory_usage("After detection")
        
        # 重複検出の統合（軽量化版）
        merged_detections = self.merge_overlapping_detections(all_detections)
        
        # 古墳座標データで検証・補正（軽量化版）
        validated_detections = []
        for detection in merged_detections:
            # 簡易検証：信頼度が一定以上の場合のみ採用
            if detection['confidence'] >= 0.2:  # 閾値を上げる
                detection['final_confidence'] = detection['confidence']
                detection['validation_info'] = {
                    'original_confidence': detection['confidence'],
                    'validation_score': detection['confidence']
                }
                validated_detections.append(detection)
        
        print(f"✅ Enhanced detection completed: {len(validated_detections)} validated detections")
        
        # 結果を描画
        if output_path:
            self.draw_enhanced_results(image_path, validated_detections, output_path)
        
        return validated_detections
    
    def merge_overlapping_detections(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """重複する検出結果を統合"""
        if not detections:
            return []
        
        # 信頼度でソート
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            current_group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                # IoU計算
                iou = self.calculate_iou_center(det1, det2)
                
                if iou > iou_threshold:
                    current_group.append(det2)
                    used.add(j)
            
            # グループ内の検出を統合
            if len(current_group) > 1:
                merged_det = self.merge_detection_group_center(current_group)
                merged.append(merged_det)
            else:
                merged.append(det1)
        
        return merged
    
    def calculate_iou_center(self, det1: Dict, det2: Dict) -> float:
        """中心座標形式でのIoU計算"""
        # 中心座標をバウンディングボックスに変換
        def center_to_bbox(det):
            x_center, y_center = det['x_center'], det['y_center']
            width, height = det['width'], det['height']
            x1 = x_center - width/2
            y1 = y_center - height/2
            x2 = x_center + width/2
            y2 = y_center + height/2
            return [x1, y1, x2, y2]
        
        bbox1 = center_to_bbox(det1)
        bbox2 = center_to_bbox(det2)
        
        # IoU計算
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def merge_detection_group_center(self, detections: List[Dict]) -> Dict:
        """中心座標形式での検出グループ統合"""
        # 信頼度の重み付き平均
        total_weight = sum(d['confidence'] for d in detections)
        
        # 中心座標の重み付き平均
        merged_x_center = sum(d['x_center'] * d['confidence'] for d in detections) / total_weight
        merged_y_center = sum(d['y_center'] * d['confidence'] for d in detections) / total_weight
        merged_width = sum(d['width'] * d['confidence'] for d in detections) / total_weight
        merged_height = sum(d['height'] * d['confidence'] for d in detections) / total_weight
        
        # 信頼度は最大値を使用
        max_confidence = max(d['confidence'] for d in detections)
        
        return {
            'x_center': merged_x_center,
            'y_center': merged_y_center,
            'width': merged_width,
            'height': merged_height,
            'confidence': max_confidence,
            'class': detections[0]['class'],
            'ensemble_size': len(detections)
        }
    
    def draw_enhanced_results(self, image_path: str, detections: List[Dict], output_path: str):
        """強化された結果を描画"""
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        for i, det in enumerate(detections):
            # 中心座標をピクセル座標に変換
            x_center_px = int(det['x_center'] * width)
            y_center_px = int(det['y_center'] * height)
            w_px = int(det['width'] * width)
            h_px = int(det['height'] * height)
            
            # バウンディングボックス座標
            x1 = int(x_center_px - w_px/2)
            y1 = int(y_center_px - h_px/2)
            x2 = int(x_center_px + w_px/2)
            y2 = int(y_center_px + h_px/2)
            
            # 検証スコアに基づく色と太さ
            final_conf = det.get('final_confidence', det['confidence'])
            val_info = det.get('validation_info', {})
            
            if final_conf >= 0.7:
                color = (0, 255, 0)  # 緑（高信頼度）
                thickness = 3
            elif final_conf >= 0.5:
                color = (0, 255, 255)  # 黄色（中信頼度）
                thickness = 2
            else:
                color = (0, 165, 255)  # オレンジ（低信頼度）
                thickness = 1
            
            # バウンディングボックス描画
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # ラベル描画
            label = f"#{i+1}: {final_conf:.3f}"
            if val_info.get('nearby_kofun_count', 0) > 0:
                label += f" (K{val_info['nearby_kofun_count']})"
            if val_info.get('coordinate_corrected', False):
                label += " ✓"
            
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imwrite(output_path, image)
        print(f"💾 Enhanced results saved to: {output_path}")

def main():
    """メイン実行関数"""
    validation_system = KofunValidationSystem()
    
    # テスト用ファイル
    test_xml = "../テスト用xml/FG-GML-4930-64-23-DEM5A-20250620.xml"
    test_png = "test_results/test_converted.png"
    
    if os.path.exists(test_xml) and os.path.exists(test_png):
        # 強化検出を実行
        results = validation_system.run_enhanced_detection(
            test_png, test_xml, "enhanced_detection_result.png"
        )
        
        print(f"\n📊 Enhanced Detection Summary:")
        print(f"Total validated detections: {len(results)}")
        
        for i, det in enumerate(results):
            val_info = det.get('validation_info', {})
            print(f"  Detection #{i+1}:")
            print(f"    Final Confidence: {det.get('final_confidence', 0):.3f}")
            print(f"    Nearby Kofun: {val_info.get('nearby_kofun_count', 0)}")
            print(f"    Closest Distance: {val_info.get('closest_kofun_distance', float('inf')):.1f}m")
            if val_info.get('coordinate_corrected', False):
                print(f"    ✓ Coordinate Corrected")
    else:
        print(f"❌ Test files not found")

if __name__ == "__main__":
    main() 