#!/usr/bin/env python3
"""
検出結果の詳細分析
"""

import os
import sys
import torch
import cv2
import numpy as np

def analyze_detections():
    """
    検出結果を詳細に分析
    """
    print("🔍 Analyzing detection results in detail...")
    
    # YOLOv5のインポート
    sys.path.insert(0, os.path.abspath('yolov5'))
    from yolov5.models.common import DetectMultiBackend
    from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
    from yolov5.utils.torch_utils import select_device
    
    # 設定
    weights = 'yolov5/weights/best.pt'
    source = 'second_test_results/second_test.png'
    imgsz = (640, 640)
    conf_thres = 0.01  # 超低閾値
    iou_thres = 0.45
    max_det = 1000
    device = select_device('')
    
    # モデルの読み込み
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    
    # ウォームアップ
    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))
    
    # 画像の読み込み
    im0s = cv2.imread(source)
    if im0s is None:
        print(f"❌ Could not read image: {source}")
        return
    
    # 前処理
    im = cv2.resize(im0s, imgsz)
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]
    
    # 推論
    pred = model(im, augment=False, visualize=False)
    
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=max_det)
    
    # 結果の処理
    detections = []
    for i, det in enumerate(pred):
        if len(det):
            # 座標を元の画像サイズにスケール
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
            
            # 検出結果を保存
            for *xyxy, conf, cls in reversed(det):
                detection = {
                    'bbox': [int(x) for x in xyxy],
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': names[int(cls)] if int(cls) < len(names) else 'unknown'
                }
                detections.append(detection)
    
    # 詳細分析
    print(f"\n📊 Detailed Analysis of {len(detections)} detections:")
    
    # 信頼度別の分類
    high_conf = [d for d in detections if d['confidence'] >= 0.05]  # 5%以上
    medium_conf = [d for d in detections if 0.03 <= d['confidence'] < 0.05]  # 3-5%
    low_conf = [d for d in detections if d['confidence'] < 0.03]  # 3%未満
    
    print(f"🔴 High confidence (≥5%): {len(high_conf)} detections")
    print(f"🟡 Medium confidence (3-5%): {len(medium_conf)} detections")
    print(f"🟢 Low confidence (<3%): {len(low_conf)} detections")
    
    # 信頼度の高い検出を表示
    if high_conf:
        print(f"\n🔴 High Confidence Detections (≥5%):")
        for i, det in enumerate(high_conf[:10]):  # 上位10個
            print(f"   #{i+1}: Confidence {det['confidence']:.4f}, BBox {det['bbox']}")
        if len(high_conf) > 10:
            print(f"   ... and {len(high_conf) - 10} more")
    
    # 信頼度の分布
    confidences = [d['confidence'] for d in detections]
    if confidences:
        print(f"\n📈 Confidence Statistics:")
        print(f"   Max: {max(confidences):.4f}")
        print(f"   Min: {min(confidences):.4f}")
        print(f"   Mean: {sum(confidences)/len(confidences):.4f}")
        print(f"   Median: {sorted(confidences)[len(confidences)//2]:.4f}")
    
    # 推奨閾値の提案
    print(f"\n💡 Recommendations:")
    if len(high_conf) > 0:
        print(f"   ✅ Use confidence threshold ≥ 0.05 for reliable detections")
        print(f"   📊 This would give you {len(high_conf)} high-confidence detections")
    elif len(medium_conf) > 0:
        print(f"   ⚠️  Use confidence threshold ≥ 0.03 for moderate detections")
        print(f"   📊 This would give you {len(medium_conf)} medium-confidence detections")
    else:
        print(f"   ❌ No high-confidence detections found")
        print(f"   💡 Consider model retraining or different images")
    
    return detections

if __name__ == "__main__":
    if not os.path.exists('second_test_results/second_test.png'):
        print("❌ Please run test_second_xml.py first")
    else:
        detections = analyze_detections() 