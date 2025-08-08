#!/usr/bin/env python3
"""
超低信頼度閾値での推論テスト
"""

import os
import sys
import torch
import cv2
import numpy as np

def ultra_low_threshold_inference():
    """
    超低信頼度閾値（0.01）で推論を実行
    """
    print("🚀 Running ultra-low threshold inference (0.01)...")
    
    # YOLOv5のインポート
    sys.path.insert(0, os.path.abspath('yolov5'))
    from yolov5.models.common import DetectMultiBackend
    from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
    from yolov5.utils.torch_utils import select_device
    
    # 設定 - 超低閾値
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
    
    # 結果の表示
    print(f"\n🎯 Ultra-Low Threshold Results (conf_thres=0.01):")
    print(f"📊 Total detections: {len(detections)}")
    
    if detections:
        print("✅ DETECTIONS FOUND!")
        for i, det in enumerate(detections):
            print(f"   Detection #{i+1}:")
            print(f"     Class: {det['class_name']}")
            print(f"     Confidence: {det['confidence']:.4f}")
            print(f"     BBox: {det['bbox']}")
    else:
        print("❌ Still no detections even with ultra-low threshold")
        print("💡 This might indicate:")
        print("   1. The model needs retraining")
        print("   2. The image doesn't contain kofun")
        print("   3. Model compatibility issues")
    
    return detections

if __name__ == "__main__":
    if not os.path.exists('second_test_results/second_test.png'):
        print("❌ Please run test_second_xml.py first")
    else:
        detections = ultra_low_threshold_inference()
        
        if detections:
            print("\n🎉 SUCCESS! Detections found with ultra-low threshold!")
            print("This confirms the system is working, but the model might need adjustment.")
        else:
            print("\n⚠️  No detections even with ultra-low threshold.")
            print("Consider testing with known positive samples or model retraining.") 