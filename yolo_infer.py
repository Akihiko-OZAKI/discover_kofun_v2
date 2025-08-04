# yolo_infer.py

import torch
import cv2
import os

def run_inference(
    image_path='outputs/dem_image.png',
    weights_path='weights/best.pt',
    conf_thres=0.25,
    iou_thres=0.45,
    save_output_path='outputs/infer_result.png',
):
    # YOLOv5のモデルをロード（自動で yolov5s.yaml を推論）
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    model.conf = conf_thres
    model.iou = iou_thres

    # 推論実行
    results = model(image_path)

    # 画像と結果を取得
    img = cv2.imread(image_path)
    detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, cls]

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 赤枠
        cv2.putText(img, f"{model.names[int(cls)]} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 結果画像保存
    cv2.imwrite(save_output_path, img)
    print(f"[INFO] 推論完了: {save_output_path}")

    # 推論結果（バウンディングボックス）を返す
    return detections.tolist()

# 実行テスト
if __name__ == '__main__':
    run_inference()
