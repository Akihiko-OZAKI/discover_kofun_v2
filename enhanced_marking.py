#!/usr/bin/env python3
"""
地形図画像に古墳候補を美しくマーキングする機能
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_enhanced_detections(image_path, detections, output_path, xml_path=None):
    """
    地形図画像に古墳候補を美しくマーキング
    """
    # 画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return False
    
    # BGR to RGB 変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # PIL Imageに変換
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # フォント設定（日本語対応）
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    height, width = image.shape[:2]
    
    # 検出結果を描画
    for i, detection in enumerate(detections):
        x_center = detection['x_center']
        y_center = detection['y_center']
        w = detection['width']
        h = detection['height']
        confidence = detection.get('confidence', 0.0)
        
        # 座標をピクセルに変換
        x_center_px = int(x_center * width)
        y_center_px = int(y_center * height)
        w_px = int(w * width)
        h_px = int(h * height)
        
        # バウンディングボックスの座標を計算
        x1 = int(x_center_px - w_px/2)
        y1 = int(y_center_px - h_px/2)
        x2 = int(x_center_px + w_px/2)
        y2 = int(y_center_px + h_px/2)
        
        # 信頼度に基づいて色を決定
        if confidence >= 0.3:
            color = (255, 0, 0)  # 赤（高信頼度）
            thickness = 3
        elif confidence >= 0.2:
            color = (255, 165, 0)  # オレンジ（中信頼度）
            thickness = 2
        else:
            color = (255, 255, 0)  # 黄色（低信頼度）
            thickness = 1
        
        # 矩形を描画
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        # 信頼度テキストを描画
        conf_text = f"#{i+1}: {confidence:.3f}"
        text_bbox = draw.textbbox((0, 0), conf_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # テキスト背景を描画
        text_bg_x1 = x1
        text_bg_y1 = y1 - text_height - 5
        text_bg_x2 = x1 + text_width + 10
        text_bg_y2 = y1 - 5
        
        draw.rectangle([text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2], 
                      fill=color, outline=color)
        
        # テキストを描画
        draw.text((x1 + 5, y1 - text_height - 5), conf_text, 
                 fill=(255, 255, 255), font=font)
    
    # 凡例を追加
    legend_y = 30
    legend_items = [
        ("高信頼度 (≥30%)", (255, 0, 0)),
        ("中信頼度 (20-30%)", (255, 165, 0)),
        ("低信頼度 (<20%)", (255, 255, 0))
    ]
    
    for item_text, item_color in legend_items:
        # 凡例の背景
        text_bbox = draw.textbbox((0, 0), item_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        legend_bg_x1 = 20
        legend_bg_y1 = legend_y - text_height - 5
        legend_bg_x2 = 20 + text_width + 20
        legend_bg_y2 = legend_y + 5
        
        draw.rectangle([legend_bg_x1, legend_bg_y1, legend_bg_x2, legend_bg_y2], 
                      fill=(0, 0, 0, 128), outline=item_color)
        
        # 凡例のテキスト
        draw.text((25, legend_y - text_height - 5), item_text, 
                 fill=(255, 255, 255), font=font)
        
        legend_y += 30
    
    # 統計情報を追加
    if detections:
        high_conf = len([d for d in detections if d.get('confidence', 0) >= 0.3])
        medium_conf = len([d for d in detections if 0.2 <= d.get('confidence', 0) < 0.3])
        low_conf = len([d for d in detections if d.get('confidence', 0) < 0.2])
        
        stats_text = f"検出結果: 高信頼度 {high_conf}個, 中信頼度 {medium_conf}個, 低信頼度 {low_conf}個"
        text_bbox = draw.textbbox((0, 0), stats_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 統計情報の背景
        stats_bg_x1 = width - text_width - 30
        stats_bg_y1 = 30 - text_height - 5
        stats_bg_x2 = width - 10
        stats_bg_y2 = 30 + 5
        
        draw.rectangle([stats_bg_x1, stats_bg_y1, stats_bg_x2, stats_bg_y2], 
                      fill=(0, 0, 0, 128), outline=(255, 255, 255))
        
        # 統計情報のテキスト
        draw.text((width - text_width - 25, 30 - text_height - 5), stats_text, 
                 fill=(255, 255, 255), font=font)
    
    # 結果を保存
    pil_image.save(output_path)
    print(f"✅ Enhanced marking saved to: {output_path}")
    return True

def create_matplotlib_visualization(image_path, detections, output_path):
    """
    matplotlibを使用したより詳細な可視化
    """
    # 画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return False
    
    # BGR to RGB 変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # プロット作成
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    
    height, width = image.shape[:2]
    
    # 検出結果を描画
    for i, detection in enumerate(detections):
        x_center = detection['x_center']
        y_center = detection['y_center']
        w = detection['width']
        h = detection['height']
        confidence = detection.get('confidence', 0.0)
        
        # 座標をピクセルに変換
        x_center_px = x_center * width
        y_center_px = y_center * height
        w_px = w * width
        h_px = h * height
        
        # バウンディングボックスの座標を計算
        x1 = x_center_px - w_px/2
        y1 = y_center_px - h_px/2
        
        # 信頼度に基づいて色を決定
        if confidence >= 0.05:
            color = 'red'
            linewidth = 3
            alpha = 0.8
        elif confidence >= 0.03:
            color = 'orange'
            linewidth = 2
            alpha = 0.6
        else:
            color = 'yellow'
            linewidth = 1
            alpha = 0.4
        
        # 矩形を描画
        rect = patches.Rectangle((x1, y1), w_px, h_px, 
                               linewidth=linewidth, edgecolor=color, 
                               facecolor='none', alpha=alpha)
        ax.add_patch(rect)
        
        # ラベルを追加
        ax.text(x1, y1-5, f"#{i+1}: {confidence:.3f}", 
               color=color, fontsize=8, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    # 凡例を追加
    legend_elements = [
        patches.Patch(color='red', label='高信頼度 (≥5%)'),
        patches.Patch(color='orange', label='中信頼度 (3-5%)'),
        patches.Patch(color='yellow', label='低信頼度 (<3%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title('古墳候補検出結果', fontsize=16, weight='bold')
    ax.axis('off')
    
    # 結果を保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Matplotlib visualization saved to: {output_path}")
    return True

if __name__ == "__main__":
    # テスト用
    test_image = "second_test_results/second_test.png"
    test_detections = [
        {'x_center': 0.5, 'y_center': 0.5, 'width': 0.1, 'height': 0.1, 'confidence': 0.08},
        {'x_center': 0.3, 'y_center': 0.7, 'width': 0.08, 'height': 0.08, 'confidence': 0.04},
        {'x_center': 0.7, 'y_center': 0.3, 'width': 0.12, 'height': 0.12, 'confidence': 0.02}
    ]
    
    if os.path.exists(test_image):
        # PIL版のマーキング
        enhanced_output = "enhanced_marking.png"
        draw_enhanced_detections(test_image, test_detections, enhanced_output)
        
        # matplotlib版の可視化
        matplotlib_output = "matplotlib_visualization.png"
        create_matplotlib_visualization(test_image, test_detections, matplotlib_output)
    else:
        print(f"❌ Test image not found: {test_image}") 