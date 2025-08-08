#!/usr/bin/env python3
"""
超高感度モードテスト - さきたま史跡編
既知の古墳が存在する地域で超高感度モードをテスト
"""

import os
import sys
sys.path.insert(0, os.path.abspath('yolov5'))

import cv2
import numpy as np
from xml_to_png import convert_xml_to_png
from my_utils import parse_latlon_range, bbox_to_latlon
from kofun_validation_system import KofunValidationSystem
from model_optimization import KofunDetectionOptimizer

def test_sakitama_ultra_sensitive():
    """
    さきたま史跡で超高感度モードをテスト
    """
    print("🏛️ さきたま史跡 超高感度モードテスト開始")
    
    # さきたま史跡の座標範囲（確認済み）
    sakitama_lat_range = (36.158333333, 36.166666667)
    sakitama_lon_range = (139.45, 139.4625)
    
    print(f"📍 さきたま史跡座標範囲:")
    print(f"   緯度: {sakitama_lat_range[0]:.6f} - {sakitama_lat_range[1]:.6f}")
    print(f"   経度: {sakitama_lon_range[0]:.6f} - {sakitama_lon_range[1]:.6f}")
    print(f"   📍 既知の古墳: 9基（さきたま古墳群）")
    
    # さきたま史跡のタイルファイルを選択
    sakitama_dir = "static/uploads/sakitama"
    if not os.path.exists(sakitama_dir):
        print(f"❌ さきたま史跡ディレクトリが見つかりません: {sakitama_dir}")
        return
    
    # 複数のタイルをテスト
    test_files = [
        "FG-GML-5439-13-96-DEM5A-20250620.xml",  # 中心部
        "FG-GML-5439-13-97-DEM5A-20250620.xml",  # 隣接タイル
        "FG-GML-5439-13-98-DEM5A-20250620.xml",  # 隣接タイル
        "FG-GML-5439-13-85-DEM5A-20250620.xml",  # 別のタイル
        "FG-GML-5439-13-86-DEM5A-20250620.xml",  # 別のタイル
    ]
    
    total_detections = 0
    
    for test_file in test_files:
        test_xml = os.path.join(sakitama_dir, test_file)
        
        if not os.path.exists(test_xml):
            print(f"⚠️ ファイルが見つかりません: {test_xml}")
            continue
        
        print(f"\n📁 テストファイル: {test_file}")
        
        # 出力ディレクトリの準備
        output_dir = f"sakitama_ultra_test_{test_file.split('.')[0]}"
        os.makedirs(output_dir, exist_ok=True)
        
        # XML → PNG 変換
        png_path = os.path.join(output_dir, f"{test_file.split('.')[0]}_converted.png")
        print("📊 XML → PNG 変換中...")
        
        try:
            convert_xml_to_png(test_xml, png_path)
            print(f"✅ PNG変換完了: {png_path}")
        except Exception as e:
            print(f"❌ PNG変換失敗: {e}")
            continue
        
        # 座標範囲の確認
        try:
            lat0, lon0, lat1, lon1 = parse_latlon_range(test_xml)
            print(f"📍 タイル座標範囲:")
            print(f"   緯度: {lat0:.6f} - {lat1:.6f}")
            print(f"   経度: {lon0:.6f} - {lon1:.6f}")
            
            # さきたま史跡の範囲内かチェック
            if (sakitama_lat_range[0] <= lat0 <= sakitama_lat_range[1] and 
                sakitama_lon_range[0] <= lon0 <= sakitama_lon_range[1]):
                print("✅ さきたま史跡範囲内のタイルです")
            else:
                print("⚠️ さきたま史跡範囲外のタイルの可能性があります")
                
        except Exception as e:
            print(f"❌ 座標解析失敗: {e}")
        
        # 超高感度モードで検出実行
        print("🔍 超高感度モード検出開始...")
        
        try:
            # 検証システムを初期化
            validation_system = KofunValidationSystem()
            optimizer = KofunDetectionOptimizer()
            
            # 超高感度検出実行
            enhanced_detections = validation_system.run_enhanced_detection(
                png_path, test_xml, 
                os.path.join(output_dir, f'{test_file.split(".")[0]}_ultra_result.png')
            )
            
            print(f"🔍 超高感度検出結果: {len(enhanced_detections)} 件")
            
            # アンサンブル検出も実行
            img = cv2.imread(png_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ensemble_detections = optimizer.apply_ensemble_detection(img_rgb)
            
            print(f"🔍 アンサンブル検出結果: {len(ensemble_detections)} 件")
            
            # 結果を統合
            all_detections = enhanced_detections + ensemble_detections
            print(f"🔍 統合検出結果: {len(all_detections)} 件")
            
            total_detections += len(all_detections)
            
            # 結果をファイルに保存
            results_file = os.path.join(output_dir, f"{test_file.split('.')[0]}_detection_results.txt")
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write(f"さきたま史跡 超高感度モード検出結果\n")
                f.write(f"テストファイル: {test_xml}\n")
                f.write(f"座標範囲: 緯度 {lat0:.6f}-{lat1:.6f}, 経度 {lon0:.6f}-{lon1:.6f}\n")
                f.write(f"既知の古墳: 9基（さきたま古墳群）\n")
                f.write(f"検出結果: {len(all_detections)} 件\n\n")
                
                for i, detection in enumerate(all_detections):
                    f.write(f"検出 {i+1}:\n")
                    f.write(f"  信頼度: {detection['confidence']:.4f}\n")
                    f.write(f"  座標: ({detection['bbox'][0]:.1f}, {detection['bbox'][1]:.1f}, {detection['bbox'][2]:.1f}, {detection['bbox'][3]:.1f})\n")
                    
                    # 座標変換
                    try:
                        lat, lon = bbox_to_latlon(detection['bbox'], png_path, test_xml)
                        f.write(f"  緯度経度: ({lat:.6f}, {lon:.6f})\n")
                    except:
                        f.write(f"  緯度経度: 変換エラー\n")
                    f.write("\n")
            
            print(f"📁 結果保存: {results_file}")
            
            # 成功判定
            if len(all_detections) > 0:
                print("🎉 成功！古墳が検出されました！")
                print(f"   検出数: {len(all_detections)} 件")
                break  # 1つでも検出されれば成功
            
        except Exception as e:
            print(f"❌ 検出実行エラー: {e}")
            import traceback
            traceback.print_exc()
    
    # 全体の結果
    print(f"\n📊 全体結果:")
    print(f"   テストしたタイル数: {len(test_files)}")
    print(f"   総検出数: {total_detections} 件")
    
    if total_detections > 0:
        print("🎉 成功！古墳が検出されました！")
        print(f"   既知の古墳: 9基")
        print(f"   検出率: {total_detections/9*100:.1f}%")
    else:
        print("❌ 全てのタイルで検出されませんでした")
        print("   次のステップ:")
        print("   1. 閾値をさらに下げる（0.001以下）")
        print("   2. モデルの再学習を検討")
        print("   3. データセットの見直し")

if __name__ == "__main__":
    test_sakitama_ultra_sensitive() 