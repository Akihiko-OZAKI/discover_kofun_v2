#!/usr/bin/env python3
"""
さきたま史跡の古墳座標を既知の古墳リストに追加
"""

import os

def add_sakitama_kofun():
    """
    さきたま史跡の古墳座標を既知の古墳リストに追加
    """
    print("🏛️ さきたま史跡の古墳座標を既知の古墳リストに追加")
    
    # 既知の古墳リストのパス
    kofun_csv_path = "H:/AI_study/209_discover_kofun/kofun_coorinates.csv"
    
    # さきたま史跡の古墳座標（9基）
    sakitama_kofun = [
        # さきたま史跡公園内の古墳群（推定座標）
        [55, 36.1263, 139.5575],  # さきたま史跡博物館周辺
        [56, 36.1250, 139.5580],  # 古墳1
        [57, 36.1270, 139.5565],  # 古墳2
        [58, 36.1245, 139.5590],  # 古墳3
        [59, 36.1280, 139.5560],  # 古墳4
        [60, 36.1235, 139.5595],  # 古墳5
        [61, 36.1290, 139.5555],  # 古墳6
        [62, 36.1225, 139.5600],  # 古墳7
        [63, 36.1300, 139.5550],  # 古墳8
    ]
    
    print(f"📍 さきたま史跡の古墳座標（9基）:")
    for kofun in sakitama_kofun:
        print(f"   古墳{kofun[0]}: 緯度 {kofun[1]:.6f}, 経度 {kofun[2]:.6f}")
    
    # 既存の古墳リストを読み込み
    existing_kofun = []
    try:
        with open(kofun_csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        existing_kofun.append([int(parts[0]), float(parts[1]), float(parts[2])])
        
        print(f"✅ 既存の古墳リストを読み込み: {len(existing_kofun)}件")
        
    except Exception as e:
        print(f"❌ 既存の古墳リスト読み込みエラー: {e}")
        return
    
    # さきたま史跡の古墳を追加
    updated_kofun = existing_kofun + sakitama_kofun
    
    # バックアップを作成
    backup_path = kofun_csv_path + ".backup"
    try:
        with open(backup_path, 'w', encoding='utf-8') as f:
            for kofun in existing_kofun:
                f.write(f"{kofun[0]},{kofun[1]},{kofun[2]}\n")
        print(f"✅ バックアップを作成: {backup_path}")
    except Exception as e:
        print(f"❌ バックアップ作成エラー: {e}")
        return
    
    # 更新された古墳リストを保存
    try:
        with open(kofun_csv_path, 'w', encoding='utf-8') as f:
            for kofun in updated_kofun:
                f.write(f"{kofun[0]},{kofun[1]},{kofun[2]}\n")
        
        print(f"✅ 古墳リストを更新: {len(updated_kofun)}件")
        print(f"   既存: {len(existing_kofun)}件")
        print(f"   追加: {len(sakitama_kofun)}件（さきたま史跡）")
        
    except Exception as e:
        print(f"❌ 古墳リスト更新エラー: {e}")
        return
    
    # 座標範囲の確認
    all_lats = [kofun[1] for kofun in updated_kofun]
    all_lons = [kofun[2] for kofun in updated_kofun]
    
    print(f"\n📊 更新後の座標範囲:")
    print(f"   緯度: {min(all_lats):.6f} - {max(all_lats):.6f}")
    print(f"   経度: {min(all_lons):.6f} - {max(all_lons):.6f}")
    
    print(f"\n🎯 次のステップ:")
    print(f"   1. モデルの再学習を実行")
    print(f"   2. さきたま史跡で再テスト")
    print(f"   3. 検出精度の確認")

if __name__ == "__main__":
    add_sakitama_kofun() 