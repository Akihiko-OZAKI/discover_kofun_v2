#!/usr/bin/env python3
"""
古墳の緯度経度データを収集するスクリプト
"""

import requests
import json
import csv
import os
from typing import List, Dict

class KofunDataCollector:
    def __init__(self, output_dir="collected_data"):
        self.output_dir = output_dir
        self.kofun_dir = os.path.join(output_dir, "kofun_data")
        os.makedirs(self.kofun_dir, exist_ok=True)
    
    def collect_from_cultural_agency(self, prefecture: str = None):
        """
        文化庁データベースから古墳情報を収集
        """
        # 文化庁の遺跡データベースAPI（例）
        base_url = "https://bunka.nii.ac.jp/api/v1/heritages"
        
        kofun_data = []
        
        try:
            # 古墳関連のキーワードで検索
            params = {
                'keyword': '古墳',
                'type': 'burial_mound',
                'limit': 1000
            }
            
            if prefecture:
                params['prefecture'] = prefecture
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for item in data.get('items', []):
                if '古墳' in item.get('name', ''):
                    kofun_info = {
                        'name': item.get('name', ''),
                        'latitude': item.get('latitude'),
                        'longitude': item.get('longitude'),
                        'prefecture': item.get('prefecture', ''),
                        'municipality': item.get('municipality', ''),
                        'period': item.get('period', ''),
                        'source': 'cultural_agency'
                    }
                    kofun_data.append(kofun_info)
            
            print(f"文化庁から {len(kofun_data)} 件の古墳データを収集")
            
        except Exception as e:
            print(f"文化庁データ収集エラー: {e}")
        
        return kofun_data
    
    def collect_from_local_government(self, prefecture: str):
        """
        地方自治体の遺跡台帳から古墳情報を収集
        """
        # 各自治体のオープンデータAPI（例）
        kofun_data = []
        
        # 奈良県の例
        if prefecture == "奈良県":
            try:
                # 奈良県遺跡台帳API
                url = "https://www.pref.nara.jp/opendata/api/archaeological_sites"
                response = requests.get(url)
                response.raise_for_status()
                
                data = response.json()
                
                for site in data:
                    if '古墳' in site.get('site_type', ''):
                        kofun_info = {
                            'name': site.get('site_name', ''),
                            'latitude': site.get('lat'),
                            'longitude': site.get('lon'),
                            'prefecture': '奈良県',
                            'municipality': site.get('municipality', ''),
                            'period': site.get('period', ''),
                            'source': 'nara_prefecture'
                        }
                        kofun_data.append(kofun_info)
                
                print(f"奈良県から {len(kofun_data)} 件の古墳データを収集")
                
            except Exception as e:
                print(f"奈良県データ収集エラー: {e}")
        
        return kofun_data
    
    def save_to_csv(self, kofun_data: List[Dict], filename: str):
        """
        古墳データをCSVファイルに保存
        """
        output_path = os.path.join(self.kofun_dir, filename)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['name', 'latitude', 'longitude', 'prefecture', 
                         'municipality', 'period', 'source']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for kofun in kofun_data:
                writer.writerow(kofun)
        
        print(f"✅ 古墳データを保存: {output_path}")
    
    def save_to_json(self, kofun_data: List[Dict], filename: str):
        """
        古墳データをJSONファイルに保存
        """
        output_path = os.path.join(self.kofun_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(kofun_data, jsonfile, ensure_ascii=False, indent=2)
        
        print(f"✅ 古墳データを保存: {output_path}")

def main():
    collector = KofunDataCollector()
    
    print("🏛️ 古墳データ収集開始")
    
    # 文化庁から収集
    cultural_data = collector.collect_from_cultural_agency()
    
    # 奈良県から収集
    nara_data = collector.collect_from_local_government("奈良県")
    
    # データを統合
    all_kofun_data = cultural_data + nara_data
    
    # 重複除去
    unique_kofun = []
    seen_coords = set()
    
    for kofun in all_kofun_data:
        coord = (kofun.get('latitude'), kofun.get('longitude'))
        if coord not in seen_coords and coord[0] and coord[1]:
            unique_kofun.append(kofun)
            seen_coords.add(coord)
    
    print(f"重複除去後: {len(unique_kofun)} 件の古墳データ")
    
    # ファイルに保存
    collector.save_to_csv(unique_kofun, "kofun_database.csv")
    collector.save_to_json(unique_kofun, "kofun_database.json")

if __name__ == "__main__":
    main() 