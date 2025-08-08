#!/usr/bin/env python3
"""
地形データと古墳データの収集を統合管理
"""

import os
import json
from gsi_data_collector import GSIDataCollector
from kofun_data_collector import KofunDataCollector

class DataCollectionManager:
    def __init__(self, output_dir="collected_data"):
        self.output_dir = output_dir
        self.gsi_collector = GSIDataCollector(output_dir)
        self.kofun_collector = KofunDataCollector(output_dir)
        
    def collect_training_data(self, region_config):
        """
        学習用データを一括収集
        """
        print("🚀 学習用データ収集開始")
        
        # 1. 古墳データ収集
        print("\n🏛️ 古墳データ収集...")
        kofun_data = self.kofun_collector.collect_from_cultural_agency()
        
        # 2. 地形データ収集
        print("\n🗺️ 地形データ収集...")
        lat_min = region_config['lat_min']
        lat_max = region_config['lat_max']
        lon_min = region_config['lon_min']
        lon_max = region_config['lon_max']
        
        dem_count = self.gsi_collector.download_range(
            lat_min, lat_max, lon_min, lon_max
        )
        
        # 3. 収集結果のサマリー
        self.create_collection_summary(kofun_data, dem_count, region_config)
        
        return {
            'kofun_count': len(kofun_data),
            'dem_count': dem_count,
            'region': region_config
        }
    
    def create_collection_summary(self, kofun_data, dem_count, region_config):
        """
        収集結果のサマリーを作成
        """
        summary = {
            'collection_date': '2025-08-07',
            'region': region_config,
            'kofun_data': {
                'total_count': len(kofun_data),
                'sources': list(set([k.get('source', 'unknown') for k in kofun_data]))
            },
            'dem_data': {
                'total_count': dem_count,
                'source': 'GSI_DEM5A'
            },
            'training_ready': len(kofun_data) > 0 and dem_count > 0
        }
        
        summary_path = os.path.join(self.output_dir, 'collection_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 収集サマリー保存: {summary_path}")
        print(f"古墳データ: {len(kofun_data)} 件")
        print(f"地形データ: {dem_count} ファイル")
        print(f"学習準備完了: {'✅' if summary['training_ready'] else '❌'}")

def main():
    # 奈良県周辺のデータ収集設定
    nara_config = {
        'name': '奈良県周辺',
        'lat_min': 34.0,
        'lat_max': 35.0,
        'lon_min': 135.5,
        'lon_max': 136.5,
        'description': '古墳密集地域として知られる奈良県周辺'
    }
    
    manager = DataCollectionManager()
    result = manager.collect_training_data(nara_config)
    
    print(f"\n🎯 収集完了!")
    print(f"地域: {result['region']['name']}")
    print(f"古墳データ: {result['kofun_count']} 件")
    print(f"地形データ: {result['dem_count']} ファイル")

if __name__ == "__main__":
    main() 