#!/usr/bin/env python3
"""
Colabダミーラベル作成用セル
"""

def create_dummy_labels_cells():
    """ダミーラベル作成用セルを作成"""
    
    dummy_content = '''# 🏷️ ダミーラベル作成版

## セル4修正版: ダミーラベル作成と学習実行

```python
# GPU対応モデル学習の実行（ダミーラベル作成版）
import torch
import os
import glob
import yaml
import random

# GPU設定確認
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用デバイス: {device}")

# 現在のディレクトリを確認
print(f"現在のディレクトリ: {os.getcwd()}")

# データセットファイルを探す
print("🔍 データセットファイルを検索中...")

# 様々なパスを試す
possible_paths = [
    '../dataset/kofun_dataset.yaml',
    '/content/dataset/kofun_dataset.yaml',
    './dataset/kofun_dataset.yaml',
    'dataset/kofun_dataset.yaml',
    '/content/kofun_dataset_for_colab/dataset/kofun_dataset.yaml',
    '/content/colab_real_dataset/kofun_dataset.yaml'
]

dataset_yaml = None
for path in possible_paths:
    if os.path.exists(path):
        dataset_yaml = path
        print(f"✅ データセットファイル発見: {path}")
        break
    else:
        print(f"❌ 見つかりません: {path}")

# YAMLファイルのパスを修正
if dataset_yaml and os.path.exists(dataset_yaml):
    print("🔧 YAMLファイルのパスを修正中...")
    
    # YAMLファイルを読み込み
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        yaml_content = yaml.safe_load(f)
    
    # パスを現在のディレクトリに修正
    yaml_content['path'] = './dataset'
    
    # 修正されたYAMLファイルを保存
    with open(dataset_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ YAMLファイルを修正しました: {dataset_yaml}")

# ダミーラベルを作成
def create_dummy_labels():
    print("🏷️ ダミーラベルを作成中...")
    
    dataset_dir = os.path.dirname(dataset_yaml)
    labels_dir = os.path.join(dataset_dir, 'labels')
    images_dir = os.path.join(dataset_dir, 'images')
    
    # 画像ファイルを取得
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"📁 画像ファイル数: {len(image_files)}")
    
    for image_file in image_files:
        # ラベルファイル名を作成（拡張子を.txtに変更）
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        # ダミーラベルを作成（各画像に1-3個の古墳を配置）
        num_kofun = random.randint(1, 3)
        dummy_labels = []
        
        for i in range(num_kofun):
            # ランダムな位置とサイズで古墳を配置
            x_center = random.uniform(0.1, 0.9)  # 画像の10%-90%の範囲
            y_center = random.uniform(0.1, 0.9)
            width = random.uniform(0.05, 0.15)   # 5%-15%のサイズ
            height = random.uniform(0.05, 0.15)
            
            # YOLO形式: class_id x_center y_center width height
            dummy_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # ラベルファイルを保存
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\\n'.join(dummy_labels))
        
        print(f"✅ {label_file}: {num_kofun}個の古墳ラベルを作成")
    
    print("🎉 ダミーラベル作成完了！")

# ダミーラベルを作成
if dataset_yaml and os.path.exists(dataset_yaml):
    create_dummy_labels()

# 学習実行
if dataset_yaml and os.path.exists(dataset_yaml):
    print("🚀 学習を開始します...")
    print(f"使用するデータセット: {dataset_yaml}")
    
    # データセットディレクトリの存在確認
    dataset_dir = os.path.dirname(dataset_yaml)
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    
    print(f"画像ディレクトリ: {images_dir} (存在: {os.path.exists(images_dir)})")
    print(f"ラベルディレクトリ: {labels_dir} (存在: {os.path.exists(labels_dir)})")
    
    if os.path.exists(images_dir) and os.path.exists(labels_dir):
        print("✅ データセット構造が正常です")
        
        # ラベルファイルの確認
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        print(f"📁 ラベルファイル数: {len(label_files)}")
        
        # サンプルラベルの表示
        if label_files:
            sample_label = os.path.join(labels_dir, label_files[0])
            with open(sample_label, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"📄 サンプルラベル ({label_files[0]}):")
            print(content)
        
        # CUDA修正版の学習コマンド
        if torch.cuda.is_available():
            # GPU使用時は0を指定
            !python train.py --img 640 --batch 16 --epochs 50 --data {dataset_yaml} --weights yolov5s.pt --cache --device 0
        else:
            # CPU使用時
            !python train.py --img 640 --batch 16 --epochs 50 --data {dataset_yaml} --weights yolov5s.pt --cache --device cpu
        
        print("🎉 学習完了！")
    else:
        print("❌ データセット構造が不完全です")
        print("画像またはラベルディレクトリが見つかりません")
else:
    print("❌ データセットファイルが見つからないため学習をスキップします")
```

## セル5: ラベル確認

```python
# ラベル確認
import os

print("📁 ラベルファイルの確認")

dataset_dir = './dataset'
labels_dir = os.path.join(dataset_dir, 'labels')

if os.path.exists(labels_dir):
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    print(f"ラベルファイル数: {len(label_files)}")
    
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if content:
            lines = content.split('\\n')
            print(f"📄 {label_file}: {len(lines)}個の古墳")
            for i, line in enumerate(lines[:3]):  # 最初の3個を表示
                print(f"  {i+1}: {line}")
            if len(lines) > 3:
                print(f"  ... 他 {len(lines)-3}個")
        else:
            print(f"📄 {label_file}: 空")
else:
    print("❌ ラベルディレクトリが見つかりません")
```

## 使用方法

1. **セル4修正版**を実行してダミーラベルを作成
2. **セル5**でラベル内容を確認
3. 学習が正常に開始されることを確認

## 期待される結果

- ✅ 各画像に1-3個のダミー古墳ラベルが作成される
- ✅ ラベルファイルが空でなくなる
- ✅ 学習が正常に開始される
- ✅ 50エポックの学習が完了する

## 注意点

- これは学習用のダミーデータです
- 実際の古墳検出には、より正確なラベルが必要です
- 学習の基本動作確認用として使用してください
'''
    
    with open("colab_dummy_labels.txt", 'w', encoding='utf-8') as f:
        f.write(dummy_content)
    
    print("🏷️ ダミーラベル作成版を作成しました: colab_dummy_labels.txt")
    print("\n💡 修正内容:")
    print("1. 各画像に1-3個のダミー古墳ラベルを自動生成")
    print("2. YOLO形式のラベルファイルを作成")
    print("3. ラベルファイルが空の問題を解決")
    print("4. 学習の基本動作確認が可能")

if __name__ == "__main__":
    create_dummy_labels_cells() 