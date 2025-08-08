
# データセット準備スクリプト
import os
import zipfile
import shutil
from pathlib import Path

def prepare_dataset_for_colab(training_dataset_path="training_dataset", output_zip="kofun_dataset.zip"):
    """学習用データセットをColab用に準備"""
    
    # 出力ディレクトリ作成
    colab_dataset_dir = "colab_dataset"
    os.makedirs(colab_dataset_dir, exist_ok=True)
    
    # 画像とラベルのディレクトリ作成
    train_img_dir = os.path.join(colab_dataset_dir, "images", "train")
    train_label_dir = os.path.join(colab_dataset_dir, "labels", "train")
    val_img_dir = os.path.join(colab_dataset_dir, "images", "val")
    val_label_dir = os.path.join(colab_dataset_dir, "labels", "val")
    
    for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # データセット情報を読み込み
    dataset_info_path = os.path.join(training_dataset_path, "dataset_info.json")
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    # データを分割（80%学習、20%検証）
    total_images = len(dataset_info['image_paths'])
    train_count = int(total_images * 0.8)
    
    # 学習データをコピー
    for i, (img_path, label_path) in enumerate(zip(dataset_info['image_paths'], dataset_info['label_paths'])):
        if i < train_count:
            # 学習データ
            dst_img = os.path.join(train_img_dir, f"train_{i:04d}.png")
            dst_label = os.path.join(train_label_dir, f"train_{i:04d}.txt")
        else:
            # 検証データ
            dst_img = os.path.join(val_img_dir, f"val_{i:04d}.png")
            dst_label = os.path.join(val_label_dir, f"val_{i:04d}.txt")
        
        # ファイルをコピー
        shutil.copy2(img_path, dst_img)
        shutil.copy2(label_path, dst_label)
    
    # ZIPファイルを作成
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(colab_dataset_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, colab_dataset_dir)
                zipf.write(file_path, arcname)
    
    print(f"✅ Colab用データセットを作成しました: {output_zip}")
    print(f"   学習データ: {train_count}件")
    print(f"   検証データ: {total_images - train_count}件")

# 実行
if __name__ == "__main__":
    prepare_dataset_for_colab()
