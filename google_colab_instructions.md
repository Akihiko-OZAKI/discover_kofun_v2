# 🚀 Google Colab 古墳検出モデル学習ガイド

## 📋 準備ファイル

以下のファイルをGoogle Colabにアップロードしてください：

### 1. 学習スクリプト
- `kofun_colab_script.py` - メインの学習スクリプト

### 2. 既知の古墳リスト
- `kofun_coordinates_updated.csv` - 更新された古墳座標（63件）

## 🔧 Google Colabでの実行手順

### ステップ1: 新しいノートブックを作成
1. [Google Colab](https://colab.research.google.com/) にアクセス
2. 「新しいノートブック」をクリック

### ステップ2: ファイルをアップロード
```python
# セル1: ファイルアップロード
from google.colab import files

# 学習スクリプトをアップロード
uploaded = files.upload()
```

### ステップ3: 必要なライブラリをインストール
```python
# セル2: 依存関係のインストール
!pip install opencv-python
!pip install tqdm
!pip install requests
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install pillow
```

### ステップ4: YOLOv5をクローン
```python
# セル3: YOLOv5のセットアップ
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt
%cd ..
```

### ステップ5: 学習スクリプトを実行
```python
# セル4: 学習スクリプトの実行
!python kofun_colab_script.py
```

### ステップ6: 学習の実行
```python
# セル5: YOLOv5学習の実行
%cd yolov5
!python train.py --img 640 --batch 16 --epochs 150 --data ../kofun_dataset_updated.yaml --weights yolov5s.pt --cache --patience 30 --save-period 10 --project runs/train --name kofun_updated --exist-ok
%cd ..
```

### ステップ7: 学習結果の確認
```python
# セル6: 学習結果の確認
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 学習曲線の表示
img = mpimg.imread('yolov5/runs/train/kofun_updated/results.png')
plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.axis('off')
plt.show()

print("✅ 学習完了！")
```

### ステップ8: モデルのダウンロード
```python
# セル7: 学習済みモデルのダウンロード
from google.colab import files

# best.ptをダウンロード
files.download('yolov5/runs/train/kofun_updated/weights/best.pt')
```

## 📊 期待される結果

### データセット情報
- **訓練データ**: 約80%の画像
- **検証データ**: 約20%の画像
- **古墳座標**: 63件（既存54件 + さきたま史跡9件）

### 学習設定
- **エポック数**: 150
- **バッチサイズ**: 16
- **画像サイズ**: 640x640
- **早期停止**: 30エポック（改善なし）

### 収集地域
1. **大阪・奈良周辺**: (34.4, 35.0, 135.4, 136.0)
2. **京都周辺**: (34.8, 35.2, 135.6, 135.9)
3. **堺市周辺**: (34.6, 34.8, 135.7, 135.9)
4. **さきたま史跡周辺**: (36.1, 36.2, 139.4, 139.6) ⭐新規追加

## 🔍 学習後のテスト

学習完了後、以下の手順でさきたま史跡での検出精度をテストしてください：

1. `best.pt`をローカル環境にダウンロード
2. `yolov5/weights/`ディレクトリに配置
3. `ultra_low_threshold_test.py`で再テスト

## ⚠️ 注意事項

- **GPU使用**: Google ColabのGPUを有効にしてください
- **実行時間**: 約2-3時間（データ収集 + 学習）
- **ストレージ**: 十分な空き容量を確保してください
- **ネットワーク**: 安定したインターネット接続が必要です

## 🎯 目標

この学習により、さきたま史跡の既知の古墳（9基）を「古墳あり」として検出できるようになることを目指します。

---

**📞 サポート**: 問題が発生した場合は、エラーメッセージを確認して対処してください。 