# 古墳再発見 Webアプリ

国土地理院のGMLデータを元に、標高画像を生成 → YOLOv5で古墳の候補地形を検出する Flask Webアプリです。

## 🔍 機能
- GML (DEM5A) → 標高画像（PNG）自動生成
- YOLOv5 による推論
- 検出結果を赤枠＋緯度経度付きで表示
- Webアプリ経由で結果を可視化

## 🧪 ローカル起動方法

```bash
git clone https://github.com/Akihiko-OZAKI/discover_kofun_v1.git
cd discover_kofun_v1
python -m venv .venv
.venv\Scripts\activate   # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
