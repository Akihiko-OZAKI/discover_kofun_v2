import sys
print(sys.executable)
print(sys.path)

# app.py

from flask import Flask, render_template, request, redirect, url_for
from yolo_infer import run_inference
import shutil
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # YOLO推論実行
        detections = run_inference()

        # 結果画像を static にコピー
        src = 'outputs/infer_result.png'
        dst = os.path.join('static', 'infer_result.png')
        shutil.copyfile(src, dst)

        return render_template('index.html', result=True, image_path=dst, detections=detections)

    return render_template('index.html', result=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
