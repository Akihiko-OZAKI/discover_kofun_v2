import sys
import os
sys.path.insert(0, os.path.abspath('yolov5'))  # これでyolov5がパスに入る

from detect import run
from flask import Flask, render_template, request, redirect, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import json

from xml_to_png import convert_xml_to_png
from my_utils import parse_latlon_range, bbox_to_latlon, read_yolo_labels
from enhanced_marking import draw_enhanced_detections, create_matplotlib_visualization
from kofun_validation_system import KofunValidationSystem
from model_optimization import KofunDetectionOptimizer

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
ALLOWED_EXTENSIONS = {'xml'}

# Logging setup
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_detections_on_image(image_path, detections, output_path):
    """
    検出結果を画像に描画（強化版）
    """
    # 新しいマーキング機能を使用
    return draw_enhanced_detections(image_path, detections, output_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect('/upload')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # 古いPNG/JPGファイル削除
    for f in os.listdir(app.config['RESULT_FOLDER']):
        path = os.path.join(app.config['RESULT_FOLDER'], f)
        if os.path.isfile(path) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                os.remove(path)
            except PermissionError:
                print(f"Permission denied: {path}")

    if 'file' not in request.files:
        return render_template('index.html', error='ファイルが選択されていません')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='ファイルが選択されていません')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        xml_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(xml_path)

        try:
            # XML → PNG 変換
            png_path = os.path.join(app.config['RESULT_FOLDER'], 'converted.png')
            convert_xml_to_png(xml_path, png_path)

            # 最適化された検出システムを初期化
            print("🚀 Running optimized detection with enhanced validation...")
            validation_system = KofunValidationSystem()
            optimizer = KofunDetectionOptimizer()
            
            # 最適化された検出を実行
            enhanced_detections = validation_system.run_enhanced_detection(
                png_path, xml_path, 
                os.path.join(app.config['RESULT_FOLDER'], 'enhanced_result.png')
            )
            
            # アンサンブル検出による精度向上
            if enhanced_detections:
                # 画像を読み込んでアンサンブル検出を実行
                img = cv2.imread(png_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                ensemble_detections = optimizer.apply_ensemble_detection(img_rgb)
                
                # 既存の検出結果とアンサンブル結果を統合
                all_detections = enhanced_detections + ensemble_detections
                
                # 重複検出を統合
                final_detections = optimizer.merge_ensemble_detections(all_detections)
            else:
                final_detections = enhanced_detections
            
            # 検出結果を処理
            detections = []
            for det in final_detections:
                detection_info = {
                    'x_center': det['x_center'],
                    'y_center': det['y_center'],
                    'width': det['width'],
                    'height': det['height'],
                    'confidence': det.get('final_confidence', det['confidence']),
                    'validation_info': det.get('validation_info', {}),
                    'optimization_info': {
                        'ensemble_boosted': 'ensemble' in det,
                        'validation_score': det.get('validation_score', 0.0)
                    }
                }
                detections.append(detection_info)

            # 検出結果を画像に描画
            result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result.png')
            draw_detections_on_image(png_path, detections, result_image_path)

            # 座標変換と結果処理
            latlon_range = parse_latlon_range(xml_path)
            processed_results = process_detection_results(xml_path, png_path, detections)

            return render_template('results.html', 
                                results=processed_results,
                                image_path='results/result.png',
                                enhanced_image_path='results/enhanced_result.png',
                                optimization_info={
                                    'total_detections': len(detections),
                                    'ensemble_detections': len([d for d in detections if d.get('optimization_info', {}).get('ensemble_boosted', False)]),
                                    'validation_enhanced': len([d for d in detections if d.get('validation_info')])
                                })

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return render_template('index.html', error=f'処理中にエラーが発生しました: {str(e)}')

    return render_template('index.html', error='無効なファイル形式です')

def process_detection_results(xml_path, png_path, detections=None):
    """
    YOLOv5の推論結果を処理し、緯度経度情報を追加
    """
    result_info = {
        'detections': [],
        'total_detections': 0,
        'has_kofun': False,
        'coordinates': [],
        'debug_info': {}  # デバッグ情報を追加
    }
    
    try:
        # XMLから緯度経度範囲を取得
        lat0, lon0, lat1, lon1 = parse_latlon_range(xml_path)
        result_info['debug_info']['lat_range'] = f"{lat0:.6f} - {lat1:.6f}"
        result_info['debug_info']['lon_range'] = f"{lon0:.6f} - {lon1:.6f}"
        
        if detections is None:
            # テキストファイルから読み込み（フォールバック）
            result_dir = os.path.join(app.config['RESULT_FOLDER'], '')
            txt_files = [f for f in os.listdir(result_dir) if f.endswith('.txt') and f != 'labels.txt']
            result_info['debug_info']['found_txt_files'] = txt_files
            
            if txt_files:
                txt_file = os.path.join(result_dir, txt_files[0])
                detections = read_yolo_labels(txt_file)
            else:
                detections = []
                result_info['debug_info']['error'] = "No detection result files found"
        
        result_info['total_detections'] = len(detections)
        result_info['has_kofun'] = len(detections) > 0
        result_info['debug_info']['raw_detections'] = detections
        
        # 各検出結果に緯度経度を追加
        for i, detection in enumerate(detections):
            lat, lon = bbox_to_latlon(detection, lat0, lon0, lat1, lon1)
            detection['latitude'] = lat
            detection['longitude'] = lon
            detection['detection_id'] = i + 1
            result_info['detections'].append(detection)
            result_info['coordinates'].append({
                'lat': lat,
                'lon': lon,
                'confidence': detection.get('confidence', 0.0)
            })
        
        return result_info
        
    except Exception as e:
        logger.error(f"Error processing detection results: {str(e)}", exc_info=True)
        result_info['debug_info']['error'] = "処理中にエラーが発生しました。詳細はログを確認してください。"
        return result_info

@app.route('/api/detection_results')
def get_detection_results():
    """
    APIエンドポイント：検出結果をJSONで返す
    """
    result_dir = app.config['RESULT_FOLDER']
    result_info = {
        'detections': [],
        'total_detections': 0,
        'has_kofun': False
    }
    
    try:
        txt_files = [f for f in os.listdir(result_dir) if f.endswith('.txt') and f != 'labels.txt']
        if txt_files:
            txt_file = os.path.join(result_dir, txt_files[0])
            detections = read_yolo_labels(txt_file)
            result_info['detections'] = detections
            result_info['total_detections'] = len(detections)
            result_info['has_kofun'] = len(detections) > 0
    except Exception as e:
        logger.error(f"Error getting detection results: {str(e)}", exc_info=True)
        result_info['error'] = "検出結果の取得中にエラーが発生しました。"
    
    return jsonify(result_info)

if __name__ == '__main__':
    app.run(debug=True)
