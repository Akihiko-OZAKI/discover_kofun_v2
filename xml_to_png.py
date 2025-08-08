import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import tempfile

# GDALとrasterioのインポートを試行
try:
    import rasterio
    from osgeo import gdal, osr
    GDAL_AVAILABLE = True
except ImportError:
    print("Warning: GDAL/rasterio not available. Using fallback method.")
    GDAL_AVAILABLE = False

def parse_dem_xml(xml_file):
    """XMLから標高データと座標範囲を抽出"""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    ns = {'gml': 'http://www.opengis.net/gml/3.2', 'def': 'http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema'}

    # 座標範囲
    lower_corner = root.find('.//gml:lowerCorner', ns).text.split()
    upper_corner = root.find('.//gml:upperCorner', ns).text.split()
    lat_min, lon_min = map(float, lower_corner)
    lat_max, lon_max = map(float, upper_corner)

    # gridサイズ取得（low, high）
    low = root.find('.//gml:low', ns).text.split()
    high = root.find('.//gml:high', ns).text.split()
    cols = int(high[0]) - int(low[0]) + 1
    rows = int(high[1]) - int(low[1]) + 1

    # 標高値の文字列
    tuple_list_text = root.find('.//gml:tupleList', ns).text.strip()
    # 標高値だけ抽出（「地表面,標高」形式を分解）
    elevation_strs = [item.split(',')[1] for item in tuple_list_text.split()]
    elevation_values = np.array([float(val) if val != '-9999.' else np.nan for val in elevation_strs])

    elevation_grid = elevation_values.reshape((rows, cols))

    return lat_min, lon_min, lat_max, lon_max, rows, cols, elevation_grid

def create_geotiff(output_path, elevation_grid, lat_min, lon_min, lat_max, lon_max):
    """標高グリッドをGeoTIFFに変換"""
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL not available")
        
    rows, cols = elevation_grid.shape
    # 解像度計算（緯度経度単位）
    pixel_width = (lon_max - lon_min) / cols
    pixel_height = (lat_max - lat_min) / rows

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)

    # 左上の座標をセット（経度最小、緯度最大）
    dataset.SetGeoTransform([lon_min, pixel_width, 0, lat_max, 0, -pixel_height])

    # 空間参照（WGS84）
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('WGS84')
    dataset.SetProjection(srs.ExportToWkt())

    # バンドにデータ書き込み
    band = dataset.GetRasterBand(1)
    band.WriteArray(elevation_grid)
    band.SetNoDataValue(np.nan)

    dataset.FlushCache()
    dataset = None

def convert_xml_to_png(xml_path, png_path):
    """
    国土地理院DEM XMLを読み込み、標高値に応じたカラーのヒートマップPNGを保存する関数。
    Google Colabのコードを基に実装。

    - xml_path: 入力XMLファイルパス
    - png_path: 出力PNGファイルパス
    """
    try:
        # XMLから標高データと座標範囲を取得
        lat_min, lon_min, lat_max, lon_max, rows, cols, elevation_grid = parse_dem_xml(xml_path)
        
        if GDAL_AVAILABLE:
            # GDALが利用可能な場合：TIFF経由で高品質変換
            try:
                # 一時的なTIFFファイルを作成
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_tiff:
                    tiff_path = tmp_tiff.name
                
                # GeoTIFF作成
                create_geotiff(tiff_path, elevation_grid, lat_min, lon_min, lat_max, lon_max)
                
                # TIFFを読み込んでカラー画像に変換
                with rasterio.open(tiff_path) as src:
                    elevation_data = src.read(1)  # 最初のバンドを読み込み
                    
                    # NaN値を処理
                    elevation_data = np.where(elevation_data == src.nodata, np.nan, elevation_data)
                    
                    # 有効なデータのみで正規化
                    valid_data = elevation_data[~np.isnan(elevation_data)]
                    if len(valid_data) > 0:
                        vmin, vmax = np.nanmin(valid_data), np.nanmax(valid_data)
                    else:
                        vmin, vmax = 0, 1
                    
                    # カラーマップでヒートマップ作成
                    plt.figure(figsize=(10, 10), dpi=100)
                    plt.axis('off')
                    
                    # カラーマップを適用（地形に適したカラーマップ）
                    im = plt.imshow(elevation_data, cmap='terrain', vmin=vmin, vmax=vmax)
                    
                    plt.tight_layout(pad=0)
                    
                    # PNG保存（高品質で保存）
                    plt.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=150, 
                               facecolor='white', edgecolor='none')
                    plt.close()
                
                # 一時ファイルを削除
                os.unlink(tiff_path)
                
                print(f"Successfully converted {xml_path} to {png_path} (GDAL method)")
                return
                
            except Exception as e:
                print(f"GDAL method failed: {e}. Falling back to direct method.")
        
        # GDALが利用できない場合または失敗した場合：直接変換
        fallback_convert_xml_to_png(xml_path, png_path, elevation_grid)
        
    except Exception as e:
        print(f"Error converting {xml_path}: {e}")
        # 最終的なフォールバック
        basic_convert_xml_to_png(xml_path, png_path)

def fallback_convert_xml_to_png(xml_path, png_path, elevation_grid=None):
    """
    改良されたフォールバック版XML→PNG変換
    """
    try:
        if elevation_grid is None:
            # elevation_gridが提供されていない場合は解析
            lat_min, lon_min, lat_max, lon_max, rows, cols, elevation_grid = parse_dem_xml(xml_path)
        
        # NaN値を処理
        elevation_data = np.where(np.isnan(elevation_grid), np.nanmin(elevation_grid), elevation_grid)
        
        # 有効なデータのみで正規化
        valid_data = elevation_data[~np.isnan(elevation_data)]
        if len(valid_data) > 0:
            vmin, vmax = np.nanmin(valid_data), np.nanmax(valid_data)
        else:
            vmin, vmax = 0, 1
        
        # カラーマップでヒートマップ作成
        plt.figure(figsize=(10, 10), dpi=100)
        plt.axis('off')
        
        # カラーマップを適用（地形に適したカラーマップ）
        im = plt.imshow(elevation_data, cmap='terrain', vmin=vmin, vmax=vmax)
        
        plt.tight_layout(pad=0)
        
        # PNG保存（高品質で保存）
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=150, 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Successfully converted {xml_path} to {png_path} (fallback method)")
        
    except Exception as e:
        print(f"Fallback method failed: {e}. Using basic method.")
        basic_convert_xml_to_png(xml_path, png_path)

def basic_convert_xml_to_png(xml_path, png_path):
    """
    元の簡易版XML→PNG変換（最終フォールバック用）
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 標高値抽出（例：<Elevation>タグの値を2D配列で取得する想定）
        elevations = []
        for elevation in root.iter('Elevation'):
            text = elevation.text
            if text is not None:
                elevations.append(float(text))

        # 仮に10x10の標高値として整形（実際はXML構造に合わせて）
        size = int(len(elevations) ** 0.5)
        if size * size != len(elevations):
            # きれいな正方形に整形できない場合は例外
            raise ValueError("Elevationデータ数が正方形になりません。")

        elevation_array = np.array(elevations).reshape((size, size))

        # ヒートマップ作成 matplotlibでカラー化
        plt.figure(figsize=(6,6), dpi=100)
        plt.axis('off')
        plt.imshow(elevation_array, cmap='jet')  # 'jet'はよくあるヒートマップカラー
        plt.tight_layout(pad=0)

        # PNG保存（透明部分なしで保存）
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Successfully converted {xml_path} to {png_path} (basic method)")
        
    except Exception as e:
        print(f"All conversion methods failed: {e}")
        # 最後の手段：空の画像を作成
        create_empty_image(png_path)

def create_empty_image(png_path):
    """
    エラー時の最終手段：空の画像を作成
    """
    try:
        # 空の画像を作成
        img = Image.new('RGB', (400, 400), color='lightgray')
        img.save(png_path)
        print(f"Created empty image at {png_path}")
    except Exception as e:
        print(f"Failed to create empty image: {e}")
