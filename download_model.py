import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """
    学習済みモデルをダウンロードする関数
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs('weights', exist_ok=True)
    filepath = os.path.join('weights', filename)
    
    if os.path.exists(filepath):
        print(f"File {filename} already exists. Skipping download.")
        return
    
    progress_bar = tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        desc=f'Downloading {filename}'
    )
    
    with open(filepath, 'wb') as file:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    
    progress_bar.close()

if __name__ == '__main__':
    # モデルのダウンロードURL（要更新）
    MODEL_URL = "https://your-model-storage-url/best.pt"
    download_file(MODEL_URL, 'best.pt')