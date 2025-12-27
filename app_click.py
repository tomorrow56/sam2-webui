import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import requests
import os
import base64
from io import BytesIO

# ページ設定
st.set_page_config(
    page_title="SAM 2 Click Segmentation",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 SAM 2: Click to Segment")
st.markdown("画像上をクリックして、その位置のオブジェクトをセグメンテーションします！")

# サイドバー設定
st.sidebar.header("設定")

# デバイス選択
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"使用デバイス: {device}")

def download_model():
    """モデルをダウンロードする関数"""
    model_name = "sam2.1_hiera_small.pt"
    model_path = f"checkpoints/{model_name}"
    
    if not os.path.exists(model_path):
        os.makedirs("checkpoints", exist_ok=True)
        
        with st.spinner(f"SAM 2.1モデルをダウンロード中..."):
            url = f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{model_name}"
            response = requests.get(url, stream=True)
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        st.success("モデルのダウンロードが完了しました！")
    
    return model_path

def load_sam2_model():
    """SAM 2モデルを読み込む関数"""
    try:
        model_path = download_model()
        
        # SAM 2のインポート
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", model_path, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        return predictor
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {e}")
        return None

def image_to_base64(image):
    """画像をbase64に変換"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def main():
    # モデル読み込み
    if 'predictor' not in st.session_state:
        with st.spinner("SAM 2モデルを読み込み中..."):
            st.session_state.predictor = load_sam2_model()
    
    if st.session_state.predictor is None:
        st.error("モデルの読み込みに失敗しました。")
        return
    
    # 画像アップロード
    uploaded_file = st.file_uploader(
        "画像をアップロードしてください",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        # 画像の読み込みと表示
        image = Image.open(uploaded_file)
        
        # RGBAをRGBに変換
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        # セッション状態に画像を保存
        st.session_state.image_array = image_array
        
        # 画像のサイズを取得
        height, width = image_array.shape[:2]
        
        # 画像をbase64に変換
        img_base64 = image_to_base64(image)
        
        # HTMLとJavaScriptでクリックイベントを処理
        st.subheader("クリックしてオブジェクトを選択")
        
        # 座標入力用のテキストフィールド（一意のIDを付与）
        coords_input = st.text_input("クリック位置（手動入力）", placeholder="X,Y形式で入力（例: 100,200）", key="coords_input", help="画像上をクリックすると自動で入力されます")
        
        # クリック位置を表示するHTML
        html_code = f"""
        <div style="position: relative; display: inline-block;">
            <img id="clickable-image" src="data:image/png;base64,{img_base64}" 
                 style="max-width: 100%; cursor: crosshair;">
            <div id="click-marker" style="position: absolute; width: 10px; height: 10px; 
                 background-color: red; border-radius: 50%; border: 2px solid white; 
                 display: none; pointer-events: none;"></div>
        </div>
        
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const img = document.getElementById('clickable-image');
            const marker = document.getElementById('click-marker');
            
            img.addEventListener('click', function(e) {{
                const rect = img.getBoundingClientRect();
                const scaleX = img.naturalWidth / rect.width;
                const scaleY = img.naturalHeight / rect.height;
                
                const x = Math.round((e.clientX - rect.left) * scaleX);
                const y = Math.round((e.clientY - rect.top) * scaleY);
                
                // マーカーを表示
                marker.style.left = (e.clientX - rect.left - 5) + 'px';
                marker.style.top = (e.clientY - rect.top - 5) + 'px';
                marker.style.display = 'block';
                
                // Streamlitの入力フィールドを見つけて値を設定
                const inputs = window.parent.document.querySelectorAll('input[data-testid="stTextInput"]');
                for (let input of inputs) {{
                    if (input.placeholder && input.placeholder.includes('X,Y形式')) {{
                        input.value = x + ',' + y;
                        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        console.log('Set coords to:', x + ',' + y);
                        break;
                    }}
                }}
                
                console.log('Clicked at:', x, y);
            }});
        }});
        </script>
        """
        
        st.components.v1.html(html_code, height=600)
        
        # 座標の解析
        x_coord = None
        y_coord = None
        
        if coords_input:
            try:
                parts = coords_input.split(',')
                if len(parts) == 2:
                    x_coord = int(parts[0].strip())
                    y_coord = int(parts[1].strip())
                    
                    # 座標の範囲チェック
                    x_coord = max(0, min(x_coord, width - 1))
                    y_coord = max(0, min(y_coord, height - 1))
                    
                    st.write(f"クリック位置: X={x_coord}, Y={y_coord}")
            except ValueError:
                st.error("座標の形式が正しくありません。X,Y形式で入力してください。")
        
        if x_coord is not None and y_coord is not None:
            # セグメンテーション実行ボタン
            if st.button("この位置でセグメンテーション"):
                with st.spinner("セグメンテーションを実行中..."):
                    try:
                        # 画像を予測器に設定
                        st.session_state.predictor.set_image(st.session_state.image_array)
                        
                        # クリック位置でセグメンテーション
                        point_coords = np.array([[x_coord, y_coord]])
                        point_labels = np.array([1])  # 前景ポイント
                        
                        masks, scores, logits = st.session_state.predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=True
                        )
                        
                        # 結果をセッション状態に保存
                        st.session_state.masks = masks
                        st.session_state.scores = scores
                        st.session_state.click_coords = (x_coord, y_coord)
                        
                    except Exception as e:
                        st.error(f"セグメンテーション中にエラーが発生しました: {e}")
        
        # 手動入力も残す
        st.subheader("または手動で座標を入力")
        col1, col2 = st.columns(2)
        with col1:
            manual_x = st.number_input("X座標", min_value=0, max_value=width, value=width//2, key="manual_x")
        with col2:
            manual_y = st.number_input("Y座標", min_value=0, max_value=height, value=height//2, key="manual_y")
        
        if st.button("手動入力でセグメンテーション"):
            with st.spinner("セグメンテーションを実行中..."):
                try:
                    # 画像を予測器に設定
                    st.session_state.predictor.set_image(st.session_state.image_array)
                    
                    # クリック位置でセグメンテーション
                    point_coords = np.array([[manual_x, manual_y]])
                    point_labels = np.array([1])  # 前景ポイント
                    
                    masks, scores, logits = st.session_state.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True
                    )
                    
                    # 結果をセッション状態に保存
                    st.session_state.masks = masks
                    st.session_state.scores = scores
                    st.session_state.click_coords = (manual_x, manual_y)
                    
                except Exception as e:
                    st.error(f"セグメンテーション中にエラーが発生しました: {e}")
        
        # 結果表示
        if 'masks' in st.session_state:
            st.subheader("セグメンテーション結果")
            
            masks = st.session_state.masks
            scores = st.session_state.scores
            
            if len(masks) > 0:
                # 3カラムで結果を表示
                col_list = st.columns(3)
                
                for i in range(min(3, len(masks))):
                    mask = masks[i]
                    score = scores[i]
                    
                    mask_bool = mask.astype(bool)
                    result_image = st.session_state.image_array.copy()
                    overlay = st.session_state.image_array.copy()
                    overlay[mask_bool] = [255, 100, 100]  # 薄い赤
                    
                    alpha = 0.4
                    result = cv2.addWeighted(overlay, alpha, st.session_state.image_array, 1-alpha, 0)
                    
                    # 各カラムに画像を表示
                    col_list[i].image(result, use_column_width=True)
                    col_list[i].write(f"スコア: {score:.3f}")
                    
                    if i == 0:  # 最初のマスクをダウンロード
                        # 切り抜き画像の作成
                        original_image = st.session_state.image_array.copy()
                        cutout = np.zeros_like(original_image)
                        cutout[mask_bool] = original_image[mask_bool]
                        
                        if np.any(mask_bool):
                            rows = np.any(mask_bool, axis=1)
                            cols_check = np.any(mask_bool, axis=0)
                            ymin, ymax = np.where(rows)[0][[0, -1]]
                            xmin, xmax = np.where(cols_check)[0][[0, -1]]
                            
                            padding = 10
                            ymin = max(0, ymin - padding)
                            ymax = min(original_image.shape[0], ymax + padding)
                            xmin = max(0, xmin - padding)
                            xmax = min(original_image.shape[1], xmax + padding)
                            
                            final_cutout = cutout[ymin:ymax, xmin:xmax]
                            cutout_pil = Image.fromarray(final_cutout)
                            
                            col_list[i].download_button(
                                label="切り抜き画像をダウンロード",
                                data=cutout_pil.tobytes(),
                                file_name="segmented_object.png",
                                mime="image/png"
                            )

if __name__ == "__main__":
    main()
