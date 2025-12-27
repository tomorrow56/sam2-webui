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
from streamlit_javascript import st_javascript

# ページ設定
st.set_page_config(
    page_title="SAM 2 Click Segmentation",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 SAM 2: Click Segmentation")
st.markdown("画像をアップロードして、クリック位置のオブジェクトをセグメンテーションします！")

# サイドバー設定
st.sidebar.header("設定")

# デバイス選択
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"使用デバイス: {device}")

def download_model():
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
    try:
        model_path = download_model()
        
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", model_path, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        return predictor
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {e}")
        return None

def image_to_base64(image):
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
    
    # セッション状態の初期化
    if 'click_x' not in st.session_state:
        st.session_state.click_x = None
    if 'click_y' not in st.session_state:
        st.session_state.click_y = None
    
    # 画像アップロード
    uploaded_file = st.file_uploader(
        "画像をアップロードしてください",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        # 画像の読み込み
        image = Image.open(uploaded_file)
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        st.session_state.image_array = image_array
        
        height, width = image_array.shape[:2]
        img_base64 = image_to_base64(image)
        
        # JavaScriptでクリック座標を取得
        click_result = st_javascript(f"""
        (function() {{
            // クリック座標を保存する変数
            if (!window.clickCoords) {{
                window.clickCoords = null;
            }}
            
            // 既存のクリックハンドラを削除
            if (window.clickHandler) {{
                document.removeEventListener('click', window.clickHandler);
            }}
            
            // クリックハンドラを設定
            window.clickHandler = function(e) {{
                const img = e.target;
                if (img.classList.contains('clickable-seg-image')) {{
                    const rect = img.getBoundingClientRect();
                    const x = Math.round((e.clientX - rect.left) * (img.naturalWidth / rect.width));
                    const y = Math.round((e.clientY - rect.top) * (img.naturalHeight / rect.height));
                    
                    const finalX = Math.max(0, Math.min({width} - 1, x));
                    const finalY = Math.max(0, Math.min({height} - 1, y));
                    
                    window.clickCoords = {{x: finalX, y: finalY}};
                    console.log('クリック座標:', finalX, finalY);
                    
                    // 座標表示を更新
                    const infoDiv = document.getElementById('click-coord-info');
                    if (infoDiv) {{
                        infoDiv.innerHTML = '✅ クリック位置: X=' + finalX + ', Y=' + finalY;
                        infoDiv.style.background = 'linear-gradient(135deg, #28a745 0%, #20c997 100%)';
                    }}
                }}
            }};
            
            document.addEventListener('click', window.clickHandler);
            
            return window.clickCoords;
        }})();
        """)
        
        # 2カラムレイアウト
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 画像をクリックして座標を指定")
            
            # クリック可能な画像を表示
            st.markdown("""
            <style>
            .clickable-seg-image {
                cursor: crosshair;
                border: 3px solid #007bff;
                border-radius: 8px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 8px rgba(0,123,255,0.2);
                max-width: 100%;
            }
            .clickable-seg-image:hover {
                border-color: #0056b3;
                box-shadow: 0 6px 12px rgba(0,123,255,0.4);
            }
            #click-coord-info {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                text-align: center;
                margin-top: 10px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <img src="data:image/png;base64,{img_base64}" 
                 class="clickable-seg-image" 
                 style="max-width: 100%; height: auto;">
            <div id="click-coord-info">
                🎯 画像をクリックして座標を設定
            </div>
            """, unsafe_allow_html=True)
            
            # クリック座標の取得と表示
            if click_result and isinstance(click_result, dict):
                st.session_state.click_x = click_result.get('x')
                st.session_state.click_y = click_result.get('y')
            
            # 座標入力（手動入力も可能）
            st.markdown("---")
            st.write("### 📍 座標入力（手動でも設定可能）")
            
            col_x, col_y = st.columns(2)
            with col_x:
                default_x = st.session_state.click_x if st.session_state.click_x is not None else width // 2
                x_coord = st.number_input("X座標", min_value=0, max_value=width, value=default_x, key="input_x")
            with col_y:
                default_y = st.session_state.click_y if st.session_state.click_y is not None else height // 2
                y_coord = st.number_input("Y座標", min_value=0, max_value=height, value=default_y, key="input_y")
            
            # 座標情報の表示
            st.info(f"🎯 現在の座標: X={x_coord}, Y={y_coord}")
            
            # 画像上にクリック位置を表示
            st.write("### 📊 クリック位置の確認")
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(image_array)
            ax.plot(x_coord, y_coord, 'ro', markersize=15, markeredgecolor='yellow', markeredgewidth=3)
            ax.set_title(f"Segmentation Point: ({x_coord}, {y_coord})", fontsize=14, fontweight='bold')
            ax.axis('off')
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
            # セグメンテーションボタン
            st.markdown("---")
            st.write("### 🚀 セグメンテーション実行")
            st.write(f"📍 実行座標: X={x_coord}, Y={y_coord} | 📏 画像サイズ: {width} x {height}")
            
            if st.button("🎯 この位置でセグメンテーション実行", type="primary", use_container_width=True):
                with st.spinner("セグメンテーションを実行中..."):
                    try:
                        st.session_state.predictor.set_image(st.session_state.image_array)
                        point_coords = np.array([[int(x_coord), int(y_coord)]])
                        point_labels = np.array([1])
                        
                        masks, scores, logits = st.session_state.predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=True
                        )
                        
                        st.session_state.masks = masks
                        st.session_state.scores = scores
                        st.session_state.seg_coords = (int(x_coord), int(y_coord))
                        
                        st.success(f"🎉 セグメンテーション完了！位置: ({int(x_coord)}, {int(y_coord)})")
                        
                    except Exception as e:
                        st.error(f"❌ エラー: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with col2:
            st.subheader("セグメンテーション結果")
            
            if 'masks' in st.session_state:
                masks = st.session_state.masks
                scores = st.session_state.scores
                
                if len(masks) > 0:
                    best_mask_idx = np.argmax(scores)
                    mask = masks[best_mask_idx]
                    mask_bool = mask.astype(bool)
                    
                    result_image = image_array.copy()
                    overlay = image_array.copy()
                    overlay[mask_bool] = [255, 100, 100]
                    
                    alpha = 0.4
                    result = cv2.addWeighted(overlay, alpha, image_array, 1-alpha, 0)
                    
                    st.image(result, use_container_width=True)
                    st.success(f"スコア: {scores[best_mask_idx]:.3f}")
                    
                    # 切り抜き画像の作成
                    st.subheader("切り抜き結果")
                    cutout = np.zeros_like(image_array)
                    cutout[mask_bool] = image_array[mask_bool]
                    
                    if np.any(mask_bool):
                        rows = np.any(mask_bool, axis=1)
                        cols_check = np.any(mask_bool, axis=0)
                        ymin, ymax = np.where(rows)[0][[0, -1]]
                        xmin, xmax = np.where(cols_check)[0][[0, -1]]
                        
                        padding = 10
                        ymin = max(0, ymin - padding)
                        ymax = min(image_array.shape[0], ymax + padding)
                        xmin = max(0, xmin - padding)
                        xmax = min(image_array.shape[1], xmax + padding)
                        
                        final_cutout = cutout[ymin:ymax, xmin:xmax]
                        cutout_pil = Image.fromarray(final_cutout)
                        
                        st.image(cutout_pil, use_container_width=True)
                        
                        # ダウンロードボタン
                        st.download_button(
                            label="切り抜き画像をダウンロード",
                            data=cutout_pil.tobytes(),
                            file_name="segmented_object.png",
                            mime="image/png"
                        )
                else:
                    st.warning("セグメンテーション結果が見つかりませんでした。")
            else:
                st.info("左側で座標を指定して「この位置でセグメンテーション」をクリックしてください。")

if __name__ == "__main__":
    main()
