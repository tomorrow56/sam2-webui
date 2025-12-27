import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
import requests

# ページ設定
st.set_page_config(
    page_title="SAM 2 Web UI",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 SAM 2: Segment Anything Web UI")
st.markdown("画像をアップロードして、SAM 2でオブジェクトセグメンテーションを試してみましょう！")

# サイドバー設定
st.sidebar.header("設定")

# モデル選択
model_options = {
    "sam2_hiera_small": "sam2_hiera_small.pt",
    "sam2_hiera_base_plus": "sam2_hiera_base_plus.pt", 
    "sam2_hiera_large": "sam2_hiera_large.pt"
}

selected_model = st.sidebar.selectbox(
    "モデルを選択",
    list(model_options.keys()),
    index=0
)

# デバイス選択
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"使用デバイス: {device}")

def download_model(model_name):
    """モデルをダウンロードする関数"""
    model_path = f"checkpoints/{model_options[model_name]}"
    
    if not os.path.exists(model_path):
        os.makedirs("checkpoints", exist_ok=True)
        
        with st.spinner(f"{model_name}をダウンロード中..."):
            url = f"https://dl.fbaipublicfiles.com/segment_anything_2/{model_options[model_name]}"
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        st.success(f"{model_name}のダウンロードが完了しました！")
    
    return model_path

def load_sam2_model(model_name):
    """SAM 2モデルを読み込む関数"""
    try:
        model_path = download_model(model_name)
        
        sam2_config = f"sam2_hiera_s.yaml"
        if "base_plus" in model_name:
            sam2_config = "sam2_hiera_b+.yaml"
        elif "large" in model_name:
            sam2_config = "sam2_hiera_l.yaml"
            
        sam2_model = build_sam2(sam2_config, model_path, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        return predictor
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {e}")
        return None

def main():
    # モデル読み込み
    if 'predictor' not in st.session_state:
        with st.spinner("SAM 2モデルを読み込み中..."):
            st.session_state.predictor = load_sam2_model(selected_model)
    
    if st.session_state.predictor is None:
        st.error("モデルの読み込みに失敗しました。アプリケーションを再起動してください。")
        return
    
    # 画像アップロード
    uploaded_file = st.file_uploader(
        "画像をアップロードしてください",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        # 画像の読み込みと表示
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # 2カラムレイアウト
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("元画像")
            st.image(image, use_column_width=True)
            
            # クリック位置の取得
            st.info("画像上でクリックしてセグメンテーション対象を選択してください")
            
        with col2:
            st.subheader("セグメンテーション結果")
            
            # 画像を予測器に設定
            st.session_state.predictor.set_image(image_array)
            
            # クリックイベントの処理
            if st.button("自動セグメンテーションを実行"):
                with st.spinner("セグメンテーションを実行中..."):
                    # 画像全体のセグメンテーション（自動）
                    masks, scores, logits = st.session_state.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=None,
                        multimask_output=True
                    )
                    
                    # 結果の表示
                    if len(masks) > 0:
                        # 最もスコアの高いマスクを選択
                        best_mask_idx = np.argmax(scores)
                        mask = masks[best_mask_idx]
                        
                        # マスクを可視化
                        result_image = image_array.copy()
                        result_image[mask] = [255, 0, 0]  # 赤色でマスク
                        
                        # 半透明のオーバーレイ
                        overlay = image_array.copy()
                        overlay[mask] = [255, 100, 100]  # 薄い赤
                        
                        # 結果の結合
                        alpha = 0.4
                        result = cv2.addWeighted(overlay, alpha, image_array, 1-alpha, 0)
                        
                        st.image(result, use_column_width=True)
                        st.success(f"セグメンテーション完了！スコア: {scores[best_mask_idx]:.3f}")
                    else:
                        st.warning("セグメンテーション結果が見つかりませんでした。")
        
        # ポイントクリック方式のセグメンテーション
        st.subheader("ポイントクリックによるセグメンテーション")
        
        click_col1, click_col2 = st.columns(2)
        
        with click_col1:
            st.write("クリック位置を入力してください:")
            x_coord = st.number_input("X座標", min_value=0, max_value=image_array.shape[1], value=image_array.shape[1]//2)
            y_coord = st.number_input("Y座標", min_value=0, max_value=image_array.shape[0], value=image_array.shape[0]//2)
            
            if st.button("指定位置でセグメンテーション"):
                with st.spinner("セグメンテーションを実行中..."):
                    point_coords = np.array([[x_coord, y_coord]])
                    point_labels = np.array([1])  # 前景ポイント
                    
                    masks, scores, logits = st.session_state.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True
                    )
                    
                    with click_col2:
                        if len(masks) > 0:
                            # 結果の表示
                            fig, axes = plt.subplots(1, min(3, len(masks)), figsize=(15, 5))
                            
                            for i in range(min(3, len(masks))):
                                if len(masks) == 1:
                                    ax = axes
                                else:
                                    ax = axes[i]
                                
                                result_image = image_array.copy()
                                result_image[masks[i]] = [255, 0, 0]
                                
                                ax.imshow(result_image)
                                ax.set_title(f"マスク {i+1} (スコア: {scores[i]:.3f})")
                                ax.axis('off')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning("指定位置でのセグメンテーションに失敗しました。")

if __name__ == "__main__":
    main()
