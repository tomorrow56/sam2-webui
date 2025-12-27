import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import requests
import os

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
        
        # 2カラムレイアウト
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("元画像")
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader("セグメンテーション結果")
            
            # 画像を予測器に設定
            st.session_state.predictor.set_image(image_array)
            
            if st.button("自動セグメンテーションを実行"):
                with st.spinner("セグメンテーションを実行中..."):
                    try:
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
                            
                            # マスクをboolean型に変換
                            mask_bool = mask.astype(bool)
                            
                            # マスクを可視化
                            result_image = image_array.copy()
                            overlay = image_array.copy()
                            overlay[mask_bool] = [255, 100, 100]  # 薄い赤
                            
                            # 結果の結合
                            alpha = 0.4
                            result = cv2.addWeighted(overlay, alpha, image_array, 1-alpha, 0)
                            
                            st.image(result, use_column_width=True)
                            st.success(f"セグメンテーション完了！スコア: {scores[best_mask_idx]:.3f}")
                        else:
                            st.warning("セグメンテーション結果が見つかりませんでした。")
                    except Exception as e:
                        st.error(f"セグメンテーション中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
