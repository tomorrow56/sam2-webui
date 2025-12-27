import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import requests
import os
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# ページ設定
st.set_page_config(
    page_title="SAM 2 Interactive Web UI",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 SAM 2: Interactive Segmentation")
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
        
        # 3カラムレイアウト
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("元画像")
            
            # Plotlyでインタラクティブな画像表示
            fig = go.Figure()
            
            # 画像を追加
            fig.add_trace(go.Image(
                z=image_array,
                colormodel='rgb',
                hoverinfo='skip'
            ))
            
            # クリックイベントを設定
            fig.update_layout(
                width=400,
                height=400,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x", scaleratio=1),
                clickmode='event+select'
            )
            
            # 画像を表示
            clicked_point = plotly_events(fig, click_event=True)
            
            # 元画像も表示
            st.image(image, use_column_width=True)
            
            # クリック位置の処理
            if clicked_point:
                # Plotlyの座標を取得
                plotly_x = clicked_point[0]['x']
                plotly_y = clicked_point[0]['y']
                
                # 元画像の座標に変換（Plotlyは画像中心が原点）
                original_x = int(plotly_x + width / 2)
                original_y = int(height / 2 - plotly_y)
                
                # 座標を範囲内に制限
                original_x = max(0, min(original_x, width - 1))
                original_y = max(0, min(original_y, height - 1))
                
                st.write(f"クリック位置: X={original_x}, Y={original_y}")
                
                # セッション状態に座標を保存
                st.session_state.click_x = original_x
                st.session_state.click_y = original_y
                
                # 自動でセグメンテーションを実行
                if st.button("この位置でセグメンテーション", key="auto_segment"):
                    if 'predictor' in st.session_state and 'image_array' in st.session_state:
                        with st.spinner("セグメンテーションを実行中..."):
                            try:
                                # 画像を予測器に設定
                                st.session_state.predictor.set_image(st.session_state.image_array)
                                
                                # クリック位置でセグメンテーション
                                point_coords = np.array([[original_x, original_y]])
                                point_labels = np.array([1])  # 前景ポイント
                                
                                masks, scores, logits = st.session_state.predictor.predict(
                                    point_coords=point_coords,
                                    point_labels=point_labels,
                                    multimask_output=True
                                )
                                
                                # 結果をセッション状態に保存
                                st.session_state.masks = masks
                                st.session_state.scores = scores
                                st.session_state.click_coords = (original_x, original_y)
                                
                            except Exception as e:
                                st.error(f"セグメンテーション中にエラーが発生しました: {e}")
            else:
                st.info("画像上をクリックして位置を指定してください")
                
                # 手動入力も残す
                st.write("または手動で座標を入力:")
                x_coord = st.number_input("X座標", min_value=0, max_value=width, value=width//2, key="x_coord")
                y_coord = st.number_input("Y座標", min_value=0, max_value=height, value=height//2, key="y_coord")
                
                if st.button("手動入力でセグメンテーション"):
                    if 'predictor' in st.session_state and 'image_array' in st.session_state:
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
        
        with col2:
            st.subheader("セグメンテーション結果")
            
            if 'masks' in st.session_state:
                masks = st.session_state.masks
                scores = st.session_state.scores
                
                if len(masks) > 0:
                    # 最もスコアの高いマスクを選択
                    best_mask_idx = np.argmax(scores)
                    mask = masks[best_mask_idx]
                    mask_bool = mask.astype(bool)
                    
                    # マスクを可視化
                    result_image = st.session_state.image_array.copy()
                    overlay = st.session_state.image_array.copy()
                    overlay[mask_bool] = [255, 100, 100]  # 薄い赤
                    
                    # 結果の結合
                    alpha = 0.4
                    result = cv2.addWeighted(overlay, alpha, st.session_state.image_array, 1-alpha, 0)
                    
                    st.image(result, use_column_width=True)
                    st.success(f"スコア: {scores[best_mask_idx]:.3f}")
                else:
                    st.warning("セグメンテーション結果が見つかりませんでした。")
            else:
                st.info("左側でクリック位置を指定して「クリック位置でセグメンテーション」をクリックしてください。")
        
        with col3:
            st.subheader("切り抜き結果")
            
            if 'masks' in st.session_state:
                masks = st.session_state.masks
                scores = st.session_state.scores
                
                if len(masks) > 0:
                    best_mask_idx = np.argmax(scores)
                    mask = masks[best_mask_idx]
                    mask_bool = mask.astype(bool)
                    
                    # マスク領域を切り抜き
                    original_image = st.session_state.image_array.copy()
                    
                    # 黒背景にマスク領域を描画
                    cutout = np.zeros_like(original_image)
                    cutout[mask_bool] = original_image[mask_bool]
                    
                    # バウンディングボックスを計算して余白を削除
                    if np.any(mask_bool):
                        rows = np.any(mask_bool, axis=1)
                        cols = np.any(mask_bool, axis=0)
                        ymin, ymax = np.where(rows)[0][[0, -1]]
                        xmin, xmax = np.where(cols)[0][[0, -1]]
                        
                        # 少し余白を追加
                        padding = 10
                        ymin = max(0, ymin - padding)
                        ymax = min(original_image.shape[0], ymax + padding)
                        xmin = max(0, xmin - padding)
                        xmax = min(original_image.shape[1], xmax + padding)
                        
                        # 切り抜き
                        final_cutout = cutout[ymin:ymax, xmin:xmax]
                        
                        st.image(final_cutout, use_column_width=True)
                        
                        # ダウンロードボタン
                        cutout_pil = Image.fromarray(final_cutout)
                        st.download_button(
                            label="切り抜き画像をダウンロード",
                            data=cutout_pil.tobytes(),
                            file_name="segmented_object.png",
                            mime="image/png"
                        )
                    else:
                        st.warning("マスク領域が見つかりませんでした。")
        
        # 複数のマスクを表示
        if 'masks' in st.session_state and len(st.session_state.masks) > 1:
            st.subheader("すべてのセグメンテーション結果")
            
            cols = st.columns(min(3, len(st.session_state.masks)))
            for i, (mask, score) in enumerate(zip(st.session_state.masks, st.session_state.scores)):
                if i < 3:  # 最大3つまで表示
                    with cols[i]:
                        mask_bool = mask.astype(bool)
                        result_image = st.session_state.image_array.copy()
                        overlay = st.session_state.image_array.copy()
                        overlay[mask_bool] = [255, 100, 100]
                        
                        alpha = 0.4
                        result = cv2.addWeighted(overlay, alpha, st.session_state.image_array, 1-alpha, 0)
                        
                        st.image(result, use_column_width=True)
                        st.write(f"スコア: {score:.3f}")

if __name__ == "__main__":
    main()
