import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import requests
import os

# ページ設定
st.set_page_config(
    page_title="SAM 2 Click Segmentation",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 SAM 2: Click Segmentation")
st.markdown("画像をアップロードして、**画像上をクリック**するとセグメンテーションが実行されます！")

# サイドバー設定
st.sidebar.header("設定")

# デバイス選択
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"使用デバイス: {device}")

# セグメンテーション境界検出の調整
st.sidebar.subheader("🎚️ セグメンテーション調整")
boundary_mode = st.sidebar.radio(
    "境界検出モード",
    options=["狭い（精密）", "標準", "広い（大まか）"],
    index=1,
    help="セグメンテーションの境界検出の範囲を調整します"
)

# 境界モードに応じたパラメータ設定
boundary_params = {
    "狭い（精密）": {"mask_threshold": 0.5, "description": "オブジェクトの境界を精密に検出"},
    "標準": {"mask_threshold": 0.0, "description": "標準的な境界検出"},
    "広い（大まか）": {"mask_threshold": -0.5, "description": "オブジェクトを広めに検出"}
}

st.sidebar.caption(boundary_params[boundary_mode]["description"])

# 詳細設定（折りたたみ）
with st.sidebar.expander("🔧 詳細設定"):
    custom_threshold = st.slider(
        "カスタム閾値",
        min_value=-2.0,
        max_value=2.0,
        value=boundary_params[boundary_mode]["mask_threshold"],
        step=0.1,
        help="マスクの閾値を細かく調整（負の値で広く、正の値で狭く）"
    )
    use_custom = st.checkbox("カスタム閾値を使用", value=False)
    
    if use_custom:
        st.info(f"カスタム閾値: {custom_threshold}")
        boundary_params[boundary_mode]["mask_threshold"] = custom_threshold

# 境界スムージング設定
with st.sidebar.expander("✨ 境界スムージング"):
    smooth_enabled = st.checkbox("境界をスムーズにする", value=True, help="マスクの境界を滑らかにします")
    
    if smooth_enabled:
        smooth_method = st.radio(
            "スムージング方法",
            options=["ガウシアンブラー", "モルフォロジー（開閉）", "両方"],
            index=0,
            help="境界を滑らかにする方法を選択"
        )
        
        blur_kernel = st.slider(
            "ブラー強度",
            min_value=1,
            max_value=15,
            value=5,
            step=2,
            help="ガウシアンブラーのカーネルサイズ（奇数）"
        )
        
        morph_kernel = st.slider(
            "モルフォロジーカーネル",
            min_value=1,
            max_value=11,
            value=3,
            step=2,
            help="モルフォロジー処理のカーネルサイズ（奇数）"
        )
    else:
        smooth_method = None
        blur_kernel = 5
        morph_kernel = 3

# スムージング設定を再適用するボタン
if st.sidebar.button("🔄 スムージング設定を適用", use_container_width=True):
    if hasattr(st.session_state, 'logits') and st.session_state.logits is not None:
        st.session_state.force_recompute = True

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

def create_plotly_image(image_array, click_x=None, click_y=None):
    """Plotlyでクリック可能な画像を作成"""
    fig = px.imshow(image_array)
    
    # クリック位置にマーカーを追加
    if click_x is not None and click_y is not None:
        fig.add_trace(go.Scatter(
            x=[click_x],
            y=[click_y],
            mode='markers',
            marker=dict(
                size=20,
                color='red',
                symbol='circle',
                line=dict(color='yellow', width=3)
            ),
            name='クリック位置',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(text="🖱️ 画像をクリックしてセグメンテーション", font=dict(size=16)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=40, b=0),
        height=450,
        showlegend=False
    )
    
    return fig

def run_segmentation(predictor, image_array, x, y, mask_threshold=0.0):
    """セグメンテーションを実行"""
    predictor.set_image(image_array)
    point_coords = np.array([[int(x), int(y)]])
    point_labels = np.array([1])
    
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )
    
    # 元のマスクを返す（logitsは境界調整用に保存）
    return masks, scores, logits

def adjust_masks_with_threshold(logits, mask_threshold, target_shape, smooth_enabled=False, smooth_method=None, blur_kernel=5, morph_kernel=3):
    """logitsから閾値を適用してマスクを調整し、画像サイズにリサイズ"""
    adjusted_masks = []
    for i in range(len(logits)):
        # logitsを画像サイズにリサイズしてから閾値を適用（より滑らかな境界）
        logit = logits[i]
        
        # logitsを画像サイズにリサイズ（バイキュービック補間でより滑らかに）
        if logit.shape != target_shape[:2]:
            logit_resized = cv2.resize(
                logit.astype(np.float32), 
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_CUBIC
            )
        else:
            logit_resized = logit.astype(np.float32)
        
        # リサイズ後に閾値を適用
        adjusted_mask = (logit_resized > mask_threshold).astype(np.uint8)
        
        # スムージング処理
        if smooth_enabled and smooth_method:
            adjusted_mask = smooth_mask(adjusted_mask, smooth_method, blur_kernel, morph_kernel)
        
        adjusted_masks.append(adjusted_mask.astype(bool))
    
    return np.array(adjusted_masks)

def smooth_mask(mask, method, blur_kernel=5, morph_kernel=3):
    """マスクの境界をスムーズにする"""
    mask_uint8 = mask.astype(np.uint8) * 255
    
    if method == "ガウシアンブラー" or method == "両方":
        # ガウシアンブラーを適用
        blurred = cv2.GaussianBlur(mask_uint8, (blur_kernel, blur_kernel), 0)
        mask_uint8 = (blurred > 127).astype(np.uint8) * 255
    
    if method == "モルフォロジー（開閉）" or method == "両方":
        # モルフォロジー処理（開閉操作でノイズ除去と穴埋め）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        # Opening: 小さなノイズを除去
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        # Closing: 小さな穴を埋める
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    return (mask_uint8 > 127).astype(np.uint8)

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
    if 'masks' not in st.session_state:
        st.session_state.masks = None
    if 'scores' not in st.session_state:
        st.session_state.scores = None
    
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
        
        # 2カラムレイアウト
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 画像をクリックして座標を指定")
            st.write("画像上をクリックすると、その位置でセグメンテーションが実行されます")
            
            # Plotlyで画像を表示
            fig = create_plotly_image(
                image_array, 
                st.session_state.click_x, 
                st.session_state.click_y
            )
            
            # plotly_eventsでクリックイベントを取得
            clicked_points = plotly_events(fig, click_event=True, key="plotly_click")
            
            # クリックイベントの処理 - 即座にセグメンテーションを実行
            if clicked_points and len(clicked_points) > 0:
                point = clicked_points[0]
                click_x = int(point.get('x', 0))
                click_y = int(point.get('y', 0))
                
                # 座標の範囲チェック
                click_x = max(0, min(width - 1, click_x))
                click_y = max(0, min(height - 1, click_y))
                
                # 前回と異なる座標の場合のみ実行
                if (st.session_state.click_x != click_x or 
                    st.session_state.click_y != click_y):
                    
                    st.session_state.click_x = click_x
                    st.session_state.click_y = click_y
                    
                    st.info(f"🎯 クリック位置: X={click_x}, Y={click_y}")
                    
                    # 即座にセグメンテーションを実行
                    with st.spinner(f"座標 ({click_x}, {click_y}) でセグメンテーションを実行中..."):
                        try:
                            # 境界モードのパラメータを取得
                            mask_threshold = boundary_params[boundary_mode]["mask_threshold"]
                            
                            masks, scores, logits = run_segmentation(
                                st.session_state.predictor,
                                st.session_state.image_array,
                                click_x, click_y,
                                mask_threshold
                            )
                            
                            # logitsから閾値を適用してマスクを生成（滑らかな境界）
                            adjusted_masks = adjust_masks_with_threshold(
                                logits, mask_threshold, 
                                st.session_state.image_array.shape,
                                smooth_enabled, smooth_method, blur_kernel, morph_kernel
                            )
                            
                            st.session_state.masks = adjusted_masks
                            st.session_state.scores = scores
                            st.session_state.logits = logits
                            st.session_state.seg_coords = (click_x, click_y)
                            st.session_state.boundary_mode = boundary_mode
                            
                            st.success(f"🎉 セグメンテーション完了！")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ エラー: {e}")
                            import traceback
                            st.code(traceback.format_exc())
            
            # 現在の座標情報
            if st.session_state.click_x is not None:
                st.markdown("---")
                st.write(f"📍 **現在の座標**: X={st.session_state.click_x}, Y={st.session_state.click_y}")
                st.write(f"📏 **画像サイズ**: {width} x {height}")
                
                # スムージング設定が変更された場合、再計算
                if (hasattr(st.session_state, 'force_recompute') and 
                    st.session_state.force_recompute and
                    hasattr(st.session_state, 'logits') and
                    st.session_state.logits is not None):
                    
                    st.info(f"🔄 スムージング設定を適用中...")
                    mask_threshold = boundary_params[boundary_mode]["mask_threshold"]
                    
                    adjusted_masks = adjust_masks_with_threshold(
                        st.session_state.logits, 
                        mask_threshold,
                        st.session_state.image_array.shape,
                        smooth_enabled, smooth_method, blur_kernel, morph_kernel
                    )
                    
                    st.session_state.masks = adjusted_masks
                    st.session_state.force_recompute = False
                    st.rerun()
                
                # 境界モードが変更された場合、再セグメンテーション
                if (hasattr(st.session_state, 'boundary_mode') and 
                    st.session_state.boundary_mode != boundary_mode and
                    hasattr(st.session_state, 'logits') and
                    st.session_state.logits is not None):
                    
                    st.info(f"🔄 境界モードを「{boundary_mode}」に変更中...")
                    mask_threshold = boundary_params[boundary_mode]["mask_threshold"]
                    
                    # logitsから新しいマスクを計算（画像サイズにリサイズ）
                    adjusted_masks = adjust_masks_with_threshold(
                        st.session_state.logits, 
                        mask_threshold,
                        st.session_state.image_array.shape,
                        smooth_enabled, smooth_method, blur_kernel, morph_kernel
                    )
                    
                    st.session_state.masks = adjusted_masks
                    st.session_state.boundary_mode = boundary_mode
                    st.rerun()
        
        with col2:
            st.subheader("セグメンテーション結果")
            
            if st.session_state.masks is not None and len(st.session_state.masks) > 0:
                masks = st.session_state.masks
                scores = st.session_state.scores
                
                best_mask_idx = np.argmax(scores)
                mask = masks[best_mask_idx]
                mask_bool = mask.astype(bool)
                
                # マスクを可視化
                overlay = image_array.copy()
                overlay[mask_bool] = [255, 100, 100]
                
                alpha = 0.4
                result = cv2.addWeighted(overlay, alpha, image_array, 1-alpha, 0)
                
                st.image(result, use_container_width=True)
                st.success(f"スコア: {scores[best_mask_idx]:.3f}")
                
                if hasattr(st.session_state, 'seg_coords'):
                    st.write(f"📍 セグメンテーション位置: {st.session_state.seg_coords}")
                
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
                
                # 他のマスクも表示
                if len(masks) > 1:
                    st.subheader("他のセグメンテーション結果")
                    for i in range(min(3, len(masks))):
                        if i != best_mask_idx:
                            other_mask = masks[i].astype(bool)
                            other_overlay = image_array.copy()
                            other_overlay[other_mask] = [255, 100, 100]
                            other_result = cv2.addWeighted(other_overlay, alpha, image_array, 1-alpha, 0)
                            
                            st.markdown(f"#### 結果 {i+1}")
                            st.image(other_result, use_container_width=True)
                            st.write(f"スコア: {scores[i]:.3f}")
                            
                            # 切り抜き画像の作成
                            other_cutout = np.zeros_like(image_array)
                            other_cutout[other_mask] = image_array[other_mask]
                            
                            if np.any(other_mask):
                                other_rows = np.any(other_mask, axis=1)
                                other_cols = np.any(other_mask, axis=0)
                                other_ymin, other_ymax = np.where(other_rows)[0][[0, -1]]
                                other_xmin, other_xmax = np.where(other_cols)[0][[0, -1]]
                                
                                padding = 10
                                other_ymin = max(0, other_ymin - padding)
                                other_ymax = min(image_array.shape[0], other_ymax + padding)
                                other_xmin = max(0, other_xmin - padding)
                                other_xmax = min(image_array.shape[1], other_xmax + padding)
                                
                                other_final_cutout = other_cutout[other_ymin:other_ymax, other_xmin:other_xmax]
                                other_cutout_pil = Image.fromarray(other_final_cutout)
                                
                                st.write("切り抜き結果:")
                                st.image(other_cutout_pil, use_container_width=True)
                                
                                # ダウンロードボタン
                                st.download_button(
                                    label=f"切り抜き画像をダウンロード (結果{i+1})",
                                    data=other_cutout_pil.tobytes(),
                                    file_name=f"segmented_object_{i+1}.png",
                                    mime="image/png",
                                    key=f"download_other_{i}"
                                )
                            
                            st.markdown("---")
            else:
                st.info("👆 左側の画像をクリックしてセグメンテーションを実行してください。")

if __name__ == "__main__":
    main()
