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
from translations import get_text

# ページ設定
st.set_page_config(
    page_title="SAM 2 Click Segmentation",
    page_icon="🎯",
    layout="wide"
)

# 言語選択
lang = st.sidebar.selectbox(
    "Language / 言語",
    options=["日本語", "English"],
    index=0,
    key="language"
)
lang_code = "ja" if lang == "日本語" else "en"

st.title("🎯 " + get_text("title", lang_code))
st.markdown(
    get_text("main.upload_description", lang_code) if lang_code == "en" 
    else "画像をアップロードして、**画像上をクリック**するとセグメンテーションが実行されます！"
)

# サイドバー設定
st.sidebar.header(get_text("sidebar.title", lang_code))

# デバイス選択
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"使用デバイス: {device}" if lang_code == "ja" else f"Using device: {device}")

# セグメンテーション境界検出の調整
st.sidebar.subheader("🎚️ " + get_text("sidebar.segmentation_adjustment", lang_code))
boundary_mode = st.sidebar.radio(
    get_text("sidebar.boundary_detection_mode", lang_code),
    options=[
        get_text("sidebar.boundary_options.narrow", lang_code),
        get_text("sidebar.boundary_options.standard", lang_code),
        get_text("sidebar.boundary_options.wide", lang_code)
    ],
    index=1,
    help=get_text("sidebar.boundary_descriptions.standard", lang_code)
)

# 境界モードに応じたパラメータ設定
boundary_params = {
    get_text("sidebar.boundary_options.narrow", lang_code): {"mask_threshold": 0.5, "description": get_text("sidebar.boundary_descriptions.narrow", lang_code)},
    get_text("sidebar.boundary_options.standard", lang_code): {"mask_threshold": 0.0, "description": get_text("sidebar.boundary_descriptions.standard", lang_code)},
    get_text("sidebar.boundary_options.wide", lang_code): {"mask_threshold": -0.5, "description": get_text("sidebar.boundary_descriptions.wide", lang_code)}
}

st.sidebar.caption(boundary_params[boundary_mode]["description"])

# 詳細設定（折りたたみ）
with st.sidebar.expander("🔧 " + get_text("sidebar.detailed_settings", lang_code)):
    custom_threshold = st.slider(
        get_text("sidebar.custom_threshold", lang_code),
        min_value=-2.0,
        max_value=2.0,
        value=boundary_params[boundary_mode]["mask_threshold"],
        step=0.1,
        help=get_text("sidebar.custom_threshold_help", lang_code)
    )
    use_custom = st.checkbox(get_text("sidebar.use_custom_threshold", lang_code), value=False)
    
    if use_custom:
        st.info(get_text("sidebar.custom_threshold_info", lang_code).format(custom_threshold))
        boundary_params[boundary_mode]["mask_threshold"] = custom_threshold

# 境界スムージング設定
with st.sidebar.expander("✨ " + get_text("sidebar.boundary_smoothing", lang_code)):
    smooth_enabled = st.checkbox(get_text("sidebar.enable_smoothing", lang_code), value=True, help=get_text("sidebar.enable_smoothing_help", lang_code))
    
    if smooth_enabled:
        smooth_method = st.radio(
            get_text("sidebar.smoothing_method", lang_code),
            options=[
                get_text("sidebar.smoothing_options.gaussian", lang_code),
                get_text("sidebar.smoothing_options.morphology", lang_code),
                get_text("sidebar.smoothing_options.both", lang_code)
            ],
            index=0,
            help=get_text("sidebar.smoothing_help", lang_code)
        )
        
        blur_kernel = st.slider(
            get_text("sidebar.blur_intensity", lang_code),
            min_value=1,
            max_value=15,
            value=5,
            step=2,
            help=get_text("sidebar.blur_help", lang_code)
        )
        
        morph_kernel = st.slider(
            get_text("sidebar.morphology_kernel", lang_code),
            min_value=1,
            max_value=11,
            value=3,
            step=2,
            help=get_text("sidebar.morphology_help", lang_code)
        )
    else:
        smooth_method = None
        blur_kernel = 5
        morph_kernel = 3

# スムージング設定を再適用するボタン
if st.sidebar.button("🔄 " + get_text("sidebar.apply_smoothing", lang_code), use_container_width=True):
    if hasattr(st.session_state, 'logits') and st.session_state.logits is not None:
        st.session_state.force_recompute = True
        st.rerun()

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

def create_plotly_image(image_array, click_x=None, click_y=None, lang_code="ja"):
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
            name='Click position' if lang_code == 'en' else 'クリック位置',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(text="🖱️ " + ("Click on image for segmentation" if lang_code == "en" else "画像をクリックしてセグメンテーション"), font=dict(size=16)),
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

def adjust_masks_with_threshold(logits, mask_threshold, target_shape, smooth_enabled=False, smooth_method=None, blur_kernel=5, morph_kernel=3, lang_code="ja"):
    """logitsから閾値を適用してマスクを生成（滑らかな境界）"""
    adjusted_masks = []
    
    for i, logit in enumerate(logits):
        # logitsを画像サイズにリサイズ（INTER_CUBICで滑らかに）
        logit_resized = cv2.resize(
            logit, 
            (target_shape[1], target_shape[0]), 
            interpolation=cv2.INTER_CUBIC
        )
        
        # 閾値を適用してマスクを生成
        adjusted_mask = (logit_resized > mask_threshold).astype(np.uint8)
        
        # スムージング処理
        if smooth_enabled and smooth_method:
            adjusted_mask = smooth_mask(adjusted_mask, smooth_method, blur_kernel, morph_kernel, lang_code)
        
        adjusted_masks.append(adjusted_mask.astype(bool))
    
    return np.array(adjusted_masks)

def smooth_mask(mask, method, blur_kernel=5, morph_kernel=3, lang_code="ja"):
    """マスクの境界をスムーズにする"""
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # 英語と日本語のメソッド名マッピング
    method_map = {
        "ja": {"gaussian": "ガウシアンブラー", "morphology": "モルフォロジー（開閉）", "both": "両方"},
        "en": {"gaussian": "Gaussian Blur", "morphology": "Morphology (Open/Close)", "both": "Both"}
    }
    
    if method == method_map[lang_code]["gaussian"] or method == method_map[lang_code]["both"]:
        # ガウシアンブラーを適用
        blurred = cv2.GaussianBlur(mask_uint8, (blur_kernel, blur_kernel), 0)
        mask_uint8 = (blurred > 127).astype(np.uint8) * 255
    
    if method == method_map[lang_code]["morphology"] or method == method_map[lang_code]["both"]:
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
        with st.spinner("SAM 2モデルを読み込み中..." if lang_code == "ja" else "Loading SAM 2 model..."):
            st.session_state.predictor = load_sam2_model()
    
    if st.session_state.predictor is None:
        st.error(get_text("main.error", lang_code).format("モデルの読み込みに失敗しました。" if lang_code == "ja" else "Failed to load model."))
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
    if 'force_recompute' not in st.session_state:
        st.session_state.force_recompute = False
    if 'last_smooth_settings' not in st.session_state:
        st.session_state.last_smooth_settings = {
            'smooth_enabled': True,
            'smooth_method': get_text("sidebar.smoothing_options.gaussian", lang_code),
            'blur_kernel': 5,
            'morph_kernel': 3
        }
    
    # 画像アップロード
    uploaded_file = st.file_uploader(
        get_text("sidebar.image_upload", lang_code),
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help=get_text("sidebar.upload_help", lang_code)
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
            st.subheader("📸 " + ("画像をクリックして座標を指定" if lang_code == "ja" else "Click on Image to Set Coordinates"))
            st.write("画像上をクリックすると、その位置でセグメンテーションが実行されます" if lang_code == "ja" else "Click on the image to run segmentation at that position")
            
            # Plotlyで画像を表示
            fig = create_plotly_image(
                image_array, 
                st.session_state.click_x, 
                st.session_state.click_y,
                lang_code
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
                    
                    st.info(get_text("main.click_position", lang_code).format(click_x, click_y))
                    
                    # 即座にセグメンテーションを実行
                    with st.spinner(get_text("main.segmentation_running", lang_code).format(click_x, click_y)):
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
                                smooth_enabled, smooth_method, blur_kernel, morph_kernel, lang_code
                            )
                            
                            st.session_state.masks = adjusted_masks
                            st.session_state.scores = scores
                            st.session_state.logits = logits
                            st.session_state.seg_coords = (click_x, click_y)
                            st.session_state.boundary_mode = boundary_mode
                            
                            st.success(get_text("main.segmentation_complete", lang_code))
                            st.rerun()
                            
                        except Exception as e:
                            st.error(get_text("main.error", lang_code).format(str(e)))
                            import traceback
                            st.code(traceback.format_exc())
            
            # 現在の座標情報
            if st.session_state.click_x is not None:
                st.markdown("---")
                st.write(get_text("main.current_position", lang_code).format(st.session_state.click_x, st.session_state.click_y))
                st.write(get_text("main.image_size", lang_code).format(width, height))
                
                # スムージング設定が変更されたかチェック
                current_smooth_settings = {
                    'smooth_enabled': smooth_enabled,
                    'smooth_method': smooth_method,
                    'blur_kernel': blur_kernel,
                    'morph_kernel': morph_kernel
                }
                
                # スムージング設定が変更された場合、またはforce_recomputeフラグが立っている場合
                if ((hasattr(st.session_state, 'last_smooth_settings') and 
                     st.session_state.last_smooth_settings != current_smooth_settings) or
                    (hasattr(st.session_state, 'force_recompute') and 
                     st.session_state.force_recompute)) and \
                    hasattr(st.session_state, 'logits') and st.session_state.logits is not None:
                    
                    st.info(get_text("main.applying_smoothing", lang_code))
                    mask_threshold = boundary_params[boundary_mode]["mask_threshold"]
                    
                    adjusted_masks = adjust_masks_with_threshold(
                        st.session_state.logits, 
                        mask_threshold,
                        st.session_state.image_array.shape,
                        smooth_enabled, smooth_method, blur_kernel, morph_kernel, lang_code
                    )
                    
                    st.session_state.masks = adjusted_masks
                    st.session_state.last_smooth_settings = current_smooth_settings
                    st.session_state.force_recompute = False
                    st.rerun()
                
                # 境界モードが変更された場合、再セグメンテーション
                if (hasattr(st.session_state, 'boundary_mode') and 
                    st.session_state.boundary_mode != boundary_mode and
                    hasattr(st.session_state, 'logits') and
                    st.session_state.logits is not None):
                    
                    st.info(get_text("main.changing_boundary_mode", lang_code).format(boundary_mode))
                    mask_threshold = boundary_params[boundary_mode]["mask_threshold"]
                    
                    # logitsから新しいマスクを計算（画像サイズにリサイズ）
                    adjusted_masks = adjust_masks_with_threshold(
                        st.session_state.logits, 
                        mask_threshold,
                        st.session_state.image_array.shape,
                        smooth_enabled, smooth_method, blur_kernel, morph_kernel, lang_code
                    )
                    
                    st.session_state.masks = adjusted_masks
                    st.session_state.boundary_mode = boundary_mode
                    st.rerun()
        
        with col2:
            st.subheader(get_text("main.right_column", lang_code))
            
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
                st.success(get_text("main.score", lang_code).format(scores[best_mask_idx]))
                
                if hasattr(st.session_state, 'seg_coords'):
                    st.write(get_text("main.segmentation_position", lang_code).format(str(st.session_state.seg_coords)))
                
                # 切り抜き画像の作成
                st.subheader(get_text("main.cutout_result", lang_code))
                # RGBA画像を作成（透明チャネル付き）
                cutout_rgba = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
                # RGBチャネルに元の画像をコピー
                cutout_rgba[:, :, :3][mask_bool] = image_array[mask_bool]
                # Alphaチャネルにマスクを設定
                cutout_rgba[:, :, 3][mask_bool] = 255  # 不透明
                cutout_rgba[:, :, 3][~mask_bool] = 0   # 透明
                
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
                    
                    final_cutout = cutout_rgba[ymin:ymax, xmin:xmax]
                    cutout_pil = Image.fromarray(final_cutout, mode='RGBA')
                    
                    # チェック柄の背景を表示して透明部分を確認しやすく
                    st.markdown("""
                    <style>
                    .transparent-bg {
                        background-image: 
                            linear-gradient(45deg, #ccc 25%, transparent 25%),
                            linear-gradient(-45deg, #ccc 25%, transparent 25%),
                            linear-gradient(45deg, transparent 75%, #ccc 75%),
                            linear-gradient(-45deg, transparent 75%, #ccc 75%);
                        background-size: 20px 20px;
                        background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
                        padding: 20px;
                        border-radius: 10px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('<div class="transparent-bg">', unsafe_allow_html=True)
                    st.image(cutout_pil, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # ダウンロードボタン
                    st.download_button(
                        label=get_text("main.download_cutout", lang_code),
                        data=cutout_pil.tobytes(),
                        file_name="segmented_object.png",
                        mime="image/png"
                    )
                
                # 他のマスクも表示
                if len(masks) > 1:
                    st.subheader(get_text("main.other_results", lang_code))
                    for i in range(min(3, len(masks))):
                        if i != best_mask_idx:
                            other_mask = masks[i].astype(bool)
                            other_overlay = image_array.copy()
                            other_overlay[other_mask] = [255, 100, 100]
                            other_result = cv2.addWeighted(other_overlay, alpha, image_array, 1-alpha, 0)
                            
                            st.markdown(f"#### {get_text('main.result_number', lang_code).format(i+1)}")
                            st.image(other_result, use_container_width=True)
                            st.write(get_text("main.score", lang_code).format(scores[i]))
                            
                            # 切り抜き画像の作成（透過PNG）
                            other_cutout_rgba = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
                            other_cutout_rgba[:, :, :3][other_mask] = image_array[other_mask]
                            other_cutout_rgba[:, :, 3][other_mask] = 255  # 不透明
                            other_cutout_rgba[:, :, 3][~other_mask] = 0   # 透明
                            
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
                                
                                other_final_cutout = other_cutout_rgba[other_ymin:other_ymax, other_xmin:other_xmax]
                                other_cutout_pil = Image.fromarray(other_final_cutout, mode='RGBA')
                                
                                st.write("Cutout result:" if lang_code == "en" else "切り抜き結果:")
                                st.markdown('<div class="transparent-bg">', unsafe_allow_html=True)
                                st.image(other_cutout_pil, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # ダウンロードボタン
                                st.download_button(
                                    label=get_text("main.download_other", lang_code).format(i+1),
                                    data=other_cutout_pil.tobytes(),
                                    file_name=f"segmented_object_{i+1}.png",
                                    mime="image/png",
                                    key=f"download_other_{i}"
                                )
                            
                            st.markdown("---")
            else:
                st.info(get_text("main.click_to_segment", lang_code))

if __name__ == "__main__":
    main()
