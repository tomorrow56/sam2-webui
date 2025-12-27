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
    page_title="SAM 2 Simple Click",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 SAM 2: Simple Click Segmentation")
st.markdown("画像をアップロードして、クリック位置のオブジェクトをセグメンテーションします！")

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
        
        # 2カラムレイアウト
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 元画像と座標指定")
            
            # 座標入力（マウスクリックで自動入力）
            st.write("### 📍 座標入力（マウスクリックで自動設定）")
            col_x, col_y = st.columns(2)
            with col_x:
                x_coord = st.number_input("X座標", min_value=0, max_value=width, value=width//2, key="click_x")
            with col_y:
                y_coord = st.number_input("Y座標", min_value=0, max_value=height, value=height//2, key="click_y")
            
            # 座標情報の表示
            st.info(f"🎯 現在の座標: X={x_coord}, Y={y_coord}")
            
            # 画像上にクリック位置を表示
            st.write("### 📊 クリック位置の確認")
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(image_array)
            ax.plot(x_coord, y_coord, 'ro', markersize=15, markeredgecolor='yellow', markeredgewidth=3)
            ax.set_title(f"セグメンテーション位置: ({x_coord}, {y_coord})", fontsize=14, fontweight='bold')
            ax.axis('off')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
            # セグメンテーションボタン（この位置が正しい）
            st.markdown("---")
            st.write("### 🚀 セグメンテーション実行")
            st.write(f"📍 実行座標: X={x_coord}, Y={y_coord} | 📏 画像サイズ: {width} x {height}")
            
            if st.button("🎯 この位置でセグメンテーション実行", type="primary", use_container_width=True):
                with st.spinner("セグメンテーションを実行中..."):
                    try:
                        st.session_state.predictor.set_image(st.session_state.image_array)
                        point_coords = np.array([[int(x_coord), int(y_coord)]])
                        point_labels = np.array([1])
                        
                        st.write("🔍 デバッグ情報:")
                        st.write(f"   - 入力座標: ({x_coord}, {y_coord})")
                        st.write(f"   - 変換後座標: ({int(x_coord)}, {int(y_coord)})")
                        
                        masks, scores, logits = st.session_state.predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=True
                        )
                        
                        st.write(f"✅ 生成されたマスク数: {len(masks)}")
                        st.session_state.masks = masks
                        st.session_state.scores = scores
                        st.session_state.click_coords = (int(x_coord), int(y_coord))
                        
                        st.success(f"🎉 セグメンテーション完了！位置: ({int(x_coord)}, {int(y_coord)})")
                        
                    except Exception as e:
                        st.error(f"❌ エラー: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # クリック可能な画像（別セクション）
            st.markdown("---")
            st.write("### 🖱️ または画像を直接クリックして座標設定")
            st.write("下の画像をクリックすると、上の座標入力フィールドに自動で値が入力されます")
            
            st.markdown("""
            <style>
            .clickable-image {
                cursor: crosshair;
                border: 3px solid #007bff;
                border-radius: 8px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 8px rgba(0,123,255,0.2);
            }
            .clickable-image:hover {
                border-color: #0056b3;
                box-shadow: 0 6px 12px rgba(0,123,255,0.4);
                transform: scale(1.02);
            }
            #click-info {
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
            
            html_code = f"""
            <div>
                <img src="data:image/png;base64,{img_base64}" 
                     class="clickable-image" 
                     style="max-width: 100%; height: auto;"
                     onclick="clickImage(event, {width}, {height})">
                <div id="click-info" style="margin-top: 10px;">
                    🎯 画像をクリックして座標を設定
                </div>
            </div>
            
            <script>
            function clickImage(event, imgWidth, imgHeight) {{
                const img = event.target;
                const rect = img.getBoundingClientRect();
                
                const x = Math.round((event.clientX - rect.left) * (img.naturalWidth / rect.width));
                const y = Math.round((event.clientY - rect.top) * (img.naturalHeight / rect.height));
                
                const finalX = Math.max(0, Math.min(imgWidth - 1, x));
                const finalY = Math.max(0, Math.min(imgHeight - 1, y));
                
                console.log('=== クリック座標情報 ===');
                console.log('クライアント座標:', event.clientX, event.clientY);
                console.log('最終座標:', finalX, finalY);
                console.log('========================');
                
                // 座標を表示
                document.getElementById('click-info').innerHTML = 
                    '✅ クリック位置: X=' + finalX + ', Y=' + finalY + '<br>' +
                    '📍 座標が入力フィールドに反映されました';
                
                document.getElementById('click-info').style.background = 'linear-gradient(135deg, #28a745 0%, #20c997 100%)';
                
                // Streamlitの入力フィールドを更新（強化版）
                setTimeout(function() {{
                    console.log('入力フィールドの更新を開始...');
                    
                    // 全てのinput要素を取得
                    const allInputs = window.parent.document.querySelectorAll('input');
                    console.log('全てのinput要素数:', allInputs.length);
                    
                    let xFound = false;
                    let yFound = false;
                    
                    // 各input要素をチェック
                    for (let i = 0; i < allInputs.length; i++) {{
                        const input = allInputs[i];
                        console.log('input[' + i + ']:', input.type, input.value, input.getAttribute('data-testid'));
                        
                        // numberタイプのinputを探す
                        if (input.type === 'number') {{
                            const label = input.closest('[data-testid="stNumberInput"]')?.parentElement?.querySelector('label');
                            const labelText = label ? label.textContent : '';
                            
                            console.log('ラベルテキスト:', labelText);
                            
                            // 最初のnumber入力をX座標、2番目をY座標として設定
                            if (!xFound) {{
                                console.log('X座標フィールドを発見、値を設定:', finalX);
                                input.value = finalX;
                                input.focus();
                                input.blur();
                                
                                // 複数のイベントを発火
                                input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                input.dispatchEvent(new Event('keyup', {{ bubbles: true }}));
                                input.dispatchEvent(new Event('blur', {{ bubbles: true }}));
                                
                                // Reactの状態更新をトリガー
                                const setter = Object.getOwnPropertyDescriptor(input.constructor.prototype, 'value')?.set;
                                if (setter) {{
                                    setter.call(input, finalX);
                                }}
                                
                                xFound = true;
                            }} else if (!yFound) {{
                                console.log('Y座標フィールドを発見、値を設定:', finalY);
                                input.value = finalY;
                                input.focus();
                                input.blur();
                                
                                // 複数のイベントを発火
                                input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                input.dispatchEvent(new Event('keyup', {{ bubbles: true }}));
                                input.dispatchEvent(new Event('blur', {{ bubbles: true }}));
                                
                                // Reactの状態更新をトリガー
                                const setter = Object.getOwnPropertyDescriptor(input.constructor.prototype, 'value')?.set;
                                if (setter) {{
                                    setter.call(input, finalY);
                                }}
                                
                                yFound = true;
                                break;
                            }}
                        }}
                    }}
                    
                    // もし見つからない場合、別の方法を試す
                    if (!xFound || !yFound) {{
                        console.log('別の方法を試します...');
                        
                        // stNumberInputを直接探す
                        const numberInputs = window.parent.document.querySelectorAll('[data-testid="stNumberInput"] input');
                        console.log('stNumberInputの数:', numberInputs.length);
                        
                        if (numberInputs.length >= 2) {{
                            numberInputs[0].value = finalX;
                            numberInputs[1].value = finalY;
                            
                            numberInputs[0].dispatchEvent(new Event('input', {{ bubbles: true }}));
                            numberInputs[1].dispatchEvent(new Event('input', {{ bubbles: true }}));
                            
                            console.log('stNumberInputで座標を設定');
                        }}
                    }}
                    
                    // ページ全体の再描画をトリガー
                    window.parent.document.dispatchEvent(new Event('resize'));
                    window.parent.dispatchEvent(new Event('resize'));
                    
                    // Streamlitの状態更新を試みる
                    if (window.parent.Streamlit) {{
                        window.parent.Streamlit.setComponentValue({{ coordinates: [finalX, finalY] }});
                    }}
                    
                    console.log('✅ 座標設定完了:', finalX, finalY);
                    console.log('X座標見つかった:', xFound, 'Y座標見つかった:', yFound);
                }}, 500);
            }}
            </script>
            """
            
            st.components.v1.html(html_code, height=450)
        
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
                    result_image = image_array.copy()
                    overlay = image_array.copy()
                    overlay[mask_bool] = [255, 100, 100]  # 薄い赤
                    
                    alpha = 0.4
                    result = cv2.addWeighted(overlay, alpha, image_array, 1-alpha, 0)
                    
                    st.image(result, use_column_width=True)
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
                        
                        st.image(cutout_pil, use_column_width=True)
                        
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
                        for i, (mask, score) in enumerate(zip(masks[1:4], scores[1:4])):
                            mask_bool = mask.astype(bool)
                            result_image = image_array.copy()
                            overlay = image_array.copy()
                            overlay[mask_bool] = [255, 100, 100]
                            
                            alpha = 0.4
                            result = cv2.addWeighted(overlay, alpha, image_array, 1-alpha, 0)
                            
                            st.image(result, use_column_width=True)
                            st.write(f"スコア: {score:.3f}")
                else:
                    st.warning("セグメンテーション結果が見つかりませんでした。")
            else:
                st.info("左側で座標を指定して「この位置でセグメンテーション」をクリックしてください。")

if __name__ == "__main__":
    main()
