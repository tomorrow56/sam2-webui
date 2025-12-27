# Translation dictionary for the SAM 2 Web UI

translations = {
    "ja": {
        "title": "SAM 2 Web UI",
        "sidebar": {
            "title": "設定",
            "image_upload": "画像をアップロード",
            "upload_help": "JPG, PNG, BMP形式の画像をアップロードしてください",
            "segmentation_adjustment": "セグメンテーション調整",
            "boundary_detection_mode": "境界検出モード",
            "boundary_options": {
                "narrow": "狭い（精密）",
                "standard": "標準",
                "wide": "広い（大まか）"
            },
            "boundary_descriptions": {
                "narrow": "オブジェクトの境界を精密に検出",
                "standard": "標準的な境界検出",
                "wide": "オブジェクトを広めに検出"
            },
            "detailed_settings": "詳細設定",
            "custom_threshold": "カスタム閾値",
            "custom_threshold_help": "マスクの閾値を細かく調整（負の値で広く、正の値で狭く）",
            "use_custom_threshold": "カスタム閾値を使用",
            "custom_threshold_info": "カスタム閾値: {}",
            "boundary_smoothing": "境界スムージング",
            "enable_smoothing": "境界をスムーズにする",
            "enable_smoothing_help": "マスクの境界を滑らかにします",
            "smoothing_method": "スムージング方法",
            "smoothing_options": {
                "gaussian": "ガウシアンブラー",
                "morphology": "モルフォロジー（開閉）",
                "both": "両方"
            },
            "smoothing_help": "境界を滑らかにする方法を選択",
            "blur_intensity": "ブラー強度",
            "blur_help": "ガウシアンブラーのカーネルサイズ（奇数）",
            "morphology_kernel": "モルフォロジーカーネル",
            "morphology_help": "モルフォロジー処理のカーネルサイズ（奇数）",
            "apply_smoothing": "スムージング設定を適用"
        },
        "main": {
            "upload_description": "Upload an image and **click on it** to run segmentation!",
            "left_column": "画像をクリック",
            "right_column": "セグメンテーション結果",
            "click_position": "クリック位置: X={}, Y={}",
            "segmentation_running": "座標 ({}, {}) でセグメンテーションを実行中...",
            "segmentation_complete": "セグメンテーション完了！",
            "current_position": "現在の座標: X={}, Y={}",
            "image_size": "画像サイズ: {} x {}",
            "changing_boundary_mode": "境界モードを「{}」に変更中...",
            "applying_smoothing": "スムージング設定を適用中...",
            "score": "スコア: {:.3f}",
            "segmentation_position": "セグメンテーション位置: {}",
            "cutout_result": "切り抜き結果",
            "download_cutout": "切り抜き画像をダウンロード（透過PNG）",
            "download_other": "切り抜き画像をダウンロード (結果{})",
            "other_results": "他のセグメンテーション結果",
            "result_number": "結果 {}",
            "click_to_segment": "👆 左側の画像をクリックしてセグメンテーションを実行してください。",
            "error": "❌ エラー: {}"
        }
    },
    "en": {
        "title": "SAM 2 Web UI",
        "sidebar": {
            "title": "Settings",
            "image_upload": "Upload Image",
            "upload_help": "Please upload an image in JPG, PNG, or BMP format",
            "segmentation_adjustment": "Segmentation Adjustment",
            "boundary_detection_mode": "Boundary Detection Mode",
            "boundary_options": {
                "narrow": "Narrow (Precise)",
                "standard": "Standard",
                "wide": "Wide (Rough)"
            },
            "boundary_descriptions": {
                "narrow": "Detect object boundaries precisely",
                "standard": "Standard boundary detection",
                "wide": "Detect objects more broadly"
            },
            "detailed_settings": "Detailed Settings",
            "custom_threshold": "Custom Threshold",
            "custom_threshold_help": "Fine-tune mask threshold (negative for wider, positive for narrower)",
            "use_custom_threshold": "Use Custom Threshold",
            "custom_threshold_info": "Custom threshold: {}",
            "boundary_smoothing": "Boundary Smoothing",
            "enable_smoothing": "Smooth Boundaries",
            "enable_smoothing_help": "Make mask boundaries smoother",
            "smoothing_method": "Smoothing Method",
            "smoothing_options": {
                "gaussian": "Gaussian Blur",
                "morphology": "Morphology (Open/Close)",
                "both": "Both"
            },
            "smoothing_help": "Choose a method to smooth boundaries",
            "blur_intensity": "Blur Intensity",
            "blur_help": "Gaussian blur kernel size (odd number)",
            "morphology_kernel": "Morphology Kernel",
            "morphology_help": "Morphology operation kernel size (odd number)",
            "apply_smoothing": "Apply Smoothing Settings"
        },
        "main": {
            "upload_description": "Upload an image and **click on it** to run segmentation!",
            "left_column": "Click on Image",
            "right_column": "Segmentation Results",
            "click_position": "Click position: X={}, Y={}",
            "segmentation_running": "Running segmentation at coordinates ({}, {})...",
            "segmentation_complete": "Segmentation complete!",
            "current_position": "Current position: X={}, Y={}",
            "image_size": "Image size: {} x {}",
            "changing_boundary_mode": "Changing boundary mode to '{}'...",
            "applying_smoothing": "Applying smoothing settings...",
            "score": "Score: {:.3f}",
            "segmentation_position": "Segmentation position: {}",
            "cutout_result": "Cutout Result",
            "download_cutout": "Download Cutout Image (Transparent PNG)",
            "download_other": "Download Cutout Image (Result {})",
            "other_results": "Other Segmentation Results",
            "result_number": "Result {}",
            "click_to_segment": "👆 Click on the image on the left to run segmentation.",
            "error": "❌ Error: {}"
        }
    }
}

def get_text(key_path, lang="ja"):
    """Get translated text by key path"""
    keys = key_path.split(".")
    try:
        value = translations[lang]
        for k in keys:
            value = value[k]
        return value
    except KeyError:
        # Fallback to Japanese if key not found
        try:
            value = translations["ja"]
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return key_path  # Return key path as fallback
