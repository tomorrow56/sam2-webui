# SAM 2 Web UI

SAM 2（Segment Anything Model 2）を使用したインタラクティブな画像セグメンテーションWebアプリケーションです。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 機能

- **インタラクティブセグメンテーション**: 画像をクリックするだけで即座にセグメンテーションを実行
- **複数の結果表示**: 最適な結果と他の候補を同時に表示
- **切り抜き画像のダウンロード**: セグメンテーション結果を透過PNG形式でダウンロード
- **境界検出モード**: 3段階（狭い/標準/広い）で境界の精度を調整
- **境界スムージング**: ガウシアンブラー、モルフォロジー処理で滑らかな境界を実現
- **カスタム閾値**: 詳細な閾値調整が可能

## スクリーンショット

アプリケーションは2カラムレイアウトで構成されています：
- **左側**: 画像をクリックして座標を指定
- **右側**: セグメンテーション結果と切り抜き画像を表示

## インストール方法

### 1. リポジトリをクローン

```bash
git clone https://github.com/YOUR_USERNAME/sam2-webui.git
cd sam2-webui
```

### 2. 仮想環境を作成（推奨）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows
```

### 3. 依存関係をインストール

```bash
pip install -r requirements.txt
```

### 4. アプリケーションを起動

```bash
streamlit run app_click_final.py
```

## 使用方法

1. ブラウザで `http://localhost:8501` にアクセス
2. サイドバーから画像をアップロード（JPG, PNG, BMP対応）
3. 左側の画像をクリックしてセグメンテーションを実行
4. 右側に結果が表示される
5. 「切り抜き画像をダウンロード」ボタンで結果を保存

### サイドバー設定

#### 🎚️ セグメンテーション調整
- **狭い（精密）**: オブジェクトの境界を精密に検出
- **標準**: 標準的な境界検出
- **広い（大まか）**: オブジェクトを広めに検出

#### 🔧 詳細設定
- **カスタム閾値**: -2.0〜2.0の範囲で細かく調整

#### ✨ 境界スムージング
- **ガウシアンブラー**: マスクの境界をぼかして滑らかに
- **モルフォロジー（開閉）**: 小さなノイズ除去と穴埋め
- **両方**: 両方の処理を適用（最も滑らか）

## ファイル構成

```
sam2-webui/
├── app_click_final.py    # メインアプリケーション（推奨）
├── app.py                # 基本版アプリケーション
├── requirements.txt      # 依存関係
├── checkpoints/          # モデルファイル（自動ダウンロード）
└── README.md
```

## システム要件

- **Python**: 3.8以上
- **GPU**: CUDA対応GPU（推奨）またはCPU
- **RAM**: 8GB以上（Largeモデルの場合は16GB以上推奨）

## モデル

初回起動時に以下のモデルが自動ダウンロードされます：

| モデル | サイズ | 特徴 |
|--------|--------|------|
| sam2.1_hiera_small | ~150MB | 高速、軽量 |
| sam2.1_hiera_base_plus | ~300MB | バランス型 |
| sam2.1_hiera_large | ~800MB | 高精度 |

## 依存関係

- streamlit >= 1.28.0
- torch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- pillow >= 10.0.0
- matplotlib >= 3.7.0
- plotly >= 5.0.0
- streamlit-plotly-events >= 0.0.6
- [SAM 2](https://github.com/facebookresearch/sam2)

## 注意事項

- 初回起動時にモデルのダウンロードが行われます（数分かかる場合があります）
- GPU環境でより高速に動作します
- 大きな画像は処理に時間がかかる場合があります

## ライセンス

MIT License

## 謝辞

- [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/sam2) - Meta AI Research
- [Streamlit](https://streamlit.io/) - Web UIフレームワーク
