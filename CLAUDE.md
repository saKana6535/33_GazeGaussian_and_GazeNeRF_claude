# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## リポジトリ概要

このリポジトリには2つの視線リダイレクション手法の実装が含まれています:

1. **GazeGaussian** (`GazeGaussian-main/`): 3D Gaussian Splattingを用いた高忠実度視線リダイレクション手法 (ICCV2025 Highlight)
2. **GazeNeRF** (`GazeNeRF-main/`): Neural Radiance Fieldsを用いた3D視線リダイレクション手法 (CVPR2023)

両方とも視線推定の汎化性能向上のためのデータ拡張を目的としています。

## 環境構築

### GazeGaussian

```bash
cd GazeGaussian-main
conda env create -f env.yaml
conda activate gazegaussian

# Python 3.8, CUDA 11.6, PyTorch 1.12.0
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu116_pyt1120/download.html
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.0_cu116.html

# サブモジュールのインストール
cd submodules/diff-gaussian-rasterization
pip install -e .
cd ../simple-knn
pip install -e .
```

### GazeNeRF

```bash
cd GazeNeRF-main
pip install -r requirements.txt

# データ前処理用のライブラリについては HeadNeRF リポジトリを参照
```

## データセット準備

### GazeGaussian

処理済みデータはHugging Faceから入手可能: https://huggingface.co/ucwxb/GazeGaussian/

```bash
mkdir -p GazeGaussian-main/data
cd GazeGaussian-main/data

# ETH-XGaze訓練セット (14パートに分割)
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze.zip.part*
cat ETH-XGaze.zip.part* > ETH-XGaze.zip
unzip ETH-XGaze.zip

# ETH-XGazeテストセット
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/ETH-XGaze_test.zip
unzip ETH-XGaze_test.zip

# クロスデータセット評価用
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/{ColumbiaGaze,MPIIFaceGaze,gazecapture}.zip
unzip *.zip

# 必要な設定ファイル
cd ../configs
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/config_models.zip
unzip config_models.zip

# 事前学習済みチェックポイント
cd ../checkpoint
wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/gazegaussian_ckp.pth
```

### GazeNeRF

```bash
cd GazeNeRF-main
python dataset_pre_processing.py \
  --dataset_dir=/path/to/dataset \
  --dataset_name=eth_xgaze \
  --output_dir=/path/to/output
```

## 訓練コマンド

### GazeGaussian (2段階訓練)

```bash
cd GazeGaussian-main

# ステップ1: Canonical Mesh Headの事前訓練
bash scripts/train/train_meshhead.sh <GPU_ID>
# チェックポイントは work_dirs/meshhead/checkpoints/ に保存される

# ステップ2: GazeGaussianパイプラインの訓練
bash scripts/train/train_gazegaussian.sh <GPU_ID>
# デフォルトでは事前訓練済みチェックポイントをロード
# スクラッチから訓練する場合は load_meshhead_checkpoint パラメータを変更
```

直接Pythonで実行する場合:
```bash
# Mesh Head訓練
python train_meshhead.py \
  --batch_size 1 \
  --name 'meshhead' \
  --img_dir './data/ETH-XGaze' \
  --num_epochs 10 \
  --num_workers 8

# GazeGaussian訓練
python train_gazegaussian.py \
  --batch_size 1 \
  --name 'gazegaussian' \
  --img_dir './data/ETH-XGaze' \
  --num_epochs 20 \
  --num_workers 2 \
  --lr 0.0001 \
  --clip_grad \
  --load_gazegaussian_checkpoint ./checkpoint/gazegaussian_ckp.pth
```

### GazeNeRF

```bash
cd GazeNeRF-main

python train.py \
  --batch_size=2 \
  --log=true \
  --learning_rate=0.0001 \
  --img_dir='/path/to/ETH-XGaze/training/dataset'

# NVIDIA A40 GPU 1台で訓練
```

評価:
```bash
# メトリクス評価
python evaluate_metrics.py \
  --log=true \
  --num_epochs=75 \
  --model_path=checkpoints/your_checkpoint.json

# 補間デモ生成
python evaluate.py \
  --model_path=checkpoints/your_checkpoint.json \
  --img_dir='/path/to/ETH-XGaze/test/dataset'
```

## アーキテクチャ概要

### GazeGaussian

**2ストリーム3D Gaussian Splattingアーキテクチャ:**

- **Mesh Head Module** (`models/mesh_head.py`): DMTet (Differentiable Marching Tetrahedra) を使用して正準的な頭部メッシュを生成
  - `geo_mlp`: ジオメトリ予測
  - `shape_color_mlp`, `pose_color_mlp`, `eye_color_mlp`: 各種カラー予測
  - `shape_deform_mlp`, `pose_deform_mlp`, `eye_deform_mlp`: 変形予測

- **Gaussian Model** (`models/gaussian_model.py`): 3Dガウシアンの表現とレンダリング
  - 顔領域用と目領域用の2つのストリームに分離

- **GazeGaussian Net** (`models/gaze_gaussian.py`): メインネットワーク
  - `fg_CD_predictor_face`: 顔領域のガウシアンモデル
  - `fg_CD_predictor_eyes`: 目領域のガウシアンモデル (剛体回転による視線制御)
  - `camera`: カメラモジュール
  - `neural_render`: ニューラルレンダラー

- **Trainer** (`trainer/gazegaussian_trainer.py`, `trainer/meshhead_trainer.py`): 訓練ループと最適化

### GazeNeRF

**NeRFベースのアーキテクチャ:**

- **GazeNeRF Net** (`models/gaze_nerf.py`): メインネットワーク
  - 顔領域と目領域で別々のMLPforNeRFを使用
  - Positional encoding (vp_encoder) とView direction encoding (vd_encoder)
  - Hierarchical sampling (coarse + fine) をサポート

- **MLP for NeRF** (`models/mlp_nerf.py`): 密度とRGBを予測するMLP

- **Neural Renderer** (`models/neural_renderer.py`): 特徴マップから最終画像を生成
  - PixelShuffle upsampling を使用

- **Trainer** (`trainer/gazenerf_trainer.py`): 訓練ループ、損失計算、最適化

### 共通コンポーネント

- **Datasets** (`datasets/`, `dataloader/`):
  - `eth_xgaze.py`: ETH-XGazeデータセット
  - `columbia.py`: Columbia Gazeデータセット
  - `mpii_face_gaze.py`: MPII Face Gazeデータセット
  - `gaze_capture.py`: GazeCaptureデータセット

- **Face Recognition** (`face_recognition/`): 顔検出、ランドマーク検出、顔認識モデル
  - RetinaFace, PFLD, MobileFaceNet等のモデル定義

- **Gaze Estimation** (`gaze_estimation/`): 視線推定ベースラインモデル
  - ResNet/VGGベースのXGazeベースライン実装

## ディレクトリ構造

```
GazeGaussian-main/
├── configs/           # 設定ファイル (options, データセット分割)
├── dataloader/        # データローダー実装
├── models/            # モデル定義 (GazeGaussian, MeshHead, Gaussian)
├── trainer/           # 訓練ロジック
├── losses/            # 損失関数
├── utils/             # ユーティリティ関数
├── scripts/train/     # 訓練スクリプト
├── submodules/        # diff-gaussian-rasterization, simple-knn
├── data/              # データセット (手動ダウンロード)
└── checkpoint/        # モデルチェックポイント

GazeNeRF-main/
├── configs/           # 設定ファイル
├── datasets/          # データセット実装
├── models/            # モデル定義 (GazeNeRF, MLPNeRF)
├── trainer/           # 訓練ロジック
├── losses/            # 損失関数
├── utils/             # ユーティリティ関数
├── pre_processing/    # データ前処理スクリプト
├── surface_fitting/   # NL3DMM表面フィッティング
└── data/              # データセット (手動ダウンロード)
```

## 重要な設定ファイル

### GazeGaussian
- `configs/gazegaussian_options.py`: GazeGaussian���訓練オプション
- `configs/meshhead_options.py`: MeshHeadの訓練オプション
- `configs/dataset/*/train_test_split.json`: データセット分割情報

### GazeNeRF
- `configs/gazenerf_options.py`: GazeNeRFの訓練オプション

## 主要な技術的詳細

### GazeGaussian
- **3D Gaussian Splatting**: 非構造化3Dガウシアンによる高速レンダリング
- **2ストリーム設計**: 顔と目を分離してモデル化することで精密な視線制御を実現
- **剛体回転**: ターゲット視線方向に基づく目領域の剛体回転
- **Expression-conditional module**: 異なる被験者間での汎化性能向上

### GazeNeRF
- **Neural Radiance Fields**: ボリュームレンダリングによる3D表現
- **Hierarchical sampling**: CoarseとFineの2段階サンプリング
- **Positional encoding**: 高周波詳細の学習のための位置エンコーディング

## 事前学習モデル

### GazeGaussian
- チェックポイント: https://huggingface.co/ucwxb/GazeGaussian/resolve/main/gazegaussian_ckp.pth

### GazeNeRF
- GazeNeRFモデル: https://drive.google.com/file/d/100ksmOoWc5kFB0V4eT0RZecI9N1Hr2vu/view
- 視線推定器: https://drive.google.com/file/d/1YFQjLYx187XyhGj6SGEONmgV3lBieJsn/view
