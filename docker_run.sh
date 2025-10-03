#!/bin/bash
# GazeGaussian Docker推論の実行スクリプト

set -e

echo "========================================="
echo "GazeGaussian Docker推論スクリプト"
echo "========================================="

# チェックポイントの確認
CHECKPOINT="GazeGaussian-main/checkpoint/gazegaussian_ckp.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo ""
    echo "エラー: チェックポイントが見つかりません: $CHECKPOINT"
    echo ""
    echo "以下のコマンドでダウンロードしてください:"
    echo "  mkdir -p GazeGaussian-main/checkpoint"
    echo "  cd GazeGaussian-main/checkpoint"
    echo "  wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/gazegaussian_ckp.pth"
    echo "  cd ../.."
    exit 1
fi

# 入力ファイルの確認（GazeGaussianの命名規則に従う）
INPUT="preprocessed_output/xgaze_single_image.h5"
if [ ! -f "$INPUT" ]; then
    echo ""
    echo "エラー: 入力ファイルが見つかりません: $INPUT"
    echo ""
    echo "先に前処理を実行してください:"
    echo "  python preprocess_single_image.py --input <画像パス> --output_dir ./preprocessed_output"
    echo ""
    echo "注意: GazeGaussianは 'xgaze_*.h5' という命名規則を期待します"
    exit 1
fi

# 出力ディレクトリの作成
mkdir -p output

echo ""
echo "ステップ1: Dockerイメージのビルド"
echo "（初回のみ時間がかかります）"
echo ""
docker-compose build

echo ""
echo "ステップ2: コンテナの起動"
echo ""
docker-compose up -d

echo ""
echo "ステップ3: サブモジュールのビルド（初回のみ）"
echo ""
docker-compose exec gazegaussian bash -c '
if [ ! -f "/workspace/GazeGaussian/submodules/diff-gaussian-rasterization/.build_done" ]; then
    echo "diff-gaussian-rasterizationをビルド中..."
    cd /workspace/GazeGaussian/submodules/diff-gaussian-rasterization
    pip install -e . > /dev/null 2>&1
    touch .build_done
    cd /workspace/GazeGaussian
fi

if [ ! -f "/workspace/GazeGaussian/submodules/simple-knn/.build_done" ]; then
    echo "simple-knnをビルド中..."
    cd /workspace/GazeGaussian/submodules/simple-knn
    pip install -e . > /dev/null 2>&1
    touch .build_done
    cd /workspace/GazeGaussian
fi

echo "ビルド完了"
'

echo ""
echo "ステップ4: GazeGaussian推論の実行"
echo ""

docker-compose exec gazegaussian bash -c "
cd /workspace
python run_inference.py \
    --input /workspace/preprocessed_output/xgaze_single_image.h5 \
    --checkpoint /workspace/GazeGaussian/checkpoint/gazegaussian_ckp.pth \
    --output_dir /workspace/output \
    --gpu_id 0
"

echo ""
echo "========================================="
echo "推論が完了しました！"
echo "結果は output/single_image/ ディレクトリに保存されています。"
echo "========================================="
echo ""
echo "コンテナを停止する場合は以下を実行:"
echo "  docker-compose down"
echo ""
