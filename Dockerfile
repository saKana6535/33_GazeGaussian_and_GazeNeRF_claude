# GazeGaussian用Dockerfile
# CUDA 11.6、Python 3.8、PyTorch 1.12.0環境

FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# タイムゾーンの設定（インタラクティブプロンプトを回避）
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda3のインストール
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# 作業ディレクトリの作成
WORKDIR /workspace

# 環境ファイルとコードをコピー
COPY GazeGaussian-main/env.yaml /workspace/env.yaml

# Conda環境の作成（一部の依存関係は後でインストール）
RUN conda env create -f env.yaml && \
    conda clean -afy

# Conda環境をデフォルトで有効化
SHELL ["conda", "run", "-n", "gazegaussian", "/bin/bash", "-c"]

# diff-gaussian-rasterizationとsimple-knnのビルド準備
# （実際のビルドはコンテナ起動後に行う）
RUN echo "source activate gazegaussian" >> ~/.bashrc

# PyTorch3Dとkaiolin（環境構築済みなのでスキップ可能）
# 必要に応じて追加のパッケージをインストール
RUN conda run -n gazegaussian pip install --no-cache-dir \
    wandb \
    opencv-python \
    h5py

# 作業ディレクトリ
WORKDIR /workspace/GazeGaussian

# デフォルトコマンド
CMD ["/bin/bash"]
