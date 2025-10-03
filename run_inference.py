#!/usr/bin/env python
"""
GazeGaussian推論スクリプト
GazeGaussianの既存機能（evaluate_single_image）を使用して視線リダイレクションを実行
"""
import argparse
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='GazeGaussian視線リダイレクション推論',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, required=True,
                        help='入力HDF5ファイルのパス')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='チェックポイントファイルのパス')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='出力ディレクトリ')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='使用するGPU ID')
    parser.add_argument('--subject_name', type=str, default='single_image',
                        help='被験者名（出力サブディレクトリ名）')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='処理する開始フレーム番号')

    args = parser.parse_args()

    # 作業ディレクトリ変更前にパスを絶対パスに変換
    args.input = str(Path(args.input).resolve())
    args.checkpoint = str(Path(args.checkpoint).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())

    # GazeGaussianのconfigsディレクトリにアクセスできるように、カレントディレクトリを変更
    # (get_test_loaderが相対パスでconfigs/を探すため)
    script_dir = Path(__file__).parent.resolve()
    gazegaussian_dir = script_dir / 'GazeGaussian-main'
    os.chdir(gazegaussian_dir)

    # GazeGaussianのパスを追加（ディレクトリ変更後）
    sys.path.insert(0, str(gazegaussian_dir))

    # ディレクトリ変更後にimport
    from configs.gazegaussian_options import BaseOptions
    from trainer.gazegaussian_trainer import GazeGaussianTrainer
    from utils.recorder import GazeGaussianTrainRecorder
    from dataloader.eth_xgaze import get_test_loader

    # 入力ファイルとチェックポイントの存在確認
    if not os.path.exists(args.input):
        print(f"エラー: 入力ファイルが見つかりません: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"エラー: チェックポイントが見つかりません: {args.checkpoint}")
        print("\n以下のコマンドでダウンロードしてください:")
        print("  mkdir -p GazeGaussian-main/checkpoint")
        print("  cd GazeGaussian-main/checkpoint")
        print("  wget https://huggingface.co/ucwxb/GazeGaussian/resolve/main/gazegaussian_ckp.pth")
        sys.exit(1)

    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)

    # GazeGaussianのオプション設定
    opt = BaseOptions()
    opt.gpu_id = args.gpu_id
    opt.load_gazegaussian_checkpoint = args.checkpoint
    opt.save_folder = args.output_dir
    opt.dataset_name = "eth_xgaze"
    opt.batch_size = 1
    opt.num_workers = 0

    # 入力H5ファイルの情報を取得
    input_path = Path(args.input)
    data_dir = str(input_path.parent)
    h5_filename = input_path.stem

    # GazeGaussianは "xgaze_<subject>.h5" という命名規則を期待
    # ファイル名が "xgaze_" で始まらない場合はエラー
    if not h5_filename.startswith('xgaze_'):
        print(f"\nエラー: 入力ファイル名は 'xgaze_' で始まる必要があります")
        print(f"現在のファイル名: {h5_filename}.h5")
        print(f"\n以下のコマンドでファイル名を変更してください:")
        print(f"  mv {args.input} {input_path.parent}/xgaze_{h5_filename}.h5")
        print(f"\n変更後、以下のように実行してください:")
        print(f"  python run_inference.py --input {input_path.parent}/xgaze_{h5_filename}.h5 --checkpoint {args.checkpoint}")
        sys.exit(1)

    # subjectファイル名を抽出（"xgaze_" プレフィックスを削除）
    subject = h5_filename.replace('xgaze_', '') + '.h5'

    print("=" * 70)
    print("GazeGaussian 視線リダイレクション推論")
    print("=" * 70)
    print(f"入力ファイル: {args.input}")
    print(f"チェックポイント: {args.checkpoint}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print(f"GPU ID: {opt.gpu_id}")
    print(f"被験者名: {args.subject_name}")
    print(f"開始フレーム: {args.start_frame}")
    print("=" * 70)

    # GazeGaussianの既存データローダーを使用
    print("\nデータローダーを作成中...")
    opt.img_dir = data_dir

    data_loader = get_test_loader(
        opt,
        data_dir,
        batch_size=1,
        dataset_name=opt.dataset_name,
        num_workers=0,
        is_shuffle=False,
        subject=subject,
        evaluate='single_image'
    )
    print(f"データローダー作成完了")

    # Recorderとtrainerの初期化（GazeGaussianの既存クラスを使用）
    # チェックポイントは初期化時に自動的にロードされる
    print("モデルを初期化中（チェックポイントを自動ロード）...")
    recorder = GazeGaussianTrainRecorder(opt)
    trainer = GazeGaussianTrainer(opt, recorder)

    # GazeGaussianの既存評価関数を使用して推論実行
    print("\n推論を実行中...")
    trainer.evaluate_single_image(
        data_loader=data_loader,
        key=args.subject_name,
        start_frame=args.start_frame
    )

    print(f"\n推論が完了しました！")
    print(f"結果は {args.output_dir}/{args.subject_name}/ に保存されました。")


if __name__ == '__main__':
    main()
