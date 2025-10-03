"""
単一Webカメラ画像をGazeGaussian用のHDF5形式に前処理するスクリプト

MacBook Pro M4 14インチ内蔵カメラを想定
視線はディスプレイ中央を見ているものとする
"""

import os
import sys
import argparse
import numpy as np
import cv2
import h5py
import torch
from pathlib import Path

# GazeNeRFのモジュールをインポート
sys.path.insert(0, 'GazeNeRF-main')
from pre_processing.gen_landmark import Gen2DLandmarks
from pre_processing.gen_all_masks import GenMask
from surface_fitting.nl3dmm.fitting_nl3dmm import FittingNL3DMM
from surface_fitting.nl3dmm.dataset_fitting_3dmm import DatasetforFitting3DMM
from utils.gaze_estimation_utils import estimateHeadPose


class SingleImagePreprocessor:
    """単一画像の前処理を行うクラス"""

    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id

        # MacBook Pro M4 14インチのカメラパラメータ推定値
        # 画面解像度: 3024×1964, 一般的なWebカメラFOV: 65度
        self.camera_params = {
            'image_width': 1920,
            'image_height': 1080,
            'fx': 1050.0,  # 焦点距離 (ピクセル単位)
            'fy': 1050.0,
            'cx': 960.0,   # 光学中心 x
            'cy': 540.0,   # 光学中心 y
            'screen_width_pixels': 3024,
            'screen_height_pixels': 1964,
            'screen_width_mm': 310.0,  # 14インチディスプレイ幅の推定値
            'screen_height_mm': 200.0,  # 14インチディスプレイ高さの推定値
        }

        # 視線方向の計算 (ディスプレイ中央を見ている想定)
        # カメラは画面上部中央に配置されているため、わずかに下向きの視線
        self.gaze_params = {
            'pitch': 5.0,   # 下向きに5度 (ディスプレイを見下ろす)
            'yaw': 0.0,     # 正面
        }

    def get_camera_matrix(self):
        """カメラ内部パラメータ行列を取得"""
        K = np.array([
            [self.camera_params['fx'], 0, self.camera_params['cx']],
            [0, self.camera_params['fy'], self.camera_params['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        return K

    def get_gaze_direction(self):
        """視線方向をpitch-yaw形式で取得"""
        pitch = np.radians(self.gaze_params['pitch'])
        yaw = np.radians(self.gaze_params['yaw'])
        return np.array([pitch, yaw], dtype=np.float32)

    def estimate_head_pose_from_camera(self, landmarks_3d):
        """
        3Dランドマークから頭部姿勢を推定
        簡易的な実装: 顔が正面を向いていると仮定
        """
        # カメラ座標系での回転 (Rodrigues形式)
        # 正面を向いている想定: わずかに下を向く
        hr = np.array([0.1, 0.0, 0.0], dtype=np.float32)  # X軸周りに少し回転

        # 並進ベクトル: カメラからの距離を推定
        # 平均的なWebカメラ使用距離: 50-70cm
        ht = np.array([0.0, 0.0, 600.0], dtype=np.float32)  # 60cm

        return hr, ht

    def process_image(self, image_path, output_dir, temp_dir='temp_single_image'):
        """
        単一画像を処理してGazeGaussian用のHDF5ファイルを生成

        Args:
            image_path: 入力画像のパス
            output_dir: 出力ディレクトリ
            temp_dir: 一時ファイル用ディレクトリ
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        # 画像の読み込みと前処理
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")

        # 画像サイズを調整 (512x512に正規化)
        target_size = 512
        h, w = image.shape[:2]

        # アスペクト比を保持してリサイズ
        if h > w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)

        image_resized = cv2.resize(image, (new_w, new_h))

        # 中央にパディングして512x512にする
        top = (target_size - new_h) // 2
        bottom = target_size - new_h - top
        left = (target_size - new_w) // 2
        right = target_size - new_w - left

        image_padded = cv2.copyMakeBorder(
            image_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        # 一時ディレクトリに保存
        temp_image_path = os.path.join(temp_dir, 'input.png')
        cv2.imwrite(temp_image_path, image_padded)

        print(f"画像を前処理しました: {image_padded.shape}")

        # ステップ1: 顔ランドマーク検出
        print("顔ランドマークを検出中...")
        landmark_detector = Gen2DLandmarks(gpu_id=self.gpu_id, log=False)
        landmarks = landmark_detector.main_process(temp_dir, None)

        if len(landmarks) == 0:
            raise ValueError("顔ランドマークが検出できませんでした")

        print(f"検出されたランドマーク数: {len(landmarks)}")

        # ステップ2: セグメンテーションマスク生成
        print("セグメンテーションマスクを生成中...")
        mask_generator = GenMask(gpu_id=self.gpu_id, log=False)
        head_masks, left_eye_masks, right_eye_masks = mask_generator.main_process(img_dir=temp_dir)

        print(f"マスク生成完了: head={len(head_masks)}, left_eye={len(left_eye_masks)}, right_eye={len(right_eye_masks)}")

        # ステップ3: NL3DMM表面フィッティング
        print("3D顔モデルをフィッティング中...")
        fitting_3dmm = FittingNL3DMM(
            img_size=512,
            intermediate_size=256,
            gpu_id=self.gpu_id,
            batch_size=1,
            img_dir=temp_dir
        )
        surface_list = fitting_3dmm.main_process()

        if len(surface_list) == 0:
            raise ValueError("3DMMフィッティングが失敗しました")

        surface_data = surface_list[0]
        print(f"3DMMフィッティング完了: {list(surface_data.keys())}")

        # ステップ4: HDF5ファイルの生成
        print("HDF5ファイルを生成中...")
        output_h5_path = os.path.join(output_dir, 'single_image_preprocessed.h5')

        self.create_h5_file(
            output_h5_path,
            image_padded,
            landmarks[0],
            head_masks[0],
            left_eye_masks[0],
            right_eye_masks[0],
            surface_data
        )

        print(f"前処理完了! 出力ファイル: {output_h5_path}")

        # 一時ファイルのクリーンアップ
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return output_h5_path

    def create_h5_file(self, output_path, image, landmarks, head_mask,
                       left_eye_mask, right_eye_mask, surface_data):
        """
        GazeGaussian用のHDF5ファイルを作成

        Args:
            output_path: 出力HDF5ファイルのパス
            image: RGB画像 (512x512x3)
            landmarks: 2D顔ランドマーク (68x2)
            head_mask: 頭部マスク (512x512)
            left_eye_mask: 左目マスク (512x512)
            right_eye_mask: 右目マスク (512x512)
            surface_data: NL3DMMフィッティング結果
        """
        with h5py.File(output_path, 'w') as h5f:
            # 1サンプルのデータとして保存 (バッチ次元を追加)

            # 画像 (BGR形式で保存)
            image_bgr = image[:, :, [2, 1, 0]]
            h5f.create_dataset('face_patch', data=image_bgr[np.newaxis, ...], dtype=np.uint8)

            # ランドマーク
            h5f.create_dataset('facial_landmarks', data=landmarks[np.newaxis, ...], dtype=np.float32)

            # 3Dランドマーク (surface_dataから取得)
            if 'lm68' in surface_data:
                facial_landmarks_3d = surface_data['lm68'].cpu().numpy()
                h5f.create_dataset('facial_landmarks_3d', data=facial_landmarks_3d[np.newaxis, ...], dtype=np.float32)
            else:
                # ダミーデータ
                h5f.create_dataset('facial_landmarks_3d', data=np.zeros((1, 68, 3)), dtype=np.float32)

            # マスク
            h5f.create_dataset('head_mask', data=head_mask[np.newaxis, ...], dtype=np.uint8)
            h5f.create_dataset('left_eye_mask', data=left_eye_mask[np.newaxis, ...], dtype=np.uint8)
            h5f.create_dataset('right_eye_mask', data=right_eye_mask[np.newaxis, ...], dtype=np.uint8)

            # 潜在コード (NL3DMM)
            if 'code' in surface_data:
                code = surface_data['code'].cpu().numpy()
                h5f.create_dataset('latent_codes', data=code, dtype=np.float32)
            else:
                # ダミーデータ (256次元)
                h5f.create_dataset('latent_codes', data=np.zeros((1, 256)), dtype=np.float32)

            # カメラパラメータ
            K = self.get_camera_matrix()
            h5f.create_dataset('inmat', data=K[np.newaxis, ...], dtype=np.float32)
            h5f.create_dataset('inv_inmat', data=np.linalg.inv(K)[np.newaxis, ...], dtype=np.float32)

            # World to Camera変換 (正面を向いている想定)
            R_w2c = np.eye(3, dtype=np.float32)
            T_w2c = np.array([0.0, 0.0, 600.0], dtype=np.float32)  # 60cm離れている想定
            h5f.create_dataset('w2c_Rmat', data=R_w2c[np.newaxis, ...], dtype=np.float32)
            h5f.create_dataset('w2c_Tvec', data=T_w2c[np.newaxis, ...], dtype=np.float32)

            # Camera to World変換
            R_c2w = R_w2c.T
            T_c2w = -R_c2w @ T_w2c
            h5f.create_dataset('c2w_Rmat', data=R_c2w[np.newaxis, ...], dtype=np.float32)
            h5f.create_dataset('c2w_Tvec', data=T_c2w[np.newaxis, ...], dtype=np.float32)

            # 視線方向 (pitch-yaw)
            gaze = self.get_gaze_direction()
            h5f.create_dataset('pitchyaw', data=gaze[np.newaxis, ...], dtype=np.float32)

            # 3D視線方向
            gaze_3d = np.array([
                -np.cos(gaze[0]) * np.sin(gaze[1]),
                -np.sin(gaze[0]),
                -np.cos(gaze[0]) * np.cos(gaze[1])
            ], dtype=np.float32)
            h5f.create_dataset('pitchyaw_3d', data=gaze_3d[np.newaxis, ...], dtype=np.float32)

            # 頭部姿勢 (pitch-yaw形式)
            head_pose = np.array([5.0, 0.0], dtype=np.float32)  # 少し下向き
            h5f.create_dataset('pitchyaw_head', data=head_pose[np.newaxis, ...], dtype=np.float32)

            # 顔の頭部姿勢 (Rodrigues形式)
            face_head_pose = np.array([0.1, 0.0, 0.0], dtype=np.float32)
            h5f.create_dataset('face_head_pose', data=face_head_pose[np.newaxis, ...], dtype=np.float32)

            # Pose (6D: rotation + translation)
            pose = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 600.0], dtype=np.float32)
            h5f.create_dataset('pose', data=pose[np.newaxis, ...], dtype=np.float32)

            # Scale
            scale = np.array([1.0], dtype=np.float32)
            h5f.create_dataset('scale', data=scale[np.newaxis, ...], dtype=np.float32)

            # 頂点 (surface_dataから取得)
            if 'verts' in surface_data:
                vertices = surface_data['verts'].cpu().numpy()
                h5f.create_dataset('vertice', data=vertices, dtype=np.float32)
            else:
                # ダミーデータ
                h5f.create_dataset('vertice', data=np.zeros((1, 5023, 3)), dtype=np.float32)

            # カメラインデックス
            h5f.create_dataset('cam_index', data=np.array([[0]], dtype=np.int32))

            # フレームインデックス
            h5f.create_dataset('frame_index', data=np.array([[0]], dtype=np.int32))

        print(f"HDF5ファイルを作成しました: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='単一Webカメラ画像をGazeGaussian用に前処理')
    parser.add_argument('--input', type=str, required=True, help='入力画像のパス')
    parser.add_argument('--output_dir', type=str, default='./preprocessed_output', help='出力ディレクトリ')
    parser.add_argument('--gpu_id', type=int, default=0, help='使用するGPU ID')
    parser.add_argument('--temp_dir', type=str, default='./temp_single_image', help='一時ファイル用ディレクトリ')

    args = parser.parse_args()

    # 前処理の実行
    preprocessor = SingleImagePreprocessor(gpu_id=args.gpu_id)
    output_path = preprocessor.process_image(
        args.input,
        args.output_dir,
        args.temp_dir
    )

    print(f"\n前処理が完了しました!")
    print(f"出力ファイル: {output_path}")
    print(f"\nGazeGaussianで使用するには:")
    print(f"  python GazeGaussian-main/inference.py --input {output_path}")


if __name__ == '__main__':
    main()
