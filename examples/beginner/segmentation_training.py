#!/usr/bin/env python3
"""
iVIT 2.0 SDK - Segmentation 訓練範例
"""

import os
import sys
import argparse

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ivit.trainer.segmentation import SegmentationTrainer


def run_segmentation_training(data_path: str, device: str, epochs: int = 50,
                            batch_size: int = 16, learning_rate: float = 0.01,
                            model_name: str = 'yolov8n-seg.pt', img_size: int = 640,
                            num_classes: int = None,
                            progress_log_path: str = None,
                            print_callbacks: bool = False,
                            verbose: bool = True):
    if verbose:
        print("🚀 iVIT 2.0 Segmentation 訓練")
        print("=" * 50)
        print(f"📁 資料路徑: {data_path}")
        print(f"🔧 設備: {device}")
        print(f"📊 模型: {model_name}")
        print(f"📐 圖片尺寸: {img_size}")
        print(f"🔄 Epochs: {epochs}")
        print(f"📦 批次大小: {batch_size}")
        print(f"📈 學習率: {learning_rate}")
        print("=" * 50)

    # 檢查是否指定多張 GPU（目前不支援）
    if isinstance(device, str) and ',' in device:
        gpu_ids = [x.strip() for x in device.split(',') if x.strip() != '']
        if len(gpu_ids) > 1:
            print("❌ Segmentation 訓練目前不支援多張 GPU，請改為指定單一卡，例如 --device 0 或 --device 1")
            return False

    # 創建訓練器
    trainer = SegmentationTrainer(
        model_name=model_name,
        img_size=img_size,
        learning_rate=learning_rate,
        device=device,
        verbose=verbose
    )

    # 驗證資料集格式
    if verbose:
        print("\n📋 驗證資料集格式...")
    if not trainer.validate_dataset_format(data_path):
        if verbose:
            print("❌ 資料集格式不正確，請確保是 YOLO 格式")
        return False

    if verbose:
        print("✅ 資料集格式正確")

    # 準備 callbacks（選擇性列印）
    callbacks = None
    if print_callbacks:
        import time
        training_start_time = None
        current_batch = 0

        def _on_epoch_end(payload):
            print("[on_epoch_end]", payload.get('val_metrics') or payload.get('metrics'))

        def _on_batch_end(payload):
            nonlocal training_start_time, current_batch
            if training_start_time is None:
                training_start_time = time.time()
            current_batch += 1
            elapsed_time = time.time() - training_start_time
            iter_per_sec = current_batch / elapsed_time if elapsed_time > 0 else 0

            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.1f}s"
                elif seconds < 3600:
                    return f"{seconds/60:.1f}m"
                else:
                    return f"{seconds/3600:.1f}h"

            info = {
                'epoch': payload.get('epoch'),
                'progress': payload.get('progress') or payload.get('progress_percent'),
                'batch': f"{current_batch}",
                'loss_items': payload.get('loss_items'),
                'lr': payload.get('lr'),
                'elapsed': format_time(elapsed_time),
                'speed': f"{iter_per_sec:.2f} it/s"
            }
            print("[on_batch_end]", info)

        callbacks = {
            'on_epoch_end': [_on_epoch_end],
            'on_batch_end': [_on_batch_end]
        }

    # 開始訓練
    if verbose:
        print("\n🎯 開始訓練...")
    try:
        results = trainer.train(
            dataset_path=data_path,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            progress_log_path=progress_log_path
        )

        if verbose:
            print("\n🎉 訓練完成！")
            print("📊 最終結果:")
            if 'final_metrics' in results:
                for metric, value in results['final_metrics'].items():
                    print(f"   {metric}: {value:.4f}")

        return True

    except Exception as e:
        if verbose:
            print(f"❌ 訓練失敗: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='iVIT 2.0 Segmentation 訓練')
    parser.add_argument('--data_path', type=str, required=True,
                       help='資料集路徑 (YOLO 格式)')
    parser.add_argument('--device', type=str, default='0',
                       help='設備配置 (單卡: "0", 多卡: "0,1,2,3")')
    parser.add_argument('--epochs', type=int, default=50,
                       help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小 (預設 4，適合小數據集)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='學習率')
    parser.add_argument('--model', type=str, default='yolov8n-seg.pt',
                       help='模型名稱 (yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt)')
    parser.add_argument('--img_size', type=int, default=640,
                       help='圖片尺寸')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='類別數量 (自動檢測如果未指定)')
    parser.add_argument('--progress_log_path', type=str, default=None,
                       help='將訓練事件寫成 JSONL 的檔案路徑')
    parser.add_argument('--print_callbacks', action='store_true',
                       help='在終端列印 on_epoch_end/on_batch_end 回呼資料')
    parser.add_argument('--verbose', action='store_true',
                       help='顯示詳細的訓練進度資訊')
    parser.add_argument('--quiet', action='store_true',
                       help='關閉所有標準輸出，只顯示 callback 訊息')

    args = parser.parse_args()

    verbose = not args.quiet

    success = run_segmentation_training(
        data_path=args.data_path,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_name=args.model,
        img_size=args.img_size,
        num_classes=args.num_classes,
        progress_log_path=args.progress_log_path,
        print_callbacks=args.print_callbacks,
        verbose=verbose
    )

    if not args.quiet:
        if success:
            print("\n✅ 訓練成功完成！")
        else:
            print("\n❌ 訓練失敗！")
            sys.exit(1)


if __name__ == "__main__":
    main()


