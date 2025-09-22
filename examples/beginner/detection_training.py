#!/usr/bin/env python3
"""
iVIT 2.0 SDK - Object Detection 訓練範例
支援單卡和多卡訓練
"""

import os
import sys
import argparse

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ivit.trainer.detection import DetectionTrainer


def run_detection_training(data_path: str, device: str, epochs: int = 50,
                         batch_size: int = 16, learning_rate: float = 0.01,
                         model_name: str = 'yolov8n.pt', img_size: int = 640,
                         num_classes: int = None,
                         progress_log_path: str = None,
                         print_callbacks: bool = False,
                         suppress_yolo_logging: bool = True,
                         verbose: bool = True):
    """
    執行物件偵測訓練
    """
    
    if verbose:
        print("🚀 iVIT 2.0 Object Detection 訓練")
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
            print("❌ Detection 訓練目前不支援多張 GPU，請改為指定單一卡，例如 --device 0 或 --device 1")
            return False

    # 創建訓練器
    trainer = DetectionTrainer(
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
    
    # 確保模型目錄存在
    os.makedirs("models", exist_ok=True)
    
    # 準備 callbacks（選擇性列印）
    callbacks = None
    if print_callbacks:
        import time
        from pathlib import Path
        training_start_time = None
        current_batch = 0
        total_batches = 0
        batches_per_epoch = 0

        # 以資料集圖片數推估每個 epoch 的 batch 數與總 batch 數
        def calculate_total_batches(dataset_path, batch_size, epochs):
            try:
                train_images_dir = Path(dataset_path) / 'images' / 'train'
                if train_images_dir.exists():
                    image_count = len([f for f in train_images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                else:
                    # 兼容另一種目錄結構 train/images
                    alt_dir = Path(dataset_path) / 'train' / 'images'
                    if alt_dir.exists():
                        image_count = len([f for f in alt_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                    else:
                        image_count = 0

                bpe = max(1, (image_count + batch_size - 1) // batch_size)
                tb = bpe * max(1, int(epochs))
                return tb, bpe
            except Exception:
                # 回退：假設每個 epoch 1600 個 batch
                bpe = 1600
                tb = bpe * max(1, int(epochs))
                return tb, bpe

        total_batches, batches_per_epoch = calculate_total_batches(data_path, batch_size, epochs)
        
        def _on_epoch_end(payload):
            print("[on_epoch_end]", payload.get('val_metrics') or payload.get('metrics'))
        
        def _on_batch_end(payload):
            nonlocal training_start_time, current_batch
            if training_start_time is None:
                training_start_time = time.time()
            current_batch += 1
            elapsed_time = time.time() - training_start_time
            iter_per_sec = current_batch / elapsed_time if elapsed_time > 0 else 0

            # 取得 epoch 與 progress（若無 progress，嘗試由 progress_percent 組成）
            # 以推估的 batches_per_epoch 計算 epoch 與進度
            if batches_per_epoch > 0:
                epoch_idx = (current_batch - 1) // batches_per_epoch + 1
                epoch = min(int(epochs), int(epoch_idx))
                progress_ratio = min(1.0, current_batch / max(1, total_batches))
                progress_val = f"{progress_ratio * 100:.1f}%"
            else:
                epoch = payload.get('epoch')
                progress_val = payload.get('progress') or None
                if not progress_val:
                    pp = payload.get('progress_percent')
                    if isinstance(pp, (int, float)):
                        progress_val = f"{pp:.1f}%"

            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.1f}s"
                elif seconds < 3600:
                    return f"{seconds/60:.1f}m"
                else:
                    return f"{seconds/3600:.1f}h"

            info = {
                'epoch': epoch,
                'progress': progress_val,
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
            progress_log_path=progress_log_path,
            suppress_yolo_logging=suppress_yolo_logging
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
    """主函數"""
    parser = argparse.ArgumentParser(description='iVIT 2.0 Object Detection 訓練')
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
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='模型名稱 (yolov8n.pt, yolov8s.pt, yolov8m.pt)')
    parser.add_argument('--img_size', type=int, default=416,
                       help='圖片尺寸 (預設 416，避免 OOM)')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='類別數量 (自動檢測如果未指定)')
    parser.add_argument('--progress_log_path', type=str, default=None,
                       help='將訓練事件寫成 JSONL 的檔案路徑')
    parser.add_argument('--print_callbacks', action='store_true',
                       help='在終端列印 on_epoch_end 回呼資料')
    parser.add_argument('--suppress_yolo_logging', action='store_true', default=False,
                       help='關閉 Ultralytics 訓練日誌')
    parser.add_argument('--verbose', action='store_true',
                       help='顯示詳細的訓練進度資訊')
    parser.add_argument('--quiet', action='store_true',
                       help='關閉所有標準輸出，只顯示 callback 訊息')
    
    args = parser.parse_args()
    
    # 處理 verbose 和 quiet 參數
    verbose = not args.quiet
    
    # 執行訓練
    success = run_detection_training(
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
        suppress_yolo_logging=args.suppress_yolo_logging,
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

#!/usr/bin/env python3
"""
iVIT 2.0 SDK - Object Detection 訓練範例
支援單卡和多卡訓練
"""

import os
import sys
import argparse

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ivit.trainer.detection import DetectionTrainer


def run_detection_training(data_path: str, device: str, epochs: int = 50,
                         batch_size: int = 16, learning_rate: float = 0.01,
                         model_name: str = 'yolov8n.pt', img_size: int = 640,
                         num_classes: int = None,
                         progress_log_path: str = None,
                         print_callbacks: bool = False,
                         suppress_yolo_logging: bool = True,
                         verbose: bool = True):
    """
    執行物件偵測訓練
    
    Args:
        data_path: 資料集路徑 (YOLO 格式)
        device: 設備配置 ('0' 單卡, '0,1' 多卡)
        epochs: 訓練輪數
        batch_size: 批次大小
        learning_rate: 學習率
        model_name: 模型名稱
        img_size: 圖片尺寸
        verbose: 是否顯示詳細資訊
    """
    
    if verbose:
        print("🚀 iVIT 2.0 Object Detection 訓練")
        print("=" * 50)
        print(f"📁 資料路徑: {data_path}")
        print(f"🔧 設備: {device}")
        print(f"📊 模型: {model_name}")
        print(f"📐 圖片尺寸: {img_size}")
        print(f"🔄 Epochs: {epochs}")
        print(f"📦 批次大小: {batch_size}")
        print(f"📈 學習率: {learning_rate}")
        print("=" * 50)
    
    # 檢查是否指定多張 GPU
    if isinstance(device, str) and ',' in device:
        gpu_ids = [x.strip() for x in device.split(',') if x.strip() != '']
        if len(gpu_ids) > 1:
            print("❌ Detection 訓練目前不支援多張 GPU，請改為指定單一卡，例如 --device 0 或 --device 1")
            return False

    # 創建訓練器
    trainer = DetectionTrainer(
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
            print("YOLO 格式結構:")
            print("dataset/")
            print("├── images/")
            print("│   ├── train/")
            print("│   └── val/")
            print("├── labels/")
            print("│   ├── train/")
            print("│   └── val/")
            print("└── data.yaml")
        return False
    
    if verbose:
        print("✅ 資料集格式正確")
    
    # 確保模型目錄存在
    os.makedirs("models", exist_ok=True)
    
    # 準備 callbacks（選擇性列印）
    callbacks = None
    if print_callbacks:
        import time
        training_start_time = None
        current_batch = 0
        total_batches = 0
        
        # 計算實際的總 batch 數
        def calculate_total_batches(dataset_path, batch_size, epochs):
            """計算實際的總 batch 數"""
            try:
                import os
                from pathlib import Path
                
                # 計算訓練圖片數量
                train_images_dir = Path(dataset_path) / 'images' / 'train'
                if train_images_dir.exists():
                    image_count = len([f for f in train_images_dir.iterdir() 
                                     if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                else:
                    # 嘗試其他可能的目錄結構
                    train_dir = Path(dataset_path) / 'train' / 'images'
                    if train_dir.exists():
                        image_count = len([f for f in train_dir.iterdir() 
                                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                    else:
                        # 如果找不到，使用估算值
                        image_count = 1000
                
                # 計算每個 epoch 的 batch 數
                batches_per_epoch = (image_count + batch_size - 1) // batch_size  # 向上取整
                total_batches = batches_per_epoch * epochs
                
                return total_batches, batches_per_epoch
            except Exception as e:
                # 如果計算失敗，使用估算值
                estimated_batches_per_epoch = 1600
                total_batches = estimated_batches_per_epoch * epochs
                return total_batches, estimated_batches_per_epoch
        
        # 計算實際的總 batch 數
        total_batches, batches_per_epoch = calculate_total_batches(data_path, batch_size, epochs)
        
        def _on_epoch_end(payload):
            print("[on_epoch_end]", payload.get('val_metrics') or payload.get('metrics'))
        
        def _on_batch_end(payload):
            nonlocal training_start_time, current_batch, total_batches
            
            # 初始化訓練開始時間
            if training_start_time is None:
                training_start_time = time.time()
            
            # 更新 batch 計數器
            current_batch += 1
            
            # 使用實際計算的總 batch 數來計算準確的 progress
            batch_progress = f"{current_batch}"
            progress_percent = min(100.0, (current_batch / total_batches * 100)) if total_batches > 0 else 0
            
            # 計算當前 epoch（基於實際的 batches_per_epoch）
            current_epoch = min(epochs, (current_batch - 1) // batches_per_epoch + 1)
            epoch_progress = f"{current_epoch}"  # 只顯示當前 epoch
            
            elapsed_time = time.time() - training_start_time
            
            # 計算速度 (epochs per second)
            epoch_per_sec = current_epoch / elapsed_time if elapsed_time > 0 else 0
            
            # 格式化時間顯示
            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.1f}s"
                elif seconds < 3600:
                    return f"{seconds/60:.1f}m"
                else:
                    return f"{seconds/3600:.1f}h"
            
            # 構建顯示資訊 - 使用準確的 progress 計算
            info = {
                'epoch': epoch_progress,
                'progress': f"{progress_percent:.1f}%",
                'batch': f"{current_batch}",  # 當前 batch 數
                'loss_items': payload.get('loss_items'),
                'lr': payload.get('lr'),
                'elapsed': format_time(elapsed_time),
                'speed': f"{epoch_per_sec:.2f} epoch/s"
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
            progress_log_path=progress_log_path,
            suppress_yolo_logging=suppress_yolo_logging
        )
        
        if verbose:
            print("\n🎉 訓練完成！")
            print("📊 最終結果:")
            if 'final_metrics' in results:
                for metric, value in results['final_metrics'].items():
                    print(f"   {metric}: {value:.4f}")
        
        # 處理 YOLO 模型保存
        yolo_save_dir = results.get('model_path', '')
        if yolo_save_dir and os.path.exists(yolo_save_dir):
            # 尋找 YOLO 保存的最佳模型
            best_model_path = None
            for file in os.listdir(yolo_save_dir):
                if file.endswith('.pt') and 'best' in file:
                    best_model_path = os.path.join(yolo_save_dir, file)
                    break
            
            # 如果沒找到 best 模型，尋找最後的模型
            if not best_model_path:
                for file in sorted(os.listdir(yolo_save_dir), reverse=True):
                    if file.endswith('.pt'):
                        best_model_path = os.path.join(yolo_save_dir, file)
                        break
            
            # 顯示模型保存位置
            if best_model_path and os.path.exists(best_model_path):
                if verbose:
                    print(f"💾 模型已保存至: {os.path.abspath(best_model_path)}")
            else:
                if verbose:
                    print(f"⚠️ 未找到訓練好的模型檔案")
        else:
            if verbose:
                print(f"⚠️ 未找到 YOLO 保存目錄")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"❌ 訓練失敗: {str(e)}")
        return False


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='iVIT 2.0 Object Detection 訓練')
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
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='模型名稱 (yolov8n.pt, yolov8s.pt, yolov8m.pt)')
    parser.add_argument('--img_size', type=int, default=416,
                       help='圖片尺寸 (預設 416，避免 OOM)')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='類別數量 (自動檢測如果未指定)')
    parser.add_argument('--progress_log_path', type=str, default=None,
                       help='將訓練事件寫成 JSONL 的檔案路徑')
    parser.add_argument('--print_callbacks', action='store_true',
                       help='在終端列印 on_epoch_end 回呼資料')
    parser.add_argument('--suppress_yolo_logging', action='store_true', default=False,
                       help='關閉 Ultralytics 訓練日誌')
    parser.add_argument('--verbose', action='store_true',
                       help='顯示詳細的訓練進度資訊')
    parser.add_argument('--quiet', action='store_true',
                       help='關閉所有標準輸出，只顯示 callback 訊息')
    
    args = parser.parse_args()
    
    # 處理 verbose 和 quiet 參數
    # 預設為 True，除非明確指定 --quiet
    verbose = not args.quiet
    
    # 執行訓練
    success = run_detection_training(
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
        suppress_yolo_logging=args.suppress_yolo_logging,
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
