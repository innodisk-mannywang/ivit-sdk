"""
iVIT 2.0 SDK - YOLO Wrapper
提供一個最小包裝以統一 Ultralytics YOLO 物件的呼叫介面，
使其符合 BaseInference 預期的 callable 介面與屬性存取。
"""

from typing import Any


class YOLOWrapper:
    """Minimal wrapper to adapt ultralytics.YOLO to BaseInference interface."""

    def __init__(self, yolo_model: Any):
        self.yolo_model = yolo_model
        # 供外部讀取類別名稱（若可用）
        self.names = getattr(yolo_model, 'names', None)

    def __call__(self, *args, **kwargs):
        # 直接轉呼叫 YOLO 模型的 __call__（等同於 model.predict 簡寫）
        return self.yolo_model(*args, **kwargs)

    # 供 BaseInference.get_model_info 使用，回傳一個可疊代的參數生成器
    def parameters(self):
        # Ultralytics YOLO 物件通常包含 .model (nn.Module)
        model = getattr(self.yolo_model, 'model', None)
        if model is not None and hasattr(model, 'parameters'):
            return model.parameters()
        # 回傳空生成器避免 AttributeError
        return iter(())


