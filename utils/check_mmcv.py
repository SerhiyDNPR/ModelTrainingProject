import numpy
print(f"NumPy Version: {numpy.__version__}")
import mmcv
try:
    from mmcv import _ext
    print("✅ mmcv._ext успішно імпортовано! Проблема вирішена.")
except ImportError:
    print("❌ mmcv._ext все ще не знайдено.")