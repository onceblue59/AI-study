import sklearn
from sklearn.ensemble import RandomForestRegressor

# 检查 sklearn 是否会用 GPU
rf = RandomForestRegressor()
print(f"Sklearn 版本: {sklearn.__version__}")
print(f"是否支持 GPU: {hasattr(rf, 'n_streams')}")  # GPU 版本会有 CUDA 相关属性，sklearn 没有