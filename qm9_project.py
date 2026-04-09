import pandas as pd
import numpy as np
import threading
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import xgboost as xgb

# ================= 1. 加载数据 =================
print("🔬 启动完整科研流程（GPU 加速 + 自动调参版）...")

data = pd.read_csv("qm9.csv")
features = ["rotational_constant_a", "rotational_constant_b", "rotational_constant_c",
            "dipole_moment", "polarizability", "homo", "lumo", "r2", "zero_point_energy"]
target = "u0"

X = data[features]
y = data[target]

# 转换为 float32（XGBoost GPU 版要求）
X = X.astype('float32')
y = y.astype('float32')

# ================= 2. 划分数据 =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= 3. 线性回归（基准）=================
print("\n📊 训练线性回归...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"   线性回归 R² = {r2_lr:.4f} （基准线）")

# ================= 4. 强制 GPU + 手动 5折 CV（避开 xgb.cv bug）=================
print("\n🔍 STEP 0: 5折交叉验证选择最佳 max_depth (GPU 加速)...")
print("   测试深度: [3, 6, 9, 12, 15, 20] (6×5=30次训练)")

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 5折交叉验证划分器
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_train_var = np.var(y_train)

depth_candidates = [3, 6, 9, 12, 15, 20]
cv_results = []
cv_start_time = time.time()

for depth in depth_candidates:
    depth_start = time.time()
    fold_scores = []  # 存5折的R²
    
    print(f"   测试 depth={depth:2d}...", end=' ')
    
    # 手动5折循环
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        # 划分当前折
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 训练（用XGBRegressor，device='cuda'确定有效）
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=depth,
            learning_rate=0.1,
            device='cuda',  # ✅ GPU模式，确定有效
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        
        # 验证
        val_pred = model.predict(X_val)
        fold_r2 = r2_score(y_val, val_pred)
        fold_scores.append(fold_r2)
    
    # 计算5折平均
    mean_r2 = np.mean(fold_scores)
    std_r2 = np.std(fold_scores)
    
    cv_results.append({
        'depth': depth,
        'cv_r2_mean': mean_r2,
        'cv_r2_std': std_r2
    })
    
    depth_time = time.time() - depth_start
    print(f"CV R²={mean_r2:.4f}±{std_r2:.4f} | {depth_time:.1f}s")

# 选最优
best_cv = max(cv_results, key=lambda x: x['cv_r2_mean'])
best_depth = best_cv['depth']
cv_total_time = time.time() - cv_start_time

print(f"\n🎯 交叉验证选择: max_depth = {best_depth} (CV R² = {best_cv['cv_r2_mean']:.4f})")
print(f"⏱️  6个depth × 5折 CV 总耗时: {cv_total_time:.1f} 秒")

# ================= 5. XGBoost GPU 最终训练 =================
print(f"\n🚀 训练 XGBoost (GPU, depth={best_depth})...")

xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=best_depth,      # 使用交叉验证选择的最佳深度
    learning_rate=0.1,
    device='cuda',
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror'
)

# 伪进度条
stop_flag = threading.Event()
def progress_bar():
    with tqdm(total=100, desc="🚀 GPU 训练中", ascii=True, ncols=80, leave=True) as pbar:
        while not stop_flag.is_set() and pbar.n < 100:
            time.sleep(0.02)
            pbar.update(1)
        if pbar.n < 100:
            pbar.update(100 - pbar.n)

progress_thread = threading.Thread(target=progress_bar, daemon=True)
progress_thread.start()

# 训练
start_time = time.time()
xgb_model.fit(X_train, y_train)
train_time = time.time() - start_time

stop_flag.set()
progress_thread.join()

# 预测
y_pred_rf = xgb_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f"\n✅ XGBoost (GPU) 训练完成！耗时: {train_time:.3f} 秒")
print(f"   测试集 R² = {r2_rf:.4f}")
print(f"   测试集 MAE = {mae_rf:.4f} eV")

# 特征重要性
importances = xgb_model.feature_importances_
print(f"\n🔍 XGBoost 特征重要性（基于增益）:")
for f, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
    print(f"   {f:25s}: {imp:.3f}")

# ================= 6. 论文级可视化 =================
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_rf, alpha=0.5, s=20, c='steelblue', edgecolors='none')
min_val = min(y_test.min(), y_pred_rf.min())
max_val = max(y_test.max(), y_pred_rf.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')

# 统计标注
plt.text(0.05, 0.95, f'R² = {r2_rf:.4f}\nMAE = {mae_rf:.2f} eV\nDepth = {best_depth}', 
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.xlabel("True U0 (eV)")
plt.ylabel("Predicted U0 (eV)")
plt.title("XGBoost (GPU) Prediction vs True")
plt.legend()
plt.tight_layout()
plt.savefig("result01.png", dpi=300)
plt.close()
print("\n✅ 结果图已保存: result01.png")

# ================= 7. 科研验证（数据泄露检测）=================
print("\n" + "="*60)
print("🛡️  科研验证阶段：检测数据泄露 + 对照实验")
print("="*60)

# 相关性矩阵
print("\n📊 STEP 1: 特征-目标相关性检查")
correlation_matrix = data[features + [target]].corr()
target_corr = correlation_matrix[target].abs().sort_values(ascending=False)

print("与 U0 的绝对相关性排名：")
for feat, corr in target_corr.items():
    if feat != target:
        flag = "🚨" if corr > 0.95 else "⚠️ " if corr > 0.8 else "  "
        print(f"   {flag} {feat:25s}: {corr:.4f}")

# 保存相关性热图
plt.figure(figsize=(10, 8))
corr_values = correlation_matrix.values
mask = np.triu(np.ones_like(corr_values, dtype=bool))
cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'white', 'red'], N=100)

im = plt.imshow(np.ma.array(corr_values, mask=mask), cmap=cmap, vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, label='Correlation')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        text = plt.text(j, i, f'{corr_values[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.title('Feature-Target Correlation Matrix')
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150)
print("✅ 相关性图已保存")

# ================= 8. 对照实验（GPU 加速）=================
print("\n🔬 STEP 2: 对照实验（检测泄露影响）")

experiments = {
    "原始特征(可能含泄露)": features,
    "去除零点能(清洁)": [f for f in features if f != "zero_point_energy"],
    "去除旋转常数c(去重)": [f for f in features if f != "rotational_constant_c"],
    "严格特征(去泄露+去重)": [f for f in features if f not in ["zero_point_energy", "rotational_constant_c"]]
}

exp_results = []

for exp_name, feat_list in experiments.items():
    print(f"\n   测试: {exp_name} ({len(feat_list)}个特征)...")
    
    # GPU 加速的对照实验
    rf_exp = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=best_depth,  # 使用找到的最佳深度
        device='cuda',
        random_state=42
    )
    rf_exp.fit(X_train[feat_list], y_train)
    y_pred_exp = rf_exp.predict(X_test[feat_list])
    
    r2 = r2_score(y_test, y_pred_exp)
    mae = mean_absolute_error(y_test, y_pred_exp)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_exp))
    
    exp_results.append({
        "实验组": exp_name,
        "特征数": len(feat_list),
        "R²": r2,
        "MAE(eV)": mae,
        "RMSE(eV)": rmse
    })

df_exp = pd.DataFrame(exp_results)
print("\n📋 对照实验结果：")
print(df_exp.to_string(index=False))

# ================= 9. 误差分析 =================
print("\n📈 STEP 3: 误差分析")
clean_features = [f for f in features if f not in ["zero_point_energy", "rotational_constant_c"]]

rf_clean = xgb.XGBRegressor(
    n_estimators=100, 
    max_depth=best_depth, 
    device='cuda',
    random_state=42
)
rf_clean.fit(X_train[clean_features], y_train)
y_pred_clean = rf_clean.predict(X_test[clean_features])
residuals = y_test - y_pred_clean

# 找异常值
worst_5 = np.abs(residuals).nlargest(5)
print(f"   预测最差的5个样本，误差: {worst_5.values}")

# 画图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(y_test, y_pred_clean, alpha=0.5, s=20)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel("True U0")
axes[0].set_ylabel("Predicted U0")
axes[0].set_title("Clean Model: Prediction vs True")

axes[1].hist(residuals, bins=50, edgecolor='black', color='skyblue', alpha=0.7)
axes[1].set_xlabel("Residual (eV)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual Distribution")

axes[2].scatter(y_pred_clean, residuals, alpha=0.5, s=20)
axes[2].axhline(y=0, color='r', linestyle='--', lw=2)
axes[2].set_xlabel("Predicted U0")
axes[2].set_ylabel("Residual")
axes[2].set_title("Residual Plot")

plt.tight_layout()
plt.savefig("error_analysis.png", dpi=150)
print("✅ 误差分析图已保存: error_analysis.png")

# ================= 10. 最终报告 =================
print("\n" + "="*60)
print("📑 科研验证报告摘要")
print("="*60)
print(f"1. 基准模型: 线性回归 R² = {r2_lr:.4f} （对照组）")  # ✅ 打印线性 R2
print(f"2. 优化模型: XGBoost R² = {r2_rf:.4f} （提升 {r2_rf - r2_lr:+.4f}）")
print(f"3. 超参数: max_depth = {best_depth} (5折CV自动选择)")
print(f"4. GPU 训练时间: {train_time:.3f} 秒")
print(f"5. 严格特征 R²: {df_exp[df_exp['实验组']=='严格特征(去泄露+去重)']['R²'].values[0]:.4f}")
print(f"6. 生成文件: result01.png, correlation_matrix.png, error_analysis.png")
print("="*60)