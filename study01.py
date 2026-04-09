"""
MOF吸附性能预测 - sklearn入门完整代码
基于CoRE MOF数据库的简化数据
目标：用结构特征预测CO2吸附量
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 第一步：创建/加载模拟数据
# ============================================
# 如果你没有下载真实数据，先用这个模拟数据练习
# 这些特征对应论文中的"几何描述符"

def create_sample_data(n_samples=1000):
    """
    创建模拟的MOF数据集
    特征：孔径、表面积、孔体积、密度
    目标：CO2吸附量 (mmol/g)
    """
    np.random.seed(42)
    
    data = {
        # 孔径 (Å) - 论文中范围 0-70Å
        'pore_size': np.random.exponential(8, n_samples),
        
        # 比表面积 (m²/g) - 论文中范围 0-8000
        'surface_area': np.random.gamma(2, 800, n_samples),
        
        # 孔体积 (cm³/g) - 论文中范围 0-17
        'pore_volume': np.random.gamma(2, 1.5, n_samples),
        
        # 密度 (g/cm³) - 论文中范围 0.5-1.8
        'density': np.random.normal(1.0, 0.3, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 模拟CO2吸附量：与孔体积正相关，与密度负相关（简化模型）
    # 论文图3.1显示MPI与孔体积负相关，这里反过来模拟
    df['co2_uptake'] = (
        2.0 * df['pore_volume'] +                    # 孔体积越大，吸附越多
        0.001 * df['surface_area'] +                 # 表面积贡献
        0.05 * df['pore_size'] +                     # 孔径贡献
        -1.5 * df['density'] +                       # 密度越大，吸附越少
        np.random.normal(0, 0.5, n_samples)          # 噪声
    )
    
    # 截断负值（物理上吸附量不能为负）
    df['co2_uptake'] = df['co2_uptake'].clip(lower=0)
    
    return df

# 加载数据
print("=" * 50)
print("MOF吸附性能预测 - sklearn入门")
print("=" * 50)

df = create_sample_data(n_samples=2000)
print(f"\n数据集大小: {df.shape}")
print(f"特征: {list(df.columns[:-1])}")
print(f"目标: {df.columns[-1]}")
print(f"\n数据预览:")
print(df.head(10))

# ============================================
# 第二步：数据探索（EDA）
# ============================================
print("\n" + "=" * 50)
print("第二步：数据探索")
print("=" * 50)

# 统计信息
print("\n数据统计:")
print(df.describe())

# 相关性分析（对应论文图3.1的矩阵散点图简化版）
print("\n特征与目标的相关性:")
correlation = df.corr()['co2_uptake'].sort_values(ascending=False)
print(correlation)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 各特征分布
axes[0, 0].hist(df['co2_uptake'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('CO2 Adsorption (mmol/g)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Distribution of CO2 Uptake')

# 2. 孔体积 vs 吸附量（论文核心发现：强相关）
axes[0, 1].scatter(df['pore_volume'], df['co2_uptake'], alpha=0.5, s=10)
axes[0, 1].set_xlabel('Pore Volume (cm³/g)')
axes[0, 1].set_ylabel('CO2 Uptake (mmol/g)')
axes[0, 1].set_title('Pore Volume vs CO2 Uptake')

# 3. 表面积 vs 吸附量
axes[1, 0].scatter(df['surface_area'], df['co2_uptake'], alpha=0.5, s=10, color='orange')
axes[1, 0].set_xlabel('Surface Area (m²/g)')
axes[1, 0].set_ylabel('CO2 Uptake (mmol/g)')
axes[1, 0].set_title('Surface Area vs CO2 Uptake')

# 4. 密度 vs 吸附量（负相关）
axes[1, 1].scatter(df['density'], df['co2_uptake'], alpha=0.5, s=10, color='green')
axes[1, 1].set_xlabel('Density (g/cm³)')
axes[1, 1].set_ylabel('CO2 Uptake (mmol/g)')
axes[1, 1].set_title('Density vs CO2 Uptake (Negative Correlation)')

plt.tight_layout()
plt.savefig('mof_eda.png', dpi=150)
print("\n探索性分析图已保存: mof_eda.png")
plt.show()

# ============================================
# 第三步：数据准备
# ============================================
print("\n" + "=" * 50)
print("第三步：数据准备")
print("=" * 50)

# 特征和标签
feature_cols = ['pore_size', 'surface_area', 'pore_volume', 'density']
X = df[feature_cols]
y = df['co2_uptake']

# 划分训练集/验证集/测试集（论文常用8:1:1或7:1.5:1.5）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"训练集: {len(X_train)} 样本")
print(f"验证集: {len(X_val)} 样本")
print(f"测试集: {len(X_test)} 样本")

# ============================================
# 第四步：训练模型（对应论文3.3.3节）
# ============================================
print("\n" + "=" * 50)
print("第四步：模型训练与对比")
print("=" * 50)

# 模型1：线性回归（最简单基线）
print("\n--- 模型1: 线性回归 ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 模型2：随机森林（论文RF模型，对应论文Table 3.3）
print("\n--- 模型2: 随机森林 ---")
rf_model = RandomForestRegressor(
    n_estimators=100,      # 树的数量
    max_depth=15,          # 最大深度
    min_samples_split=5,   # 节点分裂最小样本
    random_state=42,
    n_jobs=-1              # 使用所有CPU
)
rf_model.fit(X_train, y_train)

# ============================================
# 第五步：模型评估（对应论文的R²和MAE）
# ============================================
print("\n" + "=" * 50)
print("第五步：模型评估")
print("=" * 50)

def evaluate_model(model, X, y, dataset_name):
    """评估模型性能"""
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"\n{dataset_name}:")
    print(f"  R²  = {r2:.4f}  (论文主要指标)")
    print(f"  MAE = {mae:.4f} mmol/g")
    print(f"  RMSE = {rmse:.4f} mmol/g")
    
    return y_pred, r2, mae

# 线性回归评估
print("\n>>> 线性回归结果:")
lr_val_pred, lr_val_r2, _ = evaluate_model(lr_model, X_val, y_val, "验证集")
lr_test_pred, lr_test_r2, _ = evaluate_model(lr_model, X_test, y_test, "测试集")

# 随机森林评估
print("\n>>> 随机森林结果:")
rf_val_pred, rf_val_r2, _ = evaluate_model(rf_model, X_val, y_val, "验证集")
rf_test_pred, rf_test_r2, _ = evaluate_model(rf_model, X_test, y_test, "测试集")

# ============================================
# 第六步：结果可视化（对应论文图3.4B）
# ============================================
print("\n" + "=" * 50)
print("第六步：结果可视化")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 线性回归预测 vs 真实值
axes[0].scatter(y_test, lr_test_pred, alpha=0.5, s=15)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('True CO2 Uptake (mmol/g)')
axes[0].set_ylabel('Predicted CO2 Uptake (mmol/g)')
axes[0].set_title(f'Linear Regression (R² = {lr_test_r2:.3f})')
axes[0].legend()

# 随机森林预测 vs 真实值
axes[1].scatter(y_test, rf_test_pred, alpha=0.5, s=15, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('True CO2 Uptake (mmol/g)')
axes[1].set_ylabel('Predicted CO2 Uptake (mmol/g)')
axes[1].set_title(f'Random Forest (R² = {rf_test_r2:.3f})')
axes[1].legend()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
print("\n模型对比图已保存: model_comparison.png")
plt.show()

# ============================================
# 第七步：特征重要性分析（论文关键分析）
# ============================================
print("\n" + "=" * 50)
print("第七步：特征重要性分析")
print("=" * 50)

importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n随机森林特征重要性:")
print(importance)

# 可视化
plt.figure(figsize=(8, 5))
plt.barh(importance['Feature'], importance['Importance'], color='skyblue', edgecolor='black')
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("\n特征重要性图已保存: feature_importance.png")
plt.show()

# ============================================
# 第八步：交叉验证（更稳健的评估）
# ============================================
print("\n" + "=" * 50)
print("第八步：5折交叉验证")
print("=" * 50)

rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print(f"\n随机森林5折交叉验证R²: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std()*2:.4f})")

# ============================================
# 总结
# ============================================
print("\n" + "=" * 50)
print("总结")
print("=" * 50)
print(f"""
你完成了论文3.3.3节的简化复现！

关键指标对比:
- 线性回归 R²: {lr_test_r2:.3f}
- 随机森林 R²: {rf_test_r2:.3f}  ← 论文中RF模型作为深度学习基线

论文中对应结果（Table 3.3）:
- RF + S-120描述符: R² ≈ 0.6
- Matformer: R² = 0.939

下一步学习方向:
1. 尝试更多特征工程（如论文中的RACs、SOAP描述符）
2. 学习PyTorch，尝试图神经网络（CGCNN）
3. 使用真实CoRE MOF数据替换模拟数据
""")