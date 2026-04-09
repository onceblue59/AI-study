from mp_api.client import MPRester  # 导入MPRester类
import pandas as pd  # 导入pandas库
import numpy as np  # 导入numpy库

API_KEY = "JmVXzeByVsSA0kyLNnzXTeZAbXZjaCBv"  # API密钥

def extract_features(structure):  # 定义特征提取函数
    # 晶格参数（最稳定、永远不报错）
    a, b, c, alpha, beta, gamma = structure.lattice.parameters  # 获取晶格参数
    
    return {  # 返回特征字典
        "volume": structure.volume,  # 体积
        "density": structure.density,  # 密度
        "num_sites": len(structure),  # 原子位点数
        "a": a,          # 晶胞a轴 ✅ 每个结构都不同
        "b": b,          # 晶胞b轴 ✅ 每个结构都不同
        "c": c,          # 晶胞c轴 ✅ 每个结构都不同
        "alpha": alpha,  # 晶胞角度
        "beta": beta,  # 晶胞角度
        "gamma": gamma  # 晶胞角度
    }  # 结束字典

with MPRester(API_KEY) as mpr:  # 连接Materials API
    docs = mpr.materials.summary.search(  # 搜索材料
        formula=["SiO2"],  # 搜索SiO2材料
        fields=["material_id", "structure", "band_gap"],  # 获取字段
        num_chunks=1,  # 数据块数量
        chunk_size=10  # 每块大小
    )  # 结束搜索

data = []   # 创建空列表
for doc in docs:  # 遍历文档
    feat = extract_features(doc.structure)  # 提取特征
    feat["band_gap"] = doc.band_gap  # 添加带隙
    data.append(feat)  # 追加到列表

df = pd.DataFrame(data)  # 创建数据框
print(df)  # 打印数据