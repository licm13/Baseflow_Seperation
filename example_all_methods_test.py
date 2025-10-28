import pandas as pd
import os
from tqdm import tqdm
from baseflow.separation import single  # 确保你有这个函数和依赖模块

# 设置路径
input_file = "Z:/LCM/Baseflow_seperation/baseflow/example.csv"
output_dir = "Z:/LCM/Baseflow_seperation/baseflow/outputs"
os.makedirs(output_dir, exist_ok=True)

# 读取数据
df = pd.read_csv(input_file)
df["time"] = pd.to_datetime(df["time"])
station_columns = [col for col in df.columns if col != "time"]

# 所有方法
methods = [
    "LH", "Chapman", "CM", "Boughton", "Furey", "Eckhardt",
    "EWMA", "Willems", "UKIH", "Fixed", "Local", "Slide"
]

# 设置面积参数
area = 150.0

# 站点总数
total_stations = len(station_columns)

# 处理每个站点（带外层进度条）
for idx, station in enumerate(tqdm(station_columns, desc="🚀 总体进度", unit="站点")):
    print(f"\n📍 正在处理第 {idx+1}/{total_stations} 个站点：{station}")

    # 构造时间索引流量序列
    series = pd.Series(df[station].values, index=df["time"])
    
    # 创建空的结果 DataFrame
    output_df = pd.DataFrame({"time": df["time"], "Q": df[station]})
    
    # 每个方法单独调用（显示子进度条）
    for method in tqdm(methods, desc=f"    ➤ {station} 方法计算中", leave=False, unit="方法"):
        try:
            # 调用单方法
            b, _ = single(series, area=area, method=[method], return_kge=False)
            output_df[method] = b[method].values
        except Exception as e:
            print(f"❌ 方法 {method} 在站点 {station} 上出错：{e}")
            output_df[method] = None  # 避免中断，填空列

    # 保存结果
    output_path = os.path.join(output_dir, f"baseflow_separation_{station}.csv")
    output_df.to_csv(output_path, index=False)
    print(f"✅ 已保存：{output_path}")
