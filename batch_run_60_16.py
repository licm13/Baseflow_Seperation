import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from baseflow.separation import single  # 替换为你的真实路径

# ------------------ 配置路径 ------------------
input_folder = r"Z:\Runoff_Flood\China_runoff\daily_Q_60_16"
output_dir = r"Z:\LCM\Baseflow_seperation\daily60_16_outputs"
info_file = r"Z:\Runoff_Flood\China_runoff\info2.xlsx"
os.makedirs(output_dir, exist_ok=True)

# ------------------ 加载面积信息 ------------------
info_df = pd.read_excel(info_file)
missing_area_sites = []

def get_area_from_info(site_id: str, info_df: pd.DataFrame) -> float:
    """从 info2.xlsx 中提取流域面积，优先取 area 和 area2 平均"""
    id8_matches = info_df[info_df['id8'].astype(str) == site_id]
    id5_matches = info_df[info_df['id5'].astype(str).str.replace('.0', '', regex=False) == site_id]
    match = pd.concat([id8_matches, id5_matches]).drop_duplicates()

    if match.empty:
        print(f"⚠️ 找不到站点 {site_id} 的面积信息，使用默认值 150.0")
        missing_area_sites.append(site_id)
        return 150.0

    row = match.iloc[0]
    area = row.get('area', np.nan)
    area2 = row.get('area2', np.nan)

    if pd.notna(area) and pd.notna(area2):
        return (area + area2) / 2
    elif pd.notna(area):
        return area
    elif pd.notna(area2):
        return area2
    else:
        print(f"⚠️ 站点 {site_id} 存在，但 area/area2 都缺失，使用默认值 150.0")
        missing_area_sites.append(site_id)
        return 150.0

# ------------------ 定义方法 ------------------
methods = [
    "LH", "Chapman", "CM", "Boughton", "Furey", "Eckhardt",
    "EWMA", "Willems", "UKIH", "Fixed", "Local", "Slide"
]

# ------------------ 扫描输入文件 ------------------
csv_files = [f for f in os.listdir(input_folder)
             if f.endswith(".csv") and f.split('.')[0].isdigit()]

kge_all = []

# ------------------ 主循环 ------------------
for idx, filename in enumerate(tqdm(csv_files, desc="📦 处理daily_Q_60_16站点", unit="站点")):
    site_id = filename.replace(".csv", "")
    filepath = os.path.join(input_folder, filename)

    try:
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        print(f"❌ 读取失败 {filename}：{e}")
        continue

    # 获取面积（km²）
    area = get_area_from_info(site_id, info_df)
    if area <= 0:
        print(f"❌ 面积非法 {area} km²，跳过站点 {site_id}")
        continue

    # 计算径流深 R（单位：mm）
    # Q (m³/s) × 86400 s/day / A (km²) × 1e6 m²/km² × 1000 mm/m = Q * 86400 / A (mm/day)
    df["R"] = df["Q"] * 86400 / area

    series = pd.Series(df["R"].values, index=df["date"])
    output_df = pd.DataFrame({"time": df["date"], "Q": df["Q"], "R": df["R"]})
    kge_record = {"site": site_id}

    print(f"\n📍 处理站点 {site_id}（{idx+1}/{len(csv_files)}），流域面积: {area:.2f} km²")

    for method in tqdm(methods, desc=f"    ➤ {site_id} 方法计算中", leave=False, unit="方法"):
        try:
            b, kge = single(series, area=area, method=[method], return_kge=True)
            output_df[method] = b[method].values
            kge_record[method] = kge[method]
        except Exception as e:
            print(f"❌ 方法 {method} 出错：{e}")
            output_df[method] = np.nan
            kge_record[method] = np.nan

    # 保存站点结果
    output_csv = os.path.join(output_dir, f"baseflow_separation_{site_id}.csv")
    output_df.to_csv(output_csv, index=False)
    print(f"✅ 保存完毕：{output_csv}")
    kge_all.append(kge_record)

# ------------------ 保存KGE与缺失面积记录 ------------------
kge_df = pd.DataFrame(kge_all)
kge_df.to_csv(os.path.join(output_dir, "baseflow_kge_all_sites.csv"), index=False)

if missing_area_sites:
    pd.DataFrame(missing_area_sites, columns=["site_id"]).to_csv(
        os.path.join(output_dir, "missing_area_sites.csv"), index=False
    )
    print(f"⚠️ 共 {len(missing_area_sites)} 个站点缺失面积，已记录。")
else:
    print("✅ 所有站点均成功匹配面积信息。")
