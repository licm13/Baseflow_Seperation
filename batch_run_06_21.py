import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from baseflow.separation import single  # 替换为你实际的导入路径

# ------------------ 配置路径 ------------------
input_folder = r"Z:\Runoff_Flood\China_runoff\daily_Q_06_21"
output_dir = r"Z:\LCM\Baseflow_seperation\daily06_21_outputs"
info_file = r"Z:\Runoff_Flood\China_runoff\info2.xlsx"
os.makedirs(output_dir, exist_ok=True)

# ------------------ 加载面积信息 ------------------
info_df = pd.read_excel(info_file)
missing_area_sites = []

def get_area_from_info(site_id: str, info_df: pd.DataFrame) -> float:
    id8_matches = info_df[info_df['id8'].astype(str) == site_id]
    id5_matches = info_df[info_df['id5'].astype(str).str.replace('.0', '', regex=False) == site_id]
    match = pd.concat([id8_matches, id5_matches]).drop_duplicates()
    
    if match.empty:
        print(f"⚠️ 无面积信息：{site_id}，使用默认值 150.0")
        missing_area_sites.append(site_id)
        return 150.0

    row = match.iloc[0]
    area, area2 = row.get('area', np.nan), row.get('area2', np.nan)

    if pd.notna(area) and pd.notna(area2):
        return (area + area2) / 2
    elif pd.notna(area):
        return area
    elif pd.notna(area2):
        return area2
    else:
        print(f"⚠️ 存在记录但无有效面积：{site_id}，使用默认值 150.0")
        missing_area_sites.append(site_id)
        return 150.0

# ------------------ 方法列表 ------------------
methods = [
    "LH", "Chapman", "CM", "Boughton", "Furey", "Eckhardt",
    "EWMA", "Willems", "UKIH", "Fixed", "Local", "Slide"
]

# ------------------ 读取所有xlsx文件 ------------------
xlsx_files = [f for f in os.listdir(input_folder) if f.endswith(".xlsx") and f.split('.')[0].isdigit()]
kge_all = []

for idx, filename in enumerate(tqdm(xlsx_files, desc="📦 处理daily_Q_06_21站点", unit="站点")):
    site_id = filename.replace(".xlsx", "")
    filepath = os.path.join(input_folder, filename)

    try:
        df = pd.read_excel(filepath)
        df.columns = ["year", "month", "day", "Q"]  # 强制列名
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    except Exception as e:
        print(f"❌ 读取失败 {filename}: {e}")
        continue

    area = get_area_from_info(site_id, info_df)
    if area <= 0:
        print(f"❌ 非法面积 {area} km²，跳过 {site_id}")
        continue

    # Q → R 转换
    df["R"] = df["Q"] * 86400 / area
    series = pd.Series(df["R"].values, index=df["date"])
    output_df = pd.DataFrame({"time": df["date"], "Q": df["Q"], "R": df["R"]})
    kge_record = {"site": site_id}

    print(f"\n📍 处理站点 {site_id}（{idx+1}/{len(xlsx_files)}），面积: {area:.2f} km²")

    for method in tqdm(methods, desc=f"    ➤ {site_id} 方法中", leave=False, unit="方法"):
        try:
            b, kge = single(series, area=area, method=[method], return_kge=True)
            output_df[method] = b[method].values
            kge_record[method] = kge[method]
        except Exception as e:
            print(f"❌ 方法 {method} 出错：{e}")
            output_df[method] = np.nan
            kge_record[method] = np.nan

    # 保存分割结果
    output_csv = os.path.join(output_dir, f"baseflow_separation_{site_id}.csv")
    output_df.to_csv(output_csv, index=False)
    print(f"✅ 保存完成：{output_csv}")

    kge_all.append(kge_record)

# ------------------ 保存KGE和缺失面积列表 ------------------
pd.DataFrame(kge_all).to_csv(os.path.join(output_dir, "baseflow_kge_all_sites.csv"), index=False)

if missing_area_sites:
    pd.DataFrame(missing_area_sites, columns=["site_id"]).to_csv(
        os.path.join(output_dir, "missing_area_sites.csv"), index=False
    )
    print(f"⚠️ 共 {len(missing_area_sites)} 个站点缺失面积，已保存。")
else:
    print("✅ 所有站点均成功匹配面积信息。")
