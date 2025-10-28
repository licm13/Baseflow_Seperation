# 基流分割批处理系统说明文档

- **作者**：Changming Li  
- **编写时间**：2024年7月  
- **最近修改**：2024年7月15日  
- **项目路径**：
  - 数据路径（输入）：  
    - `Z:\Runoff_Flood\China_runoff\daily_Q_60_16`（CSV格式，含`date+Q`）  
    - `Z:\Runoff_Flood\China_runoff\daily_Q_06_21`（XLSX格式，含`年/月/日/Q`）  
  - 面积匹配数据：`Z:\Runoff_Flood\China_runoff\info2.xlsx`
  - 输出路径：如 `Z:\LCM\Baseflow_seperation\daily60_16_outputs` 等

---

## 🧠 设计目的

本代码系统用于批量处理中国站点级日尺度径流数据，采用12种主流**基流分割方法**，得到基流序列与KGE评估结果，用于后续干旱分析、径流归因等水文分析任务。

---

## ⚙️ 输入数据说明

### 1. `daily_Q_60_16` 目录（CSV格式）  
- 每个文件名为**站点编号**（如 `15009000.csv`）  
- 列格式如下：  
  - `date`: 日期（YYYY-MM-DD）  
  - `Q`: 日均流量（单位：m³/s）

### 2. `daily_Q_06_21` 目录（XLSX格式）  
- 文件名为站点编号  
- 列格式为：
  - `年`, `月`, `日`, `Q(m³/s)`

### 3. 面积信息文件：`info2.xlsx`  
- 站点匹配字段：
  - `id8`（8位编号）或 `id5`（5位编号）
- 面积字段：
  - `area`, `area2`（单位：km²，若两者都有则取平均）

---

## 🔄 数据预处理逻辑

### ✅ Q → R 转换（单位转换）

将流量（m³/s）转换为径流深（mm/day）：

R = (Q × 86400) / Area

> 其中 Area 单位为 km²，R 单位为 mm/day。

### ✅ 时间列标准化：
- 对于 CSV 文件：直接使用 `date` 字段；
- 对于 XLSX 文件：将 `年/月/日` 合并为标准 `datetime` 对象。

---

## 🔧 使用的12种基流分割方法

| 方法名   | 是否需要 `area` 参数 | 特点                     |
| -------- | -------------------- | ------------------------ |
| LH       | 否                   | 一阶数字滤波器           |
| Chapman  | 否                   | 递归数字滤波器           |
| CM       | 否                   | Chapman改进型            |
| Boughton | 否                   | 多参数递推法             |
| Furey    | 否                   | 双参数拟合法             |
| Eckhardt | 否                   | 常用递归法，含BFImax     |
| EWMA     | 否                   | 指数滑动滤波             |
| Willems  | 否                   | 增强版递推滤波           |
| UKIH     | 否                   | 英国水文学会图解法       |
| Fixed    | ✅ 是                 | 固定窗口法（滑动极小值） |
| Local    | ✅ 是                 | 基于局部极小值           |
| Slide    | ✅ 是                 | 滑动图解法               |

---

## 📥 输出文件说明

### 每个站点生成一个分割结果文件

文件名：`baseflow_separation_站号.csv`

字段：time, Q, R, LH, Chapman, ..., Slide

### KGE指标总表：

- `baseflow_kge_all_sites.csv`

### 缺失面积站点列表（如有）：

- `missing_area_sites.csv`

---

## ❗ 注意事项与历史问题记录

- ⚠️ 面积缺失将使用默认值150.0，但会在 `missing_area_sites.csv` 中标记；
- ⚠️ XLSX格式字段不一致，需标准化列名为 `year/month/day/Q`；
- ⚠️ 某些方法可能在序列过短或全0等情况下报错，已自动跳过并填NaN；

---

## 💡 后续建议

- 添加图形绘制与 PDF 报告；
- 支持月尺度或年尺度分割；
- 多线程并行加速；
- 图形交互界面（GUI）支持；