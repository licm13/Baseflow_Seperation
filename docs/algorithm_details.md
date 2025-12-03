# 基流分割算法详解

## 概述

本文档详细介绍了 `src/baseflow/methods/` 中实现的 12 种基流分割方法的数学原理和算法逻辑。这些方法用于将河流总径流分解为基流（地下水补给）和快速径流（降雨产生的地表径流）两部分。

---

## 1. LH 数字滤波器 (Lyne & Hollick, 1979)

### 算法概述

LH 滤波器是最经典的基流分割方法之一，采用双向递归数字滤波技术。该方法首先正向扫描时间序列，然后反向扫描，以消除相位偏移。

### 数学公式

**第一遍（正向扫描）**：

$$
b_{t+1} = \beta \cdot b_t + \frac{1-\beta}{2} \cdot (Q_t + Q_{t+1})
$$

**第二遍（反向扫描）**：

$$
b_t = \beta \cdot b_{t+1} + \frac{1-\beta}{2} \cdot (b'_{t} + b'_{t+1})
$$

其中 $b'$ 表示第一遍扫描的结果。

**约束条件**：

$$
b_t \leq Q_t \quad \forall t
$$

### 参数说明

| 参数 | 符号 | 物理意义 | 典型范围 |
|------|------|----------|----------|
| 流量 | $Q_t$ | 第 $t$ 天的总流量 | > 0 |
| 基流 | $b_t$ | 第 $t$ 天的基流 | 0 ~ $Q_t$ |
| 滤波参数 | $\beta$ | 控制滤波器的平滑程度，值越大越平滑 | 0.9 ~ 0.95 |

### 推荐值

- Nathan & McMahon (1990) 推荐 $\beta = 0.925$

### 实现细节

```python
# 源码位置: src/baseflow/methods/LH.py
@njit
def LH(Q, beta=0.925, return_exceed=False):
    # 第一遍：正向扫描
    b[0] = Q[0]
    for i in range(Q.shape[0] - 1):
        b[i + 1] = beta * b[i] + (1 - beta) / 2 * (Q[i] + Q[i + 1])
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]

    # 第二遍：反向扫描
    b1 = np.copy(b)
    for i in range(Q.shape[0] - 2, -1, -1):
        b[i] = beta * b[i + 1] + (1 - beta) / 2 * (b1[i + 1] + b1[i])
        if b[i] > b1[i]:
            b[i] = b1[i]
    return b
```

### 参考文献

- Lyne, V., & Hollick, M. (1979). *Stochastic time-variable rainfall-runoff modelling*. Institute of Engineers Australia National Conference.
- Nathan, R. J., & McMahon, T. A. (1990). Evaluation of automated techniques for base flow and recession analyses. *Water Resources Research*, 26(7), 1465-1473.

---

## 2. Eckhardt 滤波器 (Eckhardt, 2005)

### 算法概述

Eckhardt 滤波器是一种单向递归数字滤波方法，引入了最大基流指数 (BFI_max) 作为物理约束，使得算法更符合实际水文过程。

### 数学公式

$$
b_{t+1} = \frac{(1 - BFI_{max}) \cdot \alpha \cdot b_t + (1 - \alpha) \cdot BFI_{max} \cdot Q_{t+1}}{1 - \alpha \cdot BFI_{max}}
$$

**约束条件**：

$$
b_t \leq Q_t \quad \forall t
$$

### 参数说明

| 参数 | 符号 | 物理意义 | 典型范围 |
|------|------|----------|----------|
| 消退系数 | $\alpha$ | 描述地下水释放速率，值越大释放越慢 | 0.90 ~ 0.995 |
| 最大基流指数 | $BFI_{max}$ | 基流占总流量的最大比例 | 0.50 ~ 0.80 |
| 流量 | $Q_t$ | 第 $t$ 天的总流量 | > 0 |
| 基流 | $b_t$ | 第 $t$ 天的基流 | 0 ~ $Q_t$ |

### 推荐值

- 多孔介质流域（砂岩、碳酸盐岩）：$BFI_{max} = 0.80$
- 混合型流域：$BFI_{max} = 0.50$
- 硬岩流域（花岗岩、片岩）：$BFI_{max} = 0.25$

### 实现细节

```python
# 源码位置: src/baseflow/methods/Eckhardt.py
@njit
def Eckhardt(Q, b_LH, a, BFImax, return_exceed=False):
    b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    for i in range(Q.shape[0] - 1):
        b[i + 1] = ((1 - BFImax) * a * b[i] + (1 - a) * BFImax * Q[i + 1]) / (1 - a * BFImax)
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
    return b
```

### 参考文献

- Eckhardt, K. (2005). How to construct recursive digital filters for baseflow separation. *Hydrological Processes*, 19(2), 507-515.

---

## 3. Chapman 滤波器 (Chapman, 1991)

### 算法概述

Chapman 滤波器基于线性储存模型，考虑了相邻时刻的流量变化。

### 数学公式

$$
b_{t+1} = \frac{3\alpha - 1}{3 - \alpha} \cdot b_t + \frac{1 - \alpha}{3 - \alpha} \cdot (Q_{t+1} + Q_t)
$$

**约束条件**：

$$
b_t \leq Q_t \quad \forall t
$$

### 参数说明

| 参数 | 符号 | 物理意义 | 典型范围 |
|------|------|----------|----------|
| 消退系数 | $\alpha$ | 描述流域的储存特性 | 0.90 ~ 0.995 |
| 流量 | $Q_t$ | 第 $t$ 天的总流量 | > 0 |
| 基流 | $b_t$ | 第 $t$ 天的基流 | 0 ~ $Q_t$ |

### 消退系数估计

消退系数 $\alpha$ 可以从流量消退曲线估计：

$$
\alpha = \frac{Q_{t+1}}{Q_t} \quad \text{(在消退期)}
$$

### 实现细节

```python
# 源码位置: src/baseflow/methods/Chapman.py
@njit
def Chapman(Q, b_LH, a, return_exceed=False):
    b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    for i in range(Q.shape[0] - 1):
        b[i + 1] = (3 * a - 1) / (3 - a) * b[i] + (1 - a) / (3 - a) * (Q[i + 1] + Q[i])
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
    return b
```

### 参考文献

- Chapman, T. (1991). Comment on "Evaluation of automated techniques for base flow and recession analyses" by Nathan, R. J., & McMahon, T. A. *Water Resources Research*, 27(7), 1783-1784.

---

## 4. Chapman-Maxwell (CM) 滤波器 (Chapman & Maxwell, 1996)

### 算法概述

CM 滤波器是 Chapman 滤波器的改进版本，通过不同的系数配置提高了分离精度。

### 数学公式

$$
b_{t+1} = \frac{\alpha}{2 - \alpha} \cdot b_t + \frac{1 - \alpha}{2 - \alpha} \cdot Q_{t+1}
$$

**约束条件**：

$$
b_t \leq Q_t \quad \forall t
$$

### 参数说明

| 参数 | 符号 | 物理意义 | 典型范围 |
|------|------|----------|----------|
| 消退系数 | $\alpha$ | 流域储存特性参数 | 0.90 ~ 0.995 |
| 流量 | $Q_t$ | 第 $t$ 天的总流量 | > 0 |
| 基流 | $b_t$ | 第 $t$ 天的基流 | 0 ~ $Q_t$ |

### 实现细节

```python
# 源码位置: src/baseflow/methods/CM.py
@njit
def CM(Q, b_LH, a, return_exceed=False):
    b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    for i in range(Q.shape[0] - 1):
        b[i + 1] = a / (2 - a) * b[i] + (1 - a) / (2 - a) * Q[i + 1]
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
    return b
```

### 参考文献

- Chapman, T. G., & Maxwell, A. I. (1996). Baseflow separation-comparison of numerical methods with tracer experiments. *Institute Engineers Australia National Conference*. Pub. 96/05, 539-545.

---

## 5. Boughton 双参数滤波器 (Boughton, 2004)

### 算法概述

Boughton 滤波器引入了校准参数 $C$，通过优化提高分离精度。

### 数学公式

$$
b_{t+1} = \frac{\alpha}{1 + C} \cdot b_t + \frac{C}{1 + C} \cdot Q_{t+1}
$$

**约束条件**：

$$
b_t \leq Q_t \quad \forall t
$$

### 参数说明

| 参数 | 符号 | 物理意义 | 典型范围 |
|------|------|----------|----------|
| 消退系数 | $\alpha$ | 流域储存特性 | 0.90 ~ 0.995 |
| 校准参数 | $C$ | 通过优化确定的参数，控制基流比例 | 自动校准 |
| 流量 | $Q_t$ | 第 $t$ 天的总流量 | > 0 |
| 基流 | $b_t$ | 第 $t$ 天的基流 | 0 ~ $Q_t$ |

### 参数校准

参数 $C$ 通过最小化与参考基流（如 LH 滤波结果）的差异自动校准：

$$
C^* = \arg\min_C \sum_{t} (b_t - b_{LH,t})^2
$$

### 实现细节

```python
# 源码位置: src/baseflow/methods/Boughton.py
@njit
def Boughton(Q, b_LH, a, C, return_exceed=False):
    b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    for i in range(Q.shape[0] - 1):
        b[i + 1] = a / (1 + C) * b[i] + C / (1 + C) * Q[i + 1]
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
    return b
```

### 参考文献

- Boughton, W. C. (2004). The Australian water balance model. *Environmental Modelling & Software*, 19(10), 943-956.

---

## 6. Furey 数字滤波器 (Furey & Gupta, 2001, 2003)

### 算法概述

Furey 滤波器基于水量平衡原理，引入参数 $A$ 来调整基流响应。

### 数学公式

$$
b_{t+1} = [\alpha - A(1 - \alpha)] \cdot b_t + A(1 - \alpha) \cdot Q_t
$$

**约束条件**：

$$
b_t \leq Q_t \quad \forall t
$$

### 参数说明

| 参数 | 符号 | 物理意义 | 典型范围 |
|------|------|----------|----------|
| 消退系数 | $\alpha$ | 流域储存特性 | 0.90 ~ 0.995 |
| 校准参数 | $A$ | 控制基流响应速度的参数 | 自动校准 |
| 流量 | $Q_t$ | 第 $t$ 天的总流量 | > 0 |
| 基流 | $b_t$ | 第 $t$ 天的基流 | 0 ~ $Q_t$ |

### 实现细节

```python
# 源码位置: src/baseflow/methods/Furey.py
@njit
def Furey(Q, b_LH, a, A, return_exceed=False):
    b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    for i in range(Q.shape[0] - 1):
        b[i + 1] = (a - A * (1 - a)) * b[i] + A * (1 - a) * Q[i]
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
    return b
```

### 参考文献

- Furey, P. R., & Gupta, V. K. (2001). A physically based filter for separating base flow from streamflow time series. *Water Resources Research*, 37(11), 2709-2722.
- Furey, P. R., & Gupta, V. K. (2003). Tests of two physically based filters for base flow separation. *Water Resources Research*, 39(10).

---

## 7. EWMA 滤波器 (Tularam & Ilahee, 2008)

### 算法概述

指数加权移动平均（EWMA）滤波器是一种简单的平滑方法，通过指数权重对流量进行平滑处理。

### 数学公式

$$
b_{t+1} = (1 - e) \cdot b_t + e \cdot Q_{t+1}
$$

**约束条件**：

$$
b_t \leq Q_t \quad \forall t
$$

### 参数说明

| 参数 | 符号 | 物理意义 | 典型范围 |
|------|------|----------|----------|
| 平滑参数 | $e$ | 控制对新观测值的响应速度，值越大响应越快 | 0.01 ~ 0.30 |
| 流量 | $Q_t$ | 第 $t$ 天的总流量 | > 0 |
| 基流 | $b_t$ | 第 $t$ 天的基流 | 0 ~ $Q_t$ |

### 实现细节

```python
# 源码位置: src/baseflow/methods/EWMA.py
@njit
def EWMA(Q, b_LH, a, e, return_exceed=False):
    b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    for i in range(Q.shape[0] - 1):
        b[i + 1] = (1 - e) * b[i] + e * Q[i + 1]
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
    return b
```

### 参考文献

- Tularam, G. A., & Ilahee, M. (2008). Exponential smoothing method of base flow separation and its impact on continuous loss estimates. *American Journal of Environmental Sciences*, 4(2), 136-144.

---

## 8. Willems 数字滤波器 (Willems, 2009)

### 算法概述

Willems 滤波器引入了快速径流比例参数 $w$，提供了更灵活的基流分离方案。

### 数学公式

首先计算中间参数：

$$
v = \frac{(1 - w)(1 - \alpha)}{2w}
$$

然后计算基流：

$$
b_{t+1} = \frac{\alpha - v}{1 + v} \cdot b_t + \frac{v}{1 + v} \cdot (Q_t + Q_{t+1})
$$

**约束条件**：

$$
b_t \leq Q_t \quad \forall t
$$

### 参数说明

| 参数 | 符号 | 物理意义 | 典型范围 |
|------|------|----------|----------|
| 消退系数 | $\alpha$ | 流域储存特性 | 0.90 ~ 0.995 |
| 快速径流比例 | $w$ | 快速径流占总流量的平均比例 | 自动校准 |
| 流量 | $Q_t$ | 第 $t$ 天的总流量 | > 0 |
| 基流 | $b_t$ | 第 $t$ 天的基流 | 0 ~ $Q_t$ |

### 实现细节

```python
# 源码位置: src/baseflow/methods/Willems.py
@njit
def Willems(Q, b_LH, a, w, return_exceed=False):
    b = np.zeros(Q.shape[0])
    b[0] = b_LH[0]
    v = (1 - w) * (1 - a) / (2 * w)
    for i in range(Q.shape[0] - 1):
        b[i + 1] = (a - v) / (1 + v) * b[i] + v / (1 + v) * (Q[i] + Q[i + 1])
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
    return b
```

### 参考文献

- Willems, P. (2009). A time series tool to support the multi-criteria performance evaluation of rainfall-runoff models. *Environmental Modelling & Software*, 24(3), 311-321.

---

## 9. UKIH 图解法 (UK Institute of Hydrology, 1980)

### 算法概述

UKIH 方法是一种基于图解的基流分割技术，通过识别流量序列中的转折点并进行线性插值来估计基流。

### 算法步骤

1. **分段**：将流量序列分成每段 N=5 天的块
2. **识别最小值**：找出每段的最小流量
3. **识别转折点**：应用 0.9 规则识别有效转折点
4. **线性插值**：在转折点之间进行线性插值

### 转折点识别规则

对于连续三个最小值 $Q_{min,i}$、$Q_{min,i+1}$、$Q_{min,i+2}$，如果满足：

$$
0.9 \cdot Q_{min,i+1} < Q_{min,i} \quad \text{且} \quad 0.9 \cdot Q_{min,i+1} < Q_{min,i+2}
$$

则 $Q_{min,i+1}$ 被识别为转折点。

### 线性插值公式

在转折点 $t_n$ 和 $t_{n+1}$ 之间：

$$
b_t = b_{t_n} + \frac{b_{t_{n+1}} - b_{t_n}}{t_{n+1} - t_n} \cdot (t - t_n)
$$

**约束条件**：

$$
b_t \leq Q_t \quad \forall t
$$

### 参数说明

| 参数 | 符号 | 物理意义 | 典型范围 |
|------|------|----------|----------|
| 块大小 | $N$ | 分段的天数 | 5（固定） |
| 流量 | $Q_t$ | 第 $t$ 天的总流量 | > 0 |
| 基流 | $b_t$ | 第 $t$ 天的基流 | 0 ~ $Q_t$ |

### 实现细节

```python
# 源码位置: src/baseflow/methods/UKIH.py
def UKIH(Q, b_LH, return_exceed=False):
    N = 5
    block_end = Q.shape[0] // N * N
    # 找出每段的最小值
    idx_min = np.argmin(Q[:block_end].reshape(-1, N), axis=1)
    idx_min = idx_min + np.arange(0, block_end, N)
    # 识别转折点
    idx_turn = UKIH_turn(Q, idx_min)
    # 线性插值
    b = linear_interpolation(Q, idx_turn, return_exceed=return_exceed)
    b[:idx_turn[0]] = b_LH[:idx_turn[0]]
    b[idx_turn[-1] + 1:] = b_LH[idx_turn[-1] + 1:]
    return b
```

### 参考文献

- Institute of Hydrology (1980). *Low flow studies*. Wallingford, UK.

---

## 10. Fixed Interval 方法 (HYSEP, Sloto & Crouse, 1996)

### 算法概述

Fixed Interval 方法来自 HYSEP 程序，通过固定时间间隔识别最小流量来估计基流。

### 算法步骤

1. **计算时间间隔**：基于流域面积计算间隔 $N$
2. **分段**：将流量序列分成每段 $N$ 天的块
3. **识别最小值**：找出每段的最小流量作为该段的基流

### 时间间隔计算

$$
N = \begin{cases}
\text{int}(A^{0.2}) & \text{if metric units (km}^2\text{)} \\
\text{int}((0.386 \cdot A)^{0.2}) & \text{if imperial units (mi}^2\text{)}
\end{cases}
$$

其中 $A$ 是流域面积。

### 基流估计

$$
b_t = \min(Q_{N \cdot i}, Q_{N \cdot i + 1}, \ldots, Q_{N \cdot (i+1) - 1}) \quad \text{for } t \in [N \cdot i, N \cdot (i+1))
$$

### 参数说明

| 参数 | 符号 | 物理意义 | 典型范围 |
|------|------|----------|----------|
| 流域面积 | $A$ | 流域面积（km²） | > 0 |
| 时间间隔 | $N$ | 分段长度（天） | 自动计算 |
| 流量 | $Q_t$ | 第 $t$ 天的总流量 | > 0 |
| 基流 | $b_t$ | 第 $t$ 天的基流 | 0 ~ $Q_t$ |

### 实现细节

```python
# 源码位置: src/baseflow/methods/Fixed.py
def Fixed(Q, area=None):
    inN = hysep_interval(area)  # 计算间隔
    return Fixed_interpolation(Q, inN)

@njit
def Fixed_interpolation(Q, inN):
    b = np.zeros(Q.shape[0])
    n = Q.shape[0] // inN
    for i in prange(n):
        b[inN * i : inN * (i + 1)] = np.min(Q[inN * i : inN * (i + 1)])
    if n * inN != Q.shape[0]:
        b[n * inN :] = np.min(Q[n * inN :])
    return b
```

### 参考文献

- Sloto, R. A., & Crouse, M. Y. (1996). *HYSEP: A computer program for streamflow hydrograph separation and analysis* (Vol. 96). US Geological Survey.

---

## 11. Local Minimum 方法 (HYSEP, Sloto & Crouse, 1996)

### 算法概述

Local Minimum 方法也来自 HYSEP 程序，通过识别局部最小值并进行线性插值来估计基流。

### 算法步骤

1. **计算时间间隔**：与 Fixed Interval 方法相同
2. **识别局部最小值**：在滑动窗口 $[t - N/2, t + N/2]$ 内找最小值
3. **线性插值**：在局部最小值之间进行线性插值

### 局部最小值判定

对于时刻 $t$，如果：

$$
Q_t = \min(Q_{t-N/2}, \ldots, Q_{t+N/2})
$$

则 $Q_t$ 是局部最小值。

### 参数说明

| 参数 | 符号 | 物理意义 | 典型范围 |
|------|------|----------|----------|
| 流域面积 | $A$ | 流域面积（km²） | > 0 |
| 时间间隔 | $N$ | 滑动窗口半径（天） | 自动计算 |
| 流量 | $Q_t$ | 第 $t$ 天的总流量 | > 0 |
| 基流 | $b_t$ | 第 $t$ 天的基流 | 0 ~ $Q_t$ |

### 参考文献

- Sloto, R. A., & Crouse, M. Y. (1996). *HYSEP: A computer program for streamflow hydrograph separation and analysis* (Vol. 96). US Geological Survey.

---

## 12. Slide Interval 方法 (HYSEP, Sloto & Crouse, 1996)

### 算法概述

Slide Interval 方法结合了 Fixed 和 Local 方法的优点，通过滑动窗口在每个位置找最小值。

### 算法步骤

1. **计算时间间隔**：基于流域面积
2. **滑动窗口**：在每个位置 $t$ 应用长度为 $N$ 的窗口
3. **识别最小值**：找出窗口内的最小值作为基流

### 基流估计

$$
b_t = \min(Q_{t}, Q_{t+1}, \ldots, Q_{t+N-1})
$$

### 参数说明

| 参数 | 符号 | 物理意义 | 典型范围 |
|------|------|----------|----------|
| 流域面积 | $A$ | 流域面积（km²） | > 0 |
| 时间间隔 | $N$ | 滑动窗口长度（天） | 自动计算 |
| 流量 | $Q_t$ | 第 $t$ 天的总流量 | > 0 |
| 基流 | $b_t$ | 第 $t$ 天的基流 | 0 ~ $Q_t$ |

### 参考文献

- Sloto, R. A., & Crouse, M. Y. (1996). *HYSEP: A computer program for streamflow hydrograph separation and analysis* (Vol. 96). US Geological Survey.

---

## 方法对比总结

### 按复杂度分类

| 类别 | 方法 | 参数数量 | 是否需要校准 |
|------|------|----------|--------------|
| **简单滤波器** | LH | 1 ($\beta$) | 否 |
| | EWMA | 1 ($e$) | 是 |
| **递归滤波器** | Chapman | 1 ($\alpha$) | 否 |
| | CM | 1 ($\alpha$) | 否 |
| | Eckhardt | 2 ($\alpha$, $BFI_{max}$) | 否 |
| **双参数滤波器** | Boughton | 2 ($\alpha$, $C$) | 是 |
| | Furey | 2 ($\alpha$, $A$) | 是 |
| | Willems | 2 ($\alpha$, $w$) | 是 |
| **图解法** | UKIH | 0 | 否 |
| | Fixed | 1 (面积) | 否 |
| | Local | 1 (面积) | 否 |
| | Slide | 1 (面积) | 否 |

### 按数据需求分类

| 数据需求 | 方法 |
|---------|------|
| **仅需流量数据** | LH, Chapman, CM, Eckhardt, Boughton, Furey, EWMA, Willems, UKIH |
| **需要流域面积** | Fixed, Local, Slide |

### 推荐使用场景

1. **通用场景**：Eckhardt（精度高、物理意义明确）
2. **快速分析**：LH（计算快、稳定性好）
3. **小流域**：Chapman、CM（响应快）
4. **大流域**：HYSEP 系列（Fixed, Local, Slide）
5. **研究对比**：使用多种方法ensemble

---

## 参考文献汇总

1. Lyne, V., & Hollick, M. (1979). *Stochastic time-variable rainfall-runoff modelling*. Institute of Engineers Australia National Conference.

2. Nathan, R. J., & McMahon, T. A. (1990). Evaluation of automated techniques for base flow and recession analyses. *Water Resources Research*, 26(7), 1465-1473.

3. Chapman, T. (1991). Comment on "Evaluation of automated techniques for base flow and recession analyses". *Water Resources Research*, 27(7), 1783-1784.

4. Chapman, T. G., & Maxwell, A. I. (1996). Baseflow separation-comparison of numerical methods with tracer experiments. *Institute Engineers Australia National Conference*, Pub. 96/05, 539-545.

5. Furey, P. R., & Gupta, V. K. (2001). A physically based filter for separating base flow from streamflow time series. *Water Resources Research*, 37(11), 2709-2722.

6. Furey, P. R., & Gupta, V. K. (2003). Tests of two physically based filters for base flow separation. *Water Resources Research*, 39(10).

7. Boughton, W. C. (2004). The Australian water balance model. *Environmental Modelling & Software*, 19(10), 943-956.

8. Eckhardt, K. (2005). How to construct recursive digital filters for baseflow separation. *Hydrological Processes*, 19(2), 507-515.

9. Tularam, G. A., & Ilahee, M. (2008). Exponential smoothing method of base flow separation and its impact on continuous loss estimates. *American Journal of Environmental Sciences*, 4(2), 136-144.

10. Willems, P. (2009). A time series tool to support the multi-criteria performance evaluation of rainfall-runoff models. *Environmental Modelling & Software*, 24(3), 311-321.

11. Sloto, R. A., & Crouse, M. Y. (1996). *HYSEP: A computer program for streamflow hydrograph separation and analysis* (Vol. 96). US Geological Survey.

12. Institute of Hydrology (1980). *Low flow studies*. Wallingford, UK.

---

**文档版本**: 1.0
**最后更新**: 2025-12-03
**维护者**: Baseflow Separation Team
