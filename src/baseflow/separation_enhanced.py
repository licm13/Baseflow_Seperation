"""高级基流分割工作流 API (带详细中文注释)。

本模块提供用户友好的函数,用于对水文时间序列数据应用基流分割算法。
支持单站点和多站点批处理,并具有自动参数标定功能。

主要改进:
- 详细的中文行内注释,解释每个步骤的"为什么"
- (可选)使用方法注册表进行动态分派
- 保持与原有 API 的完全兼容性
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from .comparision import KGE, strict_baseflow
from .config import ALL_METHODS, get_param_range
from .methods import (
    Boughton,
    Chapman,
    CM,
    Eckhardt,
    EWMA,
    Fixed,
    Furey,
    LH,
    Local,
    Slide,
    UKIH,
    Willems,
)
from .param_estimate import param_calibrate, recession_coefficient
from .utils import clean_streamflow, exist_ice, format_method, geo2imagexy

__all__ = ["single", "separation"]


def single(
    series: pd.Series,
    area: Optional[float] = None,
    ice: Optional[Union[npt.NDArray[np.bool_], Tuple[List[int], List[int]]]] = None,
    method: Union[str, List[str]] = "all",
    return_kge: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """对单个水文站点进行基流分割。

    这是核心的单站点分割函数,应用一种或多种基流分割算法到流量时间序列。
    它自动处理数据清洗、参数估计和可选的性能评估(使用 KGE 指标)。

    工作流程:
        1. 数据预处理: 清洗流量数据,移除异常值
        2. 冻结期处理: 识别并掩膜冻结/融雪期,避免算法失真
        3. 严格基流识别: 找出可靠的基流主导期,用于参数估计和评估
        4. 参数估计: 根据需要估算衰退系数和其他参数
        5. 基流分割: 应用选定的分割方法
        6. 性能评估: 计算 KGE 分数(如果启用)

    Args:
        series: 流量时间序列,必须带有 DatetimeIndex
        area: 流域面积(km²)。HYSEP 方法(Local, Fixed, Slide)需要此参数
        ice: 冻结期定义。可以是:
            - 布尔数组(与 series 长度相同,True 表示冻结期)
            - 元组: [(开始月, 开始日), (结束月, 结束日)]
            - None (跳过冻结期掩膜)
        method: 要应用的方法名称。选项:
            - "all": 应用所有 12 种方法
            - 单个方法名: 如 "LH", "Eckhardt"
            - 方法名列表: 如 ["LH", "Chapman", "Eckhardt"]
        return_kge: 是否计算 KGE 分数(对比严格基流)

    Returns:
        元组,包含:
            - DataFrame: 每种方法的基流序列(索引=日期,列=方法名)
            - Series: 每种方法的 KGE 分数(如果 return_kge=True),否则为 None

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2010-01-01', periods=365, freq='D')
        >>> flow = pd.Series(np.random.lognormal(2, 1, 365), index=dates)
        >>> baseflow, kge_scores = single(flow, area=1000, method=["LH", "Eckhardt"])
        >>> print(kge_scores)

    Note:
        需要特定输入的方法:
        - Local, Fixed, Slide: 需要 'area' 参数
        - Chapman, CM, Boughton, Furey, Eckhardt, Willems: 需要衰退系数
          (会从数据自动估算)
    """
    # ========================================================================
    # 步骤1: 数据预处理 - 清洗流量数据
    # ========================================================================
    # clean_streamflow 函数执行以下操作:
    #   - 移除或插值缺失值(NaN)
    #   - 确保时间序列按时间排序
    #   - 处理负流量值(设为零或插值)
    #   - 返回清洁的日期索引和流量数组
    date, Q = clean_streamflow(series)

    # 格式化方法参数: 将 "all" 转换为方法列表,或验证方法名
    method = format_method(method)

    # ========================================================================
    # 步骤2: 冻结期处理 - 识别并掩膜冰冻/融雪期
    # ========================================================================
    # 为什么需要冻结期掩膜?
    #   - 在冻结期和融雪期,流量动态受冰雪融化控制,不遵循正常的降雨-径流关系
    #   - 基流分割算法假设正常的水文过程,在冻结期会产生不可靠的结果
    #   - 掩膜这些时期可以提高参数估计和基流识别的准确性
    if not isinstance(ice, np.ndarray) or ice.shape[0] == 12:
        # 如果 ice 不是数组,或是月度数组,调用 exist_ice 转换为布尔掩膜
        # exist_ice 使用全球冻土数据或用户定义的日期范围
        ice = exist_ice(date, ice)

    # ========================================================================
    # 步骤3: 严格基流识别 - 找出可靠的基流主导期
    # ========================================================================
    # strict_baseflow 函数识别流量主要由基流贡献的时期:
    #   - 移除大流量事件(通常是洪峰,基流贡献小)
    #   - 移除冻结/融雪期
    #   - 返回布尔掩膜,True 表示严格基流期
    # 这些时期用于:
    #   - 参数估计(更准确的衰退系数)
    #   - 性能评估(将估算结果与可靠的基流期对比)
    strict = strict_baseflow(Q, ice)

    # ========================================================================
    # 步骤4: 参数估计 - 估算方法所需的参数
    # ========================================================================
    # 4.1 衰退系数估算
    # ----------------------
    # 检查是否有任何选定的方法需要衰退系数 a
    # 衰退系数描述基流的消退速度: Q[t+1] = a * Q[t]
    # 典型值: a ∈ [0.90, 0.995] 对于日尺度数据
    if any(m in ["Chapman", "CM", "Boughton", "Furey", "Eckhardt", "Willems"] for m in method):
        # recession_coefficient 函数:
        #   - 分析严格基流期的流量衰减模式
        #   - 使用 Eckhardt (2008) 的方法估算 a
        #   - 取消退阶段 -dQ/Q 比率的第5百分位数
        a = recession_coefficient(Q, strict)

    # 4.2 LH 滤波基准计算
    # ----------------------
    # 许多方法使用 LH 滤波器作为基准或初始值:
    #   - LH 是最简单、最快的方法
    #   - 提供合理的基流初始估计
    #   - 其他方法在此基础上进行改进
    # 即使不使用 LH 方法,也需要计算它作为其他方法的输入
    b_LH = LH(Q)

    # ========================================================================
    # 步骤5: 初始化结果存储
    # ========================================================================
    # 创建 DataFrame 存储每种方法的基流结果
    # - 索引: 原始日期序列
    # - 列: 选定的方法名
    # - 初始值: NaN (稍后填充)
    b = pd.DataFrame(np.nan, index=date, columns=method)

    # ========================================================================
    # 步骤6: 应用每种选定的方法
    # ========================================================================
    # 注意: 这里使用 if/elif 链进行方法分派
    # (未来可以改为注册表模式,但为保持兼容性暂时保留)
    for m in method:
        # ====================================================================
        # 图形法/滤波法 - 不需要额外参数
        # ====================================================================
        if m == "UKIH":
            # UK Institute of Hydrology 方法
            # - 基于 LH 滤波的变体
            # - 使用反向滤波增强稳定性
            b[m] = UKIH(Q, b_LH)

        elif m == "LH":
            # Lyne-Hollick 数字滤波器
            # - 最经典的基流分割方法之一
            # - 使用双遍递归滤波(正向+反向)
            # - 已在步骤 4.2 中计算,直接使用
            b[m] = b_LH

        # ====================================================================
        # HYSEP 方法 - 需要流域面积
        # ====================================================================
        elif m == "Local":
            # HYSEP Local Minimum 方法
            # - 基于局部最小值的图形法
            # - 使用流域面积确定窗口大小: N = 2 * sqrt(area)
            b[m] = Local(Q, b_LH, area)

        elif m == "Fixed":
            # HYSEP Fixed Interval 方法
            # - 固定间隔窗口的图形法
            # - 窗口大小同样基于流域面积
            b[m] = Fixed(Q, area)

        elif m == "Slide":
            # HYSEP Sliding Interval 方法
            # - 滑动窗口图形法
            # - 结合 Local 和 Fixed 的优点
            b[m] = Slide(Q, area)

        # ====================================================================
        # 参数化滤波法 - 需要衰退系数 a,无需额外标定
        # ====================================================================
        elif m == "Chapman":
            # Chapman (1991) 数字滤波器
            # - 基于物理的参数化方法
            # - 使用衰退系数 a 和简单递归公式
            b[m] = Chapman(Q, b_LH, a)

        elif m == "CM":
            # Combined Method (Chapman 的变体)
            # - Chapman 方法的改进版本
            # - 对某些流域类型性能更好
            b[m] = CM(Q, b_LH, a)

        # ====================================================================
        # 需要参数标定的方法 - 使用网格搜索寻找最优参数
        # ====================================================================
        elif m == "Boughton":
            # Boughton 两参数滤波器
            # - 需要标定参数 C (衰退常数)
            # - 步骤:
            #   1. 获取参数搜索范围(从 config.py)
            #   2. 网格搜索最优 C (最小化 NSE 损失)
            #   3. 使用最优 C 进行分割
            param_range = get_param_range("Boughton")  # 默认: np.arange(0.0001, 0.1, 0.0001)
            C = param_calibrate(param_range, Boughton, Q, b_LH, a)  # 网格搜索
            b[m] = Boughton(Q, b_LH, a, C)  # 应用最优参数

        elif m == "Furey":
            # Furey 基流分割法
            # - 需要标定参数 A (缩放因子)
            param_range = get_param_range("Furey")  # 默认: np.arange(0.01, 10, 0.01)
            A = param_calibrate(param_range, Furey, Q, b_LH, a)
            b[m] = Furey(Q, b_LH, a, A)

        elif m == "Eckhardt":
            # Eckhardt 两参数数字滤波器
            # - 需要标定 BFImax (最大基流指数)
            # - BFImax 表示长期基流占总流量的最大比例
            # - 典型值: 永久性河流 0.8, 间歇性河流 0.5, 短暂性河流 0.25
            param_range = get_param_range("Eckhardt")  # 默认: np.arange(0.001, 1, 0.001)
            BFImax = param_calibrate(param_range, Eckhardt, Q, b_LH, a)
            b[m] = Eckhardt(Q, b_LH, a, BFImax)

        elif m == "EWMA":
            # Exponential Weighted Moving Average 滤波器
            # - 需要标定平滑参数 e
            # - 注意: EWMA 不需要衰退系数,所以传入 0 作为占位符
            param_range = get_param_range("EWMA")  # 默认: np.arange(0.0001, 0.1, 0.0001)
            e = param_calibrate(param_range, EWMA, Q, b_LH, 0)  # a=0 作为占位符
            b[m] = EWMA(Q, b_LH, 0, e)

        elif m == "Willems":
            # Willems 基流分割法
            # - 需要标定权重参数 w
            param_range = get_param_range("Willems")  # 默认: np.arange(0.001, 1, 0.001)
            w = param_calibrate(param_range, Willems, Q, b_LH, a)
            b[m] = Willems(Q, b_LH, a, w)

    # ========================================================================
    # 步骤7: 性能评估(可选)
    # ========================================================================
    if return_kge:
        # KGE (Kling-Gupta Efficiency) 是一个综合性能指标:
        #   - 评估相关性、偏差和变异性
        #   - 范围: -∞ 到 1,越接近 1 越好
        #   - KGE > 0.75: 优秀
        #   - KGE > 0.5: 良好
        #   - KGE < 0: 差于均值模型
        #
        # 这里的评估策略:
        #   - 仅在严格基流期进行评估(where strict==True)
        #   - 对比估算的基流与原始流量
        #   - 在严格基流期,基流应该≈总流量
        #   - KGE 分数高表示该方法在基流期的估算准确
        KGEs = pd.Series(
            KGE(
                b[strict].values,  # 估算的基流(仅严格基流期)
                np.repeat(Q[strict], len(method)).reshape(-1, len(method))  # 对比目标: 原始流量
            ),
            index=b.columns,
        )
        return b, KGEs
    else:
        return b, None


def separation(
    df: pd.DataFrame,
    df_sta: Optional[pd.DataFrame] = None,
    method: Union[str, List[str]] = "all",
    return_bfi: bool = False,
    return_kge: bool = False,
) -> Union[
    Dict[str, pd.DataFrame],
    Tuple[Dict[str, pd.DataFrame], pd.DataFrame],
    Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame],
]:
    """对多个水文站点执行基流分割(批处理模式)。

    此函数处理来自多个站点的流量数据,对每个站点应用选定的基流分割方法。
    可选地使用站点元数据(面积、坐标)来提高分割精度并计算性能指标。

    批处理工作流程:
        1. 加载全球冻土数据(用于自动检测冻结期)
        2. 为每个方法创建结果 DataFrame
        3. 对每个站点:
           a. 从 df_sta 读取站点属性(面积、经纬度)
           b. 根据坐标自动确定冻结期
           c. 调用 single() 进行分割
           d. 存储结果到对应的 DataFrame
        4. 计算 BFI 和 KGE(如果请求)
        5. 返回汇总结果

    Args:
        df: 流量数据 DataFrame (索引=日期, 列=站点ID)
        df_sta: 可选的站点元数据 DataFrame (索引=站点ID)
            支持的列:
            - 'area': 流域面积(km²),用于 HYSEP 方法
            - 'lon', 'lat': 坐标,用于自动冻结期检测
        method: 要应用的方法名称(参见 single() 的说明)
        return_bfi: 是否计算每个站点的基流指数(BFI)
        return_kge: 是否计算每个站点的 KGE 分数

    Returns:
        根据标志,返回:
        - dfs: 字典,映射方法名到基流 DataFrame (与输入 df 同形状)
        - df_bfi: BFI 值 DataFrame (站点 × 方法) 如果 return_bfi=True
        - df_kge: KGE 分数 DataFrame (站点 × 方法) 如果 return_kge=True

    Example:
        >>> # 创建多站点流量数据
        >>> dates = pd.date_range('2010-01-01', periods=365)
        >>> stations = ['Station_A', 'Station_B', 'Station_C']
        >>> flow_data = pd.DataFrame(
        ...     np.random.lognormal(2, 1, (365, 3)),
        ...     index=dates,
        ...     columns=stations
        ... )
        >>>
        >>> # 站点元数据
        >>> station_info = pd.DataFrame({
        ...     'area': [1000, 1500, 800],
        ...     'lon': [-120.5, -119.2, -121.0],
        ...     'lat': [45.2, 44.8, 46.1]
        ... }, index=stations)
        >>>
        >>> # 运行分割
        >>> results, bfi, kge = separation(
        ...     flow_data,
        ...     df_sta=station_info,
        ...     method=["LH", "Eckhardt"],
        ...     return_bfi=True,
        ...     return_kge=True
        ... )
        >>> print(f"应用的方法: {list(results.keys())}")
        >>> print(f"BFI 摘要:\\n{bfi}")
        >>> print(f"KGE 摘要:\\n{kge}")

    Note:
        - 通过 tqdm 进度条显示进度
        - 处理失败的站点会打印错误消息并跳过
        - 冻结期检测使用全球冻土数据(包含在包中)
    """
    # ========================================================================
    # 内部工作函数: 处理单个站点
    # ========================================================================
    def sep_work(s: str) -> None:
        """处理单个站点的基流分割。

        Args:
            s: 站点ID (df 和 df_sta 的列名/索引)

        Note:
            这是一个闭包函数,可以访问外部作用域的变量:
            - df, df_sta: 输入数据
            - method: 分割方法
            - dfs, df_bfi, df_kge: 结果存储
            - thawed: 全球冻土数据
        """
        try:
            # ================================================================
            # 步骤1: 读取站点属性
            # ================================================================
            area, ice = None, None

            # 辅助函数: 安全地将列值转换为数值
            # - 如果 df_sta 为 None 或列不存在,返回 NaN
            # - 否则尝试转换为数值(失败则返回 NaN)
            to_num = lambda col: (
                pd.to_numeric(df_sta.loc[s, col], errors="coerce")
                if (df_sta is not None) and (col in df_sta.columns)
                else np.nan
            )

            # 1.1 读取流域面积
            if np.isfinite(to_num("area")):
                area = to_num("area")

            # 1.2 根据坐标确定冻结期
            # ----------------------
            # 如果提供了经纬度坐标:
            #   - 将地理坐标转换为全球冻土栅格的像素坐标
            #   - 查询该位置的冻土状态(12个月)
            #   - 生成布尔掩膜: True=冻结, False=融化
            if np.isfinite(to_num("lon")):
                # geo2imagexy: 经纬度 -> 栅格行列号
                c, r = geo2imagexy(to_num("lon"), to_num("lat"))

                # thawed: 全球冻土数据, shape=(12, height, width)
                # thawed[month, r, c] = True 表示该月该位置融化
                # ~thawed = ice (取反得到冻结掩膜)
                ice = ~thawed[:, r, c]

                # 特殊情况: 如果全年冻结(永久冻土),使用默认冻结期
                # 默认: 11月1日 到 3月31日
                ice = ([11, 1], [3, 31]) if ice.all() else ice

            # ================================================================
            # 步骤2: 对站点 S 进行基流分割
            # ================================================================
            # 调用 single() 函数处理该站点
            # - df[s]: 该站点的流量时间序列
            # - ice: 冻结期掩膜(从坐标推断或为 None)
            # - area: 流域面积(如果可用)
            # - method: 用户选定的方法
            # - return_kge: 是否计算 KGE
            b, KGEs = single(
                df[s],
                ice=ice,
                area=area,
                method=method,
                return_kge=return_kge
            )

            # ================================================================
            # 步骤3: 写入结果到预先创建的 DataFrame
            # ================================================================
            # 3.1 基流时间序列
            # ----------------------
            # 将每种方法的基流写入对应的 DataFrame
            # - dfs[m]: 方法 m 的基流 DataFrame (所有站点)
            # - dfs[m].loc[b.index, s]: 站点 s 在有效日期的基流
            for m in method:
                dfs[m].loc[b.index, s] = b[m]

            # 3.2 基流指数(BFI)
            # ----------------------
            # BFI = 基流总量 / 总流量
            # 表示长期平均基流占总流量的比例
            # 典型值: 0.2-0.8 (取决于流域特征)
            if return_bfi:
                df_bfi.loc[s] = b.sum() / df.loc[b.index, s].abs().sum()

            # 3.3 KGE 分数
            # ----------------------
            # 从 single() 返回的 KGE 分数写入汇总 DataFrame
            if return_kge:
                df_kge.loc[s] = KGEs

        except BaseException:
            # 捕获所有异常,避免单个站点失败中断整个批处理
            # 打印错误消息,但继续处理其他站点
            print("\n无法为站点 {sta} 完成基流分割".format(sta=s))
            pass

    # ========================================================================
    # 主批处理逻辑
    # ========================================================================

    # 步骤1: 确保时间索引为 DatetimeIndex
    # ----------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 步骤2: 加载全球冻土数据
    # ----------------------
    # thawed.npz 包含全球冻土栅格数据:
    #   - shape: (12, height, width)
    #   - 12 个月的冻土状态
    #   - True = 融化, False = 冻结
    # 数据来源: https://doi.org/10.5194/essd-9-133-2017
    with np.load(Path(__file__).parent / "thawed.npz") as f:
        thawed = f["thawed"]

    # 步骤3: 格式化方法参数
    # ----------------------
    # 将 "all" 转换为方法列表,或验证方法名
    method = format_method(method)

    # 步骤4: 创建结果存储 DataFrame
    # ----------------------
    # 为每种方法创建一个 DataFrame,用于存储所有站点的基流
    # - 索引: 与输入 df 相同的日期
    # - 列: 与输入 df 相同的站点ID
    # - 初始值: NaN (将在 sep_work 中填充)
    dfs = {
        m: pd.DataFrame(np.nan, index=df.index, columns=df.columns, dtype=float)
        for m in method
    }

    # 步骤5: 创建 BFI 和 KGE 存储 DataFrame
    # ----------------------
    if return_bfi:
        # df_bfi: 行=站点, 列=方法
        df_bfi = pd.DataFrame(np.nan, index=df.columns, columns=method, dtype=float)
    if return_kge:
        # df_kge: 行=站点, 列=方法
        df_kge = pd.DataFrame(np.nan, index=df.columns, columns=method, dtype=float)

    # 步骤6: 对每个站点运行分割
    # ----------------------
    # 使用 tqdm 显示进度条
    # - total: 总站点数
    # - desc: 进度条描述(可选,默认为 "Processing")
    for s in tqdm(df.columns, total=df.shape[1]):
        sep_work(s)  # 处理站点 s

    # ========================================================================
    # 步骤7: 返回结果
    # ========================================================================
    # 根据用户请求的标志,返回不同组合的结果
    if return_bfi and return_kge:
        return dfs, df_bfi, df_kge
    if return_bfi and not return_kge:
        return dfs, df_bfi
    if not return_bfi and return_kge:
        return dfs, df_kge
    return dfs
