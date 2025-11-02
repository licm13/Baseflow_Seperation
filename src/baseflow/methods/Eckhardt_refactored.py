"""Eckhardt 两参数数字滤波法的面向对象实现。

Eckhardt 滤波器是一种参数化的基流分割方法,结合了物理意义明确的
衰退系数和最大基流指数(BFImax)两个参数。

参考文献:
    Eckhardt, K. (2005). How to construct recursive digital filters for
    baseflow separation. Hydrological Processes, 19(2), 507-515.
"""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from numba import njit

from ._base import BaseflowMethod, register_method

__all__ = ["EckhardtMethod", "Eckhardt"]


@register_method("Eckhardt")
class EckhardtMethod(BaseflowMethod):
    """Eckhardt 两参数数字滤波法。

    Eckhardt 方法的核心是一个递归数字滤波器,其形式为:
        b[i+1] = ((1-BFImax)*a*b[i] + (1-a)*BFImax*Q[i+1]) / (1-a*BFImax)

    参数说明:
        - a (recession coefficient): 衰退系数,描述基流消退速度
          典型值: 0.90-0.995 (日尺度)
        - BFImax: 最大基流指数,表示长期基流占总流量的最大比例
          典型值: 永久性河流 0.80, 间歇性河流 0.50, 短暂性河流 0.25

    特点:
        - 参数具有明确的物理意义
        - 适用于不同流域类型(通过调整 BFImax)
        - 需要参数标定以获得最优结果
    """

    name = "Eckhardt"
    description = "Eckhardt two-parameter digital filter (两参数递归滤波)"
    requires_area = False
    requires_recession_coef = True
    requires_calibration = True

    def __init__(self, BFImax: Optional[float] = None):
        """初始化 Eckhardt 方法。

        Args:
            BFImax: 最大基流指数。如果为 None,将通过标定确定。
        """
        self.BFImax = BFImax

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        """执行 Eckhardt 基流分割。

        Args:
            Q: 流量时间序列
            b_LH: LH 滤波基准(用于初始化)
            a: 衰退系数(必需)
            area: 未使用
            **kwargs: 可选参数
                - BFImax: 最大基流指数
                - return_exceed: 是否返回超限次数

        Returns:
            基流时间序列

        Raises:
            ValueError: 如果未提供衰退系数 a
            ValueError: 如果未提供 BFImax 且未预设
        """
        if a is None:
            raise ValueError("Eckhardt method requires recession coefficient 'a'")

        BFImax = kwargs.get("BFImax", self.BFImax)
        if BFImax is None:
            raise ValueError(
                "Eckhardt method requires BFImax parameter. "
                "Either set it during initialization or pass as keyword argument."
            )

        return_exceed = kwargs.get("return_exceed", False)

        return Eckhardt(Q, b_LH, a, BFImax, return_exceed=return_exceed)

    def calibrate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        param_range: Optional[npt.NDArray[np.float64]] = None,
    ) -> float:
        """标定最优 BFImax 参数。

        使用参数估计模块进行网格搜索标定。

        Args:
            Q: 流量时间序列
            b_LH: LH 滤波基准
            a: 衰退系数
            param_range: BFImax 搜索范围

        Returns:
            最优 BFImax 值
        """
        if a is None:
            raise ValueError("Recession coefficient 'a' is required for calibration")

        if param_range is None:
            # 默认搜索范围: 0.001 到 0.999,步长 0.001
            param_range = np.arange(0.001, 1.0, 0.001)

        from ..param_estimate import param_calibrate

        optimal_BFImax = param_calibrate(param_range, Eckhardt, Q, b_LH, a)
        self.BFImax = optimal_BFImax  # 保存标定结果

        return optimal_BFImax


# ============================================================================
# 原始 Numba JIT 编译函数(保持向后兼容)
# ============================================================================

@njit
def Eckhardt(
    Q: npt.NDArray[np.float64],
    b_LH: npt.NDArray[np.float64],
    a: float,
    BFImax: float,
    return_exceed: bool = False
) -> npt.NDArray[np.float64]:
    """Eckhardt 滤波器的 Numba 加速实现。

    递归滤波公式:
        b[i+1] = ((1-BFImax)*a*b[i] + (1-a)*BFImax*Q[i+1]) / (1-a*BFImax)

    公式推导思路:
        - 基于基流的指数衰退假设: b[i+1] = a*b[i] + recharge[i+1]
        - 引入 BFImax 约束,限制基流的长期占比
        - 通过代数变换得到上述递归形式

    Args:
        Q: 流量数组
        b_LH: LH 滤波结果(用于初始化 b[0])
        a: 衰退系数
        BFImax: 最大基流指数
        return_exceed: 是否返回超限计数

    Returns:
        基流数组(如果 return_exceed=True,末尾附加超限次数)

    Note:
        - 初始值取自 LH 滤波,确保合理的启动条件
        - 物理约束: b[i] ≤ Q[i]
        - 参数范围建议: a ∈ [0.9, 0.995], BFImax ∈ [0.2, 0.9]
    """
    # 初始化结果数组
    if return_exceed:
        b = np.zeros(Q.shape[0] + 1)  # 最后一个元素存储超限次数
    else:
        b = np.zeros(Q.shape[0])

    # 初始条件: 使用 LH 滤波的第一个值
    b[0] = b_LH[0]

    # ========================================================================
    # 递归滤波
    # ========================================================================
    for i in range(Q.shape[0] - 1):
        # Eckhardt 递归公式
        # 分子: (1-BFImax)*a*b[i] 表示前一时刻基流的衰减部分
        #       + (1-a)*BFImax*Q[i+1] 表示当前流量对基流的贡献
        # 分母: 1-a*BFImax 是归一化因子
        numerator = (1 - BFImax) * a * b[i] + (1 - a) * BFImax * Q[i + 1]
        denominator = 1 - a * BFImax
        b[i + 1] = numerator / denominator

        # 物理约束: 基流不能超过总流量
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            if return_exceed:
                b[-1] += 1  # 记录超限次数

    return b
