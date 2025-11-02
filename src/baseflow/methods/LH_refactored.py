"""Lyne-Hollick (LH) 数字滤波法的面向对象实现。

LH 滤波器是最广泛使用的基流分割方法之一,通过递归数字滤波
实现高频(快速流)和低频(基流)成分的分离。

参考文献:
    Lyne, V., & Hollick, M. (1979). Stochastic time-variable rainfall-runoff modelling.
    Institute of Engineers Australia National Conference. (pp. 89-93). Perth.
"""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from numba import njit

from ._base import BaseflowMethod, register_method

__all__ = ["LHMethod", "LH"]


@register_method("LH")
class LHMethod(BaseflowMethod):
    """Lyne-Hollick 数字滤波法。

    LH 滤波器使用一个简单的递归方程,通过两遍滤波(正向+反向)
    来分离基流,确保结果在时间上对称。

    核心公式:
        b[i+1] = β * b[i] + (1-β)/2 * (Q[i] + Q[i+1])

    其中:
        - β: 滤波参数,通常为 0.925(Nathan & McMahon, 1990 推荐值)
        - 较大的 β 值会产生更平滑的基流曲线
    """

    name = "LH"
    description = "Lyne-Hollick digital filter (双遍递归滤波)"
    requires_area = False
    requires_recession_coef = False
    requires_calibration = False

    def __init__(self, beta: float = 0.925):
        """初始化 LH 方法。

        Args:
            beta: 滤波参数(0 < beta < 1)。
                  推荐值:0.925 (Nathan & McMahon, 1990)
        """
        self.beta = beta

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        """执行 LH 基流分割。

        Args:
            Q: 流量时间序列
            b_LH: 未使用(LH 方法不依赖其他滤波器)
            a: 未使用
            area: 未使用
            **kwargs: 可选参数
                - beta: 覆盖默认的滤波参数
                - return_exceed: 是否返回超限次数(用于标定)

        Returns:
            基流时间序列
        """
        beta = kwargs.get("beta", self.beta)
        return_exceed = kwargs.get("return_exceed", False)

        return LH(Q, beta=beta, return_exceed=return_exceed)


# ============================================================================
# 原始 Numba JIT 编译函数(保持向后兼容)
# ============================================================================

@njit
def LH(Q: npt.NDArray[np.float64], beta: float = 0.925, return_exceed: bool = False) -> npt.NDArray[np.float64]:
    """LH 数字滤波器的 Numba 加速实现。

    使用两遍滤波策略:
    1. 正向遍历: 计算初步基流
    2. 反向遍历: 在初步结果上再次滤波,确保时间对称性

    Args:
        Q: 流量数组
        beta: 滤波参数,默认 0.925
        return_exceed: 是否返回超限计数(基流>流量的次数)

    Returns:
        如果 return_exceed=False: 基流数组
        如果 return_exceed=True: 基流数组(末尾附加超限次数)

    Note:
        - Numba JIT 编译确保性能与原始实现相同
        - 物理约束: b[i] ≤ Q[i] (基流不能超过总流量)
    """
    # 初始化结果数组
    if return_exceed:
        b = np.zeros(Q.shape[0] + 1)  # 最后一个元素存储超限次数
    else:
        b = np.zeros(Q.shape[0])

    # ========================================================================
    # 第一遍: 正向滤波
    # ========================================================================
    b[0] = Q[0]  # 初始值: 假设初始基流等于初始流量

    for i in range(Q.shape[0] - 1):
        # LH 递归公式: 当前基流 = β * 前一时刻基流 + (1-β)/2 * (当前流量 + 前一时刻流量)
        b[i + 1] = beta * b[i] + (1 - beta) / 2 * (Q[i] + Q[i + 1])

        # 物理约束: 基流不能超过总流量
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            if return_exceed:
                b[-1] += 1  # 记录超限次数(用于标定评估)

    # ========================================================================
    # 第二遍: 反向滤波(在第一遍结果上操作)
    # ========================================================================
    b1 = np.copy(b)  # 保存第一遍结果

    for i in range(Q.shape[0] - 2, -1, -1):  # 从倒数第二个元素开始反向遍历
        # 反向递归公式: 与正向类似,但方向相反
        b[i] = beta * b[i + 1] + (1 - beta) / 2 * (b1[i + 1] + b1[i])

        # 物理约束: 反向滤波的基流也不能超过第一遍的结果
        if b[i] > b1[i]:
            b[i] = b1[i]
            if return_exceed:
                b[-1] += 1

    return b
