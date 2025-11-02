"""为现有 Numba 函数创建面向对象的包装器。

本模块为所有现有的基流分割方法创建轻量级的 OOP 包装,
实现统一接口,同时保持原有 Numba JIT 编译函数的高性能。

这是一种实用的架构升级方式:
- 保留经过优化的 Numba 函数(性能关键)
- 添加 OOP 层用于统一接口和注册管理
- 支持渐进式重构,不破坏现有代码
"""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from ._base import BaseflowMethod, register_method

# 导入所有现有的 Numba 函数
from .Boughton import Boughton as Boughton_func
from .Chapman import Chapman as Chapman_func
from .CM import CM as CM_func
from .Eckhardt import Eckhardt as Eckhardt_func
from .EWMA import EWMA as EWMA_func
from .Fixed import Fixed as Fixed_func
from .Furey import Furey as Furey_func
from .LH import LH as LH_func
from .Local import Local as Local_func
from .Slide import Slide as Slide_func
from .UKIH import UKIH as UKIH_func
from .Willems import Willems as Willems_func

__all__ = [
    "LHMethod",
    "UKIHMethod",
    "ChapmanMethod",
    "CMMethod",
    "BoughtonMethod",
    "FureyMethod",
    "EckhardtMethod",
    "EWMAMethod",
    "WillemsMethod",
    "LocalMethod",
    "FixedMethod",
    "SlideMethod",
]


# ============================================================================
# 数字滤波法 (Digital Filter Methods)
# ============================================================================

@register_method("LH")
class LHMethod(BaseflowMethod):
    """Lyne-Hollick 数字滤波法 (双遍递归滤波)。"""

    name = "LH"
    description = "Lyne-Hollick digital filter"
    requires_area = False
    requires_recession_coef = False
    requires_calibration = False

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        # LH 方法不依赖 b_LH 输入,直接对 Q 进行滤波
        return LH_func(Q)


@register_method("UKIH")
class UKIHMethod(BaseflowMethod):
    """UK Institute of Hydrology 方法 (基于 LH 的变体)。"""

    name = "UKIH"
    description = "UK Institute of Hydrology method"
    requires_area = False
    requires_recession_coef = False
    requires_calibration = False

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        # UKIH 使用预先计算的 LH 结果
        return UKIH_func(Q, b_LH)


@register_method("Chapman")
class ChapmanMethod(BaseflowMethod):
    """Chapman 数字滤波法 (1991)。"""

    name = "Chapman"
    description = "Chapman digital filter"
    requires_area = False
    requires_recession_coef = True
    requires_calibration = False

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        if a is None:
            raise ValueError("Chapman method requires recession coefficient 'a'")
        return Chapman_func(Q, b_LH, a)


@register_method("CM")
class CMMethod(BaseflowMethod):
    """Combined Method (Chapman 变体)。"""

    name = "CM"
    description = "Combined method (Chapman variant)"
    requires_area = False
    requires_recession_coef = True
    requires_calibration = False

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        if a is None:
            raise ValueError("CM method requires recession coefficient 'a'")
        return CM_func(Q, b_LH, a)


@register_method("EWMA")
class EWMAMethod(BaseflowMethod):
    """指数加权移动平均滤波器。"""

    name = "EWMA"
    description = "Exponential Weighted Moving Average filter"
    requires_area = False
    requires_recession_coef = False
    requires_calibration = True

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        # EWMA 需要参数 e,从 kwargs 获取
        e = kwargs.get("e")
        if e is None:
            raise ValueError("EWMA method requires parameter 'e'")
        # 注意: EWMA 函数签名使用 0 作为 a 的占位符
        return EWMA_func(Q, b_LH, 0, e)


# ============================================================================
# 参数化方法 (Parameterized Methods)
# ============================================================================

@register_method("Boughton")
class BoughtonMethod(BaseflowMethod):
    """Boughton 两参数滤波法。"""

    name = "Boughton"
    description = "Boughton method with parameter C"
    requires_area = False
    requires_recession_coef = True
    requires_calibration = True

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        if a is None:
            raise ValueError("Boughton method requires recession coefficient 'a'")
        C = kwargs.get("C")
        if C is None:
            raise ValueError("Boughton method requires parameter 'C'")
        return Boughton_func(Q, b_LH, a, C)


@register_method("Furey")
class FureyMethod(BaseflowMethod):
    """Furey 基流分割法。"""

    name = "Furey"
    description = "Furey method with parameter A"
    requires_area = False
    requires_recession_coef = True
    requires_calibration = True

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        if a is None:
            raise ValueError("Furey method requires recession coefficient 'a'")
        A = kwargs.get("A")
        if A is None:
            raise ValueError("Furey method requires parameter 'A'")
        return Furey_func(Q, b_LH, a, A)


@register_method("Eckhardt")
class EckhardtMethod(BaseflowMethod):
    """Eckhardt 两参数数字滤波法。"""

    name = "Eckhardt"
    description = "Eckhardt two-parameter digital filter with BFImax"
    requires_area = False
    requires_recession_coef = True
    requires_calibration = True

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        if a is None:
            raise ValueError("Eckhardt method requires recession coefficient 'a'")
        BFImax = kwargs.get("BFImax")
        if BFImax is None:
            raise ValueError("Eckhardt method requires parameter 'BFImax'")
        return Eckhardt_func(Q, b_LH, a, BFImax)


@register_method("Willems")
class WillemsMethod(BaseflowMethod):
    """Willems 基流分割法。"""

    name = "Willems"
    description = "Willems method with parameter w"
    requires_area = False
    requires_recession_coef = True
    requires_calibration = True

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        if a is None:
            raise ValueError("Willems method requires recession coefficient 'a'")
        w = kwargs.get("w")
        if w is None:
            raise ValueError("Willems method requires parameter 'w'")
        return Willems_func(Q, b_LH, a, w)


# ============================================================================
# HYSEP 图形法 (Graphical Methods)
# ============================================================================

@register_method("Local")
class LocalMethod(BaseflowMethod):
    """HYSEP Local Minimum 方法。"""

    name = "Local"
    description = "Local minimum method (HYSEP)"
    requires_area = True
    requires_recession_coef = False
    requires_calibration = False

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        if area is None:
            raise ValueError("Local method requires drainage area")
        return Local_func(Q, b_LH, area)


@register_method("Fixed")
class FixedMethod(BaseflowMethod):
    """HYSEP Fixed Interval 方法。"""

    name = "Fixed"
    description = "Fixed interval method (HYSEP)"
    requires_area = True
    requires_recession_coef = False
    requires_calibration = False

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        if area is None:
            raise ValueError("Fixed method requires drainage area")
        return Fixed_func(Q, area)


@register_method("Slide")
class SlideMethod(BaseflowMethod):
    """HYSEP Sliding Interval 方法。"""

    name = "Slide"
    description = "Sliding interval method (HYSEP)"
    requires_area = True
    requires_recession_coef = False
    requires_calibration = False

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        if area is None:
            raise ValueError("Slide method requires drainage area")
        return Slide_func(Q, area)
