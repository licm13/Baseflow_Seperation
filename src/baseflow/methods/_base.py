"""Base class and registry for baseflow separation methods.

本模块定义了基流分割方法的统一接口和注册机制。
所有分割方法都应继承 BaseflowMethod 抽象基类,以确保接口一致性。
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type

import numpy as np
import numpy.typing as npt

__all__ = ["BaseflowMethod", "METHOD_REGISTRY", "register_method", "get_method"]


class BaseflowMethod(ABC):
    """基流分割方法的抽象基类。

    所有基流分割方法都应继承此类,并实现 separate() 方法。
    这确保了所有方法具有统一的接口,便于动态调用和扩展。

    Attributes:
        name: 方法的唯一标识符(例如 'LH', 'Eckhardt')
        description: 方法的简短描述
        requires_area: 是否需要流域面积参数
        requires_recession_coef: 是否需要衰退系数参数
        requires_calibration: 是否需要参数标定
    """

    # 方法元数据 - 子类应重写这些属性
    name: str = "BaseMethod"
    description: str = "Base class for baseflow separation methods"
    requires_area: bool = False
    requires_recession_coef: bool = False
    requires_calibration: bool = False

    @abstractmethod
    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        """执行基流分割。

        这是核心方法,子类必须实现。

        Args:
            Q: 流量时间序列数组
            b_LH: LH 滤波基准结果(许多方法用作初始值)
            a: 衰退系数(对于需要该参数的方法)
            area: 流域面积 km²(对于 HYSEP 类方法)
            **kwargs: 方法特定的额外参数

        Returns:
            基流时间序列数组,长度与 Q 相同

        Raises:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement separate() method"
        )

    def calibrate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        param_range: Optional[npt.NDArray[np.float64]] = None,
    ) -> Any:
        """标定方法特定参数(可选)。

        对于需要参数标定的方法(如 Eckhardt 的 BFImax),
        此方法执行网格搜索或其他优化算法。

        Args:
            Q: 流量时间序列
            b_LH: LH 滤波基准
            a: 衰退系数
            param_range: 参数搜索范围

        Returns:
            最优参数值

        Note:
            默认实现返回 None。需要标定的方法应重写此方法。
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ============================================================================
# 方法注册表 - 使用注册器模式管理所有方法
# ============================================================================

METHOD_REGISTRY: Dict[str, Type[BaseflowMethod]] = {}
"""全局方法注册表,映射方法名称到方法类。"""


def register_method(
    name: Optional[str] = None
) -> Callable[[Type[BaseflowMethod]], Type[BaseflowMethod]]:
    """装饰器:将方法类注册到全局注册表。

    使用示例:
        @register_method("LH")
        class LHMethod(BaseflowMethod):
            ...

    Args:
        name: 方法名称(可选)。如果未提供,使用类的 name 属性。

    Returns:
        装饰器函数
    """
    def decorator(cls: Type[BaseflowMethod]) -> Type[BaseflowMethod]:
        method_name = name if name is not None else cls.name

        if method_name in METHOD_REGISTRY:
            # 警告:方法重复注册(在开发时可能有用)
            import warnings
            warnings.warn(
                f"Method '{method_name}' is being re-registered. "
                f"Previous: {METHOD_REGISTRY[method_name]}, New: {cls}",
                UserWarning
            )

        METHOD_REGISTRY[method_name] = cls
        return cls

    return decorator


def get_method(name: str) -> Type[BaseflowMethod]:
    """从注册表获取方法类。

    Args:
        name: 方法名称

    Returns:
        方法类

    Raises:
        KeyError: 如果方法未注册
    """
    if name not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY.keys()))
        raise KeyError(
            f"Method '{name}' not found in registry. "
            f"Available methods: {available}"
        )

    return METHOD_REGISTRY[name]


def list_methods() -> Dict[str, Type[BaseflowMethod]]:
    """列出所有已注册的方法。

    Returns:
        方法名称到方法类的字典
    """
    return METHOD_REGISTRY.copy()
