"""Collection of individual baseflow separation method implementations.

本模块包含12种基流分割方法的实现,采用混合架构:
- 保留原有的高性能 Numba JIT 编译函数(向后兼容)
- 新增面向对象的包装类,实现统一接口和注册管理

使用方式:
    # 方式1: 直接使用 Numba 函数(传统方式,高性能)
    from baseflow.methods import LH, Eckhardt
    baseflow = LH(Q)

    # 方式2: 使用面向对象接口(新方式,统一接口)
    from baseflow.methods import get_method
    method = get_method("LH")
    instance = method()
    baseflow = instance.separate(Q, b_LH)

    # 方式3: 查看所有可用方法
    from baseflow.methods import list_methods
    methods = list_methods()
"""

# ============================================================================
# 导入原有的 Numba 函数(保持向后兼容)
# ============================================================================
from .Boughton import *  # noqa: F401,F403
from .CM import *  # noqa: F401,F403
from .Chapman import *  # noqa: F401,F403
from .Eckhardt import *  # noqa: F401,F403
from .EWMA import *  # noqa: F401,F403
from .Fixed import *  # noqa: F401,F403
from .Furey import *  # noqa: F401,F403
from .LH import *  # noqa: F401,F403
from .Local import *  # noqa: F401,F403
from .Slide import *  # noqa: F401,F403
from .UKIH import *  # noqa: F401,F403
from .Willems import *  # noqa: F401,F403

# ============================================================================
# 导入新的面向对象架构
# ============================================================================
from ._base import (
    BaseflowMethod,
    METHOD_REGISTRY,
    get_method,
    list_methods,
    register_method,
)

# 导入所有包装类(这会自动注册到 METHOD_REGISTRY)
from ._wrappers import (
    BoughtonMethod,
    ChapmanMethod,
    CMMethod,
    EckhardtMethod,
    EWMAMethod,
    FixedMethod,
    FureyMethod,
    LHMethod,
    LocalMethod,
    SlideMethod,
    UKIHMethod,
    WillemsMethod,
)

__all__ = [
    # 原有 Numba 函数(向后兼容)
    "Boughton",
    "CM",
    "Chapman",
    "Eckhardt",
    "EWMA",
    "Fixed",
    "Furey",
    "LH",
    "Local",
    "Slide",
    "UKIH",
    "Willems",
    # 新的 OOP 架构
    "BaseflowMethod",
    "METHOD_REGISTRY",
    "get_method",
    "list_methods",
    "register_method",
    # 包装类
    "BoughtonMethod",
    "ChapmanMethod",
    "CMMethod",
    "EckhardtMethod",
    "EWMAMethod",
    "FixedMethod",
    "FureyMethod",
    "LHMethod",
    "LocalMethod",
    "SlideMethod",
    "UKIHMethod",
    "WillemsMethod",
]
