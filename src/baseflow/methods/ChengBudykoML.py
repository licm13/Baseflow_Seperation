"""Budyko-constrained machine learning baseflow separation (Cheng et al., 2023).

This module integrates the methodology described in "Global runoff partitioning
based on Budyko-constrained machine learning" (Cheng et al., 2023) into the
``baseflow_separation`` library architecture.  The implementation follows the
library's object-oriented interface so the method can be discovered through the
global registry alongside the traditional digital-filter approaches.

The original research code combines two major components:

1.  A set of Budyko-type analytical relationships linking long-term
    precipitation (``P``), potential evapotranspiration (``PET``) and runoff
    (``Q``) through the Fu-Zhang equation.
2.  A gradient-boosting regression (BRT/XGBoost) framework that regionalises the
    Budyko parameter :math:`\alpha` and the potential baseflow :math:`Q_{b,p}`
    using catchment attributes.

The implementation below exposes these steps through a single class
(:class:`ChengBudykoML`) that:

* accepts user-provided streamflow series together with collocated precipitation
  and PET information;
* optionally loads pre-trained machine learning models (saved with ``joblib``)
  to predict :math:`\alpha` and :math:`Q_{b,p}` from catchment descriptors;
* provides transparent fall-backs for situations where trained models are not
  available by estimating Budyko parameters from the supplied hydrometeorologic
  data (leveraging the numerical routines shipped with the original project);
* returns a baseflow time series that satisfies the Budyko/BFC constraints and
  is consistent with the rest of the library (``numpy.ndarray`` aligned with
  the input streamflow).

Notes
-----
* The method operates on long-term averages (annualised fluxes).  When a time
  step is not provided it defaults to daily data and converts totals to
  millimetres per year.
* Gradient boosting models are optional.  Users can either provide trained
  estimators directly (any scikit-learn compatible regressor exposing a
  ``predict`` method), joblib paths, or explicit ``alpha``/``qbp`` values.
* All heavy dependencies (``joblib``/``xgboost``/``scikit-learn``) are imported
  lazily so that the base package remains lightweight unless this method is
  exercised.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from ._base import BaseflowMethod, register_method

# ---------------------------------------------------------------------------
# Budyko helper functions (adapted from ``Cheng-3D-Budyko/utils.py``)
# ---------------------------------------------------------------------------


def _budyko_curve(
    P: Union[float, np.ndarray],
    Ep: Union[float, np.ndarray],
    alpha: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Runoff coefficient (Q/P) based on the Fu-Zhang Budyko equation."""

    P = np.asarray(P, dtype=float)
    Ep = np.asarray(Ep, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        aridity_index = np.divide(Ep, P, out=np.zeros_like(Ep, dtype=float), where=P != 0)

    term = (1.0 + np.power(aridity_index, alpha)) ** (1.0 / alpha)
    q_over_p = -aridity_index + term
    return np.clip(q_over_p, 0.0, 1.0)


def _bfc_curve(
    P: Union[float, np.ndarray],
    Ep: Union[float, np.ndarray],
    alpha: Union[float, np.ndarray],
    Qbp: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Baseflow coefficient (Qb/P) from Cheng et al. (2021) BFC equation."""

    P = np.asarray(P, dtype=float)
    Ep = np.asarray(Ep, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    Qbp = np.asarray(Qbp, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        aridity_index = np.divide(Ep, P, out=np.zeros_like(Ep, dtype=float), where=P != 0)
        qbp_over_p = np.divide(Qbp, P, out=np.zeros_like(Qbp, dtype=float), where=P != 0)

    term1 = (1.0 + np.power(aridity_index, alpha)) ** (1.0 / alpha)
    term2 = (1.0 + np.power(aridity_index + qbp_over_p, alpha)) ** (1.0 / alpha)
    qb_over_p = qbp_over_p + term1 - term2

    q_over_p = _budyko_curve(P, Ep, alpha)
    return np.clip(qb_over_p, 0.0, q_over_p)


def _estimate_alpha_from_obs(
    P: np.ndarray,
    Ep: np.ndarray,
    Q: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """Estimate Budyko :math:`\alpha` via Newton iterations."""

    P = np.asarray(P, dtype=float)
    Ep = np.asarray(Ep, dtype=float)
    Q = np.asarray(Q, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        rfc_obs = np.divide(Q, P, out=np.zeros_like(Q), where=P != 0)
        aridity_index = np.divide(Ep, P, out=np.zeros_like(Ep), where=P != 0)

    alpha = np.full_like(rfc_obs, 2.0, dtype=float)

    for _ in range(max_iter):
        rfc_calc = _budyko_curve(P, Ep, alpha)
        residual = rfc_calc - rfc_obs

        if np.all(np.abs(residual) < tol):
            break

        delta_alpha = 0.001
        rfc_plus = _budyko_curve(P, Ep, alpha + delta_alpha)
        derivative = (rfc_plus - rfc_calc) / delta_alpha

        mask = np.abs(derivative) > 1e-10
        alpha[mask] -= residual[mask] / derivative[mask]
        alpha = np.clip(alpha, 1.0, 10.0)

    # Avoid NaNs for arid catchments where P≈0 by falling back to α=2
    alpha = np.where(np.isfinite(alpha), alpha, 2.0)
    return alpha


def _estimate_qbp_from_obs(
    P: np.ndarray,
    Ep: np.ndarray,
    Qb: np.ndarray,
    alpha: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """Estimate potential baseflow :math:`Q_{b,p}` via Newton iterations."""

    P = np.asarray(P, dtype=float)
    Ep = np.asarray(Ep, dtype=float)
    Qb = np.asarray(Qb, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        bfc_obs = np.divide(Qb, P, out=np.zeros_like(Qb), where=P != 0)

    qbp = Qb.astype(float)

    for _ in range(max_iter):
        bfc_calc = _bfc_curve(P, Ep, alpha, qbp)
        residual = bfc_calc - bfc_obs

        if np.all(np.abs(residual) < tol):
            break

        delta_qbp = 1.0
        bfc_plus = _bfc_curve(P, Ep, alpha, qbp + delta_qbp)
        derivative = (bfc_plus - bfc_calc) / delta_qbp

        mask = np.abs(derivative) > 1e-10
        qbp[mask] -= residual[mask] / derivative[mask]
        qbp = np.maximum(qbp, 1.0)

    return np.where(np.isfinite(qbp), qbp, np.maximum(Qb, 1.0))


# ---------------------------------------------------------------------------
# Helper dataclass used to bundle optional ML models.
# ---------------------------------------------------------------------------


@dataclass
class _ParameterModel:
    """Container wrapping an arbitrary regressor with optional metadata."""

    estimator: Any
    feature_names: Optional[Sequence[str]] = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        predict = getattr(self.estimator, "predict", None)
        if predict is None:
            raise AttributeError(
                "Estimator does not provide a 'predict' method required for inference"
            )
        return np.asarray(predict(features)).ravel()


def _load_joblib_model(model_path: Optional[Union[str, Path]]) -> Optional[Any]:
    if model_path is None:
        return None

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        import joblib  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "joblib is required to load pre-trained ChengBudykoML models. "
            "Install it via `pip install joblib`."
        ) from exc

    return joblib.load(path)


# ---------------------------------------------------------------------------
# Main method implementation
# ---------------------------------------------------------------------------


@register_method("ChengBudykoML")
class ChengBudykoML(BaseflowMethod):
    """Budyko-constrained machine learning baseflow separation method."""

    name = "ChengBudykoML"
    description = (
        "Budyko-constrained ML regionalisation of baseflow (Cheng et al., 2023)"
    )
    requires_area = False
    requires_recession_coef = False
    requires_calibration = False

    def __init__(
        self,
        alpha_model: Optional[Any] = None,
        qbp_model: Optional[Any] = None,
        alpha_model_path: Optional[Union[str, Path]] = None,
        qbp_model_path: Optional[Union[str, Path]] = None,
        feature_scaler: Optional[Any] = None,
        feature_scaler_path: Optional[Union[str, Path]] = None,
        default_alpha: float = 2.0,
        default_qbp: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Load models/scalers from disk when paths are supplied
        if alpha_model is None and alpha_model_path is not None:
            alpha_model = _load_joblib_model(alpha_model_path)
        if qbp_model is None and qbp_model_path is not None:
            qbp_model = _load_joblib_model(qbp_model_path)
        if feature_scaler is None and feature_scaler_path is not None:
            feature_scaler = _load_joblib_model(feature_scaler_path)

        self._alpha_model = self._wrap_model(alpha_model)
        self._qbp_model = self._wrap_model(qbp_model)
        self._feature_scaler = feature_scaler

        self._default_alpha = float(default_alpha)
        self._default_qbp = float(default_qbp) if default_qbp is not None else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def separate(
        self,
        Q: npt.NDArray[np.float64],
        b_LH: npt.NDArray[np.float64],
        a: Optional[float] = None,
        area: Optional[float] = None,
        **kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Derive baseflow using the Budyko-constrained ML approach.

        Parameters
        ----------
        Q:
            Streamflow series (1-D array).  Units should match precipitation and
            PET (e.g., mm/day).
        b_LH:
            Lyne-Hollick baseline baseflow.  Used as observational support when
            no ML model is available for :math:`Q_{b,p}` estimation.
        a, area:
            Unused but kept for API compatibility with :class:`BaseflowMethod`.
        **kwargs:
            Additional inputs controlling the separation.  Recognised keys:

            ``precip`` (array-like)
                Precipitation series collocated with ``Q``.
            ``pet`` (array-like)
                Potential evapotranspiration series.
            ``features`` (array-like or mapping)
                Catchment descriptors for the ML models.  Either a NumPy array
                of shape ``(n_features,)``/``(1, n_features)`` or a mapping from
                feature name to value.
            ``alpha`` (float)
                Override for the Budyko :math:`\alpha` parameter.
            ``qbp`` (float)
                Override for potential baseflow :math:`Q_{b,p}` (mm/yr).
            ``timestep_days`` (float)
                Temporal resolution of the inputs in days.  Defaults to 1.0
                (daily data).
        """

        precip = self._extract_required_array(kwargs, "precip")
        pet = self._extract_required_array(kwargs, "pet")

        Q = np.asarray(Q, dtype=float)
        b_LH = np.asarray(b_LH, dtype=float)
        precip = np.asarray(precip, dtype=float)
        pet = np.asarray(pet, dtype=float)

        self._validate_lengths(Q=Q, precip=precip, pet=pet)

        timestep_days = float(kwargs.get("timestep_days", 1.0))
        years = max(float(len(Q) * timestep_days / 365.0), 1e-6)

        # Aggregate to long-term annualised fluxes (mm/yr)
        P_annual = np.nansum(precip) / years
        Ep_annual = np.nansum(pet) / years
        Q_annual = np.nansum(Q) / years

        alpha = self._resolve_alpha(kwargs, P_annual, Ep_annual, Q_annual)
        qbp = self._resolve_qbp(kwargs, P_annual, Ep_annual, b_LH, alpha, years)

        runoff_coeff = _budyko_curve(P_annual, Ep_annual, alpha)
        baseflow_coeff = _bfc_curve(P_annual, Ep_annual, alpha, qbp)

        with np.errstate(divide="ignore", invalid="ignore"):
            baseflow_fraction = np.divide(
                baseflow_coeff,
                runoff_coeff,
                out=np.zeros_like(baseflow_coeff, dtype=float),
                where=runoff_coeff != 0,
            )

        baseflow = baseflow_fraction * Q
        baseflow = np.clip(baseflow, 0.0, Q)
        return baseflow.astype(np.float64, copy=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_alpha(
        self,
        kwargs: Dict[str, Any],
        P_annual: float,
        Ep_annual: float,
        Q_annual: float,
    ) -> float:
        if "alpha" in kwargs and kwargs["alpha"] is not None:
            return float(kwargs["alpha"])

        features = self._prepare_features(kwargs.get("features"))
        if self._alpha_model is not None and features is not None:
            alpha_pred = self._alpha_model.predict(features[None, :])[0]
            if np.isfinite(alpha_pred):
                return float(np.clip(alpha_pred, 1.0, 10.0))

        if P_annual <= 0.0:
            return self._default_alpha

        alpha_est = _estimate_alpha_from_obs(
            np.array([P_annual]), np.array([Ep_annual]), np.array([Q_annual])
        )
        if np.isfinite(alpha_est[0]):
            return float(alpha_est[0])

        return self._default_alpha

    def _resolve_qbp(
        self,
        kwargs: Dict[str, Any],
        P_annual: float,
        Ep_annual: float,
        b_LH: np.ndarray,
        alpha: float,
        years: float,
    ) -> float:
        if "qbp" in kwargs and kwargs["qbp"] is not None:
            return float(kwargs["qbp"])

        features = self._prepare_features(kwargs.get("features"))
        if self._qbp_model is not None and features is not None:
            qbp_pred = self._qbp_model.predict(features[None, :])[0]
            if np.isfinite(qbp_pred):
                return float(max(qbp_pred, 1.0))

        if P_annual <= 0.0:
            return self._default_qbp if self._default_qbp is not None else 1.0

        # Use LH output as observational proxy when ML models are not available
        Qb_annual = np.nansum(b_LH) / years

        qbp_est = _estimate_qbp_from_obs(
            np.array([P_annual]),
            np.array([Ep_annual]),
            np.array([Qb_annual]),
            np.array([alpha]),
        )
        qbp_val = float(qbp_est[0]) if np.isfinite(qbp_est[0]) else None

        if qbp_val is not None and qbp_val > 0.0:
            return qbp_val

        if self._default_qbp is not None:
            return self._default_qbp

        # Conservative fallback: assume baseflow corresponds to 60% of total runoff
        runoff_coeff = _budyko_curve(P_annual, Ep_annual, alpha)
        return float(0.6 * runoff_coeff * P_annual)

    @staticmethod
    def _extract_required_array(kwargs: Dict[str, Any], key: str) -> Any:
        if key not in kwargs or kwargs[key] is None:
            raise ValueError(
                f"ChengBudykoML requires '{key}' data supplied via keyword argument"
            )
        return kwargs[key]

    @staticmethod
    def _validate_lengths(**series: np.ndarray) -> None:
        lengths = {name: np.asarray(values).shape[0] for name, values in series.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "Input series must share the same length. Received: "
                + ", ".join(f"{name}={length}" for name, length in lengths.items())
            )

    def _prepare_features(
        self, features: Optional[Union[Sequence[float], Dict[str, float]]]
    ) -> Optional[np.ndarray]:
        if features is None:
            return None

        if isinstance(features, dict):
            if self._alpha_model and self._alpha_model.feature_names:
                keys = self._alpha_model.feature_names
            elif self._qbp_model and self._qbp_model.feature_names:
                keys = self._qbp_model.feature_names
            else:
                keys = sorted(features.keys())
            arr = np.asarray([features[key] for key in keys], dtype=float)
            return self._scale_features(arr)

        features_array = np.asarray(features, dtype=float)
        if features_array.ndim == 1:
            return self._scale_features(features_array)
        if features_array.ndim == 2 and features_array.shape[0] == 1:
            return self._scale_features(features_array[0])

        raise ValueError(
            "Catchment features should be a 1-D array or a mapping of name→value"
        )

    def _scale_features(self, arr: np.ndarray) -> np.ndarray:
        if self._feature_scaler is None:
            return arr

        transform = getattr(self._feature_scaler, "transform", None)
        if callable(transform):
            transformed = transform(arr.reshape(1, -1))
            return np.asarray(transformed).ravel()

        # Fall back to simple attribute-based scaling (e.g. StandardScaler stats)
        mean = getattr(self._feature_scaler, "mean_", None)
        scale = getattr(self._feature_scaler, "scale_", None)
        if mean is not None and scale is not None:
            return (arr - np.asarray(mean)) / np.asarray(scale)

        return arr

    @staticmethod
    def _wrap_model(model: Optional[Any]) -> Optional[_ParameterModel]:
        if model is None:
            return None

        feature_names: Optional[Sequence[str]] = None
        estimator = model

        if isinstance(model, tuple) and len(model) == 2:
            estimator, feature_names = model  # type: ignore[assignment]
        elif hasattr(model, "feature_names_in_"):
            feature_names = list(getattr(model, "feature_names_in_"))

        return _ParameterModel(estimator=estimator, feature_names=feature_names)


__all__ = ["ChengBudykoML"]

