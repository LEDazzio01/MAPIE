from typing import cast, no_type_check

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.multiclass import (
    check_classification_targets,
    type_of_target,
)
from .regression import BaseRegressionScore
from .classification import BaseClassificationScore
from .bounds import (
    AbsoluteConformityScore,
    GammaConformityScore,
    ResidualNormalisedScore,
    StdConformityScore,
)
from .sets import (
    LACConformityScore,
    TopKConformityScore,
    APSConformityScore,
    RAPSConformityScore,
)


def _check_alpha_and_last_axis(vector: NDArray, alpha_np: NDArray):
    """Check when the dimension of vector is 3 that its last axis
    size is the same than the number of alphas.

    Parameters
    ----------
    vector: NDArray of shape (n_samples, 1, n_alphas)
        Vector on which compute the quantile.
    alpha_np: NDArray of shape (n_alphas, )
        Confidence levels.


    Raises
    ------
    ValueError
        Error is the last axis dimension is different from the
        number of alphas.
    """
    if len(alpha_np) != vector.shape[2]:
        raise ValueError(
            "In case of the vector has 3 dimensions, the dimension of its"
            + "last axis must be equal to the number of confidence levels"
        )
    else:
        return vector, alpha_np


def _compute_regression_quantile(
    conformity_scores: NDArray,
    alpha_np: NDArray,
    axis: int = 0,
    reverse: bool = False,
    unbounded: bool = False,
) -> NDArray:
    """Compute the alpha quantile of conformity scores for regression
    with finite-sample correction.

    Uses a ``ceil()``-based correction formula to ensure finite-sample
    coverage: quantile level ``ceil(alpha_ref * (n + 1)) / n`` with
    numpy ``method="lower"``. Operates on signed conformity scores to
    support directional quantiles via the ``reverse`` parameter.

    Parameters
    ----------
    conformity_scores: NDArray of shape (n_samples,) or
        (n_samples, n_estimators)
        Values from which the quantile is computed.

    alpha_np: NDArray of shape (n_alpha,)
        NDArray of floats between ``0`` and ``1``, represents the
        uncertainty of the confidence set.

    axis: int
        The axis from which to compute the quantile.

        By default ``0``.

    reverse: bool
        Boolean specifying whether we take the upper or lower
        quantile. If False, the alpha quantile; otherwise the
        (1-alpha) quantile.

        By default ``False``.

    unbounded: bool
        Boolean specifying whether infinite prediction sets
        could be produced (when alpha_np is greater than or
        equal to 1.).

        By default ``False``.

    Returns
    -------
    NDArray of shape (1, n_alpha) or (n_samples, n_alpha)
        The quantiles of the conformity scores.

    Raises
    ------
    ValueError
        If all conformity scores are NaN along the reduction axis.
    """
    n_ref = conformity_scores.shape[1 - axis] if conformity_scores.ndim > 1 else 1
    n_calib: int = np.min(np.sum(~np.isnan(conformity_scores), axis=axis))
    if n_calib == 0:
        raise ValueError(
            "All conformity scores are NaN along the reduction axis. "
            "Cannot compute quantile correction."
        )
    signed = 1 - 2 * reverse

    # Adapt alpha w.r.t upper/lower : alpha vs. 1-alpha
    alpha_ref = (1 - 2 * alpha_np) * reverse + alpha_np

    # Adjust alpha w.r.t quantile correction
    alpha_cor = np.ceil(alpha_ref * (n_calib + 1)) / n_calib
    alpha_cor = np.clip(alpha_cor, a_min=0, a_max=1)

    # Compute the target quantiles:
    # If unbounded is True and alpha is greater than or equal to 1,
    # the quantile is set to infinity.
    # Otherwise, the quantile is calculated as the corrected lower
    # quantile of the signed conformity scores.
    quantile: NDArray = signed * np.column_stack(
        [
            np.nanquantile(
                signed * conformity_scores,
                _alpha_cor,
                axis=axis,
                method="lower",
            )
            if not (unbounded and _alpha >= 1)
            else np.inf * np.ones(n_ref)
            for _alpha, _alpha_cor in zip(alpha_ref, alpha_cor)
        ]
    )
    return quantile


def _compute_classification_quantile(
    conformity_scores: NDArray, alpha_np: NDArray
) -> NDArray:
    """Compute the desired quantiles of conformity scores for classification.

    Parameters
    ----------
    conformity_scores: NDArray of shape Union[(n_samples, 1),
        (n_samples, 1, n_alphas)]
        Values from which the quantile is computed. If the array has
        3 dimensions, then each 1-alpha quantile will be computed on
        its corresponding matrix selected on the last axis of the matrix.
    alpha_np: NDArray for shape (n_alphas, )
        Risk levels.

    Returns
    -------
    NDArray of shape (n_alphas, )
        Quantiles of the conformity scores.
    """
    n = len(conformity_scores)
    if len(conformity_scores.shape) <= 2:
        quantiles_ = np.stack(
            [
                np.quantile(
                    conformity_scores,
                    ((n + 1) * (1 - _alpha)) / n,
                    method="higher",
                )
                for _alpha in alpha_np
            ]
        )

    else:
        _check_alpha_and_last_axis(conformity_scores, alpha_np)
        quantiles_ = np.stack(
            [
                _compute_classification_quantile(
                    conformity_scores[:, :, i], np.array([alpha_])
                )
                for i, alpha_ in enumerate(alpha_np)
            ]
        )[:, 0]
    return cast(NDArray, quantiles_)


CONFORMITY_SCORES_STRING_MAP = {
    BaseRegressionScore: {
        "absolute": AbsoluteConformityScore,
        "gamma": GammaConformityScore,
        "residual_normalized": ResidualNormalisedScore,
        "std_normalized": StdConformityScore,
    },
    BaseClassificationScore: {
        "lac": LACConformityScore,
        "top_k": TopKConformityScore,
        "aps": APSConformityScore,
        "raps": RAPSConformityScore,
    },
}


@no_type_check  # Cumbersome to type
def check_and_select_conformity_score(conformity_score, conformity_score_type):
    if isinstance(conformity_score, conformity_score_type):
        return conformity_score
    elif conformity_score in CONFORMITY_SCORES_STRING_MAP[conformity_score_type]:
        return CONFORMITY_SCORES_STRING_MAP[conformity_score_type][conformity_score]()
    else:
        raise ValueError("Invalid conformity_score parameter")


def check_regression_conformity_score(
    conformity_score: BaseRegressionScore,
    sym: bool = True,
) -> BaseRegressionScore:
    """
    Check parameter `conformity_score` for regression task.
    By default, return a AbsoluteConformityScore instance.

    Parameters
    ----------
    conformity_score: BaseClassificationScore
        Conformity score function.

        By default, `None`.

    sym: bool
        Whether to use symmetric bounds.

        By default, `True`.

    Raises
    ------
    ValueError
        If parameters are not valid.

    Examples
    --------
    >>> from mapie.conformity_scores.utils import (
    ...     check_regression_conformity_score
    ... )
    >>> try:
    ...     check_regression_conformity_score(1)
    ... except Exception as exception:
    ...     print(exception)
    ...
    Invalid conformity_score argument.
    Must be None or a BaseRegressionScore instance.
    """
    if conformity_score is None:
        return AbsoluteConformityScore(sym=sym)
    elif isinstance(conformity_score, BaseRegressionScore):
        return conformity_score
    else:
        raise ValueError(
            "Invalid conformity_score argument.\n"
            "Must be None or a BaseRegressionScore instance."
        )


def check_target(conformity_score: BaseClassificationScore, y: ArrayLike) -> None:
    """
    Check that if the type of target is binary,
    (then the method have to be `"lac"`), or multi-class.

    Parameters
    ----------
    conformity_score: BaseClassificationScore
        Conformity score function.

    y: NDArray of shape (n_samples,)
        Training labels.

    Raises
    ------
    ValueError
        If type of target is binary and method is not `"lac"`
        or `"score"` or if type of target is not multi-class.
    """
    check_classification_targets(y)
    if type_of_target(y) == "binary" and not isinstance(
        conformity_score, LACConformityScore
    ):
        raise ValueError(
            "Invalid conformity score for binary target. The only valid score is 'lac'."
        )


def check_classification_conformity_score(
    conformity_score: BaseClassificationScore = None,
) -> BaseClassificationScore:
    """
    Check parameter `conformity_score` for classification task.
    By default, return a LACConformityScore instance.

    Parameters
    ----------
    conformity_score: BaseClassificationScore
        Conformity score function.

        By default, `None`.

    Raises
    ------
    ValueError
        If conformity_score is not valid.
    """
    if conformity_score is not None:
        if isinstance(conformity_score, BaseClassificationScore):
            return conformity_score
        else:
            raise ValueError(
                "Invalid conformity_score argument.\n"
                "Must be None or a BaseClassificationScore instance."
            )
    else:
        return LACConformityScore()
