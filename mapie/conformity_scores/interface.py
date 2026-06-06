from abc import ABCMeta, abstractmethod
from typing import Optional

from sklearn.base import BaseEstimator

from numpy.typing import NDArray

from mapie.utils import _compute_quantile


class BaseConformityScore(metaclass=ABCMeta):
    """
    Base class for conformity scores.

    This class should not be used directly. Use derived classes instead.
    """

    def __init__(self) -> None:
        pass

    def set_external_attributes(self, **kwargs) -> None:
        """
        Set attributes that are not provided by the user.

        Must be overloaded by subclasses if necessary to add more attributes,
        particularly when the attributes are known after the object has been
        instantiated.
        """

    def set_ref_predictor(self, predictor: BaseEstimator):
        """
        Set the reference predictor.

        Parameters
        ----------
        predictor: BaseEstimator
            Reference predictor.
        """
        self.predictor = predictor

    def split_data(
        self,
        X: NDArray,
        y: NDArray,
        y_enc: NDArray,
        sample_weight: Optional[NDArray] = None,
        groups: Optional[NDArray] = None,
    ):
        """
        Split data. Keeps part of the data for the calibration estimator
        (separate from the calibration data).

        Parameters
        ----------
        *args: Tuple of NDArray

        Returns
        -------
        Tuple of NDArray
            Split data for training and calibration.
        """
        self.n_samples_ = len(X)
        return X, y, y_enc, sample_weight, groups

    @abstractmethod
    def get_conformity_scores(self, y: NDArray, y_pred: NDArray, **kwargs) -> NDArray:
        """
        Placeholder for `get_conformity_scores`.
        Subclasses should implement this method!

        Compute the sample conformity scores given the predicted and
        observed targets.

        Parameters
        ----------
        y: NDArray of shape (n_samples,)
            Observed target values.

        y_pred: NDArray of shape (n_samples,)
            Predicted target values.

        Returns
        -------
        NDArray of shape (n_samples,)
            Conformity scores.
        """

    @staticmethod
    def get_quantile(
        conformity_scores: NDArray,
        alpha_np: NDArray,
        axis: int = 0,
        reversed: bool = False,
        unbounded: bool = False,
    ) -> NDArray:
        """
        Compute the alpha quantile of the conformity scores.

        Parameters
        ----------
        conformity_scores: NDArray of shape (n_samples,)
            Values from which the quantile is computed.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between `0` and `1`, represents the
            uncertainty of the confidence set.

        axis: int
            The axis from which to compute the quantile.

            By default `0`.

        reversed: bool
            Boolean specifying whether we take the upper or lower quantile,
            if False, the alpha quantile, otherwise the (1-alpha) quantile.

            By default `False`.

        unbounded: bool
            Boolean specifying whether infinite prediction sets
            could be produced (when alpha_np is greater than or equal to 1.).

            By default `False`.

        Returns
        -------
        NDArray of shape (1, n_alpha) or (n_samples, n_alpha)
            The quantiles of the conformity scores.
        """
        return _compute_quantile(
            conformity_scores,
            alpha_np,
            axis=axis,
            reverse=reversed,
            unbounded=unbounded,
        )

    @abstractmethod
    def predict_set(self, X: NDArray, alpha_np: NDArray, **kwargs):
        """
        Compute the prediction sets on new samples based on the uncertainty of
        the target confidence set.

        Parameters:
        -----------
        X: NDArray of shape (n_samples,)
            The input data or samples for prediction.

        alpha_np: NDArray of shape (n_alpha, )
            Represents the uncertainty of the confidence set to produce.

        **kwargs: dict
            Additional keyword arguments.

        Returns:
        --------
        result
            The prediction sets for each sample and each alpha level.
            The output structure depends on the subclass.
        """
