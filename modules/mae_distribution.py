import logging
from typing import Callable, List, Tuple, Dict, Optional
import pandas as pd
import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod

from pywatts.core.base_summary import BaseSummary
from pywatts.core.summary_object import SummaryObjectList, SummaryObjectTable
from pywatts.core.filemanager import FileManager
from pywatts.core.exceptions.input_not_available import InputNotAvailable
from pywatts.core.exceptions.invalid_input_exception import InvalidInputException
logger = logging.getLogger(__name__)

class MetricBase(BaseSummary, ABC):
    """
    Base Class for all Metrics

    :param offset: Offset, which determines the number of ignored values in the beginning for calculating the Metric.
                   Default 0
    :type offset: int
    :param filter_method: Filter which should performed on the data before calculating the Metric.
    :type filter_method: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    """

    def __init__(self,
                 name: str = None,
                 filter_method: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = None,
                 offset: int = 0,
                 cuts: List[Tuple[pd.Timestamp, pd.Timestamp]] = None):
        super().__init__(name if name is not None else self.__class__.__name__)
        self.offset = offset
        self.filter_method = filter_method
        if cuts is None:
            self.cuts = []
        else:
            self.cuts = cuts

    def get_params(self) -> Dict[str, object]:
        """
        Returns a dict of parameters used in the Metric.

        :return: Parameters set for the Metric
        :rtype: Dict[str, object]
        """
        return {"offset": self.offset,
                "cuts": self.cuts}

    def set_params(self, offset: Optional[int] = None, cuts=Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]]):
        """
        Set parameters of the Metric.

        :param offset: Offset, which determines the number of ignored values in the beginning for calculating the Metric.
        :type offset: int
        :param cuts: The cutouts on which the metric should be additionally calculated.
        :type cuts: List[Tuple[pd.Timestamp, pd.Timestamp]]
        """
        if offset:
            self.offset = offset
        if cuts is not None:
            self.cuts = cuts

    def transform(self, file_manager: FileManager, y: xr.DataArray, **kwargs: xr.DataArray) -> SummaryObjectList:
        """
        Calculates the MAE based on the predefined target and predictions variables.
        :param file_manager: The filemanager, it can be used to store data that corresponds to the summary as a file.
        :type: file_manager: FileManager
        :param y: the input dataset
        :type y: xr.DataArray
        :param kwargs: the predictions
        :type kwargs: xr.DataArray

        :return: The calculated MAE
        :rtype: xr.DataArray
        """

        t = y.values
        summary = SummaryObjectList(self.name)
        if kwargs == {}:
            error_message = f"No predictions are provided as input for the {self.__class__.__name__}.  You should add the predictions by a " \
                            f"seperate key word arguments if you add the {self.__class__.__name__} to the pipeline."
            logger.error(error_message)
            raise InputNotAvailable(error_message)

        y_ = y[self.offset:]
        kwargs_ = {key: value[self.offset:] for key, value in kwargs.items()}
        self._transform({key: value for key, value in kwargs_.items()}, "", summary, y_.values, file_manager)
        return summary

    def _transform(self, kwargs, suffix, summary, t, fm:FileManager):
        for key, y_hat in kwargs.items():
            p = y_hat.values
            if p.shape != t.shape:
                try:
                    p = p.reshape(t.shape)
                except ValueError:
                    raise InvalidInputException(
                        f"The prediction {key} does not match to the shape of the ground truth y in the instance "
                        f"{self.name} of class {self.__class__.__name__}.")
                self.logger.info(f"Reshaped prediction {key} in {self.name}")
            if self.filter_method:
                p_, t_ = self.filter_method(p, t)
                mae = self._apply_metric(p_, t_)

            else:
                mae = self._apply_metric(p, t)
            path = fm.get_path(key + suffix + "hist.png")
            #plt.hist(mae, bins=20)
            #plt.savefig(path)
            #plt.close()
            path = fm.get_path(key + suffix + "hist.csv")
            pd.DataFrame(np.histogram(mae, bins=20)).to_csv(path)
            summary.set_kv(key + suffix, path)
        return summary

    def save(self, fm: FileManager) -> Dict:
        json = super().save(fm)
        if self.filter_method is not None:
            filter_path = fm.get_path(f"{self.name}_filter.pickle")
            with open(filter_path, 'wb') as outfile:
                cloudpickle.dump(self.filter_method, outfile)
            json["filter"] = filter_path
        return json

    @classmethod
    def load(cls, load_information: Dict):
        params = load_information["params"]
        name = load_information["name"]
        filter_method = None
        if "filter" in load_information:
            with open(load_information["filter"], 'rb') as pickle_file:
                filter_method = cloudpickle.load(pickle_file)
        return cls(name=name, filter_method=filter_method, **params)

    @abstractmethod
    def _apply_metric(self, p, t):
        pass


class MAEDistribution(MetricBase):
    """
    Module to calculate the Mean Absolute Error (MAE)

    :param offset: Offset, which determines the number of ignored values in the beginning for calculating the MAE.
                   Default 0
    :type offset: int
    :param filter_method: Filter which should performed on the data before calculating the MAE.
    :type filter_method: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    """


    def _apply_metric(self, p, t):
        return np.mean(np.abs((p - t)), axis=1)
