from typing import Dict

import pandas as pd
import xarray as xr
from pywatts.core.base_summary import BaseSummary
from pywatts.core.filemanager import FileManager
from pywatts.core.summary_object import SummaryObject, SummaryObjectList, SummaryObjectTable
from scipy.stats import stats

from modules.utils_Forecast_OP import *


class Stats(BaseSummary):

    def __init__(self, name: str = "Stats"):
        super().__init__(name)

    def get_params(self) -> Dict[str, object]:
        pass

    def set_params(self, **kwargs):
        pass

    def transform(self, file_manager: FileManager, **kwargs: xr.DataArray) -> SummaryObject:
        summary = SummaryObjectTable(self.name + 'columns are: "Name", "Mean", "Max", "Min", "Var", "Skew", "Kurt"')
        df = pd.DataFrame(columns=["Name", "Mean", "Max", "Min", "Var", "Skew", "Kurt"])
        for pred_name, prediction in kwargs.items():

            df = df.append({
                "Name": pred_name,
                "Mean": prediction.values.mean(),
                "Max": prediction.values.max(),
                "Min": prediction.values.min(),
                "Var": prediction.values.var(),
                "Skew": stats.skew(prediction.values, axis=None),
                "Kurt": stats.kurtosis(prediction, axis=None),
            }, ignore_index=True)

        summary.set_kv("Statistics", df.values)
        return summary
