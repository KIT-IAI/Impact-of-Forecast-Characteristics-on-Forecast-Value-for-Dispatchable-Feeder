from typing import Dict

import pandas as pd
import time
import xarray as xr
import functools
from multiprocessing import Process, Manager, Pool

from pywatts.core.base_summary import BaseSummary
from pywatts.core.filemanager import FileManager
from pywatts.core.summary_object import SummaryObject, SummaryObjectList

from modules.utils_Forecast_OP import *

import matplotlib.pyplot as plt
class OP(BaseSummary):

    def __init__(self,building_id: int=0, train: bool=True, name: str="OP",capacity=13.5):
        self.building_id = building_id
        self.train = train
        self.capacity = capacity
        super().__init__(name)

    def get_params(self) -> Dict[str, object]:
        pass

    def set_params(self, **kwargs):
        pass

    def set_train(self,train):
        self.train = train

    def transform(self, file_manager: FileManager, gt:xr.DataArray, **kwargs: xr.DataArray) -> SummaryObject:
        summary = SummaryObjectList(self.name)

        # for gt and all kwargs solve op and evaluate it
        for prediction_method, pred_name in enumerate(kwargs):
            with Manager() as m:
                costs_imbalances10_unrestricted = m.list()
                costs_imbalances10_restricted = m.list()
                costs_imbalances2_unrestricted = m.list()
                costs_imbalances2_restricted = m.list()
                costs_houses_unrestricted = m.list()
                costs_houses_restricted = m.list()
                g_max_unrestricted = m.list()
                g_max_restricted = m.list()
                messages = m.list()

                to_save_dict = m.dict()
                
                #
                #Criminal Fixes
                #
                kwargs[pred_name]["time"] = kwargs[pred_name]["time"] + pd.Timedelta("1h")
                gt["time"] = gt["time"] + pd.Timedelta("1h")  

                prediction = kwargs[pred_name]
                prediction["time"] = prediction["time"]
                

                #
                #
                #
                
                
                pool = Pool(16) # TODO change depending on the number of cores
                start_time = time.time()
                pool.map(functools.partial( 
                    _solve_week, costs_imbalances10_unrestricted, costs_imbalances10_restricted,
                        costs_imbalances2_unrestricted, costs_imbalances2_restricted, costs_houses_unrestricted, 
                        costs_houses_restricted, prediction, g_max_restricted, 
                        g_max_unrestricted,to_save_dict,messages, gt, self.capacity)
                    ,kwargs[pred_name]["time"][::7]
                    )
                pool.close()
                pool.join()
                summary.set_kv(pred_name + "OP Time", time.time() - start_time)

                print(str(len(to_save_dict)) +" Lenght of the dict:)")

                if self.train:
                    path = file_manager.get_path("train_" + pred_name +".csv")
                    df = pd.DataFrame.from_dict(to_save_dict,orient="index").sort_index()
                    df.to_csv(path)

                else:
                    path = file_manager.get_path("test_" + pred_name +".csv")
                    df = pd.DataFrame.from_dict(to_save_dict,orient="index").sort_index()
                    df.to_csv(path)

                summary.set_kv(pred_name + "Daily average cost DS unrestricted", sum(costs_houses_unrestricted) / len(costs_houses_unrestricted))
                summary.set_kv(pred_name + "Daily average cost imbalances factor 2 unrestricted", sum(costs_imbalances2_unrestricted) / len(costs_imbalances2_unrestricted))
                summary.set_kv(pred_name + "Daily average cost imbalances factor 10 unrestricted", sum(costs_imbalances10_unrestricted) / len(costs_imbalances10_unrestricted))
                summary.set_kv(pred_name + "g_max unrestricted", max(g_max_unrestricted))
                summary.set_kv(pred_name + "Messages", list(messages))
                path = file_manager.get_path(pred_name + "Weekly Costs ds.png")
                #plt.plot(costs_houses_unrestricted)
                #plt.savefig(path)
                #plt.close()
                path = file_manager.get_path(pred_name + "Weekly Costs imbalances 2.png")
                #plt.plot(costs_imbalances2_unrestricted)
                #plt.savefig(path)
                #plt.close()
                path = file_manager.get_path(pred_name + "Weekly Costs impalances 10.png")
                #plt.plot(costs_imbalances10_unrestricted)
                #plt.savefig(path)
                #plt.close()
                # TODO add name of summary module.
                path = file_manager.get_path(pred_name + "Weekly Costs ds.csv")
                pd.DataFrame(list(costs_houses_unrestricted)).to_csv(path)
                path = file_manager.get_path(pred_name + "Weekly Costs imbalances 2.csv")
                pd.DataFrame(list(costs_imbalances2_unrestricted)).to_csv(path)
                path = file_manager.get_path(pred_name + "Weekly Costs impalances 10.csv")
                pd.DataFrame(list(costs_imbalances10_unrestricted)).to_csv(path)

                print(f"{pred_name} op finished")
        return summary

def _solve_week(costs_imbalances10_unrestricted, costs_imbalances10_restricted,
                costs_imbalances2_unrestricted, costs_imbalances2_restricted, costs_houses_unrestricted, 
                costs_houses_restricted, prediction,  g_max_restricted_list, 
                g_max_unrestricted_list, to_save_dict,messages, gt,capacity, start_compute):
        try:
            # strange fix
            #start_compute = start_compute - pd.Timedelta("1h")


            p_max = [5]
            p_min = [-i for i in p_max]
            e_min = [0]
            e_max = [capacity]
            e_start = [capacity / 2]
            houses = []
            end_compute = start_compute + pd.Timedelta(weeks=1, hours=11)
            gt = gt.loc[start_compute:end_compute].values
            for _ in range(parameter['number_houses']):
                houses.append(DeterministicHouse(parameter, pd.Timestamp(start_compute.to_pandas()), pd.Timestamp(end_compute.to_pandas())))

                forecast_prosumption = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
                actual_prosumption = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
                for i, house in enumerate(houses):

                    forecast_prosumption['house' + str(i)] = {}
                    forecast_prosumption['house' + str(i)] = prediction.loc[start_compute:end_compute].values

                    actual_prosumption['house' + str(i)] = {}
                    actual_prosumption['house' + str(i)] = gt[:,0:24].reshape(-1,1)

                    house.p_min = p_min[i]
                    house.p_max = p_max[i]
                    house.e_min = e_min[i]
                    house.e_max = e_max[i]
                    house.e_start = e_start[i]

            houses_unrestricted = copy.deepcopy(houses)
            #houses_restricted_error = copy.deepcopy(houses)
            parameter['g_max'] = 200000
            try:
                run_all_function(parameter, houses_unrestricted, forecast_prosumption, actual_prosumption,to_save_dict)
            except AssertionError as e:
                messages.append(f"{start_compute} to {end_compute} not solved")
                return
            g_max_unrestricted = max(abs(sum(house.actual_g['value'] for house in houses_unrestricted)))
            parameter['g_max'] = g_max_unrestricted
            #print(parameter['g_max'])

            costs_imbalances10_unrestricted.append(costs_imbalances10(houses_unrestricted[0])/7)
            costs_imbalances2_unrestricted.append(costs_imbalances2(houses_unrestricted[0])/7)
            costs_houses_unrestricted.append(costs_DS(houses_unrestricted[0])/7)
            g_max_unrestricted_list.append(g_max_unrestricted)
        except Exception as e:
            messages.append(f"{start_compute} to {end_compute} not solved: {e}")


