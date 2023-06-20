import argparse
from calendar import c
import functools
from pyclbr import Function
from secrets import choice
import sys
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
from pyparsing import col
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from keras.losses import MeanSquaredError, MeanSquaredLogarithmicError, Huber
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.modules import SKLearnWrapper, Sampler, ClockShift, KerasWrapper, Slicer, RollingMAE, RollingRMSE, FunctionModule
from pywatts.summaries import RMSE
from pywatts.summaries.mae_summary import MAE
from pywatts.summaries.mase_summary import MASE
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model

from modules.mae_distribution import MAEDistribution
from modules.optimization_problem import OP
from modules.stat_summary import Stats
from pywatts.core.summary_formatter import SummaryJSON
from pywatts.callbacks import CSVCallback
from modules.mae_horizon_dependent import MAEHorizon

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
  except RuntimeError as e:
    print(e)

def flatten(d):
    result = {}
    if isinstance(d, dict):
        for o_key, o_value in d.items():
            result.update({o_key + "_" + i_key: i_value for i_key, i_value in flatten(o_value).items()})
        return result
    else:
        return {"": d}



HORIZON = 42
METRICS = {
    "huber": Huber(),
    "MAE": "MAE",
    "MSE": MeanSquaredError(),
    "pinball 0.1": tfa.losses.PinballLoss(tau=0.1),
    "pinball 0.25": tfa.losses.PinballLoss(tau=0.25),
    "pinball 0.75": tfa.losses.PinballLoss(tau=0.75),
    "pinball 0.9": tfa.losses.PinballLoss(tau=0.9),
}

def remove_unnecessary_data(x):
    return x[::24]

def get_forecasting_pipeline(model, column, metrics):
    forecast_pipeline = Pipeline(f"results/{column}")
    input_data = ClockShift(-HORIZON)(x=forecast_pipeline[column])
    scaler = SKLearnWrapper(StandardScaler())
    scaled_data = scaler(x=input_data)
    hist_data = ClockShift(HORIZON)(x=scaled_data)
    historical_data = Sampler(24)(x=hist_data)
    target_sampled = Sampler(HORIZON)(x=scaled_data)
    model_results = {}
    historical_data = Slicer(start=48 * 2 + 11, end=-48 * 2 -1)(x=historical_data)
    target_sampled = Slicer(start=48 * 2 + 11, end=-48 * 2 -1)(x=target_sampled)
    historical_data = FunctionModule(remove_unnecessary_data)(x=historical_data)
    target_sampled = FunctionModule(remove_unnecessary_data)(x=target_sampled)

    for name, metric in metrics.items():
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights = True)
        forecast = KerasWrapper(model=model(), compile_kwargs={"loss": metric}, name=name,
                                fit_kwargs={"epochs": 1000, "batch_size": 512, "verbose":0, "validation_split":0.2, "callbacks":[early_stopping]})(input=historical_data, target=target_sampled)
        model_results[name] = scaler(x=forecast, use_inverse_transform=True, computation_mode=ComputationMode.Transform,
                                     callbacks=[CSVCallback(f"{name}")])

    return forecast_pipeline, model_results, scaler(x=target_sampled, use_inverse_transform=True, computation_mode=ComputationMode.Transform,
                                     callbacks=[CSVCallback(f"{name}")])


parser = ArgumentParser()
parser.add_argument("--model", default="fc", choices=["fc", "lstm", "cnn"])
parser.add_argument("--factor", default="", type=str)



def get_fc_model(hist_length, horizon):
    input = layers.Input(shape=(hist_length,),
                         name='input')
    hidden = layers.Dense(16,
                          activation="relu",
                          name='hidden')(input)
    output = layers.Dense(horizon,
                          activation='linear',
                          name='target')(hidden)  # layer name must match time series name
    model = Model(inputs=[input], outputs=output)
    return model


def get_model(model):
    """
    Gibt nur eine Methode zurück, die ein Keras Modell zurück gibt
    :param model:
    :return:
    """


    return functools.partial(get_fc_model, 24, HORIZON)


if __name__ == "__main__":

    parser.add_argument('-id', type=int, help='building id')
    args = parser.parse_args()



    dateparser = lambda date: pd.Timestamp(pd.to_datetime(date, format='%Y-%m-%d %H:%M:%S'))
    data = pd.read_csv(f"data/solar_home_all_data_2010-2013{args.factor}.csv", parse_dates=["time"], index_col="time",
                       date_parser=dateparser).resample("1h").mean()
    df = pd.DataFrame()

    if args.id != None:
        column = str(args.id)

        if not column in data.columns:
            print("Building with Number " + column + " does not exist.")
            sys.exit()



        data_train = data[[column]][:pd.Timestamp(year=2012, month=7, day=1)]
        data_test = data[[column]][pd.Timestamp(year=2012, month=7, day=1):]

        pipeline, forecasts, gt = get_forecasting_pipeline(model=get_model(args.model), metrics=METRICS, column=column)
        pipeline.train(data_train, summary_formatter=SummaryJSON())

        MASE(lag=7)(y=gt, **forecasts)
        RMSE()(y=gt, **forecasts)
        MAE()(y=gt, **forecasts)
        MAEHorizon()(y=gt, **forecasts)
        MAEDistribution()(y=gt, **forecasts)
        Stats()(gt=gt, **forecasts)
        op = OP(building_id=int(column))
        op(gt=gt, **forecasts)

        RollingMAE()(y=gt, **forecasts, callbacks=[
        ])
        RollingRMSE()(y=gt, **forecasts, callbacks=[
        ])

        op.set_train(False)

        result, summary = pipeline.test(data_test, summary_formatter=SummaryJSON(), summary=True)

        op.set_train(True)
        print("Finished a building")



    else:


        for column in data.columns:

            data_train = data[[column]][:pd.Timestamp(year=2012, month=7, day=1)]
            data_test = data[[column]][pd.Timestamp(year=2012, month=7, day=1):]
            pipeline, forecasts, gt = get_forecasting_pipeline(model=get_model(args.model), metrics=METRICS, column=column)
            pipeline.train(data_train)

            MASE(lag=7)(y=gt, **forecasts)
            RMSE()(y=gt, **forecasts)
            MAE()(y=gt, **forecasts)
            MAEHorizon()(y=gt, **forecasts)
            MAEDistribution()(y=gt, **forecasts)
            Stats()(gt=gt, **forecasts)
            op = OP(building_id=int(column))
            op(gt=gt, **forecasts)
            RollingMAE()(y=gt, **forecasts, callbacks=[
                                   ])
            RollingRMSE()(y=gt, **forecasts, callbacks=[
                                                       ])

            op.set_train(False)

            result, summary = pipeline.test(data_test, summary_formatter=SummaryJSON(), summary=True)

            op.set_train(True)

            print("Finished a building")


