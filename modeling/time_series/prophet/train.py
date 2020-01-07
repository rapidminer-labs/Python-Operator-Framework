import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, CatBoost
from sklearn.preprocessing import LabelEncoder

import rm_utilities as rmu
from fbprophet import Prophet


def rm_main(params, data):
    params_dict = rmu.process_params(params)
    metadata = data.rm_metadata

    clf = Prophet(
        growth=params_dict["growth"],
        changepoints=None,
        n_changepoints=params_dict["n_changepoints"],
        changepoint_range=0.8,
        yearly_seasonality=params_dict["yearly_seasonality"],
        weekly_seasonality=params_dict["weekly_seasonality"],
        daily_seasonality=params_dict["daily_seasonality"],
        holidays=None,
        seasonality_mode='additive',
        seasonality_prior_scale=params_dict["seasonality_prior_scale"],
        holidays_prior_scale=params_dict["holidays_prior_scale"],
        changepoint_prior_scale=params_dict["changepoint_prior_scale"],
        mcmc_samples=0,
        interval_width=0.80,
        uncertainty_samples=1000

    )

    clf.fit(data[["ds", "y"]])

    model = {"clf": clf, "full_output": params_dict["full_output"]}

    return model, rmu.metadata_to_string(metadata)
