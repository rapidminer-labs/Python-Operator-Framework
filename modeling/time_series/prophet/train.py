import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, CatBoost
from sklearn.preprocessing import LabelEncoder

import rm_utilities as rmu
from fbprophet import Prophet


def rm_main(params, data):
    params = rmu.process_params(params)
    metadata = data.rm_metadata

    clf = Prophet(
        growth=params["growth"],
        changepoints=None,
        n_changepoints=params["n_changepoints"],
        changepoint_range=0.8,
        yearly_seasonality=params["yearly_seasonality"],
        weekly_seasonality=params["weekly_seasonality"],
        daily_seasonality=params["daily_seasonality"],
        holidays=None,
        seasonality_mode=params["seasonality_mode"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        holidays_prior_scale=params["holidays_prior_scale"],
        changepoint_prior_scale=params["changepoint_prior_scale"],
        mcmc_samples=params["mcmc_samples"],
        interval_width=params["interval_width"],
        uncertainty_samples=params["uncertainty_samples"]

    )

    data = data.rename(
        columns={params["index_attribute"]: "ds",
                 params["forecast_attribute"]: "y"}
    )

    clf.fit(data[["ds", "y"]])

    model = {"clf": clf, "params": params}

    return model, rmu.metadata_to_string(metadata)
