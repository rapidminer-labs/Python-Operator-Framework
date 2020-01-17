import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, CatBoost, Pool
from sklearn.preprocessing import LabelEncoder
import numpy as np

import rm_utilities as rmu


def rm_main(params, data):

    params = rmu.process_params(params)
    metadata = data.rm_metadata

    y, y_name = rmu.get_label(data, metadata)
    x, x_names = rmu.get_regular(data, metadata)
    is_classification = rmu.is_nominal(y_name, metadata)

    loss_function = params["loss_function"]
    if loss_function == "Auto":
        loss_function = __get_auto_objective__(metadata, y_name, data)
    # handle nominals in a catboost fashion
    is_cat = (x.dtypes != float)

    cat_features_index = np.where(is_cat)[0]

    if is_classification:
        label_enc = LabelEncoder()
        y = label_enc.fit_transform(y)
        pool = Pool(x, y, cat_features=cat_features_index, feature_names=list(x.columns))
        clf = CatBoostClassifier(
            loss_function=loss_function,
            iterations=params["iterations"],
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            model_size_reg=None,
            rsm=params["subsample_ratio"],
            border_count=params['border_count'],
            # feature_border_type=None, Regression only.
            fold_permutation_block_size=None,
            # od_pval=None, # only available if fitted with test
            # od_wait=None,
            # od_type=None,
            nan_mode='Forbidden',  # please handle this in RM
            leaf_estimation_iterations=params["leaf_estimation_iterations"],
            leaf_estimation_method=params["leaf_estimation_method"]
        )
    else:
        pool = Pool(x, y, cat_features=cat_features_index, feature_names=list(x.columns))
        clf = CatBoostRegressor(
            loss_function=loss_function,
            iterations=params["iterations"],
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            model_size_reg=None,
            rsm=params["subsample_ratio"],
            border_count=params['border_count'],
            feature_border_type=params["feature_border_type"],
            fold_permutation_block_size=None,
            # od_pval=None, # only available if fitted with test
            # od_wait=None,
            # od_type=None,
            nan_mode='Forbidden',  # please handle this in RM
            leaf_estimation_iterations=params[
                "leaf_estimation_iterations"],
            leaf_estimation_method=
            params["leaf_estimation_method"],
        )
    clf.fit(pool)

    model = {"clf": clf, "y_name": y_name, "x_names": x_names,
             "isClassification": is_classification, "params": params}
    if is_classification:
        model["label_enc"] = label_enc

    return model, rmu.metadata_to_string(metadata)


def __get_auto_objective__(metadata, y_name, df):
    if rmu.is_nominal(y_name, metadata):
        # this checks the data
        if rmu.is_binominal(y_name, metadata, df=df):
            return "Logloss"
        else:
            return "CrossEntropy"

    elif rmu.is_numerical(y_name, metadata):
        return "RMSE"

    else:
        raise Exception("Cannot determine objective type. Please set it manually.")
