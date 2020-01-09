from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import rm_utilities as rmu


def rm_main(params, data):
    params = rmu.process_params(params)
    metadata = data.rm_metadata

    y, y_name = rmu.get_label(data, metadata)
    x, x_names = rmu.get_regular(data, metadata)

    # check if it is classification. This is done only on the metadata we retrieve from rm
    if rmu.is_nominal(y_name, metadata):
        is_classification = True
    elif rmu.is_numerical(y_name, metadata):
        is_classification = False
    else:
        raise Exception("Cannot find a numerical or nominal label attribute. Did you set one with Set Role?")
    params["is_classification"] = is_classification

    # set objective to reg:linear for regression, binary:logistic for binominal and multi:softprob for polynominal
    # this involves a data scan, since we don't know if a RM nominal has 2 or more classes.
    if params["objective"] == "auto":
        params["objective"] = __get_auto_objective__(metadata, y_name, data)


    for xname in x_names:
        if rmu.is_nominal(xname, metadata):
            raise Exception("Nominal Attributes are not yet supported. Please use Nominal to Numerical in RM.")


    if is_classification:

        label_enc = LabelEncoder()
        label_enc.fit(y)
        y = label_enc.transform(y)

        clf = XGBClassifier(max_depth=params["max_depth"],
                            learning_rate=params["learning_rate"], n_estimators=params["n_estimators"],
                            verbosity=1, silent=None, objective=params["objective"], booster=params["booster"],
                            n_jobs=1, nthread=None,
                            gamma=params["gamma"],
                            min_child_weight=params["min_child_weight"],
                            max_delta_step=params["max_delta_step"],
                            subsample=params["max_delta_step"], colsample_bytree=1, colsample_bylevel=1,
                            colsample_bynode=1,
                            reg_alpha=params["reg_alpha"], reg_lambda=params["reg_lambda"], scale_pos_weight=1,
                            base_score=0.5, random_state=params["random_state"], missing=None,
                            importance_type="gain")
        clf.fit(x, y)
    else:
        clf = XGBRegressor(max_depth=params["max_depth"],
                           learning_rate=params["learning_rate"], n_estimators=params["n_estimators"],
                           verbosity=1, silent=None, objective=params["objective"], booster=params["booster"],
                           n_jobs=1, nthread=None,
                           gamma=params["gamma"],
                           min_child_weight=params["min_child_weight"],
                           max_delta_step=params["max_delta_step"],
                           subsample=params["max_delta_step"], colsample_bytree=1, colsample_bylevel=1,
                           colsample_bynode=1,
                           reg_alpha=params["reg_alpha"], reg_lambda=params["reg_lambda"], scale_pos_weight=1,
                           base_score=0.5, random_state=params["random_state"], missing=None,
                           importance_type="gain")
        clf.fit(x, y)

    model = {"clf": clf,
             "params": params,
             "y_name": y_name,
             "x_names": x_names,
             }
    if is_classification:
        model["label_enc"] = label_enc

    return model, rmu.metadata_to_string(metadata)


def __get_auto_objective__(metadata, y_name, df):
    if rmu.is_nominal(y_name, metadata):
        # this checks the data
        if rmu.is_binominal(y_name, metadata, df=df):
            return "binary:logistic"
        else:
            return "multi:softprob"

    elif rmu.is_numerical(y_name, metadata):
        return "reg:linear"

    else:
        raise Exception("Cannot determine objective type. Please set it manually.")
