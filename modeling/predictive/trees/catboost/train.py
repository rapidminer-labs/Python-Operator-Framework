import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, CatBoost
from sklearn.preprocessing import LabelEncoder

import rm_utilities as rmu


def rm_main(params, data):
    params_dict = dict(zip(params.key, params.value))
    metadata = data.rm_metadata

    y, y_name = rmu.get_label(data, metadata)
    x, x_names = rmu.get_regular(data, metadata)
    is_classification = rmu.is_nominal(y_name, metadata)
    if is_classification:
        label_enc = LabelEncoder()
        label_enc.fit(y)
        y = label_enc.transform(y)

        params_dict["iterations"] = int(params_dict["iterations"])
        params_dict["learning_rate"] = float(params_dict["learning_rate"])

        clf = CatBoostClassifier(iterations=params_dict["iterations"],
                                 learning_rate=params_dict["learning_rate"]
                                 )
    else:
        params_dict["iterations"] = int(params_dict["iterations"])
        params_dict["learning_rate"] = float(params_dict["learning_rate"])

        clf = CatBoostRegressor(iterations=params_dict["iterations"],
                                learning_rate=params_dict["learning_rate"]
                                )
    clf.fit(x, y)

    model = {"clf": clf, "y_name": y_name, "x_names": x_names, "isClassification": is_classification}
    if is_classification:
        model["label_enc"] = label_enc

    return model, rmu.metadata_to_string(metadata)
