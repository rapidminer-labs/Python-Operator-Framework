import pandas as pd
import rm_utilities as rmu


def rm_main(stored_model, data):
    try:
        future = data[["ds"]]
    except:
        raise Exception(
            "You need to provide a data set with the "
            "data points you want to forecast for. "
            "This can be done using a Create Example Set operator.")

    forecast = stored_model['clf'].predict(future)
    metadata = data.rm_metadata  # we loose the metadata in the merge
    print(stored_model["full_output"])
    print("####",type(stored_model["full_output"]))
    if (stored_model["full_output"] is True):
        forecast = forecast.drop(["ds"], axis=1)
        data = pd.concat([data, forecast], axis=1)
    else:
        data = pd.concat([data, forecast[["yhat", "yhat_lower", "yhat_upper"]]], axis=1)
    data.rm_metadata = metadata
    rmu.set_role(data, "yhat", "prediction")

    return data, stored_model
