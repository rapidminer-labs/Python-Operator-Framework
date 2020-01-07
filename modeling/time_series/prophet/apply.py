import pandas as pd
import rm_utilities as rmu


def rm_main(stored_model, data):
    try:
        index_attribute = stored_model["params"]["index_attribute"]
        future = data[[index_attribute]]
        future = future.rename(columns={index_attribute: "ds"})
    except:
        raise Exception(
            "You need to provide a data set with the "
            "data points you want to forecast for. "
            "This can be done using a Create Example Set operator.")

    forecast = stored_model['clf'].predict(future)
    metadata = data.rm_metadata  # we loose the metadata in the merge

    if (stored_model["params"]["full_output"] is True):
        forecast = forecast.drop(["ds"], axis=1)
        data = pd.concat([data, forecast], axis=1)
    else:
        data = pd.concat([data, forecast[["yhat", "yhat_lower", "yhat_upper"]]], axis=1)
        data = data.rename(columns={"ds": index_attribute})

    # convert everything to proper rapidminer naming and roles.

    pred_name = "prediction(" + stored_model["params"]["forecast_attribute"] + ")"
    data = data.rename(columns={"yhat": pred_name})
    data.rm_metadata = metadata
    rmu.set_role(data, pred_name, "prediction")
    # TODO names for upper and lower?
    return data, stored_model
