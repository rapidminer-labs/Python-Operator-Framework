import pandas as pd
import rm_utilities as rmu


def rm_main(stored_model, data):
    y_name = stored_model["params"]["forecast_attribute"]
    try:
        index_attribute = stored_model["params"]["index_attribute"]
        future = data[[index_attribute]]
        future = future.rename(columns={index_attribute: "ds"})
    except:
        raise Exception(
            "Cannot find a attribute with name "+stored_model["params"]["index_attribute"]+
            "in your data set"
            "You need to provide a data set with the "
            "data points you want to forecast for. "
            "You can create a new data set with time stamps using Create ExampleSet operator")

    forecast = stored_model['clf'].predict(future)
    metadata = data.rm_metadata  # we loose the metadata in the merge

    if (stored_model["params"]["full_output"] is True):
        forecast = forecast.drop(["ds"], axis=1)
        data = pd.concat([data, forecast], axis=1)
    else:
        data = pd.concat([data, forecast[["yhat", "yhat_lower", "yhat_upper"]]], axis=1)
        data = data.rename(columns={"ds": index_attribute})

    # convert everything to proper rapidminer naming and roles.

    pred_name = "prediction(" + y_name + ")"
    upper_name = "upper_bound(" + y_name + ")"
    lower_name = "lower_bound(" + y_name + ")"
    data = data.rename(columns={"yhat": pred_name, "yhat_lower": lower_name, "yhat_upper": upper_name})

    data.rm_metadata = metadata

    rmu.set_roles(data, role_dict={pred_name: "prediction", upper_name: "upper_bound", lower_name: "lower_bound"})

    return data, stored_model
