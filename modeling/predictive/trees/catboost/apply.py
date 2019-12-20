import pandas as pd
import rm_utilities as rmu


def rm_main(stored_model, data):
    yName = stored_model["y_name"]
    xNames = stored_model["x_names"]
    clf = stored_model["clf"]

    regular = data[xNames]
    y_pred = clf.predict(regular)
    if stored_model["isClassification"] is True:
        # convert back to strings
        y_pred = stored_model["label_enc"].inverse_transform(y_pred.astype(int))

        # add confidences
        confidence_names = ["confidence(%s)" % v for v in stored_model["label_enc"].classes_.tolist()]
        confidence_roles = ["confidence_%s" % v for v in stored_model["label_enc"].classes_.tolist()]
        confidence_columns = pd.DataFrame.from_records(clf.predict_proba(regular), columns=confidence_names)
        metadata = data.rm_metadata  # we loose the metadata in the merge
        data = pd.concat([data, confidence_columns], axis=1)
        data.rm_metadata = metadata

        # set the meta data so we can use Performance and so on in RM
        for name, role in zip(confidence_names, confidence_roles):
            rmu.set_role(data, name, role)

    pred_name = "prediction(" + yName + ")"
    data[pred_name] = y_pred

    rmu.set_role(data, pred_name, "prediction")

    return data, stored_model
