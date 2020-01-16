from sklearn import linear_model

import numpy as np
import pandas as pd


def rm_main(params,data):
	print(data.head())
	print(params)
	"""
	Fits the linear SGD classifier.

	Parameters
	----------
	data : pd.DataFrame
		DataFrame created from the RapidMiner ExampleSet by the Execute Python operator.

	Returns
	-------
	model
		Fitted SGD classifier.
	data
		pd.DataFrame passed as argument.
	"""
	params_dict = process_params(params)
	features, labels = separate_features_labels(data)
	clf = linear_model.SGDClassifier(loss=params_dict['loss'], penalty=params_dict['penalty'], alpha=params_dict['alpha'],
									l1_ratio=params_dict['l1_ratio'], fit_intercept=params_dict['fit_intercept'],
									max_iter=params_dict['max_iter'], shuffle=params_dict['shuffle'], verbose=params_dict['verbose'],
									epsilon=params_dict['epsilon'], random_state=1,
									learning_rate=params_dict['learning_rate'], eta0=params_dict['eta0'],
									power_t=params_dict['power_t'], class_weight=params_dict['class_weight'], warm_start=params_dict['warm_start'],
									average=params_dict['average'])
	model = clf.fit(np.array(features), np.array(labels))
	print(model)
	return model, wrapOutputHTML(model,features,labels)
	
def wrapOutputHTML(model,features,label):
	
	outputs = '<h3><span color=\'red\'><h1> name of features</h1> <br>'
	for col in features.columns: 
		outputs += str(col)
		outputs += ' -- '
		#outputs += str(coef_dict[col] )
	outputs += '</span></h3>'
	outputs += "<br><h2> Model Coef</h2><br>"
	dataFrame = pd.DataFrame.from_records(model.coef_).T
	outputs +=  dataFrame.to_html().replace("\n","").strip()
	outputs += "<br>"
	outputs += "<br>"
	return outputs

def separate_features_labels(data):
	"""
	Using the RapidMiner attribute metadata, separates features and labels in the DataFrame passed as argument.

	Parameters
	----------
	data : pd.DataFrame
		Dataset.

	Returns
	-------
	features
		pd.DataFrame with features to train the SGD classifier on.
	labels
		pd.DataFrame with labels to pass to the SGD classifier during training.
	"""

	for name in data.columns.values:
		attribute_type, attribute_role = data.rm_metadata[name]

		if attribute_role == 'label':
			labels = data[name]

	features = data.drop(labels.name, axis=1)

	return features, labels

def process_params(params):
    for i in params.index:
        print(params['type'][i])
        if (params['type'][i] == 'ParameterTypeInt'):
            params['value'][i] = int(params['value'][i])
        elif (params['type'][i] == 'ParameterTypeString'):
            params['value'][i] = __process_parameter_string__(params['value'][i])
        elif (params['type'][i] == 'ParameterTypeDouble'):
            params['value'][i] = float(params['value'][i])
        elif (params['type'][i] == 'ParameterTypeBoolean'):
            params['value'][i] = __process_parameter_string__(params['value'][i])
        elif (params['type'][i] == 'ParameterTypeStringCategory'):
            params['value'][i] = __process_parameter_string__(params['value'][i])
        elif (params['type'][i] == 'ParameterTypeCategory'):
            params['value'][i] = __process_parameter_string__(params['value'][i])
    params_dict = dict(zip(params.key, params.value))
    # replace string None with KeyWord None
    # for key,value in params_dict.items():
    #   if value == 'None':
    #       params_dict[key] = None
    return params_dict

def __process_parameter_string__(strvalue):
    if (strvalue == 'None'):
        strvalue = None
    if strvalue == "True":
        return True
    if strvalue == "False":
        return False
    return strvalue


def processString(strvalue):
	if(strvalue == 'None'):
		strvalue = None
	return strvalue
	
