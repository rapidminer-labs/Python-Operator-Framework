from sklearn import linear_model

import numpy as np
import pandas as pd


def rm_main(params,data):
	"""
	Fits the linear SGD regressor.

	Parameters
	----------
	data : pd.DataFrame
		DataFrame created from the RapidMiner ExampleSet by the Execute Python operator.

	Returns
	-------
	model
		Fitted SGD regressor model.
	data
		pd.DataFrame passed as argument.
	"""
	params_dict = process_params(params)
	features, targets = separate_features_labels(data)

	clf = linear_model.SGDRegressor(loss=params_dict['loss'], penalty=params_dict['penalty'], alpha=params_dict['alpha'],
									l1_ratio=params_dict['l1_ratio'], fit_intercept=params_dict['fit_intercept'],
									max_iter=params_dict['max_iter'], shuffle=params_dict['shuffle'], verbose=params_dict['verbose'],
									epsilon=params_dict['epsilon'], random_state=params_dict['random_state'],
									learning_rate=params_dict['learning_rate'], eta0=params_dict['eta0'],
									power_t=params_dict['power_t'], warm_start=params_dict['warm_start'],
									average=params_dict['average'])
	model = clf.fit(np.array(features), np.array(targets))

	return model,  wrapOutputHTML(model,features,targets)

def wrapOutputHTML(model,features,label):
	
	outputs = "<br><h2> Model Coef</h2><br>"
	coef_dict = {}
	for coef, feat in zip(model.coef_,features):
		coef_dict[feat] = coef
	dataFrame = pd.DataFrame.from_dict(coef_dict,orient='index')
	outputs +=  dataFrame.to_html().replace("\n","").strip()
	outputs += "<br>"
	outputs += "<br>"
	return outputs


def separate_features_labels(data):
	"""
	Using the RapidMiner attribute metadata, separates features and targets in the DataFrame passed as argument.

	Parameters
	----------
	data : pd.DataFrame
		Dataset.

	Returns
	-------
	features
		pd.DataFrame with features fit the model on.
	labels
		pd.DataFrame with targets.
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
			params['value'][i] = processString(params['value'][i])
		elif (params['type'][i] == 'ParameterTypeDouble'):
			params['value'][i] = float(params['value'][i])
		elif (params['type'][i] == 'ParameterTypeBoolean'):
			params['value'][i] = bool(params['value'][i])
		elif (params['type'][i] == 'ParameterTypeStringCategory'):
			params['value'][i] = processString(params['value'][i])
		elif (params['type'][i] == 'ParameterTypeCategory'):
			params['value'][i] = processString(params['value'][i])

	
	params_dict = dict(zip(params.key,params.value))
	#replace string None with KeyWord None
	#for key,value in params_dict.items():
	#	if value == 'None':
	#		params_dict[key] = None
	return params_dict


def processString(strvalue):
	if(strvalue == 'None'):
		strvalue = None
	return strvalue