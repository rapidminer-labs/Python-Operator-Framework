from statsmodels.tsa.arima_model import ARIMA

import numpy as np
import pandas as pd

package_dict = {'LabelPropagation': 'scikit-learn', 'PassiveAggressiveClassifier': 'scikit-learn',
				'SGDClassifier': 'scikit-learn', 'SGDRegressor': 'scikit-learn', 'Earth': 'scikit-learn',
				'SMOTE': 'scikit-learn', 'ARIMAResultsWrapper': 'statsmodels'}

def rm_main(stored_model, data):
	"""
	Applies a scikit-learn or statsmodels model to previously unseen data.

	Parameters
	----------
	stored_model
		Trained scikit-learn or statsmodels model.
	data : pd.DataFrame
		DataFrame with data to perform predictions on.

	Returns
	-------
	data
		DataFrame with model predictions.
	stored_model
		Trained scikit-learn or statsmodels model passed in method arguments.
	"""
	package_name = get_package_name(stored_model)
	if package_name == 'scikit-learn':
		data, stored_model = apply_sklearn(stored_model, data)
	elif package_name == 'statsmodels':
		data, stored_model = apply_statsmodels(stored_model, data)

	return data

def get_package_name(stored_model):
	"""
	Returns the package name of the stored_model in order to call the right method for performing classification or
	regression. The methods are different for models in scikit-learn and statsmodels packages.
	"""
	return package_dict[type(stored_model).__name__]


def preprocessing_sklearn(data):
	"""
	Using the RapidMiner attribute metadata, separates features and labels/targets in the DataFrame passed as argument.

	Parameters
	----------
	data : pd.DataFrame
		Dataset.

	Returns
	-------
	features
		pd.DataFrame with features to fit the model on.
	labels
		pd.DataFrame with labels/targets to pass to the model during training.
	"""
	for name in data.columns.values:
		attribute_type, attribute_role = data.rm_metadata[name]

		if attribute_role == 'label':
			labels = data[name]

	features = data.drop(labels.name, axis=1)
	
	return features, labels


def apply_sklearn(stored_model, data):
	"""
	Applies a scikit-learn model to previously unseen data.

	Parameters
	----------
	stored_model
		Trained scikit-learn model.
	data : pd.DataFrame
		DataFrame with data to perform predictions on.

	Returns
	-------
	data
		DataFrame with model predictions.
	stored_model
		Trained scikit-learn model passed in method arguments.
	"""
	features, labels = preprocessing_sklearn(data)

	predicted_labels = pd.DataFrame(stored_model.predict(features))
	predictionColumnName = 'Prediction(' +  labels.name + ')'
	data[predictionColumnName] = predicted_labels
	data.rm_metadata[predictionColumnName] = (data.rm_metadata[labels.name][0],"prediction")
	data.rm_metadata[labels.name] = (data.rm_metadata[labels.name][0],"label")
	return data, stored_model


def __getnewargs__(self):
	"""
	Overrides a method that causes a bug when trying to serialise the ARIMA model.
	"""
	return ((self.endog), (self.k_lags, self.k_diff, self.k_ma))


def preprocessing_statsmodels(data):
	"""
	Preprocesses the data to prepare for performing prediction with the ARIMA model.

	Parameters
	----------
	data : pd.DataFrame
		DataFrame with the dataset.
	Returns
	----------
	date_time_column
		pd.Series with date_time data.
	prediction_column
		pd.Series with target data.
	"""
	date_time_column = pd.Series()
	prediction_column = pd.Series()

	for name in data.columns.values:
		attribute_type, attribute_role = data.rm_metadata[name]

		if attribute_type == 'date_time':
			date_time_column = data[name]
		if attribute_role == 'prediction':
			prediction_column = data[name]

	return date_time_column, prediction_column


def apply_statsmodels(model, data):
	"""
	 Performs out-of-sample forecast on data.

	 Parameters
	 ----------
	 model : statsmodels.tsa.arima_model.ARIMA
		Pre-trained ARIMA model.
	 data : pd.DataFrame
		DataFrame with data on which to forecast.

	 Returns
	 -------
	 result_df
		 DataFrame with out-of-sample forecasts.
	 model
		 Fitted ARIMA model.
	 """
	ARIMA.__getnewargs__ = __getnewargs__

	date_time_column, prediction_column = preprocessing_statsmodels(data)
	start = date_time_column.iloc[0]
	end = date_time_column.iloc[-1]

	predictions = model.predict(start=start, end=end, typ='levels')
	result_df = pd.DataFrame(data=predictions.values, columns=[prediction_column.name])
	result_df[date_time_column.name] = date_time_column
	result_df.rm_metadata = {date_time_column.name: ('date_time', 'id'), prediction_column.name: ('real', 'prediction')}

	return result_df, model


