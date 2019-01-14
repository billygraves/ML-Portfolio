import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
df = pd.read_csv('cal_housing_clean.csv')
for col in df.columns:
	print(df[col].describe()) #note that describe prints the name of the column at the bottom
	# Additionally, when compiling in Sublime it is useful to loop along the columns so that all output can be seen
X_train, X_test, y_train, y_test = train_test_split(df.drop('medianHouseValue', axis = 1), df['medianHouseValue'], test_size = 0.3, random_state = 101)

s = MinMaxScaler()
s.fit(X = X_train)
scale_X_train = pd.DataFrame(s.transform(X_train), columns = X_train.columns, index = X_train.index)
scale_X_test = pd.DataFrame(s.transform(X_test), columns = X_test.columns, index = X_test.index)

# Creating feature columns
feat_cols = []
for col in X_train.columns:
	col = tf.feature_column.numeric_column(col)
	feat_cols.append(col)
print(feat_cols)

input_func = tf.estimator.inputs.pandas_input_fn(x = scale_X_train, y = y_train, batch_size = 10, num_epochs = 1000, shuffle = True)

model = tf.estimator.DNNRegressor(hidden_units = [6,6,6], feature_columns = feat_cols)

model.train(input_fn = input_func, steps = 20000)

predict_input_func = tf.estimator.inputs.pandas_input_fn(x = scale_X_test, batch_size = 10, num_epochs = 1, shuffle = False) #1 epoch because testing

pred_gen = model.predict(input_fn = predict_input_func)

predictions = list(pred_gen)
final_preds = []
for pred in predictions:
	final_preds.append(pred['predictions'])
	print(pred, '\n')

print(mean_squared_error(y_test, final_preds)**0.5)
