import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
df = pd.read_csv('census_data.csv')

le = LabelEncoder()
le.fit([' <=50K', ' >50K'])
df['income_bracket'] = le.transform(df['income_bracket'])

X_train, X_test, y_train, y_test = train_test_split(df.drop('income_bracket', axis = 1), 
	df['income_bracket'], test_size = 0.3, random_state = 101)

categorical = ['workclass', 'education', 'marital_status',
       'occupation', 'relationship', 'race', 'gender',
       'native_country']

continuous = []
for var in X_train.columns:
	if var not in categorical:
		continuous.append(var)

feat_vars = []
for col in categorical:
	col = tf.feature_column.categorical_column_with_hash_bucket(col, hash_bucket_size = 60)
	feat_vars.append(col)

for col in continuous:
	col = tf.feature_column.numeric_column(col)
	feat_vars.append(col)

#Creating input function
input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size = 10,
	num_epochs = 10000, shuffle = True)

model = tf.estimator.LinearClassifier(feature_columns = feat_vars)

model.train(input_fn = input_func, steps = 20000)

predict_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size = len(X_test),
	num_epochs = 1, shuffle = False)

pred = model.predict(input_fn = predict_input_func)

predictions = list(pred)
final_preds = []
for x in predictions:
	final_preds.append(x['class_ids'])

from sklearn.metrics import classification_report

print(classification_report(y_test, final_preds))

