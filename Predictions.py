import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#import data
data = pd.read_csv('train.csv')

#remove out independent variables
numeric = data.select_dtypes(include=[np.number])
corr = numeric.corr()
cols = corr['turnout'].sort_values(ascending=False)[:7].index
print(cols)
print(data['turnout'].describe())
plt.hist(data['turnout'])
plt.show()
data_for_model = data[cols]

#split to training and test data sets

data_train, data_test = train_test_split(data_for_model, test_size=0.5, random_state=25)

turnout_train = data_train['turnout']
data_train = data_train.drop(['turnout'], axis=1)

turnout_test = data_test['turnout']
data_test = data_test.drop(['turnout'], axis=1)

#build linear model with logorithm dependent variable
lr = linear_model.LinearRegression()
model = lr.fit(data_train, np.log(turnout_test))
print(f"R^2 is: {model.score(data_test,np.log(turnout_test))}")

#generate prediction for the test.csv file, remove out categorical
test_csv_input = pd.read_csv('test.csv')
data_for_model_test_csv = test_csv_input[cols.drop('turnout')]

pred_test_csv = model.predict(data_for_model_test_csv)
pred_price = np.exp(pred_test_csv)

#output csv
pred_output_df = pd.DataFrame(pred_price, columns=['turnout'])
pred_output_df.insert(0,'state', test_csv_input['state'])
# print(pred_output_df)
pred_output_df.to_csv('predictions.csv', index=-False)