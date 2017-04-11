#!/usr/bin/python
import pandas as pd      # Handling csv file data
import numpy as np       # Matrix operation
import math              # Mathematical computation
import xgboost as xgb    # Regression
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score # Split validation set
import matplotlib.pyplot as plt # plots
from sklearn.linear_model import Ridge
#files
train_set = 'train.csv'
test_set = 'test.csv'
result_file = 'ma_result.csv'

#data set generation 
train = pd.read_csv(train_set)
test = pd.read_csv(test_set)
output = test[['Id']].copy()

# train = train.drop(train[(train['GrLivArea'] > 3000)].index )
train = train.drop(train[((train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000))].index )
train = train.drop(train[((train['YearBuilt'] < 1900) & (train['SalePrice'] > 200000))].index )
train = train.drop(train[((train['GarageYrBlt'] < 2000) & (train['SalePrice'] > 600000))].index )
train = train.drop(train[train['GarageArea'] >= 1200].index)
train = train.drop(train[train['1stFlrSF'] >= 2500].index)
#0.126
train = train.drop(train[train['TotalBsmtSF'] >= 2500].index)
train = train.drop(train[train['OverallQual'] <= 2].index)
train = train.drop(train[((train['OverallQual'] == 4) & (train['SalePrice'] >= 200000))].index)
train = train.drop(train[((train['OverallQual'] == 8) & (train['SalePrice'] >= 500000))].index)


# var = 'YrSold'
# data = pd.concat([train['SalePrice'], train[var]], axis=1)

# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# plt.show()

# train = train.drop(train[(train['MasVnrArea'] > 1000)].index)
#overall cond


#lotfront

# print train[train['LotFrontage'] >= 100]

#drop
test = test.drop(['Utilities'], axis=1)
train = train.drop(['Utilities'], axis=1)

#drop
test = test.drop(['MSSubClass'], axis=1)
train = train.drop(['MSSubClass'], axis=1)

# #drop yrSold
# test = test.drop(['YrSold'], axis=1)
# train = train.drop(['YrSold'], axis=1)

# test = test.drop(['MasVnrArea'], axis=1)
# train = train.drop(['MasVnrArea'], axis=1)

# test = test.drop(['PoolArea'], axis=1)
# train = train.drop(['PoolArea'], axis=1)
#replace BsmtQual
train = train.replace({'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
test = test.replace({'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
#ExterQual
train = train.replace({'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
test = test.replace({'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
#ExterCond
train = train.replace({'ExterCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
test = test.replace({'ExterCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
#BsmtCond
train = train.replace({'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
test = test.replace({'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
#HeatingQC
train = train.replace({'HeatingQC': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
test = test.replace({'HeatingQC': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
#KitchenQual
train = train.replace({'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
test = test.replace({'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
#FireplaceQu
train = train.replace({'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
test = test.replace({'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
#GarageQual
train = train.replace({'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
test = test.replace({'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
#GarageCond
train = train.replace({'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})
test = test.replace({'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.NaN: 0}})

#poolQc
train = train.replace({'PoolQC': {'Ex': 500, 'Gd': 400, 'TA': 300, 'Fa': 200, 'Po': 100, np.NaN: 0}})
test = test.replace({'PoolQC': {'Ex': 500, 'Gd': 400, 'TA': 300, 'Fa': 200, 'Po': 100, np.NaN: 0}})


#fill in loat area with sqrt
test['SqrtLotArea'] = np.sqrt(test['LotArea'])
train['SqrtLotArea'] = np.sqrt(train['LotArea'])


cond = test['LotFrontage'].isnull()
test.LotFrontage[cond] = test.SqrtLotArea[cond]
cond = train['LotFrontage'].isnull()
train.LotFrontage[cond] = train.SqrtLotArea[cond]

del test['SqrtLotArea']
del train['SqrtLotArea']

# drop
train = train.drop(train[((train['LotFrontage'] > 200) | (train['SalePrice'] >= 500000))].index)
# train = train.drop(train[((train['OverallCond'] == 2) & (train['SalePrice'] >= 300000))].index)
# train = train.drop(train[((train['YearRemodAdd'] < 1990) & (train['SalePrice'] >= 300000))].index)




#fill missing value with 0
def fill_miss_zero(df):
	for col in features:
		for i in df.index:
			if pd.isnull(df[col][i]):
				if df[col].dtype == 'object':
					df = df.set_value(i,col,'missing')
				else:
					df = df.set_value(i,col,df[col].mean())

#convert categorical to numerical
def cat_to_num(df):
	for col in features:
		if df[col].dtype == 'object':
			tf = pd.Categorical(df[col].values).codes
			df[col] = tf

#find correlations between 
#features and SalePrice
def find_correlation():
	global sale_cor
	sale_cor = train.corr()['SalePrice']
	# print sale_cor
	# indexes = sale_cor.index
	# indexes = indexes.drop('Id')
	# print indexes
	indexes = sale_cor[sale_cor.abs() > 0].index
	indexes = indexes.drop('Id')
	indexes = indexes.drop('SalePrice')
	return indexes

#convert features to numerical value
def numerize(df):
	fill_miss_zero(df)
	# cat_to_num(df)
	# df = pd.get_dummies(df)

def log_transform(feature):
	global train, test
	train[feature] = np.log1p(train[feature].values)
	test[feature] = np.log1p(test[feature].values)

def quadratic(feature):
	global train, test
	train[feature] = train[feature]**2
	test[feature] = test[feature]**2

#preprocessing
def preprocessing():
	global train_label,train,test,features,indexes

	train_label = train['SalePrice']

	features = train.columns.values.tolist()
	features.remove('SalePrice')

	

	numerize(train)
	numerize(test)

	log_transform('1stFlrSF')
	# log_transform('LotArea')
	quadratic('OverallQual')
	quadratic('GrLivArea')
	quadratic('GarageCars')
	# quadratic('1stFlrSF')
	# quadratic('FullBath')
	quadratic('YearBuilt')
	quadratic('YearRemodAdd')
	quadratic('TotalBsmtSF')
	#two more quardratic add by yupeng gou on 4.10
	# quadratic('YrSold')#GarageYrBlt

	#get interesting features
	# indexes = find_correlation()
	
	#drop missing value record
	for i in train.index:
		if pd.isnull(train['Electrical'][i]):
			train = train.drop(i)
			break


	#delete useless features
	del train['Id']
	del test['Id']

	# train = train[indexes]
	

	train = train.drop(['SalePrice'], axis=1)

	all_data = pd.concat((train,test))

	all_data = all_data.drop('GarageArea', 1)
	# all_data = all_data.drop('GarageQual', 1)
	# # all_data = all_data.drop('GarageCond', 1)
	# # all_data = all_data.drop('GarageType', 1)
	# all_data = all_data.drop('GarageFinish', 1)
	all_data = all_data.drop('TotRmsAbvGrd', 1)

	all_data = pd.get_dummies(all_data)


	train = all_data[:train.shape[0]]
	test = all_data[train.shape[0]:]


	print train.shape
	# print test.shape

	# test = test[indexes]

	# train.to_csv('preprocess.csv' , index = False)
	
	#convert to ndarray
	# train = train.values
	train.to_csv('preprocess1.csv' , index = False)
	# print train.shape
	train_label = train_label.values
	# test = test.values


#model training
def training():
	global train, model

	#generate validation set
	x_train, x_validate, y_train, \
	y_validate = train_test_split(train, train_label, random_state=42)

	#generate data for regression
	dtrain = xgb.DMatrix(x_train, label=y_train)
	dvalidate = xgb.DMatrix(x_validate, label=y_validate)

	#model parameters
	params = {}
	params["objective"] = "reg:linear"
	params["silent"] = 1
	params["max_depth"] = 3
	params["eta"] = 0.01
	params["eval_metric"] = 'rmse'
	params["tree_method"] = 'approx'
	params["subsample"] = 0.5
	plst = list(params.items())

	num_rounds=40000

	#watchlist
	watchlist  = [(dtrain,'train'),(dvalidate,'eval')]

	model = xgb.train(plst, 
					  dtrain,
					  num_rounds,
					  evals=watchlist,
					  early_stopping_rounds=500)
	return model


#predict and output result
def prediction():
	global output

	dtest = xgb.DMatrix(test)
	result = model.predict(dtest)
	#check negative
	for i in range(len(result)):
		if result[i] < 0:
			print 'zero'
			result[i] = 0;
	#output
	output['SalePrice'] = pd.Series(result, index=output.index)
	output.to_csv(result_file, index=False)


#run




preprocessing()
training()
prediction()