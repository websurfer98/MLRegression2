import pandas as pd
import numpy as np
from sklearn import linear_model

# Load the King County house data

# Use the type dictionary supplied
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

###################
# Using test data #
###################
print("\n\nUsing test data...")
test = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)
output = np.array(test['price'])
input_feature = []

# Bedrooms squared
raw1 = test['bedrooms']
input_feature.append(np.multiply(raw1, raw1))

# Bedrooms * Bathrooms
raw1 = np.array(test['bedrooms'])
raw2 = np.array(test['bathrooms'])
input_feature.append(np.multiply(raw1, raw2))

# Log living area
raw1 = np.array(test['sqft_living'])
input_feature.append(np.log(raw1))


# Lat long sum
raw1 = np.array(test['lat'])
raw2 = np.array(test['long'])
input_feature.append(np.add(raw1, raw2))

# Quiz question
# Mean of 4 new variables on test data
N = len(output)
print("bedroomsSq, bedbath, logliving, latlong")
mean = [print(np.sum(x)/N, "\n") for x in input_feature]



#######################
# Using training data #
#######################
print("\n\nUsing training data...")
# Example here - http://bigdataexaminer.com/uncategorized/how-to-run-linear-regression-in-python-scikit-learn/
# Train the linear regression model - load the training data

# Model 1: sqft_living, bedrooms, bathrooms, lat, and long
train = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict, usecols=["sqft_living", "bedrooms", "bathrooms", "lat", "long", "price"], header=0)
print(train.head(5))

y = train['price']
X = train.drop('price', axis=1)
print(X.head(5)) # Print the first n rows

lm1 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
lm1.fit(X, y)
print('Intercept: ', lm1.intercept_)
coeffs = pd.DataFrame(list(zip(X.columns, lm1.coef_)), columns = ['features', 'estimatedCoefficients'])
print(coeffs)
print(lm1.predict(X)[0:5])
rss = np.sum((train['price'] - lm1.predict(X)) ** 2)
print("Rss: ", rss)


# Model 2: sqft_living, bedrooms, bathrooms, lat, long, and bed_bath_rooms
# Add this into the mix - Bedrooms * Bathrooms
raw1 = np.array(train['bedrooms'])
raw2 = np.array(train['bathrooms'])
bedbath = np.array(np.multiply(raw1, raw2))
train['bedbath'] = bedbath
print(train.head(5))
X = train.drop('price', axis=1)
print(X.head(5)) # Print the first n rows

lm2 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
lm2.fit(X, y)
print('Intercept: ', lm2.intercept_)
coeffs = pd.DataFrame(list(zip(X.columns, lm2.coef_)), columns = ['features', 'estimatedCoefficients'])
print(coeffs)
print(lm2.predict(X)[0:5])

rss = np.sum((train['price'] - lm2.predict(X)) ** 2)
print("Rss: ", rss)


# Model 3: sqft_living, bedrooms, bathrooms, lat, long, bed_bath_rooms, bedrooms_squared, log_sqft_living, and lat_plus_long
# Add the others to the mix
raw1 = train['bedrooms']
bedsquared = np.array(np.multiply(raw1, raw1))
train['bedsquared'] = bedsquared

raw1 = np.array(train['sqft_living'])
logliving = np.array(np.log(raw1))
train['logliving'] = logliving

raw1 = np.array(train['lat'])
raw2 = np.array(train['long'])
latlong = np.array((np.add(raw1, raw2)))
train['latlong'] = latlong

print(train.head(5))
X = train.drop('price', axis=1)
print(X.head(5)) # Print the first n rows

lm3 = linear_model.LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
lm3.fit(X, y)
print('Intercept: ', lm3.intercept_)
coeffs = pd.DataFrame(list(zip(X.columns, lm3.coef_)), columns = ['features', 'estimatedCoefficients'])
print(coeffs)
print(lm3.predict(X)[0:5])

rss = np.sum((train['price'] - lm3.predict(X)) ** 2)
print("Rss: ", rss)


###################
# Using test data #
###################
print("\n\nUsing test data...")
# Example here - http://bigdataexaminer.com/uncategorized/how-to-run-linear-regression-in-python-scikit-learn/
# Train the linear regression model - load the training data

# Model 1: sqft_living, bedrooms, bathrooms, lat, and long
test = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict, usecols=["sqft_living", "bedrooms", "bathrooms", "lat", "long", "price"], header=0)
print(test.head(5))

y = test['price']
X = test.drop('price', axis=1)
print(X.head(5)) # Print the first n rows

print(lm1.predict(X)[0:5])
rss = np.sum((test['price'] - lm1.predict(X)) ** 2)
print("Rss: ", rss)


# Model 2: sqft_living, bedrooms, bathrooms, lat, long, and bed_bath_rooms
# Add this into the mix - Bedrooms * Bathrooms
raw1 = np.array(test['bedrooms'])
raw2 = np.array(test['bathrooms'])
bedbath = np.array(np.multiply(raw1, raw2))
test['bedbath'] = bedbath
print(test.head(5))
X = test.drop('price', axis=1)
print(X.head(5)) # Print the first n rows

print(lm2.predict(X)[0:5])
rss = np.sum((test['price'] - lm2.predict(X)) ** 2)
print("Rss: ", rss)


# Model 3: sqft_living, bedrooms, bathrooms, lat, long, bed_bath_rooms, bedrooms_squared, log_sqft_living, and lat_plus_long
# Add the others to the mix
raw1 = test['bedrooms']
bedsquared = np.array(np.multiply(raw1, raw1))
test['bedsquared'] = bedsquared

raw1 = np.array(test['sqft_living'])
logliving = np.array(np.log(raw1))
test['logliving'] = logliving

raw1 = np.array(test['lat'])
raw2 = np.array(test['long'])
latlong = np.array((np.add(raw1, raw2)))
test['latlong'] = latlong

print(test.head(5))
X = test.drop('price', axis=1)
print(X.head(5)) # Print the first n rows

print(lm3.predict(X)[0:5])
rss = np.sum((test['price'] - lm3.predict(X)) ** 2)
print("Rss: ", rss)
