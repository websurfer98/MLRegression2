import pandas as pd
import numpy as np

# Helper functions
def get_numpy_data(df, features, output):
    df['constant'] = 1 # add a constant column 

    # prepend variable 'constant' to the features list
    features = ['constant'] + features

    # Filter by features
    fm = df[features]
    y = df[output]
   
    # convert to numpy matrix/vector whatever...
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.as_matrix.html
    features_matrix = fm.as_matrix()
    output_array = y.as_matrix()

    return(features_matrix, output_array)

def predict_outcome(feature_matrix, weights):
    result = feature_matrix.dot(weights.T)
    return result

def feature_derivative(errors, feature):
    return -2 * feature.T.dot(errors)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights
        yhat = predict_outcome(feature_matrix, weights)

        # compute the errors as predictions - output
        errors = np.subtract(yhat, output)
        
        gradient_sum_squares = 0 # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]
            delw = feature_derivative(errors, feature_matrix[:, i])
            
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares = delw ** 2
            
            # update the weight based on step size and derivative
            weights[i] += step_size * delw
        
        gradient_magnitude = np.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True

    return(weights)

# Load the King County house data

# Use the type dictionary supplied
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
train_data = pd.read_csv('kc_house_train_data.csv', dtype=dtype_dict)
test_data = pd.read_csv('kc_house_test_data.csv', dtype=dtype_dict)

#######################
# Using training data #
#######################
print("\n\nUsing training data for model 1...")

simple_features = ['sqft_living']
my_output= 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)

initial_weights = np.array([-47000., 1.]) # intercept, sqft_living respectively
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance)
print("simple_weights[1]: ", simple_weights[1])

###################
# Using test data #
###################
print("\n\nUsing test data for model 1...")

test_features = ['sqft_living']
test_output= 'price'
(test_feature_matrix, output) = get_numpy_data(test_data, test_features, test_output)

yhat = test_feature_matrix.dot(simple_weights)
print("First house prediction using model 1: ", yhat[0])

diff = yhat - output
rss = diff.T.dot(diff)
print("Rss using model 1: ", rss)

#######################
# Using training data #
#######################
print("\n\nUsing training data for model 2...")

model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)

initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

model_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
print("simple_weights[1]: ", model_weights[1])

###################
# Using test data #
###################
print("\n\nUsing test data for model 2...")

test_features = ['sqft_living', 'sqft_living15']
test_output= 'price'
(test_feature_matrix, output) = get_numpy_data(test_data, test_features, test_output)

yhat = test_feature_matrix.dot(model_weights)
print("First house prediction using model 2: ", yhat[0])

diff = yhat - output
rss = diff.T.dot(diff)
print("Rss using model 2: ", rss)

print("Actual price of house in test set: ", output[0])

