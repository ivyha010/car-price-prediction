# Car price prediction: Regression
# Download dataset "Car features and MSRP" from kaggle:
# https://www.kaggle.com/datasets/CooperUnion/cardataset/
# Save it as data_cars.csv
#
# To-do tasks:
# EDA
# Target variable distribution 
# Split dataset into train/val/test sets
# Implement the linear regression model with numpy
# Use RMSE to validate the proposed model
# Feature engineering using age, categorical features
# Regularization 

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

# RMSE
def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

def prepare_X_more(df):
    df = df.copy()
    df['age'] = 2017 - df['year']
    features = base + ['age']

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X


def prepare_X_dummy(df):
    df = df.copy()

    df['age'] = 2017 - df['year']
    features = base + ['age']

    for v in [2, 3, 4]:
        df['num_doors_%d' % v] = (df.number_of_doors == v).astype(int)
        features.append('num_doors_%d' % v)

    for name, values in categorical.items():
        for value in values:
            df['%s_%s' % (name, value)] = (df[name] == value).astype(int)
            features.append('%s_%s' % (name, value))

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values

    return X

def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


if __name__=="__main__":
    # Data preparation
    df = pd.read_csv('data_cars.csv')

    # Get lower case of column names and replace ' ' with '_'
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Get column names that are objects
    strings = list(df.dtypes[df.dtypes == 'object'].index)
    print(strings)

    # Convert to lower case and replace ' ' with '_'
    for col in strings:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    print(df.dtypes)

    # Exploratory data analysis
    for col in df.columns:
        print(col)
        print(df[col].unique())
        print(df[col].nunique())
        print()
    print(df)

    ## Distribution of price
    sns.histplot(df.msrp, bins=50)
    plt.savefig('saved_img1.png')
    plt.close()
    sns.histplot(df.msrp[df.msrp < 100000], bins=50)
    plt.savefig('saved_img2.png')
    plt.close()

    ## Get the log1p of the prices to get rid of log 0. Note log1p(x) means log(x+1)
    price_logs = np.log1p(df.msrp)
    sns.histplot(price_logs, bins=50)
    plt.savefig('saved_img3.png')
    plt.close()

    ## Count the number of values that are NULL
    print(df.isnull().sum())

    # Setting up the validation framework: Train: Test : Val = 0.6 : 0.2 : 0.2
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test
    print(n_val, n_test, n_train)

    idx = np.arange(n)
    np.random.seed(2)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train+n_val]]
    df_test = df.iloc[idx[n_train+n_val:]]

    # Reset the index
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = np.log1p(df_train.msrp.values)
    y_val = np.log1p(df_val.msrp.values)
    y_test = np.log1p(df_test.msrp.values)

    del df_train['msrp']
    del df_val['msrp']
    del df_test['msrp']

    # Car price baseline model using linear regression
    print('Columns: ', df_train.columns)
    base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

    X_train = df_train[base].fillna(0).values  # Replace NAN to 0 and get their values
    w0, w = train_linear_regression(X_train, y_train)
    y_pred = w0 + X_train.dot(w)

    ## Plot predict values and targets
    sns.histplot(y_pred, color='red', alpha=0.5, bins=50)
    sns.histplot(y_train, color='blue', alpha=0.5, bins=50)
    plt.savefig('saved_img4.png')
    plt.close()

    ## RMSE
    print(rmse(y_train, y_pred))

    ## Validating the model
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression(X_train, y_train)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    rmse(y_val, y_pred)

    # Simple feature engineering: Adding the "age"column
    X_train = prepare_X_more(df_train)
    w0, w = train_linear_regression(X_train, y_train)

    X_val = prepare_X_more(df_val)
    y_pred = w0 + X_val.dot(w)
    rmse(y_val, y_pred)

    sns.histplot(y_pred, label='prediction', color='red', alpha=0.5, bins=50)
    sns.histplot(y_val, label='target', color='blue',  alpha=0.5, bins=50)
    plt.legend()
    plt.savefig('saved_img5.png')
    plt.close()

    # Categorical variables
    categorical_columns = [
        'make', 'model', 'engine_fuel_type', 'driven_wheels', 'market_category',
        'vehicle_size', 'vehicle_style']

    categorical = {}

    for c in categorical_columns:
        categorical[c] = list(df_train[c].value_counts().head().index)

    # Dummy variables: Number of doors
    X_train = prepare_X_dummy(df_train)
    w0, w = train_linear_regression(X_train, y_train)

    X_val = prepare_X_dummy(df_val)
    y_pred = w0 + X_val.dot(w)
    rmse(y_val, y_pred)

    # Regularized linear regression with r = 0.01
    X_train = prepare_X_dummy(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r=0.01)

    X_val = prepare_X_dummy(df_val)
    y_pred = w0 + X_val.dot(w)
    rmse(y_val, y_pred)

    # Tuning the model
    for r in [0.0, 0.00001, 0.0001, 0.001, 0.1, 1, 10]:
        X_train = prepare_X_dummy(df_train)
        w0, w = train_linear_regression_reg(X_train, y_train, r=r)

        X_val = prepare_X_dummy(df_val)
        y_pred = w0 + X_val.dot(w)
        score = rmse(y_val, y_pred)

        print(r, w0, score)


    r = 0.0001  # The model achieve the best performance when r = 0.0001
    X_train = prepare_X_dummy(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = prepare_X_dummy(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    print(score)

    # Using the model
    df_full_train = pd.concat([df_train, df_val])
    df_full_train = df_full_train.reset_index(drop=True)
    X_full_train = prepare_X_dummy(df_full_train)
    y_full_train = np.concatenate([y_train, y_val])
    w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.0001)

    X_test = prepare_X_dummy(df_test)
    y_pred = w0 + X_test.dot(w)
    score = rmse(y_test, y_pred)
    print(score)

    # Test with a specific value:
    car = df_test.iloc[10].to_dict()
    df_small = pd.DataFrame([car])
    print(df_small)
    X_small = prepare_X_dummy(df_small)
    y_pred = w0 + X_small.dot(w)
    y_pred = y_pred[0]
    print(y_pred)
    print('Predicted value: ', np.expm1(y_pred))
    print('Target: ', np.expm1(y_test[10]))

