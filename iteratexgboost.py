import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import pickle
import matplotlib.pyplot as plt

ONE_YEAR = 261 
MAX_PERIOD = ONE_YEAR

def get_data():
    df = pd.read_csv("daily_returns4.csv")
    df = df.drop(columns = ["PERMNO", "BIDLO", "ASKHI", "BID", "ASK", 'gvkey', 'fyearq'])
    df = df.set_index(['TICKER', "date"])
    df.sort_index(inplace = True)

    revtq = df["revtq"]
    df = df.drop(columns = ["revtq"])
    revtq = revtq.apply(lambda x: np.log(x + np.exp(1)))
    df["modrevtq"] = revtq

    return df

def get_train_test(df, forward):
    forward = MAX_PERIOD

    s = set([])
    for row in df.iterrows():
        idx = row[0]

        if (idx[0], idx[1] + forward) in df.index:
            s.add(idx[0])
    s = list(s)

    np.random.seed(0)
    np.random.shuffle(s)

    train = s[:4 * len(s) // 5]
    test = s[4 * len(s) // 5:]
    s = set(s)

    return train, test

def get_XY(df, forward, train, test, prev_models = None):
    if prev_models == None:
        X = [[], []]
        Y = [[], []]

        for row in tqdm(df.iterrows()):
            idx = row[0]

            train_test = -1
            if idx[0] in train:
                train_test = 0
            elif idx[0] in test:
                train_test = 1
            else:
                continue

            nxtidx = (idx[0], idx[1] + forward)
            if nxtidx in df.index:
                Y[train_test].append(df.at[nxtidx, "modrevtq"] - row[1]["modrevtq"])
                x = np.asarray(row[1], dtype='float64')
                x = np.nan_to_num(x)
                X[train_test].append(x)

        for i in range(2):
            X[i] = np.asarray(X[i])
            Y[i] = np.asarray(Y[i])
            Y[i] = Y[i].reshape((Y[i].shape[0], 1))

            # if prev_model:
            #     augment = prev_model.predict(X[i]).reshape((X[i].shape[0], 1))
            #     X[i] = np.hstack((X[i], augment))

            Z = np.hstack((X[i], Y[i]))
            np.random.shuffle(Z)

            X[i] = Z[:, :Z.shape[1] - 1]
            Y[i] = Z[:, Z.shape[1] - 1]

        return X, Y

    else:
        X, Y = get_XY(df, forward, train, test)
        for i, model in enumerate(prev_models):
            res = model.predict(X[1])

            if i != 0:
                for i in range(2):
                    X[i][:, -1] =  model.predict(X[i])
            else:
                for i in range(2):
                    res = model.predict(X[i])
                    res = res.reshape((res.shape[0], 1))
                    X[i] = np.hstack((X[i], res))

        return X, Y
   
def train_model(forward, filename, prev_models = None):
    df = get_data()
    train, test = get_train_test(df, forward)
    X, Y = get_XY(df, forward, train, test, prev_models)

    xgbmodel = XGBRegressor(objective='reg:squarederror', n_estimators = 100)
    xgbmodel.fit(X[0], Y[0], verbose = True)

    print(mean_squared_error(xgbmodel.predict(X[1]), Y[1]))
    print(np.exp(np.sqrt(mean_squared_error(xgbmodel.predict(X[1]), Y[1]))))

    pickle.dump(xgbmodel, open(filename, "wb"))
    
    return xgbmodel

def iterated_model(forward_values, file_prefix):
    cur_models = None
    file_names = ["{}_{}.model".format(file_prefix, i) for i in forward_values]

    for i in range(len(forward_values)):
        forward = forward_values[i]
        file_name = file_names[i]
        next_model = train_model(forward, file_name, cur_models)

        if cur_models == None:
            cur_models = []
        cur_models.append(next_model)

def load_models(forward_values, file_prefix):
    file_names = ["{}_{}.model".format(file_prefix, i) for i in forward_values]
    return [pickle.load(open(file_name, "rb")) for file_name in file_names]

def print_test_model(forward, models, n_sample):
    df = get_data()
    train, test = get_train_test(df, forward)
    X, Y = get_XY(df, forward, train, test, models[:-1])
    model = models[-1]

    print("revenue", "predicted", "diff-factor")
    for _ in range(n_sample):
        r = random.randrange(0, X[1].shape[1])
        x, y = X[1][[r]], Y[1][[r]]
        pred = model.predict(x)[0] + x[:, -2 if len(models) != 1 else -1]
        y = y + x[:, -2 if len(models) != 1 else -1]
        pred = np.exp(pred) - np.exp(1)
        y = np.exp(y) - np.exp(1)
        print(y, pred, y / pred)

cacheXY = None
def graph_model(forward, models, n_sample):
    df = get_data()
    train, test = get_train_test(df, forward)
    global cacheXY
    if not cacheXY:
        X, Y = get_XY(df, forward, train, test, models[:-1])
        cacheXY = X, Y
    else:
        X, Y = cacheXY
    model = models[-1]

    print("revenue", "predicted", "diff-factor")
    predicted = []
    actual = []
    for _ in range(n_sample):
        r = random.randrange(0, X[1].shape[1])
        x, y = X[1][[r]], Y[1][[r]]
        pred = model.predict(x)[0] + x[:, -2 if len(models) != 1 else -1]
        y = y + x[:, -2 if len(models) != 1 else -1]
        pred = np.exp(pred) - np.exp(1)
        y = np.exp(y) - np.exp(1)
        # print(y, pred, y / pred)
        predicted.append(pred[0])
        actual.append(y[0])

    predicted = np.array(predicted)
    actual = np.array(actual)

    plt.scatter(actual, predicted, color = 'red', label = "Predicted returns")
    plt.plot(actual, actual, label = "Actual Returns")
    plt.xlabel("Actual Returns")
    plt.title("Actual Returns Versus Predicted Returns")
    plt.legend()
    plt.show()

def evaluate_model(forward, models):
    df = get_data()
    train, test = get_train_test(df, forward)
    X, Y = get_XY(df, forward, train, test, models[:-1])
    model = models[-1]

    x, y = X[1], Y[1]
    pred = model.predict(x)[0] + x[:, -2 if len(models) != 1 else -1]
    pred = np.exp(pred) - np.exp(1)
    y = y + x[:, -2 if len(models) != 1 else -1]
    y = np.exp(y) - np.exp(1)

    diff = y / pred
    for i in range(len(diff)):
        if y[i] != 0:
            diff[i] = max(diff[i], pred[i] / y[i])

    ans = np.median(diff)
    return ans

if __name__ == "__main__":
    # train_model(ONE_YEAR, "m1.model")
    # iterated_model([ONE_YEAR // 2, ONE_YEAR], "iterate")
    # models = load_models([ONE_YEAR // 2, ONE_YEAR], "iterate")
    # print_test_model(ONE_YEAR, models, 100)

    # iterated_model([ONE_YEAR // 2, ONE_YEAR], "iterate")
    # print_test_model(ONE_YEAR, models, 100)


    # iterated_model([ONE_YEAR], "notiterate")
    # models = load_models([ONE_YEAR], "notiterate")
    # print_test_model(ONE_YEAR, models, 100)

        # models = load_models([ONE_YEAR // 2, ONE_YEAR], "iterate")

    # best skip value so far: 160
    # gives predictions within factor of 1.44

    skip = 160
    forward_values = list(range(skip, ONE_YEAR, skip))
    if ONE_YEAR not in forward_values:
        forward_values.append(ONE_YEAR)

    file_prefix = "models2/iterate2_{}".format(skip)
    print(file_prefix)

    # iterated_model(forward_values, file_prefix)

    models = load_models(forward_values, file_prefix)
    # print_test_model(ONE_YEAR, models, 100)
    median = evaluate_model(ONE_YEAR, models)
    print(median)

    # graph_model(ONE_YEAR, models, 1000)



