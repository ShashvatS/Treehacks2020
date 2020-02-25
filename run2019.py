from iteratexgboost import *
import datetime 
import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_the_next_week_day(day_now):    
    if day_now.isoweekday()== 5:
        day_now += datetime.timedelta(days=3)
    elif day_now.isoweekday()== 6:
        day_now += datetime.timedelta(days=2)
    else:
        day_now += datetime.timedelta(days=1)
    return day_now

def last_day():
    return datetime.datetime(2018, 12, 31)


def calculate_final_prediction(output_file, forward, models):
    if len(models) > 1:
        raise NotImplementedError()
    
    else:
        df = get_data()
        X = get_final_X(df, forward, None)

        model = models[0]
        pred = model.predict(X)[0] + X[:, -2 if len(models) != 1 else -1]
        pred = np.exp(pred) - np.exp(1)

        ticker = None
        date = None

        with open(output_file, "w") as f:
            f.write("TICKER, date, revtq\n")

            counter = 0
            for row in tqdm(df.iterrows()):
                idx = row[0]

                nxtidx = (idx[0], idx[1] + forward)
                if nxtidx not in df.index:

                    if ticker == None or nxtidx[0] != ticker:
                        ticker = nxtidx[0]
                        date = last_day()

                    date = calculate_the_next_week_day(date)

                    f.write(idx[0] + ", ")
                    f.write(date.strftime("%Y%m%d") + ", ")
                    f.write(str(pred[counter]) + "\n")

                    counter += 1



if __name__ == "__main__":
    set_final_model(True)

    # train_model(ONE_YEAR, "final/final_{}.model".format(ONE_YEAR))
    models = load_models([ONE_YEAR], "final/final".format(ONE_YEAR))

    calculate_final_prediction("final/prediction2019.csv", ONE_YEAR, models)
     