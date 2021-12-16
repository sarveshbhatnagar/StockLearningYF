
import pickle
import random

from sklearn.linear_model import LinearRegression

import numpy as np


def load_data(path):
    """
    Loads the pickle data from dataset folder
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_data(path, data):
    """
    Save data to pickle file.
    :param path:
    :param data:
    :return:
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)


class SimpleFeatures:
    def __init__(self, hist):
        self.hist = hist
        self.close = hist["Close"]
        self.open = hist["Open"]
        self.features = []

    def get_linear_regression(self, start_index=0, size=7):
        """
        Returns the linear regression of the data.
        """
        data = self.close[start_index:start_index+size]
        reg = LinearRegression()
        data = np.array(data)
        dl = len(data)
        nd = np.array(range(dl))
        reg.fit(nd.reshape(-1, 1), data.reshape(-1, 1))
        return reg.coef_[0]

    def get_normalized(self, start_index=0, size=7):
        """
        Returns a normalized data from start index.
        """
        data = self.close[start_index:start_index+size]
        return (data-data.min())/(data.max()-data.min())

    def avg_tick_size(self, start_index=0, size=30):
        """
        Returns median of tick size, average because its better fucker!
        Its a good feature for understanding how much the stock might increase
        or decrease.
        """
        assert start_index-size >= 0, "Start index must be greater than or equal to size"
        data = self.close[start_index-size:start_index] - \
            self.open[start_index-size:start_index]
        return data.median()

    def isVolatile(self, start_index, longt=30, shortt=7):
        """
        Performs Difference of Avg_tick_size in long term and short term.
        If shortterm is more, then its volatile. otherwise its not volatile.

        Good Feature for Deep Learning
        """
        if(self.avg_tick_size(start_index=start_index, size=longt)-self.avg_tick_size(start_index=start_index, size=shortt) < 0):
            return True
        else:
            return False

    def isGood(self, start_index=0, size=7):
        """
        Returns if the current pattern is good or bad
        it peeks into the future for that.
        """
        data = self.get_normalized(start_index=start_index, size=size*2)
        if(data[size:].mean() > 0.7):
            if(data[-1] - data[size] > 0.6):
                return True
            else:
                return False
        else:
            return False


class LearnFeatures:
    def __init__(self, hist):
        self.sf = SimpleFeatures(hist)
        self.hist = hist
        self.close = hist["Close"]
        self.open = hist["Open"]

    def get_random_train(self, label=True, size=9):
        """
        Returns a random train data
        """
        dic = {
            True: [1, 0],
            False: [0, 1]
        }
        while(True):
            try:
                start_index = random.randint(30, len(self.close)-30)

                if(self.sf.isGood(start_index=start_index, size=size) == label):

                    norm = list(self.sf.get_normalized(
                        start_index=start_index, size=size-2))
                    norm.append(int(self.sf.isVolatile(
                        start_index=start_index, longt=20, shortt=7)))
                    norm.append(
                        int(self.sf.isVolatile(start_index=start_index)))

                    norm = norm + (list(self.sf.get_linear_regression(
                        start_index=start_index, size=size)))

                    return norm, int(label)
                else:
                    continue
            except:
                pass


fb = load_data("../dataset/fb.pkl")

sf = SimpleFeatures(fb)
startat = 80
# print(sf.avg_tick_size(start_index=startat, size=30))
# print(sf.get_normalized(start_index=startat, size=7))
# print(sf.isVolatile(start_index=startat))
# print(sf.isGood(start_index=startat, size=7))
# print(list(sf.get_linear_regression(start_index=20, size=7)))

lf = LearnFeatures(fb)
# tr = lf.get_random_train(label=True, size=9)
# print(tr)

train_data = []
train_label = []

for i in range(1500):
    if(i % 2 == 0):

        tr = lf.get_random_train(label=True, size=20)
        train_data.append(tr[0])
        train_label.append(tr[1])
    else:
        tr = lf.get_random_train(label=False, size=20)
        train_data.append(tr[0])
        train_label.append(tr[1])

print(train_data)
print(train_label)

save_data("../dataset/train_data.pkl", train_data)
save_data("../dataset/train_label.pkl", train_label)
