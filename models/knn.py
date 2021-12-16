from collections import Counter
from scipy.spatial.distance import cityblock


def KNN(X, y, test, k=3, distance=cityblock, verbose=False, rmax=True):
    """
    X is list of comparable values
    y is list of labels
    Default distance cityblock (Manhattan)
    Default k = 3
    """

    X = [np.array(i) for i in X]
    test = np.array(test)
    res = []
    # print("Calculating Distance")
    for i in range(len(X)):
        res.append((i, distance(test, X[i])))
        if(verbose):
            print(X[i], distance(test, X[i]))
    res = sorted(res, key=lambda x: x[1])
    labels = []
    print("Calculating Minimum")
    for i in range(k):
        if(not verbose):
            print(X[res[i][0]], res[i][1], "-> ", y[res[i][0]])
        labels.append(y[res[i][0]])

    if(rmax):

        mc = Counter(labels).most_common(1)
        print(mc)
        return mc[0][1]/len(test), mc[0][0]
    return Counter(labels)
