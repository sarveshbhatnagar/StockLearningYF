
import data_gathering.data_load as dl
import models.knn as knn
import models.dlmodel as dlm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    fb = dl.load_data("dataset/fb.pkl")
    nnm = dlm.NNModels(input_shape=(21,), output_shape=2)
    model = nnm.dl_0()
    model = nnm.dl_0_compile(model)
    data = dl.load_data("dataset/train_data.pkl")
    label = dl.load_data("dataset/train_label.pkl")

    data = np.array(data)
    label = np.array(label)
    # data = data.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=0.2, random_state=1)

    history = model.fit(X_train, y_train, batch_size=64, epochs=100)

    # test_data = dl.load_data("dataset/test_data.pkl")
    # test_label = dl.load_data("dataset/test_label.pkl")
    # test_data = np.array(test_data)
    # test_label = np.array(test_label)
    predictions = model.predict(X_test)
    predictions = [item.argmax() for item in predictions]

    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(precision_score(y_test, predictions, average=None))
