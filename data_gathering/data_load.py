import pickle


def load_data(path):
    """
    Loads the pickle data from dataset folder
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data
