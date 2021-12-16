import yfinance as yf
import pickle


def get_data(ticker, period="5y"):
    """
    Get data from Yahoo Finance.
    :param ticker:
    :param start_date:
    :param end_date:
    :return:
    """
    tkr = yf.Ticker(ticker)
    data = tkr.history(period=period)
    return data


def save_data(path="../dataset/", name="data.pkl", data=None):
    """
    Save data to pickle file.
    :param path:
    :param data:
    :return:
    """
    with open(path+name, "wb") as f:
        pickle.dump(data, f)


aapl = get_data("AAPL")
fb = get_data("FB")
msft = get_data("MSFT")

save_data(data=aapl, name="aapl.pkl")
save_data(data=fb, name="fb.pkl")
save_data(data=msft, name="msft.pkl")
