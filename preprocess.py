import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from load_dataset import load_dataset

DROP_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]

def preprocess():

    df = load_dataset()

    # clean columns
    df.columns = df.columns.str.strip()

    # drop useless columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # clean invalid values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # BINARY LABEL
    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    X = df.drop("Label", axis=1)
    y = df["Label"]

    # FAST numeric conversion (NO apply)
    X = X.astype("float32")

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # reshape for CNN-LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return train_test_split(X, y, test_size=0.2, random_state=42)