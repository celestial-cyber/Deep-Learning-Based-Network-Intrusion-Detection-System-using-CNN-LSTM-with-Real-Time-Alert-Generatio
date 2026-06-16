import pandas as pd
import glob

def load_dataset():
    files = glob.glob("Dataset/*.csv")

    df_list = []

    for file in files:
        print("Loading:", file)

        df = pd.read_csv(
            file,
            engine="python",
            encoding="latin1",
            on_bad_lines="skip"
        )

        df = df.sample(10000, random_state=42)
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    df = df.sample(50000, random_state=42)

    return df