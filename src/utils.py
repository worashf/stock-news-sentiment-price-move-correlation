import pandas as pd
import os

def load_csv_finantial_data(file_path:str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    try:
        df= pd.read_csv(file_path,engine='python')
        print("Data loaded successfully.")
        print(f"DataFrame shape: {df.shape}")
        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} does not exist")
    except Exception as e:
        raise e

