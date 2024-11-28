import pandas as pd
def load_data(file_path = "./data/abalone.data"):
    columns = [
        "Sex", "Length", "Diameter", "Height", 
        "WholeWeight", "ShuckedWeight", 
        "VisceraWeight", "ShellWeight", "Rings"
    ]

    data = pd.read_csv(file_path, header=None, names=columns)

    return data