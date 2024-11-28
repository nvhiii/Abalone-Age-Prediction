from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

def load_data(file_path = "./data/abalone.data"):
    columns = [
        "Sex", "Length", "Diameter", "Height", 
        "WholeWeight", "ShuckedWeight", 
        "VisceraWeight", "ShellWeight", "Rings"
    ]

    # preprocessing
    data = pd.read_csv(file_path, header=None, names=columns)
    data = pd.get_dummies(data, columns=["Sex"], drop_first=True)

    # features v target
    X = data.drop("Rings", axis=1)
    y = data["Rings"]

    # normalisation
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # data splits
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # pt tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype = torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))

    # dl objs
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

# more preprocessing tasks (encoding / normalizing)
# data related functionality only

