from cryptography.hazmat.primitives import hashes, hmac
from cryptography.fernet import Fernet
import pandas as pd
import numpy as np

def anonimyze_with_hmac(df: pd.DataFrame, columns: list, key: str) -> tuple:
    table = []
    for column in columns:
        datos_anonimizados = []
        for dato in df[column]:
            hash_algorithm = hashes.SHA256()
            h = hmac.HMAC(key, hash_algorithm)
            h.update(str(dato).encode())
            hash_dato = h.finalize()
            datos_anonimizados.append(hash_dato.hex())
            table.append({"dato": dato, "hash": hash_dato.hex()})
        df[column] = datos_anonimizados
    return df, table


def anonimyze_with_encryption(
    df: pd.DataFrame, columns: list, key: str
) -> pd.DataFrame:
    fernet = Fernet(key)
    for column in columns:
        df[column] = df[column].apply(
            lambda x: fernet.encrypt(str(x).encode()).decode()
        )
    return df


def perturbe_numeric_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    aux_df = df.copy()
    for column in columns:
        std_dev = df[column].std()
        noise = np.random.normal(0, std_dev / 10, df[column].size)
        aux_df[column] = aux_df[column] + noise
        aux_df[column] = aux_df[column].round(0)
        aux_df[column] = aux_df[column].apply(lambda x: 0 if x < 0 else x)

    return aux_df


def generalize_numeric_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    aux_df = df.copy()
    for column in columns:
        aux_df[column["name"]] = pd.cut(df[column["name"]], column["size"])
    return aux_df


def compute_k_anonymity(df: pd.DataFrame, quasi: list) -> int:
    groups = df.groupby(quasi).size().reset_index(name="count")
    groups = groups[groups["count"] > 0]
    return groups["count"].min()


def reach_k_anonymity(
    df: pd.DataFrame, k: int, quasi: list, full_quasi: list
) -> pd.DataFrame:
    max_steps = 30
    steps = 0
    aux_df = df.copy()
    while compute_k_anonymity(aux_df, full_quasi) < k and steps < max_steps:
        aux_df = df.copy()
        aux_df = perturbe_numeric_columns(aux_df, [column for column in quasi])
        aux_df = generalize_numeric_columns(
            aux_df,
            [
                {"name": column, "size": max_steps - steps % max_steps + 1}
                for column in quasi
            ],
        )
        steps += 1
    return aux_df


if __name__ == "__main__":
    key = None
    with open("filekey.key", "rb") as f:
        key = f.read()

    identifiers = ["CustomerID"]

    quasi = [
        "Age",
        "Annual Income ($)",
        "Work Experience",
        "Family Size",
    ]

    full_quasi = [
        "Gender",
        "Age",
        "Annual Income ($)",
        "Work Experience",
        "Family Size",
    ]

    sensitive = ["Spending Score (1-100)"]

    df = pd.read_csv("Datasets/Shop Customer Data/Customers.csv")

    print("Type desired pseudonymization method:")
    print("1. HMAC")
    print("2. Encryption")

    method = int(input())

    if method == 1:
        df, table = anonimyze_with_hmac(df, identifiers, key)

        with open("table.csv", "w") as f:
            f.write("Dato,Hash\n")
            for row in table:
                f.write(f"{row['dato']},{row['hash']}\n")

    elif method == 2:
        df = anonimyze_with_encryption(df, identifiers, key)

    print("Type desired k-anonymity:")
    k = int(input())

    tries = 0
    found = False
    max_df = pd.DataFrame()
    max_k = 0
    while not found and tries < 10:
        aux_df = reach_k_anonymity(df, k, quasi, full_quasi)

        actual_k = compute_k_anonymity(aux_df, full_quasi)

        if actual_k >= k:
            found = True
            max_df = aux_df
            max_k = actual_k
        elif actual_k > max_k:
            max_k = actual_k
            max_df = aux_df

        tries += 1

    print(f"Max k: {max_k}")

    if not found:
        print("Could not reach desired k-anonymity")

    max_df.to_csv("anonimyzed.csv", index=False)