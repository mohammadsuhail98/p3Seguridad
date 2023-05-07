from cryptography.hazmat.primitives import hashes, hmac
from cryptography.fernet import Fernet
import pandas as pd
import numpy as np
from flask import Flask, make_response, render_template, request, redirect, url_for


def anonimyze_with_hmac(df: pd.DataFrame, columns: list, key: str) -> tuple:
    table = []
    for column in columns:
        datos_anonimizados = []
        for dato in df[column]:
            hash_algorithm = hashes.SHA256()
            h = hmac.HMAC(key, hash_algorithm)
            h.update(str(dato).encode("utf-8"))
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
            lambda x: fernet.encrypt(str(x).encode("utf-8")).decode()
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


def micro_aggregation(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    aux_df = df.copy()
    group_size = 10
    for column in columns:
        sorted_column = aux_df[column].sort_values()
        grouped_column = sorted_column.groupby(pd.qcut(sorted_column.index, q=round(len(df)/group_size)))
        means = grouped_column.mean()
        means_dict = means.to_dict()
        aux_df[column] = aux_df[column].apply(lambda x: means_dict[x] if x in means_dict else x)
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
    df: pd.DataFrame, k: int, quasi: list, full_quasi: list, dataset: int, noise: int
) -> pd.DataFrame:
    max_steps = 30
    steps = 0
    aux_df = df.copy()
    hierarchies = {"state": {
        "norte": ["ME", "NH", "VT", "NY", "PA", "MI", "WI", "MN"],
        "sur": [ "TX", "AR", "MS", "AL", "TN", "KY", "WV", "VA", "NC", "SC", "GA", "FL", "LA", ],
        "este": ["MA", "RI", "CT", "NJ", "DE", "MD", "DC"],
        "oeste": [ "WA", "OR", "CA", "NV", "ID", "MT", "WY", "UT", "CO", "AZ", "NM", "AK", "HI", ],
        "centro": ["ND", "SD", "NE", "KS", "OK", "IA", "MO", "IL", "IN", "OH"],
    }
}
    while compute_k_anonymity(aux_df, full_quasi) < k and steps < max_steps:
        aux_df = df.copy()
        if noise == 1:
            aux_df = perturbe_numeric_columns(aux_df, [column for column in quasi])
        elif noise == 2:
            aux_df = micro_aggregation(aux_df, quasi)
        if dataset == 2:
            aux_df = generalize_categorical_columns(aux_df,hierarchies)
        
        aux_df = generalize_numeric_columns(
            aux_df,
            [
                {"name": column, "size": max_steps - steps % max_steps + 1}
                for column in quasi
            ],
        )
        steps += 1
    return aux_df


def drop_columns_and_save(input_file, output_file, keep_columns):
    df = pd.read_csv(input_file, encoding='latin-1')
    drop_columns = [col for col in df.columns if col not in keep_columns]
    df.drop(drop_columns, axis=1, inplace=True)
    df = df[df['age'] != 'Unknown']
    df = df.loc[~df.applymap(lambda x: '-' in str(x)).any(axis=1)]
    df = df.dropna()
    df.to_csv(output_file, index=False)


def generalize_categorical_columns(df: pd.DataFrame, columns_hierarchies: dict):
    aux_df = df.copy()
    for column in columns_hierarchies:
        for hierarchy in columns_hierarchies[column]:
            aux_df.loc[
                aux_df[column].isin(columns_hierarchies[column][hierarchy]),
                column,
            ] = hierarchy

    return aux_df

# Web server
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/anonymize", methods=["POST"])
def anonymize():
    data = request.json
    dataset = data["dataset"]
    pseudonymization_method = data["pseudonymizationMethod"]
    noise_method = data["noiseMethod"]
    k_anonymity = data["kAnonymity"]
    
    anonymize_data_web(dataset, pseudonymization_method, noise_method, k_anonymity)

    response = make_response('Success!')
    response.status = 200
    return response

def anonymize_data_web(dataset, method, noise_method, k):
    key = None
    if k != '':
        k = int(k)
    else:
        k = 0
        
    with open("filekey.key", "rb") as f:
        key = f.read()
            
    if dataset == "Customers.csv":
        dataChosen = 1
        df = pd.read_csv("Datasets/Shop Customer Data/Customers.csv")
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
    elif dataset == "police_killings.csv":
        dataChosen = 2
        drop_columns_and_save("Datasets/Police Killings Data/police_killings.csv","Datasets/Police Killings Data/police_killings_light.csv",["name","age","gender","state","year","pop","city","p_income","cause"])
        df = pd.read_csv("Datasets/Police Killings Data/police_killings_light.csv", encoding='latin-1')
        identifiers = ["name"]

        quasi = [
            "pop",
            "age"
        ]

        full_quasi = [
            "age",
            "pop",
            "state"
        ]

        sensitive = ["cause"]
        
    if method == "Hmac":
        df, table = anonimyze_with_hmac(df, identifiers, key)

        with open("table.csv", "w", encoding="utf-8") as f:
            f.write("Dato,Hash\n")
            for row in table:
                f.write(f"{row['dato']},{row['hash']}\n")

    elif method == "Encryption":
        df = anonimyze_with_encryption(df, identifiers, key)

    if noise_method == "Perturbation":
        noise = 1
    elif noise_method == "Micro-aggreation":
        noise = 2

    tries = 0
    found = False
    max_df = pd.DataFrame()
    max_k = 0
    while not found and tries < 10:
        aux_df = reach_k_anonymity(df, k, quasi, full_quasi,dataChosen,noise)
        actual_k = compute_k_anonymity(aux_df, full_quasi)

        if actual_k >= k:
            found = True
            max_df = aux_df
            max_k = actual_k
        elif actual_k > max_k:
            max_k = actual_k
            max_df = aux_df

        tries += 1

    print(f"Max k obtained: {max_k}")

    if not found:
        print("Could not reach desired k-anonymity")

    removed_rows = max_df.groupby(full_quasi).filter(lambda x: len(x) < k and len(x) > 0).size

    print(f"Removed {removed_rows} rows to reach k-anonymity")

    max_df = max_df.groupby(full_quasi).filter(lambda x: len(x) >= k)

    print(max_df.groupby(full_quasi).size().reset_index(name="count"))

    max_df.to_csv("anonimyzed.csv", index=False)

        

if __name__ == "__main__":
    
    print("Choose a launch option:")
    print("1. Launch Website")
    print("2. Launch Console")
    
    launch_option = int(input())
    
    if launch_option == 1:
            app.run()
    elif launch_option == 2:
        key = None
        with open("filekey.key", "rb") as f:
            key = f.read()

        print("Dataset chosen:")
        print("1. Shop Customer")
        print("2. Police Killings")

        dataChosen = int(input())
        while dataChosen != 1 and dataChosen!= 2:
            print("Chose a right Dataset:")
            print("1. Shop Customer")
            print("2. Police Killings")
            dataChosen = int(input())

        if dataChosen == 1:
            df = pd.read_csv("Datasets/Shop Customer Data/Customers.csv")
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
        elif dataChosen == 2:
            drop_columns_and_save("Datasets/Police Killings Data/police_killings.csv","Datasets/Police Killings Data/police_killings_light.csv",["name","age","gender","state","year","pop","city","p_income","cause"])
            df = pd.read_csv("Datasets/Police Killings Data/police_killings_light.csv", encoding='latin-1')
            identifiers = ["name"]

            quasi = [
                "pop",
                "age"
            ]

            full_quasi = [
                "age",
                "pop",
                "state"
            ]

            sensitive = ["cause"]


        print("Type desired pseudonymization method:")
        print("1. HMAC")
        print("2. Encryption")

        method = int(input())
        while method != 1 and method!= 2:
            print("Type desired pseudonymization method:")
            print("1. HMAC")
            print("2. Encryption")
            method = int(input())

        if method == 1:
            df, table = anonimyze_with_hmac(df, identifiers, key)

            with open("table.csv", "w", encoding="utf-8") as f:
                f.write("Dato,Hash\n")
                for row in table:
                    f.write(f"{row['dato']},{row['hash']}\n")

        elif method == 2:
            df = anonimyze_with_encryption(df, identifiers, key)

        print("Type desired noise method:")
        print("1. Perturbation")
        print("2. Micro-aggreation")

        noise = int(input())
        while noise != 1 and noise!= 2:
            print("Type desired noise method:")
            print("1. Perturbation")
            print("2. Micro-aggreation")
            noise = int(input())
        print("Type desired k-anonymity:")
        k = int(input())

        tries = 0
        found = False
        max_df = pd.DataFrame()
        max_k = 0
        while not found and tries < 10:
            aux_df = reach_k_anonymity(df, k, quasi, full_quasi,dataChosen,noise)
            actual_k = compute_k_anonymity(aux_df, full_quasi)

            if actual_k >= k:
                found = True
                max_df = aux_df
                max_k = actual_k
            elif actual_k > max_k:
                max_k = actual_k
                max_df = aux_df

            tries += 1

        print(f"Max k obtained: {max_k}")

        if not found:
            print("Could not reach desired k-anonymity")

        removed_rows = max_df.groupby(full_quasi).filter(lambda x: len(x) < k and len(x) > 0).size

        print(f"Removed {removed_rows} rows to reach k-anonymity")

        max_df = max_df.groupby(full_quasi).filter(lambda x: len(x) >= k)

        print(max_df.groupby(full_quasi).size().reset_index(name="count"))

        max_df.to_csv("anonimyzed.csv", index=False)
    
    