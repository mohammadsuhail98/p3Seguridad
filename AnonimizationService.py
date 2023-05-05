from cryptography.hazmat.primitives import hashes, hmac
from cryptography.fernet import Fernet
from faker import Faker
from flask import Flask, request, jsonify
import pandas as pd
import hashlib
import numpy as np

fake = Faker()

# Generate dummy data
data = {
    "Patient Name": [fake.name() for _ in range(100)],
    "Patient ID": [fake.uuid4() for _ in range(100)],
    "Medical Condition": [fake.word() for _ in range(100)],
    "Medications": [fake.word() for _ in range(100)],
    "Date of Birth": [fake.date_of_birth().strftime("%Y-%m-%d") for _ in range(100)],
    "Social Security Number": [fake.ssn() for _ in range(100)],
}


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
# Create dataframe from data
df = pd.DataFrame(data)

# Save dataframe to CSV file
df.to_csv("dummy_dataset.csv", index=False)

app = Flask(__name__)
datasets = {}


# Import dataset
@app.route("/import", methods=["POST"])
def import_data():
    file_path = request.form["file_path"]
    data = pd.read_csv(file_path)
    datasets[file_path] = data
    return jsonify(data.to_dict()), 200


# Anonymize dataset
@app.route("/anonymize", methods=["POST"])
def anonymize_data():
    file_path = request.form["file_path"]
    data = datasets[file_path]
    # Add your anonymization code here
    anonymized_data = data.copy()
    mapping_data = pd.DataFrame(
        columns=["column_name", "pseudonym", "original_identifier"]
    )
    for col in anonymized_data.columns:
        # Generate a random pseudonym for each column
        pseudonym = hashlib.md5(fake.word().encode("utf-8")).hexdigest()[:8]
        # Replace the column values with pseudonyms
        mapping_data = mapping_data.append(
            {"column_name": col, "pseudonym": pseudonym}, ignore_index=True
        )
        anonymized_data[col] = pseudonym
    # Save the mapping between the original identifiers and the pseudonyms
    mapping_data.to_csv(f"{file_path}_mapping.csv", index=False)
    datasets[file_path] = anonymized_data
    return jsonify(anonymized_data.to_dict()), 200


# Substitute pseudonyms with original identifiers
@app.route("/substitute", methods=["POST"])
def deidentify_data():
    file_path = request.form["file_path"]
    data = datasets[file_path]
    # Add your deidentification code here
    deidentified_data = data.copy()
    for col in deidentified_data.columns:
        # Retrieve the mapping between the pseudonym and the original identifier
        if "ID" in col or "Social Security Number" in col:
            mapping = datasets[f"{file_path}_{col}"].rename(columns={col: "pseudonym"})
            # Replace the pseudonyms with the original identifier values
            deidentified_data[col] = pd.merge(
                deidentified_data[[col]], mapping, on="pseudonym", how="left"
            )[col].fillna("Unknown")
    datasets[file_path] = deidentified_data
    return jsonify(deidentified_data.to_dict()), 200


# Suppress dataset
@app.route("/suppress", methods=["POST"])
def suppress_data():
    file_path = request.form["file_path"]
    data = datasets[file_path]
    # Add your suppression code here
    suppressed_data = data.copy()
    # Example suppression: remove the 'Social Security Number' and 'Patient ID' columns
    suppressed_data = suppressed_data.drop(
        ["Social Security Number", "Patient ID"], axis=1
    )
    datasets[file_path] = suppressed_data
    return jsonify(suppressed_data.to_dict()), 200


# Generalize numeric data
@app.route("/generalize", methods=["POST"])
def generalize_data():
    file_path = request.form["file_path"]
    data = datasets[file_path]
    # Add your generalization code here
    generalized_data = data.copy()
    for col in generalized_data.columns:
        # Check if the column contains numeric data
        if generalized_data[col].dtype in ["int64", "float64"]:
            # Find the minimum and maximum values in the column
            min_val = generalized_data[col].min()
            max_val = generalized_data[col].max()
            # Generalize the values in the column by mapping them to ranges
            range_size = (max_val - min_val) / 5
            generalized_data[col] = pd.cut(
                generalized_data[col],
                bins=5,
                labels=[
                    f"{min_val:.2f}-{min_val+range_size:.2f}",
                    f"{min_val+range_size:.2f}-{min_val+(range_size*2):.2f}",
                    f"{min_val+(range_size*2):.2f}-{min_val+(range_size*3):.2f}",
                    f"{min_val+(range_size*3):.2f}-{min_val+(range_size*4):.2f}",
                    f"{min_val+(range_size*4):.2f}-{max_val:.2f}",
                ],
            )
    datasets[file_path] = generalized_data
    return jsonify(generalized_data.to_dict()), 200

"""
if __name__ == "__main__":
    app.run()
"""

if __name__ == "__main__":
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