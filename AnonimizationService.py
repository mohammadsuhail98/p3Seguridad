from faker import Faker
from flask import Flask, request, jsonify
import pandas as pd
import hashlib
import numpy as np

fake = Faker()

# Generate dummy data
data = {
    'Patient Name': [fake.name() for _ in range(100)],
    'Patient ID': [fake.uuid4() for _ in range(100)],
    'Medical Condition': [fake.word() for _ in range(100)],
    'Medications': [fake.word() for _ in range(100)],
    'Date of Birth': [fake.date_of_birth().strftime('%Y-%m-%d') for _ in range(100)],
    'Social Security Number': [fake.ssn() for _ in range(100)]
}

# Create dataframe from data
df = pd.DataFrame(data)

# Save dataframe to CSV file
df.to_csv('dummy_dataset.csv', index=False)

app = Flask(__name__)
datasets = {}

# Import dataset
@app.route('/import', methods=['POST'])
def import_data():
    file_path = request.form['file_path']
    data = pd.read_csv(file_path)
    datasets[file_path] = data
    return jsonify(data.to_dict()), 200

# Anonymize dataset
@app.route('/anonymize', methods=['POST'])
def anonymize_data():
    file_path = request.form['file_path']
    data = datasets[file_path]
    # Add your anonymization code here
    anonymized_data = data.copy()
    mapping_data = pd.DataFrame(columns=['column_name', 'pseudonym', 'original_identifier'])
    for col in anonymized_data.columns:
        # Generate a random pseudonym for each column
        pseudonym = hashlib.md5(fake.word().encode('utf-8')).hexdigest()[:8]
        # Replace the column values with pseudonyms
        mapping_data = mapping_data.append({'column_name': col, 'pseudonym': pseudonym}, ignore_index=True)
        anonymized_data[col] = pseudonym
    # Save the mapping between the original identifiers and the pseudonyms
    mapping_data.to_csv(f'{file_path}_mapping.csv', index=False)
    datasets[file_path] = anonymized_data
    return jsonify(anonymized_data.to_dict()), 200

# Substitute pseudonyms with original identifiers
@app.route('/substitute', methods=['POST'])
def deidentify_data():
    file_path = request.form['file_path']
    data = datasets[file_path]
    # Add your deidentification code here
    deidentified_data = data.copy()
    for col in deidentified_data.columns:
        # Retrieve the mapping between the pseudonym and the original identifier
        if 'ID' in col or 'Social Security Number' in col:
            mapping = datasets[f'{file_path}_{col}'].rename(columns={col: 'pseudonym'})
            # Replace the pseudonyms with the original identifier values
            deidentified_data[col] = pd.merge(deidentified_data[[col]], mapping, on='pseudonym', how='left')[col].fillna('Unknown')
    datasets[file_path] = deidentified_data
    return jsonify(deidentified_data.to_dict()), 200

# Suppress dataset
@app.route('/suppress', methods=['POST'])
def suppress_data():
    file_path = request.form['file_path']
    data = datasets[file_path]
    # Add your suppression code here
    suppressed_data = data.copy()
    # Example suppression: remove the 'Social Security Number' and 'Patient ID' columns
    suppressed_data = suppressed_data.drop(['Social Security Number', 'Patient ID'], axis=1)
    datasets[file_path] = suppressed_data
    return jsonify(suppressed_data.to_dict()), 200

# Generalize numeric data
@app.route('/generalize', methods=['POST'])
def generalize_data():
    file_path = request.form['file_path']
    data = datasets[file_path]
    # Add your generalization code here
    generalized_data = data.copy()
    for col in generalized_data.columns:
        # Check if the column contains numeric data
        if generalized_data[col].dtype in ['int64', 'float64']:
            # Find the minimum and maximum values in the column
            min_val = generalized_data[col].min()
            max_val = generalized_data[col].max()
            # Generalize the values in the column by mapping them to ranges
            range_size = (max_val - min_val) / 5
            generalized_data[col] = pd.cut(generalized_data[col], bins=5, labels=[f'{min_val:.2f}-{min_val+range_size:.2f}', f'{min_val+range_size:.2f}-{min_val+(range_size*2):.2f}', f'{min_val+(range_size*2):.2f}-{min_val+(range_size*3):.2f}', f'{min_val+(range_size*3):.2f}-{min_val+(range_size*4):.2f}', f'{min_val+(range_size*4):.2f}-{max_val:.2f}'])
    datasets[file_path] = generalized_data
    return jsonify(generalized_data.to_dict()), 200


if __name__ == '__main__':
    app.run()


# Basic CLI interface
def main():
    print("1 : Import dataset")
    print("2 : Anonymize dataset")
    print("3 : Suppress dataset")
    print("4 : Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        data = import_data()
        print(data.head())

    elif choice == "2":
        anonymized_data = anonymize_data(data)
        print(anonymized_data.head())

    elif choice == "3":
        suppressed_data = suppress_data(data)
        print(suppressed_data.head())

    elif choice == "4":
        exit()

    else:
        print("Invalid comand. Please try again.")
        main()