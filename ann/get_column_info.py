import pandas as pd
import json

def get_data_info(file_path):
    # Read the CSV file
    return pd.read_csv(file_path)

def get_columns_info(df):
    n = df.shape[1]

    data = {
        'number_of_rows': df.shape[0],
        'number_of_columns': n,
        'columns_info': []
    }

    for i in range(1, n):

        dtype = str(df[df.columns[i]].dtype)

        unique = df[df.columns[i]].unique().tolist()
        unique_normalized = []
        if dtype == 'int64' or dtype == 'float64':
            unique_normalized = [x / max(unique) for x in unique]
        else:
            for j in range(len(unique)):
                unique_normalized.append(j/(len(unique) - 1))

        column_data = {
            'column_number': i,
            'column_name': df.columns[i],
            'data_type': dtype,
            'unique_values': unique,
            'unique_values_normalized': unique_normalized
        }
        data['columns_info'].append(column_data)

    with open('ann/columns_info/Y_columns_info.json', 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    csv_file_path = 'dataset/train_output_DzPxaPY.csv'
    df  = get_data_info(csv_file_path)
    get_columns_info(df)
