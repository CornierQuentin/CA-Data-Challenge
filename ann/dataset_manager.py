import pandas as pd
import json

def display_csv_info(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Display the first row of data
    print("\nFirst Row of Data:")
    print(df.iloc[0])

def read_from_json(file_path):
    # Open the file in read mode ('r' mode)
    with open(file_path, 'r') as file:
        # Load the data from the JSON file
        data = json.load(file)
        return data

def get_data_info(file_path, print_info=False):
    # Read the CSV file
    df = pd.read_csv(file_path)

    if print_info:
    
        # Display the number of rows and columns
        print("\nNumber of Rows and Columns:")
        print(df.shape)
        
        # Display the column names
        print("\nColumn Names:")
        print(df.columns)
        
        # Display the data types
        print("\nData Types:")
        print(df.dtypes)
        
        # Display the first few rows of data
        print("\nFirst Few Rows of Data:")
        print(df.head())

    return df

def colunm_information(df, column_number):
    column_name = df.columns[column_number]
    print("Desciption de la colonne: ", column_name)
    print(df[column_name].describe())
    print("Valeurs rencontrées: ")
    print(df[column_name].unique())

def data_transformation(df, columns_info):

    number_of_columns = columns_info['number_of_columns']
    number_of_rows = columns_info['number_of_rows']
    columns_info = columns_info['columns_info']

    df_normalized = df.iloc[:, 1:].copy()

    for column in range(number_of_columns):
        for row in range(number_of_rows):
            value = df_normalized.iloc[row, column]
            column_info = columns_info[column]
            index = column_info['unique_values'].index(value)
            normalized_value = column_info['unique_values_normalized'][index]
            df_normalized.iloc[row, column] = normalized_value
    
    return df_normalized

if __name__ == "__main__":
    csv_file_path = 'dataset/train_input_Z61KlZo.csv'
    df  = get_data_info(csv_file_path)
    columns_info = read_from_json('ann/columns_info/X_columns_info.json')
    df_normalized = data_transformation(df, columns_info)

    size_of_trainning = round(df_normalized.shape[0] * 0.8)
    df_train_input, df_test_input = df_normalized.iloc[:size_of_trainning], df_normalized.iloc[size_of_trainning:]
    df_train_input.to_csv('dataset/dataset_normalized/train_input_normalized.csv', index=False)
    df_test_input.to_csv('dataset/dataset_normalized/test_input_normalized.csv', index=False)

    csv_file_path = 'dataset/train_output_DzPxaPY.csv'
    df  = get_data_info(csv_file_path)
    columns_info = read_from_json('ann/columns_info/Y_columns_info.json')
    df_normalized = data_transformation(df, columns_info)

    size_of_trainning = round(df_normalized.shape[0] * 0.8)
    df_train_output, df_test_output = df_normalized.iloc[:size_of_trainning], df_normalized.iloc[size_of_trainning:]
    df_train_output.to_csv('dataset/dataset_normalized/train_output_normalized.csv', index=False)
    df_test_output.to_csv('dataset/dataset_normalized/test_output_normalized.csv', index=False)