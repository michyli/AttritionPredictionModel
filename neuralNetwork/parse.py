import pandas as pd

def data2df(file_name='Employee_Attrition.csv'):
    """
    Reads the Employee_Attrition.csv file and returns it as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_name)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' does not exist in the current directory.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def transform_columns(df):
    """
    Transforms specific columns in the DataFrame:
    - Moves 'Attrition' to the rightmost column and encodes it into binary (No -> 0, Yes -> 1).
    - Applies one-hot encoding to categorical variables.
    """
    if df is None:
        print("The DataFrame is None. Transformation skipped.")
        return None

    try:
        # Encode 'Attrition' into binary (No -> 0, Yes -> 1)
        df['Attrition'] = df['Attrition'].map({'No': 0, 'Yes': 1})

        # Identify categorical variables to be one-hot encoded
        categorical_columns = ['BusinessTravel', 'EducationField', 'Gender', 'MaritalStatus']

        # Apply one-hot encoding using pandas get_dummies
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)

        # Move 'Attrition' to the rightmost column
        columns = [col for col in df.columns if col != 'Attrition'] + ['Attrition']
        df = df[columns]

        return df
    except Exception as e:
        print(f"An error occurred during transformation: {e}")
        return None