import pandas as pd
from sklearn.preprocessing import StandardScaler

def parse_csv(file_path):
    """
    Parses the CSV file, normalizes numerical data, and one-hot encodes categorical variables.

    Parameters:
    - file_path: str, path to the CSV file.

    Returns:
    - X_processed: pandas DataFrame, processed feature data.
    - y_encoded: pandas Series, encoded target variable.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    columns_to_drop = ['JobLevel', 'Education', 'EducationField', 'Gender']
    df = df.drop(columns=columns_to_drop)

    # Separate features and target variable
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Normalize numerical variables
    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

    # Encode the target variable
    y_encoded = y.map({'Yes': 1, 'No': 0})

    return X_encoded, y_encoded