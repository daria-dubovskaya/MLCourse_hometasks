from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def split_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Splits the dataset into training and validation sets.
    
    Args:
        raw_df (pd.DataFrame): The original dataset.

    Returns:
        tuple: (train_df, val_df, input_cols),
               where input_cols is the list of feature columns.
    """
    train_df, val_df = train_test_split(raw_df, test_size=0.25, random_state=42, stratify=raw_df['Exited'])
    drop_cols = ['CustomerId', 'Surname', 'Exited']
    input_cols = train_df.columns.drop(drop_cols).tolist()
    
    return train_df, val_df, input_cols


def separate_features_targets(df: pd.DataFrame, input_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separates the dataset into input features and the target variable.
    
    Args:
        df (pd.DataFrame): The dataset.
        input_cols (list[str]): List of feature columns.

    Returns:
        tuple: (inputs, target), where target is the 'Exited' column.
    """
    inputs = df[input_cols].copy()
    target = df['Exited'].copy()
    return inputs, target


def get_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Identifies numerical and categorical features.
    
    Args:
        df (pd.DataFrame): The dataset with input features.

    Returns:
        tuple: (numeric_cols, categorical_cols) - lists of numerical and categorical feature columns.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.difference(['id']).tolist()
    categorical_cols = df.select_dtypes(include='object').columns.difference(['id']).tolist()
    return numeric_cols, categorical_cols


def train_scaler(train_df: pd.DataFrame, numeric_cols: list[str]) -> MinMaxScaler:
    """
    Trains a MinMaxScaler on the training dataset.
    
    Args:
        train_df (pd.DataFrame): Training dataset.
        numeric_cols (list[str]): List of numerical feature columns.

    Returns:
        MinMaxScaler: The trained scaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, numeric_cols: list[str], scaler: MinMaxScaler) -> pd.DataFrame:
    """
    Applies a trained MinMaxScaler to a dataset.
    
    Args:
        df (pd.DataFrame): Dataset to transform.
        numeric_cols (list[str]): List of numerical feature columns.
        scaler (MinMaxScaler): The trained scaler.

    Returns:
        pd.DataFrame: The transformed dataset.
    """
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


def encode_categorical_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, categorical_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, list[str]]:
    """
    Encodes categorical features using OneHotEncoder.
    
    Args:
        train_df (pd.DataFrame): Training dataset.
        val_df (pd.DataFrame): Validation dataset.
        categorical_cols (list[str]): List of categorical feature columns.

    Returns:
        tuple: (train_encoded, val_encoded, encoder, encoded_cols) - transformed data and the encoder object.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[categorical_cols])

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    train_df[encoded_cols] = encoder.transform(train_df[categorical_cols])
    val_df[encoded_cols] = encoder.transform(val_df[categorical_cols])
    
    return train_df, val_df, encoder, encoded_cols


def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool = False) -> Dict[str, Any]:
    """
    Performs full preprocessing: data splitting, scaling, and encoding.
    
    Args:
        raw_df (pd.DataFrame): The original dataset.
        scaler_numeric (bool): Whether to scale numerical features.

    Returns:
        dict: A dictionary containing preprocessed data and transformation objects.
    """
    train_df, val_df, input_cols = split_data(raw_df)
    train_inputs, train_targets = separate_features_targets(train_df, input_cols)
    val_inputs, val_targets = separate_features_targets(val_df, input_cols)

    numeric_cols, categorical_cols = get_feature_types(train_inputs)

    scaler = None
    if scaler_numeric:
        scaler = train_scaler(train_inputs, numeric_cols)
        train_inputs = apply_scaler(train_inputs, numeric_cols, scaler)
        val_inputs = apply_scaler(val_inputs, numeric_cols, scaler)

    train_inputs, val_inputs, encoder, encoded_cols = encode_categorical_features(train_inputs, val_inputs, categorical_cols)

    X_train = train_inputs[numeric_cols + encoded_cols]
    X_val = val_inputs[numeric_cols + encoded_cols]

    return {
        'X_train': X_train,
        'train_targets': train_targets,
        'X_val': X_val,
        'val_targets': val_targets,
        'input_cols': numeric_cols + encoded_cols,
        'scaler': scaler,
        'encoder': encoder
    }


def preprocess_new_data(new_df: pd.DataFrame, scaler: Optional[MinMaxScaler], encoder: OneHotEncoder) -> pd.DataFrame:
    """
    Transforms new data using a pre-trained scaler and encoder.
    
    Args:
        new_df (pd.DataFrame): The new dataset to preprocess.
        input_cols (list[str]): List of input columns after preprocessing.
        scaler (Optional[MinMaxScaler]): The trained scaler (if used).
        encoder (OneHotEncoder): The trained encoder.

    Returns:
        pd.DataFrame: The transformed new dataset.
    """
    drop_cols = ['CustomerId', 'Surname']
    input_cols = new_df.columns.drop(drop_cols).tolist()
    
    new_inputs = new_df[input_cols].copy()

    numeric_cols, categorical_cols = get_feature_types(new_inputs)

    if scaler:
        new_inputs = apply_scaler(new_inputs, numeric_cols, scaler)

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    new_inputs[encoded_cols] = encoder.transform(new_inputs[categorical_cols])

    return new_inputs[numeric_cols + encoded_cols]
