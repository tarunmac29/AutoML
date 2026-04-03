"""
data_loader.py
Handles loading of CSV, Excel, and JSON files into pandas DataFrames.
"""

import pandas as pd
import io


def load_data(uploaded_file) -> pd.DataFrame:
    """
    Load uploaded file (CSV, Excel, JSON) into a pandas DataFrame.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        pd.DataFrame

    Raises:
        ValueError: If file format is unsupported or file is empty
    """
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)

        elif filename.endswith(".json"):
            content = uploaded_file.read()
            df = pd.read_json(io.BytesIO(content))

        else:
            raise ValueError(
                f"Unsupported file format: '{uploaded_file.name}'. "
                "Please upload a CSV, Excel (.xlsx/.xls), or JSON file."
            )

    except Exception as e:
        raise ValueError(f"Failed to read file: {str(e)}")

    if df.empty:
        raise ValueError("The uploaded file is empty. Please upload a valid dataset.")

    return df


def get_file_info(df: pd.DataFrame) -> dict:
    """
    Return basic info about the loaded DataFrame.

    Args:
        df: pandas DataFrame

    Returns:
        dict with rows, columns, dtypes summary
    """
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }
