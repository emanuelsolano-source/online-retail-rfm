"""Carga y diagnóstico inicial del dataset Online Retail II."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "online_retail_II.csv"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Carga el CSV crudo respetando su codificación original.

    Args:
        path: Ruta absoluta o relativa al archivo CSV.

    Returns:
        DataFrame con los registros sin modificar.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    return pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)


def print_load_summary(df: pd.DataFrame) -> None:
    """Imprime un resumen rápido para validar la carga.

    Args:
        df: DataFrame recién cargado.
    """
    print(f"Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nPrimeras 5 filas:")
    print(df.head())


if __name__ == "__main__":
    raw_df: pd.DataFrame = load_raw_data()
    print_load_summary(raw_df)
