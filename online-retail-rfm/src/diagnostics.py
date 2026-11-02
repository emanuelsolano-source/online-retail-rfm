"""Diagnóstico inicial de calidad de datos del dataset Online Retail II."""

import pandas as pd

from src.data_loader import load_raw_data

NUMERIC_COLUMNS: list[str] = ["Quantity", "Price"]


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el conteo y el porcentaje de valores nulos por columna.

    Args:
        df: DataFrame a diagnosticar.

    Returns:
        DataFrame con las columnas 'nulos' y 'porcentaje', ordenado de mayor
        a menor cantidad de nulos.
    """
    missing_count: pd.Series = df.isna().sum()
    missing_pct: pd.Series = (missing_count / len(df) * 100).round(2)
    summary: pd.DataFrame = pd.DataFrame(
        {"nulos": missing_count, "porcentaje": missing_pct}
    )
    return summary.sort_values("nulos", ascending=False)


def get_duplicate_summary(df: pd.DataFrame) -> tuple[int, float]:
    """Cuenta las filas duplicadas (idénticas en todas las columnas).

    Args:
        df: DataFrame a diagnosticar.

    Returns:
        Tupla con el número de duplicados y su porcentaje sobre el total.
    """
    duplicate_count: int = int(df.duplicated().sum())
    duplicate_pct: float = round(duplicate_count / len(df) * 100, 2)
    return duplicate_count, duplicate_pct


def get_descriptive_summary(
    df: pd.DataFrame, columns: list[str] = NUMERIC_COLUMNS
) -> pd.DataFrame:
    """Genera el resumen estadístico descriptivo de las columnas numéricas.

    Args:
        df: DataFrame a diagnosticar.
        columns: Columnas numéricas a resumir.

    Returns:
        DataFrame traspuesto con count, mean, std, min, cuartiles y max.
    """
    return df[columns].describe().T


def print_diagnostic_report(df: pd.DataFrame) -> None:
    """Imprime el diagnóstico completo: dimensiones, tipos, nulos,
    duplicados y resumen descriptivo.

    Args:
        df: DataFrame recién cargado.
    """
    print("=" * 60)
    print("1. DIMENSIONES Y TIPOS DE DATOS")
    print("=" * 60)
    print(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("2. VALORES NULOS POR COLUMNA")
    print("=" * 60)
    print(get_missing_summary(df))

    print("\n" + "=" * 60)
    print("3. FILAS DUPLICADAS")
    print("=" * 60)
    duplicate_count, duplicate_pct = get_duplicate_summary(df)
    print(f"Duplicados: {duplicate_count:,} ({duplicate_pct}% del total)")

    print("\n" + "=" * 60)
    print("4. RESUMEN ESTADÍSTICO (Quantity y Price)")
    print("=" * 60)
    print(get_descriptive_summary(df).round(2))


if __name__ == "__main__":
    raw_df: pd.DataFrame = load_raw_data()
    print_diagnostic_report(raw_df)
