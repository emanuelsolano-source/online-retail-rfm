"""Limpieza y tratamiento de inconsistencias del dataset Online Retail II."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import PROJECT_ROOT, load_raw_data

PROCESSED_PATH: Path = PROJECT_ROOT / "data" / "processed" / "online_retail_clean.csv"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
IQR_OUTLIER_MULTIPLIER: float = 3.0
DAY_NAMES_ES: dict[int, str] = {
    0: "Lunes",
    1: "Martes",
    2: "Miércoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sábado",
    6: "Domingo",
}


def _report_step(step: str, before: int, after: int) -> None:
    """Imprime el número de filas antes y después de un paso de limpieza.

    Args:
        step: Nombre descriptivo del paso.
        before: Número de filas antes del paso.
        after: Número de filas después del paso.
    """
    removed: int = before - after
    print(f"{step:<45} {before:>10,} -> {after:>10,} (eliminadas: {removed:,})")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas idénticas en todas las columnas, conservando la primera.

    Args:
        df: DataFrame con las columnas originales del dataset.

    Returns:
        DataFrame sin duplicados exactos.
    """
    return df.drop_duplicates(keep="first")


def parse_invoice_date(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte 'InvoiceDate' a datetime y elimina fechas no interpretables.

    Args:
        df: DataFrame con la columna 'InvoiceDate' en texto ISO (AAAA-MM-DD HH:MM:SS).

    Returns:
        Copia del DataFrame con 'InvoiceDate' como datetime y sin valores nulos.
    """
    result: pd.DataFrame = df.copy()
    result["InvoiceDate"] = pd.to_datetime(
        result["InvoiceDate"], format=DATE_FORMAT, errors="coerce"
    )
    return result.dropna(subset=["InvoiceDate"])


def flag_registered_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Marca las transacciones que tienen un Customer ID conocido.

    Las filas sin ID se conservan para el análisis de ventas generales, pero
    se excluyen del análisis RFM, que requiere un cliente identificable.

    Args:
        df: DataFrame con la columna 'Customer ID'.

    Returns:
        Copia del DataFrame con la columna booleana 'is_registered' y
        'Customer ID' convertido a entero nullable.
    """
    result: pd.DataFrame = df.copy()
    result["is_registered"] = result["Customer ID"].notna()
    result["Customer ID"] = result["Customer ID"].astype("Int64")
    return result


def flag_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Marca las facturas de cancelación, cuyo número empieza por 'C'.

    Args:
        df: DataFrame con la columna 'Invoice'.

    Returns:
        Copia del DataFrame con la columna booleana 'is_cancelled'.
    """
    result: pd.DataFrame = df.copy()
    result["is_cancelled"] = result["Invoice"].astype(str).str.startswith("C")
    return result


def count_cancellation_inconsistencies(df: pd.DataFrame) -> tuple[int, int]:
    """Cuenta los casos en los que la factura y la cantidad no coinciden.

    Args:
        df: DataFrame con las columnas 'is_cancelled' y 'Quantity'.

    Returns:
        Tupla con (cancelaciones con cantidad positiva,
        no cancelaciones con cantidad negativa).
    """
    cancelled_positive: int = int((df["is_cancelled"] & (df["Quantity"] > 0)).sum())
    not_cancelled_negative: int = int(
        (~df["is_cancelled"] & (df["Quantity"] < 0)).sum()
    )
    return cancelled_positive, not_cancelled_negative



def remove_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina registros con cantidad o precio no positivos, o sin descripción.

    Args:
        df: DataFrame con las columnas 'Quantity', 'Price' y 'Description'.

    Returns:
        DataFrame solo con transacciones de venta válidas.
    """
    valid_mask: pd.Series = (
        (df["Quantity"] > 0) & (df["Price"] > 0) & df["Description"].notna()
    )
    return df.loc[valid_mask].copy()


def remove_iqr_outliers(
    df: pd.DataFrame,
    column: str,
    multiplier: float = IQR_OUTLIER_MULTIPLIER,
) -> tuple[pd.DataFrame, float]:
    """Elimina valores atípicos con el método IQR aplicado a log(1 + x).

    La transformación logarítmica reduce la asimetría de la distribución
    antes de calcular los cuartiles, evitando eliminar productos premium o
    pedidos al por mayor legítimos.

    Args:
        df: DataFrame con valores positivos en la columna indicada.
        column: Nombre de la columna numérica.
        multiplier: Factor que multiplica el IQR para definir el límite.

    Returns:
        Tupla con el DataFrame filtrado y el límite superior en la escala
        original.
    """
    log_values: pd.Series = np.log1p(df[column])
    q1: float = float(log_values.quantile(0.25))
    q3: float = float(log_values.quantile(0.75))
    iqr: float = q3 - q1
    lower: float = q1 - multiplier * iqr
    upper: float = q3 + multiplier * iqr
    mask: pd.Series = log_values.between(lower, upper)
    upper_original: float = float(np.expm1(upper))
    return df.loc[mask].copy(), upper_original


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae variables temporales y calcula el ingreso por línea.

    Args:
        df: DataFrame con 'InvoiceDate' como datetime, 'Quantity' y 'Price'.

    Returns:
        Copia del DataFrame con Year, Month, DayOfWeek, DayName, Hour y Revenue.
    """
    result: pd.DataFrame = df.copy()
    dates: pd.Series = result["InvoiceDate"]
    result["Year"] = dates.dt.year
    result["Month"] = dates.dt.month
    result["DayOfWeek"] = dates.dt.day_of_week
    result["DayName"] = result["DayOfWeek"].map(DAY_NAMES_ES)
    result["Hour"] = dates.dt.hour
    result["Revenue"] = result["Quantity"] * result["Price"]
    return result



def clean_retail_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta el pipeline completo de limpieza e imprime el registro de pasos.

    Args:
        raw_df: DataFrame crudo tal como se carga desde el CSV.

    Returns:
        DataFrame limpio, listo para EDA y segmentación RFM.
    """
    df: pd.DataFrame = raw_df.copy()
    print("Paso                                          antes ->    después")
    print("-" * 80)

    before: int = len(df)
    df = remove_duplicates(df)
    _report_step("1. Duplicados exactos", before, len(df))

    before = len(df)
    df = parse_invoice_date(df)
    _report_step("2. Fechas no interpretables", before, len(df))

    df = flag_registered_customers(df)
    df = flag_cancellations(df)
    cancelled_pos, not_cancelled_neg = count_cancellation_inconsistencies(df)
    print(
        f"\nInconsistencias factura/cantidad: {cancelled_pos:,} cancelaciones "
        f"con cantidad positiva y {not_cancelled_neg:,} compras con cantidad "
        "negativa.\n"
    )

    before = len(df)
    df = df.loc[~df["is_cancelled"]].copy()
    _report_step("3. Facturas canceladas", before, len(df))

    before = len(df)
    df = remove_invalid_records(df)
    _report_step("4. Cantidad/precio <= 0 o sin descripción", before, len(df))

    for column in ["Quantity", "Price"]:
        before = len(df)
        df, upper_bound = remove_iqr_outliers(df, column)
        _report_step(f"5. Atípicos IQR en {column}", before, len(df))
        print(f"   Límite superior aplicado a {column}: {upper_bound:,.2f}")

    df = add_time_features(df)
    return df.reset_index(drop=True)


def save_clean_data(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    """Guarda el dataset limpio en formato CSV UTF-8.

    Args:
        df: DataFrame limpio.
        path: Ruta de destino del archivo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


if __name__ == "__main__":
    raw_data: pd.DataFrame = load_raw_data()
    clean_df: pd.DataFrame = clean_retail_data(raw_data)
    save_clean_data(clean_df)
    print(f"\nFilas finales: {len(clean_df):,}")
    print(f"Dataset limpio guardado en: {PROCESSED_PATH}")
