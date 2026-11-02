"""Agregaciones para el análisis exploratorio de datos (EDA) de ventas."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.cleaning import PROCESSED_PATH

PARETO_THRESHOLD: float = 0.80
NON_PRODUCT_CODES: frozenset[str] = frozenset(
    {
        "POST",
        "DOT",
        "C2",
        "M",
        "m",
        "D",
        "S",
        "BANK CHARGES",
        "ADJUST",
        "TEST001",
        "TEST002",
    }
)


def is_product_code(stock_code: str) -> bool:
    """Indica si un StockCode corresponde a un producto físico o digital.

    Excluye gastos de envío, comisiones, ajustes, descuentos, muestras,
    entradas manuales, registros de prueba y vales de regalo.

    Args:
        stock_code: Código de inventario tal como aparece en el dataset.

    Returns:
        True si el código representa un producto vendible.
    """
    return stock_code not in NON_PRODUCT_CODES and not stock_code.startswith("gift_")


def load_clean_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Carga el dataset limpio con 'InvoiceDate' como datetime.

    Args:
        path: Ruta del CSV limpio generado por el módulo de limpieza.

    Returns:
        DataFrame limpio listo para agregaciones.

    Raises:
        FileNotFoundError: Si el archivo limpio no existe.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset limpio: {path}. "
            "Ejecuta primero 'python -m src.cleaning'."
        )
    return pd.read_csv(
        path, encoding="utf-8", low_memory=False, parse_dates=["InvoiceDate"]
    )


def sales_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula ingreso, pedidos y unidades vendidas por mes.

    Args:
        df: DataFrame limpio con 'InvoiceDate', 'Revenue', 'Invoice' y 'Quantity'.

    Returns:
        DataFrame indexado por 'YearMonth' (periodo mensual) con las columnas
        revenue, orders y units, ordenado cronológicamente.
    """
    monthly: pd.DataFrame = (
        df.assign(YearMonth=df["InvoiceDate"].dt.to_period("M"))
        .groupby("YearMonth")
        .agg(
            revenue=("Revenue", "sum"),
            orders=("Invoice", "nunique"),
            units=("Quantity", "sum"),
        )
    )
    return monthly.sort_index()


def sales_by_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula pedidos e ingreso por día de la semana, en orden lunes a domingo.

    Args:
        df: DataFrame limpio con 'DayOfWeek', 'DayName', 'Revenue' e 'Invoice'.

    Returns:
        DataFrame indexado por 'DayName' con revenue y orders, en orden lunes
        a domingo.
    """
    daily: pd.DataFrame = (
        df.groupby(["DayOfWeek", "DayName"])
        .agg(revenue=("Revenue", "sum"), orders=("Invoice", "nunique"))
        .reset_index()
        .sort_values("DayOfWeek")
        .set_index("DayName")
    )
    return daily[["revenue", "orders"]]


def sales_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula pedidos e ingreso por hora del día.

    Args:
        df: DataFrame limpio con 'Hour', 'Revenue' e 'Invoice'.

    Returns:
        DataFrame indexado por 'Hour' con revenue y orders.
    """
    return df.groupby("Hour").agg(
        revenue=("Revenue", "sum"), orders=("Invoice", "nunique")
    )


def sales_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Resume ventas por país, con su participación en el ingreso total.

    Args:
        df: DataFrame limpio con 'Country', 'Revenue', 'Invoice' y 'Customer ID'.

    Returns:
        DataFrame indexado por 'Country', ordenado por ingreso descendente,
        con revenue, share_pct (porcentaje del ingreso total), orders y
        customers (clientes registrados distintos).
    """
    by_country: pd.DataFrame = df.groupby("Country").agg(
        revenue=("Revenue", "sum"),
        orders=("Invoice", "nunique"),
        customers=("Customer ID", "nunique"),
    )
    total_revenue: float = float(by_country["revenue"].sum())
    by_country["share_pct"] = by_country["revenue"] / total_revenue * 100
    return by_country.sort_values("revenue", ascending=False)


def product_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resume ventas por producto (StockCode), con unidades e ingreso.

    Args:
        df: DataFrame limpio con 'StockCode', 'Description', 'Quantity',
            'Revenue' e 'Invoice'.

    Returns:
        DataFrame indexado por 'StockCode' con description (la primera
        aparición), units, revenue y orders, ordenado por ingreso descendente.
    """
    is_product: pd.Series = df["StockCode"].astype(str).map(is_product_code)
    product_rows: pd.DataFrame = df.loc[is_product]
    products: pd.DataFrame = product_rows.groupby("StockCode").agg(
        description=("Description", "first"),
        units=("Quantity", "sum"),
        revenue=("Revenue", "sum"),
        orders=("Invoice", "nunique"),
    )
    return products.sort_values("revenue", ascending=False)


def pareto_analysis(
    products: pd.DataFrame, value_column: str = "revenue"
) -> tuple[pd.DataFrame, int, float]:
    """Calcula la curva de Pareto acumulada y cuántos productos generan el 80 %.

    Args:
        products: DataFrame de productos con la columna indicada.
        value_column: Métrica sobre la que se calcula la concentración.

    Returns:
        Tupla con (DataFrame con cumulative_share, número de productos que
        alcanzan el 80 %, porcentaje de productos que representan ese número).
    """
    ordered: pd.DataFrame = products.sort_values(value_column, ascending=False)
    total: float = float(ordered[value_column].sum())
    result: pd.DataFrame = ordered.assign(
        cumulative_share=ordered[value_column].cumsum() / total
    )
    reached: np.ndarray = (result["cumulative_share"] >= PARETO_THRESHOLD).to_numpy()
    products_needed: int = int(np.argmax(reached)) + 1
    products_pct: float = products_needed / len(result) * 100
    return result, products_needed, round(products_pct, 2)
