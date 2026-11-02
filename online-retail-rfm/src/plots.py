"""Visualizaciones del EDA de ventas, exportadas a reports/figures."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns

from src.cleaning import PROJECT_ROOT
from src.eda import (
    load_clean_data,
    pareto_analysis,
    product_summary,
    sales_by_country,
    sales_by_day_of_week,
    sales_by_hour,
    sales_by_month,
)

FIGURES_PATH: Path = PROJECT_ROOT / "reports" / "figures"
CURRENCY_LABEL: str = "Ingreso (£)"
PARTIAL_MONTH: str = "2011-12"


def _save_figure(fig: plt.Figure, filename: str) -> None:
    """Guarda la figura en reports/figures con resolución adecuada para GitHub.

    Args:
        fig: Figura de Matplotlib a guardar.
        filename: Nombre del archivo PNG, sin ruta.
    """
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_PATH / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_currency_axis(ax: plt.Axes, axis: str = "y") -> None:
    """Usa separador de miles en el eje en lugar de notación científica.

    Args:
        ax: Eje de Matplotlib a formatear.
        axis: Eje a formatear, 'x' o 'y'.
    """
    formatter: FuncFormatter = FuncFormatter(lambda value, _: f"{value:,.0f}")
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)


def plot_monthly_sales(df: pd.DataFrame) -> None:
    """Grafica el ingreso mensual y marca el mes incompleto de diciembre 2011.

    Args:
        df: DataFrame limpio con 'InvoiceDate', 'Revenue' e 'Invoice'.
    """
    monthly: pd.DataFrame = sales_by_month(df).reset_index()
    monthly["Mes"] = monthly["YearMonth"].dt.to_timestamp()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=monthly, x="Mes", y="revenue", marker="o", ax=ax)
    partial: pd.Series = monthly.loc[monthly["YearMonth"].astype(str) == PARTIAL_MONTH]
    if not partial.empty:
        ax.text(
            0.01,
            0.97,
            "Dic. 2011 es un dato parcial (hasta el 9 de dic.)",
            transform=ax.transAxes,
            va="top",
        )
    ax.set_title("Ingreso mensual (dic. 2009 - dic. 2011)")
    ax.set_xlabel("Mes")
    ax.set_ylabel(CURRENCY_LABEL)
    _format_currency_axis(ax)
    fig.autofmt_xdate()
    _save_figure(fig, "01_ingreso_mensual.png")


def plot_day_and_hour(df: pd.DataFrame) -> None:
    """Grafica el ingreso por día de la semana y por hora del día.

    El sábado se muestra con un aviso porque solo tiene ventas en una fecha.

    Args:
        df: DataFrame limpio con 'DayName', 'Hour', 'Revenue' e 'Invoice'.
    """
    daily: pd.DataFrame = sales_by_day_of_week(df).reset_index()
    hourly: pd.DataFrame = sales_by_hour(df).reset_index()
    sns.set_theme(style="whitegrid")
    fig, (ax_day, ax_hour) = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=daily, x="DayName", y="revenue", color="steelblue", ax=ax_day)
    ax_day.set_title("Ingreso por día de la semana")
    ax_day.set_xlabel("Día")
    ax_day.set_ylabel(CURRENCY_LABEL)
    ax_day.tick_params(axis="x", rotation=30)
    _format_currency_axis(ax_day)
    ax_day.text(
        5,
        float(daily.loc[daily["DayName"] == "Sábado", "revenue"].iloc[0]) + 150000,
        "Sin ventas\n(solo 1 fecha)",
        ha="center",
    )

    sns.lineplot(data=hourly, x="Hour", y="revenue", marker="o", ax=ax_hour)
    ax_hour.set_title("Ingreso por hora del día")
    ax_hour.set_xlabel("Hora")
    ax_hour.set_ylabel(CURRENCY_LABEL)
    ax_hour.set_xticks(hourly["Hour"])
    _format_currency_axis(ax_hour)

    _save_figure(fig, "02_dia_y_hora.png")


def plot_country_sales(df: pd.DataFrame, top_n: int = 10) -> None:
    """Grafica el ingreso de los principales países, con su participación.

    Args:
        df: DataFrame limpio con 'Country', 'Revenue', 'Invoice' y 'Customer ID'.
        top_n: Número de países a mostrar, ordenados por ingreso.
    """
    countries: pd.DataFrame = sales_by_country(df).head(top_n).reset_index()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=countries, y="Country", x="revenue", color="steelblue", ax=ax)
    for index, row in countries.iterrows():
        ax.text(
            row["revenue"],
            index,
            f" {row['share_pct']:.1f}%",
            va="center",
        )
    ax.set_title(f"Top {top_n} países por ingreso (% del ingreso total)")
    ax.set_xlabel(CURRENCY_LABEL)
    ax.set_ylabel("País")
    _format_currency_axis(ax, axis="x")
    _save_figure(fig, "03_ingreso_por_pais.png")


def plot_pareto_products(df: pd.DataFrame) -> None:
    """Grafica la curva de Pareto acumulada del ingreso por producto.

    Args:
        df: DataFrame limpio con 'StockCode', 'Description', 'Quantity',
            'Revenue' e 'Invoice'.
    """
    products: pd.DataFrame = product_summary(df)
    curve, products_needed, products_pct = pareto_analysis(products, "revenue")
    curve = curve.reset_index()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        range(1, len(curve) + 1),
        curve["cumulative_share"] * 100,
        color="steelblue",
    )
    ax.axhline(80, color="red", linestyle="--", label="80 % del ingreso")
    ax.axvline(products_needed, color="gray", linestyle=":")
    ax.set_title(
        f"Pareto de productos: {products_needed:,} productos "
        f"({products_pct}%) generan el 80 % del ingreso"
    )
    ax.set_xlabel("Número de productos (ordenados por ingreso)")
    ax.set_ylabel("Ingreso acumulado (%)")
    ax.legend()
    _save_figure(fig, "04_pareto_productos.png")


def run_eda_figures() -> None:
    """Carga el dataset limpio y genera todas las figuras del EDA."""
    clean_df: pd.DataFrame = load_clean_data()
    plot_monthly_sales(clean_df)
    plot_day_and_hour(clean_df)
    plot_country_sales(clean_df)
    plot_pareto_products(clean_df)
    print(f"Figuras guardadas en: {FIGURES_PATH}")


if __name__ == "__main__":
    run_eda_figures()
