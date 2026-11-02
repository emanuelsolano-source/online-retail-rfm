"""Segmentación de clientes con RFM (Recencia, Frecuencia, Monetario)."""

import numpy as np
import pandas as pd

from src.eda import load_clean_data

SCORE_LABELS: list[int] = [1, 2, 3, 4, 5]
SNAPSHOT_OFFSET_DAYS: int = 1


def build_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula Recencia, Frecuencia y Monetario por cliente registrado.

    La fecha de referencia es la medianoche posterior a la última transacción
    del dataset. Una compra hecha ese mismo día tiene recencia 0.

    Args:
        df: DataFrame limpio con 'Customer ID', 'is_registered', 'InvoiceDate',
            'Invoice' y 'Revenue'.

    Returns:
        DataFrame indexado por 'Customer ID' con las columnas recency (días),
        frequency (facturas distintas) y monetary (ingreso total).
    """
    registered: pd.DataFrame = df.loc[df["is_registered"]]
    snapshot: pd.Timestamp = df["InvoiceDate"].max().normalize() + pd.Timedelta(
        days=SNAPSHOT_OFFSET_DAYS
    )
    rfm: pd.DataFrame = registered.groupby("Customer ID").agg(
        last_purchase=("InvoiceDate", "max"),
        frequency=("Invoice", "nunique"),
        monetary=("Revenue", "sum"),
    )
    rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days
    return rfm[["recency", "frequency", "monetary"]]


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Asigna puntuaciones de 1 a 5 a cada dimensión usando quintiles.

    Recencia se puntúa de forma inversa: menos días desde la última compra
    implica una puntuación más alta. Como muchos clientes tienen la misma
    frecuencia, qcut necesita valores únicos. Para desempatar sin depender del
    orden de los IDs, la frecuencia se combina con el rango percentil del
    monetario (siempre menor que 1), de modo que el monetario solo rompe los
    empates dentro de una misma frecuencia.

    Args:
        rfm: DataFrame con las columnas recency, frequency y monetary.

    Returns:
        Copia del DataFrame con las columnas R, F, M (enteros 1-5) y
        RFM_score (suma de las tres puntuaciones).
    """
    result: pd.DataFrame = rfm.copy()
    result["R"] = pd.qcut(
        result["recency"].rank(method="first", ascending=False),
        q=5,
        labels=SCORE_LABELS,
    ).astype(int)
    frequency_key: pd.Series = result["frequency"] + result["monetary"].rank(pct=True)
    result["F"] = pd.qcut(
        frequency_key.rank(method="first"),
        q=5,
        labels=SCORE_LABELS,
    ).astype(int)
    result["M"] = pd.qcut(
        result["monetary"].rank(method="first"),
        q=5,
        labels=SCORE_LABELS,
    ).astype(int)
    result["RFM_score"] = result["R"] + result["F"] + result["M"]
    return result


def assign_segment(scored: pd.DataFrame) -> pd.DataFrame:
    """Asigna un segmento de negocio a cada cliente según reglas explícitas.

    Las reglas se evalúan en orden de prioridad: la primera que se cumple
    determina el segmento. Son transparentes y fáciles de explicar, a
    diferencia de un clustering no supervisado.

    Args:
        scored: DataFrame con las puntuaciones R, F, M.

    Returns:
        Copia del DataFrame con la columna 'segment'.
    """
    conditions: list[pd.Series] = [
        (scored["R"] >= 4) & (scored["F"] >= 4) & (scored["M"] >= 4),
        (scored["R"] >= 3) & (scored["F"] >= 3),
        (scored["R"] <= 2) & (scored["F"] >= 3),
        (scored["R"] <= 2) & (scored["F"] <= 2),
    ]
    choices: list[str] = ["VIP", "Leales", "En riesgo", "Perdidos"]
    result: pd.DataFrame = scored.copy()
    result["segment"] = np.select(conditions, choices, default="Potenciales")
    return result


def summarize_segments(segmented: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas resumen por segmento.

    Args:
        segmented: DataFrame con 'segment', recency, frequency y monetary.

    Returns:
        DataFrame indexado por 'segment' con número de clientes, porcentaje de
        clientes, recencia media, frecuencia media, monetario medio y
        participación en el ingreso total.
    """
    total_revenue: float = float(segmented["monetary"].sum())
    summary: pd.DataFrame = segmented.groupby("segment").agg(
        customers=("monetary", "size"),
        recency_mean=("recency", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean"),
        revenue_share_pct=("monetary", lambda s: s.sum() / total_revenue * 100),
    )
    summary["customers_pct"] = summary["customers"] / len(segmented) * 100
    return summary.sort_values("monetary_mean", ascending=False)


def run_rfm_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta el pipeline RFM completo: métricas, puntuaciones y segmentos.

    Args:
        df: DataFrame limpio del pipeline de limpieza.

    Returns:
        DataFrame de clientes con recency, frequency, monetary, R, F, M,
        RFM_score y segment.
    """
    rfm: pd.DataFrame = build_rfm_table(df)
    scored: pd.DataFrame = score_rfm(rfm)
    return assign_segment(scored)


if __name__ == "__main__":
    clean_df: pd.DataFrame = load_clean_data()
    customers: pd.DataFrame = run_rfm_analysis(clean_df)
    print(f"Clientes registrados analizados: {len(customers):,}")
    print(summarize_segments(customers).round(2).to_string())
