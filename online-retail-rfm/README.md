# Análisis de Clientes Online Retail II: Limpieza, EDA y Segmentación RFM

Proyecto de portafolio que transforma un dataset transaccional desordenado de un comercio electrónico británico (2009-2011) en un conjunto de datos limpio, analiza su comportamiento de ventas y segmenta a los clientes con el modelo RFM (Recencia, Frecuencia, Monetario).

## 1. Resumen del negocio

El dataset contiene transacciones de una empresa minorista online que vende principalmente artículos de regalo y decoración para el hogar, con clientes mayoristas en numerosos países. El objetivo es responder tres preguntas:

- ¿Cuándo se vende más (mes, día y hora)?
- ¿Qué países y productos generan el ingreso?
- ¿Qué clientes son valiosos, cuáles están en riesgo y cuáles se han perdido?

## 2. Estructura del proyecto

```
online-retail-rfm/
├── data/
│   ├── raw/                  # CSV original (no se versiona)
│   └── processed/            # Dataset limpio generado (no se versiona)
├── reports/figures/          # Gráficos del EDA (PNG)
├── src/
│   ├── data_loader.py        # Carga del CSV crudo (ISO-8859-1)
│   ├── diagnostics.py        # Nulos, duplicados y resumen estadístico
│   ├── cleaning.py           # Pipeline de limpieza y variables temporales
│   ├── eda.py                # Agregaciones de ventas, países y productos
│   ├── plots.py              # Visualizaciones del EDA
│   └── rfm.py                # Métricas RFM, scores y segmentos
├── requirements.txt
└── README.md
```

## 3. Cómo reproducir el análisis

Desde la raíz del proyecto, con el entorno virtual creado con Python 3.12:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.diagnostics   # diagnóstico inicial
.\.venv\Scripts\python.exe -m src.cleaning      # genera data/processed
.\.venv\Scripts\python.exe -m src.plots         # genera reports/figures
.\.venv\Scripts\python.exe -m src.rfm           # resumen de segmentos
```

## 4. Metodología de limpieza

El diagnóstico inicial detectó 1,067,371 filas y los siguientes problemas. Cada decisión se tomó con criterio de negocio y se documenta en el código.

| Problema | Hallazgo | Decisión |
|---|---|---|
| `Customer ID` nulo | 22.77 % de las filas | Se conservan para ventas generales y se excluyen solo del RFM (flag `is_registered`) |
| Filas duplicadas | 34,335 (3.22 %) | Se eliminan (coinciden en las 8 columnas, incluida la fecha al minuto) |
| Facturas canceladas (`C`) | 19,104 filas | Se excluyen para no mezclar devoluciones con ventas |
| Cantidad o precio ≤ 0 | 6,019 filas | Se eliminan (ajustes contables, muestras o errores) |
| Descripción nula | Incluida en el paso anterior | Se eliminan |
| Atípicos en `Quantity` y `Price` | 63 y 1,799 filas | IQR aplicado a `log(1 + x)` con multiplicador 3 |
| Fechas | Formato ISO en texto | Se convierten a `datetime` y se extraen año, mes, día, hora e ingreso |

**Resultado:** 1,006,051 filas (94.3 % del original) y 16 columnas.

**Decisiones metodológicas que conviene conocer:**

- El IQR se aplica sobre `log(1 + x)` porque la distribución es muy asimétrica. Con el IQR directo, el límite de precio sería de 12.85 £ y eliminaría productos premium legítimos.
- Los 3,393 casos de compras con cantidad negativa sin letra `C` en la factura se eliminan en el paso de cantidades no válidas.
- El ingreso analizado es bruto de devoluciones, porque las cancelaciones se excluyen en lugar de netearse. Es una limitación consciente.

## 5. Hallazgos clave del EDA

**Estacionalidad mensual.** El ingreso crece de septiembre a noviembre en ambos años. Noviembre de 2011 fue el mes con mayor ingreso, con 1,461,518 £, por encima de noviembre de 2010 (1,416,732 £). Enero y febrero son los meses más bajos. Diciembre de 2011 es un dato parcial, porque el dataset termina el 9 de diciembre.

**Día y hora.** El jueves concentra el mayor ingreso (3,996,888 £), seguido del martes (3,798,785 £). El sábado no es un patrón de negocio: solo tiene ventas en una fecha (30 pedidos), por lo que se excluye de la interpretación. Las ventas se concentran entre las 9:00 y las 16:00, con el pico a mediodía (2,787,471 £).

**Concentración geográfica.** Reino Unido genera el 84.86 % del ingreso. Le siguen EIRE (3.27 %) y Países Bajos (2.85 %). Hay 43 países en total, pero la dependencia del mercado británico es muy alta.

**Regla de Pareto en productos.** Tras excluir códigos que no son productos (envíos, comisiones, ajustes, descuentos, muestras y vales), 1,043 de 4,888 productos (21.34 %) generan el 80 % del ingreso. Por unidades, 1,080 productos (22.09 %) alcanzan el 80 % de las unidades vendidas. Los productos más vendidos por ingreso son el soporte de pastel de tres niveles (330,590 £) y el portavelas con forma de corazón (257,725 £).

## 6. Segmentación RFM

**Modelo.** La recencia se mide en días desde la última compra hasta la medianoche posterior a la última transacción del dataset. La frecuencia cuenta facturas distintas y el monetario suma el ingreso neto por cliente. Cada dimensión recibe una puntuación de 1 a 5 por quintiles con `pd.qcut`. La recencia se puntúa de forma inversa.

**Reglas de segmentación** (se evalúan en orden de prioridad):

| Segmento | Regla | Clientes | % clientes | Recencia media (días) | Frecuencia media | Monetario medio (£) | % ingreso |
|---|---|---|---|---|---|---|---|
| VIP | R ≥ 4, F ≥ 4, M ≥ 4 | 1,333 | 22.77 % | 19.2 | 16.6 | 8,815.73 | 70.04 % |
| Leales | R ≥ 3, F ≥ 3 | 1,363 | 23.29 % | 71.5 | 5.5 | 1,893.37 | 15.38 % |
| En riesgo | R ≤ 2, F ≥ 3 | 816 | 13.94 % | 365.9 | 5.1 | 1,935.56 | 9.41 % |
| Potenciales | Resto (p. ej. compras recientes con F ≤ 2) | 816 | 13.94 % | 64.3 | 1.4 | 412.35 | 2.01 % |
| Perdidos | R ≤ 2, F ≤ 2 | 1,525 | 26.06 % | 457.1 | 1.3 | 347.74 | 3.16 % |

**Validación.** Se analizaron 5,853 clientes registrados, un ID por cliente. El monetario total de los clientes (16,778,241.93 £) coincide exactamente con el ingreso de las filas con `Customer ID`.

## 7. Conclusiones de negocio y recomendaciones

1. **El segmento VIP es pequeño y muy valioso.** El 22.77 % de los clientes genera el 70.04 % del ingreso. La prioridad es retenerlos con atención personalizada, acceso anticipado a novedades y programas de fidelización.
2. **Hay un volumen grande de clientes en riesgo.** El segmento "En riesgo" (13.94 % de clientes) tiene un monetario medio casi igual al de "Leales", pero lleva en promedio unos 366 días sin comprar. Una campaña de reactivación con incentivos dirigidos tiene un retorno potencialmente alto.
3. **Los clientes perdidos son la mayoría, pero aportan poco.** El 26.06 % de los clientes tiene un monetario medio de 348 £. Conviene usar campañas de bajo costo y no inversión intensiva.
4. **La dependencia de Reino Unido es un riesgo estructural.** Diversificar hacia los mercados europeos con más potencial sería una línea de crecimiento.
5. **Planificar inventario y personal según la estacionalidad.** Los picos de septiembre a noviembre y las horas de mediodía son los momentos de mayor demanda.
6. **Concentrar el catálogo en pocos productos.** Alrededor del 21 % de los productos genera el 80 % del ingreso. Revisar el stock de estos productos y evitar quiebres es prioritario.

## 8. Limitaciones

- El ingreso es bruto de devoluciones. Las cancelaciones se excluyen en lugar de netearse con las compras originales, lo que sobreestima ligeramente las ventas netas.
- Los datos terminan en diciembre de 2011 y no incluyen un análisis de costos ni de margen, por lo que las recomendaciones se basan en ingreso y no en rentabilidad.
- La segmentación por reglas es explicable pero arbitraria en sus umbrales. Un análisis de sensibilidad o un clustering podrían validar los cortes.
- Alrededor del 23 % de las filas originales no tienen `Customer ID` y no se pueden asignar a un cliente, por lo que quedan fuera del RFM.
- Los datos provienen de una sola empresa en un periodo corto, así que los patrones pueden no generalizarse.
