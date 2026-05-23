# -*- coding: utf-8 -*-
"""
contingency_analysis.py — Tabla de contingencia y ejercicios de probabilidad.

Genera una tabla de contingencia (filas = 15 palabras, columnas = 4 fuentes)
con totales marginales, y resuelve tres ejercicios clásicos de probabilidad:
    1. Probabilidad simple   P(A)
    2. Probabilidad conjunta P(A ∩ B)
    3. Probabilidad condicional P(A | B)
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from text_processor import build_consolidated_df


# ──────────────────────────────────────────────
# 1. Construcción de la Tabla de Contingencia
# ──────────────────────────────────────────────

# Orden fijo de las columnas-fuente para presentación consistente
SOURCE_ORDER: list[str] = ["Humano", "ChatGPT", "Gemini", "Grok"]


def build_contingency_table(df_consolidated: pl.DataFrame | None = None) -> pl.DataFrame:
    """
    Construye una tabla de contingencia a partir del DataFrame consolidado.

    Estructura resultante
    ---------------------
    - Filas: las 15 palabras + fila "Total"
    - Columnas: palabra | Humano | ChatGPT | Gemini | Grok | Total

    Las celdas contienen las **frecuencias absolutas observadas**.
    Los totales se calculan dinámicamente.

    Parameters
    ----------
    df_consolidated : pl.DataFrame | None
        DataFrame con columnas [palabra, conteo, fuente, …].
        Si es None, se genera automáticamente con build_consolidated_df().

    Returns
    -------
    pl.DataFrame
        Tabla de contingencia con totales marginales.
    """
    if df_consolidated is None:
        df_consolidated = build_consolidated_df(top_n=15)

    # Pivot: filas = palabra, columnas = fuente, valores = conteo
    pivot = (
        df_consolidated
        .select(["palabra", "fuente", "conteo"])
        .pivot(on="fuente", index="palabra", values="conteo")
        .fill_null(0)
    )

    # Reordenar columnas según SOURCE_ORDER (solo las que existan)
    present_sources = [s for s in SOURCE_ORDER if s in pivot.columns]
    pivot = pivot.select(["palabra"] + present_sources)

    # Columna Total horizontal (suma por fila)
    pivot = pivot.with_columns(
        pl.sum_horizontal(present_sources).alias("Total")
    )

    # Ordenar filas por Total descendente
    pivot = pivot.sort("Total", descending=True)

    # Fila Total vertical (suma por columna)
    total_row_data: dict[str, object] = {"palabra": "Total"}
    for col in present_sources + ["Total"]:
        total_row_data[col] = pivot[col].sum()

    total_row = pl.DataFrame(
        total_row_data,
        schema={col: pivot.schema[col] for col in pivot.columns},
    )

    # Concatenar la fila de totales al final
    contingency = pl.concat([pivot, total_row])

    return contingency


def get_contingency_without_totals(contingency: pl.DataFrame) -> pl.DataFrame:
    """Retorna la tabla de contingencia sin la fila 'Total'."""
    return contingency.filter(pl.col("palabra") != "Total")


def get_grand_total(contingency: pl.DataFrame) -> int:
    """Retorna el gran total (esquina inferior derecha de la tabla)."""
    return int(
        contingency
        .filter(pl.col("palabra") == "Total")
        .select("Total")
        .item()
    )


# ──────────────────────────────────────────────
# 2. Estructura para resultados de probabilidad
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class ProbabilityResult:
    """
    Resultado de un cálculo de probabilidad.

    Attributes
    ----------
    nombre : str
        Nombre del ejercicio.
    formula : str
        Expresión de la fórmula con valores numéricos sustituidos.
    fraccion : str
        Representación como fracción (numerador / denominador).
    valor : float
        Resultado decimal.
    descripcion : str
        Interpretación verbal del resultado.
    """
    nombre: str
    formula: str
    fraccion: str
    valor: float
    descripcion: str

    def __str__(self) -> str:
        lines = [
            f"╔══ {self.nombre} ══",
            f"║  Fórmula:   {self.formula}",
            f"║  Fracción:  {self.fraccion}",
            f"║  Resultado: {self.valor:.6f}  ({self.valor * 100:.4f}%)",
            f"║  → {self.descripcion}",
            f"╚{'═' * 50}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 3. Ejercicio 1 — Probabilidad Simple P(A)
# ──────────────────────────────────────────────

def prob_simple(
    contingency: pl.DataFrame,
    palabra: str,
) -> ProbabilityResult:
    """
    Calcula P(A): la probabilidad de que una palabra seleccionada al azar
    del corpus completo sea la palabra indicada.

    P(A) = (frecuencia total de la palabra) / (gran total)

    Parameters
    ----------
    contingency : pl.DataFrame
        Tabla de contingencia con fila 'Total'.
    palabra : str
        La palabra objetivo (ej. 'de').

    Returns
    -------
    ProbabilityResult

    Raises
    ------
    ValueError
        Si la palabra no está en la tabla.
    """
    palabra = palabra.lower()
    gran_total = get_grand_total(contingency)

    row = contingency.filter(pl.col("palabra") == palabra)
    if row.is_empty():
        available = (
            contingency
            .filter(pl.col("palabra") != "Total")
            .select("palabra")
            .to_series()
            .to_list()
        )
        raise ValueError(
            f"La palabra '{palabra}' no está en la tabla de contingencia. "
            f"Palabras disponibles: {available}"
        )

    freq_total = int(row.select("Total").item())

    return ProbabilityResult(
        nombre=f"Probabilidad Simple — P('{palabra}')",
        formula=f"P(A) = frecuencia_total('{palabra}') / gran_total",
        fraccion=f"{freq_total:,} / {gran_total:,}",
        valor=freq_total / gran_total,
        descripcion=(
            f"Hay un {freq_total / gran_total * 100:.4f}% de probabilidad de que "
            f"una palabra al azar del corpus sea '{palabra}'."
        ),
    )


# ──────────────────────────────────────────────
# 4. Ejercicio 2 — Probabilidad Conjunta P(A ∩ B)
# ──────────────────────────────────────────────

def prob_conjunta(
    contingency: pl.DataFrame,
    palabra: str,
    fuente: str,
) -> ProbabilityResult:
    """
    Calcula P(A ∩ B): la probabilidad de que una observación sea
    la palabra indicada Y pertenezca a la fuente indicada.

    P(A ∩ B) = frecuencia(palabra, fuente) / gran_total

    Parameters
    ----------
    contingency : pl.DataFrame
        Tabla de contingencia con fila 'Total'.
    palabra : str
        La palabra objetivo (ej. 'que').
    fuente : str
        La fuente objetivo (ej. 'ChatGPT').

    Returns
    -------
    ProbabilityResult

    Raises
    ------
    ValueError
        Si la palabra o la fuente no están en la tabla.
    """
    palabra = palabra.lower()
    gran_total = get_grand_total(contingency)

    # Validar que la fuente existe como columna
    if fuente not in contingency.columns:
        raise ValueError(
            f"La fuente '{fuente}' no es una columna válida. "
            f"Fuentes disponibles: {SOURCE_ORDER}"
        )

    # Validar que la palabra existe
    row = contingency.filter(pl.col("palabra") == palabra)
    if row.is_empty():
        available = (
            contingency
            .filter(pl.col("palabra") != "Total")
            .select("palabra")
            .to_series()
            .to_list()
        )
        raise ValueError(
            f"La palabra '{palabra}' no está en la tabla. "
            f"Palabras disponibles: {available}"
        )

    freq_celda = int(row.select(fuente).item())

    return ProbabilityResult(
        nombre=f"Probabilidad Conjunta — P('{palabra}' ∩ '{fuente}')",
        formula=f"P(A ∩ B) = frecuencia('{palabra}', '{fuente}') / gran_total",
        fraccion=f"{freq_celda:,} / {gran_total:,}",
        valor=freq_celda / gran_total,
        descripcion=(
            f"Hay un {freq_celda / gran_total * 100:.4f}% de probabilidad de que "
            f"una palabra al azar sea '{palabra}' y provenga de '{fuente}'."
        ),
    )


# ──────────────────────────────────────────────
# 5. Ejercicio 3 — Probabilidad Condicional P(A | B)
# ──────────────────────────────────────────────

def prob_condicional(
    contingency: pl.DataFrame,
    palabra: str,
    fuente: str,
) -> ProbabilityResult:
    """
    Calcula P(A | B): la probabilidad de que la palabra sea la indicada,
    dado que se sabe que proviene de la fuente indicada.

    P(A | B) = frecuencia(palabra, fuente) / total_fuente

    Equivale a:  P(A | B) = P(A ∩ B) / P(B)

    Parameters
    ----------
    contingency : pl.DataFrame
        Tabla de contingencia con fila 'Total'.
    palabra : str
        La palabra objetivo (ej. 'el').
    fuente : str
        La fuente condicionante (ej. 'Humano').

    Returns
    -------
    ProbabilityResult

    Raises
    ------
    ValueError
        Si la palabra o la fuente no están en la tabla.
    """
    palabra = palabra.lower()

    # Validar fuente
    if fuente not in contingency.columns:
        raise ValueError(
            f"La fuente '{fuente}' no es una columna válida. "
            f"Fuentes disponibles: {SOURCE_ORDER}"
        )

    # Validar palabra
    row = contingency.filter(pl.col("palabra") == palabra)
    if row.is_empty():
        available = (
            contingency
            .filter(pl.col("palabra") != "Total")
            .select("palabra")
            .to_series()
            .to_list()
        )
        raise ValueError(
            f"La palabra '{palabra}' no está en la tabla. "
            f"Palabras disponibles: {available}"
        )

    freq_celda = int(row.select(fuente).item())

    # Total marginal de la fuente (fila 'Total', columna fuente)
    total_fuente = int(
        contingency
        .filter(pl.col("palabra") == "Total")
        .select(fuente)
        .item()
    )

    if total_fuente == 0:
        raise ZeroDivisionError(
            f"El total de la fuente '{fuente}' es 0, no se puede condicionar."
        )

    return ProbabilityResult(
        nombre=f"Probabilidad Condicional — P('{palabra}' | '{fuente}')",
        formula=f"P(A|B) = frecuencia('{palabra}', '{fuente}') / total('{fuente}')",
        fraccion=f"{freq_celda:,} / {total_fuente:,}",
        valor=freq_celda / total_fuente,
        descripcion=(
            f"Dado que la palabra proviene de '{fuente}', hay un "
            f"{freq_celda / total_fuente * 100:.4f}% de probabilidad de que "
            f"sea '{palabra}'."
        ),
    )


# ──────────────────────────────────────────────
# 6. Ejecución directa → demostración completa
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  contingency_analysis.py — Demostración")
    print("=" * 70)

    # ── Paso 1: Construir tabla de contingencia ──
    print("\n▸ Generando tabla de contingencia …\n")
    ct = build_contingency_table()
    # Configurar Polars para mostrar todas las filas
    with pl.Config(tbl_rows=20):
        print(ct)

    # ── Paso 2: Ejercicio 1 — Probabilidad Simple ──
    print("\n" + "─" * 70)
    print("  EJERCICIO 1 — Probabilidad Simple: P('de')")
    print("─" * 70 + "\n")
    r1 = prob_simple(ct, palabra="de")
    print(r1)

    # ── Paso 3: Ejercicio 2 — Probabilidad Conjunta ──
    print("\n" + "─" * 70)
    print("  EJERCICIO 2 — Probabilidad Conjunta: P('en' ∩ 'ChatGPT')")
    print("─" * 70 + "\n")
    r2 = prob_conjunta(ct, palabra="en", fuente="ChatGPT")
    print(r2)

    # ── Paso 4: Ejercicio 3 — Probabilidad Condicional ──
    print("\n" + "─" * 70)
    print("  EJERCICIO 3 — Probabilidad Condicional: P('la' | 'Humano')")
    print("─" * 70 + "\n")
    r3 = prob_condicional(ct, palabra="la", fuente="Humano")
    print(r3)

    # ── Resumen final ──
    print("\n" + "═" * 70)
    print("  Resumen de resultados")
    print("═" * 70)
    for r in [r1, r2, r3]:
        print(f"  • {r.nombre}")
        print(f"    {r.fraccion}  =  {r.valor:.6f}")
    print("═" * 70)
