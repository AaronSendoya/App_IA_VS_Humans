# -*- coding: utf-8 -*-
"""
text_processor.py — Procesamiento y análisis de frecuencia del corpus.

Funciones principales:
    - clean_text:            Limpia un texto crudo (minúsculas, sin acentos/puntuación).
    - process_humano:        Lee y procesa el archivo Humano moby dick.txt.
    - process_ia_sources:    Recorre las carpetas IA y agrega frecuencias por fuente.
    - build_consolidated_df: Intersección de las 15 palabras más comunes en español
                             compartidas por las 4 fuentes, con conteo y frecuencia relativa.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import polars as pl

from config import (
    CHATGPT_DIR,
    GEMINI_DIR,
    GROK_DIR,
    HUMANO_FILE,
)

# ──────────────────────────────────────────────
# 1. Limpieza de texto
# ──────────────────────────────────────────────

# Patrón pre-compilado: conserva solo letras (sin dígitos) y espacios
_ONLY_ALPHA_SPACE = re.compile(r"[^a-z\s]")


def _remove_accents(text: str) -> str:
    """
    Elimina marcas diacríticas (tildes, diéresis, etc.) de un texto
    descomponiéndolo en forma NFD y filtrando la categoría 'Mn'.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


def clean_text(raw: str) -> str:
    """
    Pipeline de limpieza para un texto crudo.

    Pasos
    -----
    1. Convertir a minúsculas.
    2. Eliminar acentos / diacríticos.
    3. Reemplazar caracteres no alfabéticos por espacio.
    4. Colapsar espacios múltiples.

    Parameters
    ----------
    raw : str
        Texto sin procesar.

    Returns
    -------
    str
        Texto limpio listo para tokenizar.

    Notes
    -----
    NO se eliminan stop-words (de, la, que, el, en, …) porque son
    precisamente las palabras funcionales que el análisis estadístico
    busca comparar entre fuentes.
    """
    text = raw.lower()
    text = _remove_accents(text)
    text = _ONLY_ALPHA_SPACE.sub(" ", text)
    text = " ".join(text.split())  # colapsa espacios
    return text


# ──────────────────────────────────────────────
# 2. Utilidades internas
# ──────────────────────────────────────────────

def _read_file_safe(filepath: Path, encoding: str = "utf-8") -> str | None:
    """
    Lee un archivo de texto de forma segura.

    Retorna None si el archivo no existe, está vacío o no se puede
    decodificar, y registra una advertencia por stderr.
    """
    import sys

    if not filepath.is_file():
        print(f"  ⚠ Archivo no encontrado: {filepath}", file=sys.stderr)
        return None
    try:
        content = filepath.read_text(encoding=encoding, errors="replace")
    except OSError as exc:
        print(f"  ⚠ Error al leer {filepath}: {exc}", file=sys.stderr)
        return None

    if not content.strip():
        print(f"  ⚠ Archivo vacío: {filepath}", file=sys.stderr)
        return None

    return content


def _words_to_freq_df(words: list[str], source: str) -> pl.DataFrame:
    """
    Dado una lista de palabras y el nombre de la fuente,
    retorna un DataFrame Polars con columnas [palabra, conteo, fuente].
    """
    if not words:
        return pl.DataFrame(
            {"palabra": [], "conteo": [], "fuente": []},
            schema={"palabra": pl.Utf8, "conteo": pl.UInt32, "fuente": pl.Utf8},
        )

    df = (
        pl.DataFrame({"palabra": words})
        .filter(pl.col("palabra").str.len_chars() >= 5)
        .group_by("palabra")
        .agg(pl.len().alias("conteo"))
        .with_columns(pl.lit(source).alias("fuente"))
        .sort("conteo", descending=True)
    )
    # Asegurar tipo UInt32 para conteo
    df = df.cast({"conteo": pl.UInt32})
    return df


# ──────────────────────────────────────────────
# 3. Procesamiento del texto humano
# ──────────────────────────────────────────────

def process_humano(filepath: Path = HUMANO_FILE) -> pl.DataFrame:
    """
    Lee *Humano moby dick.txt*, lo limpia y devuelve un DataFrame
    con las frecuencias absolutas de cada palabra.

    Columns
    -------
    palabra : Utf8
    conteo  : UInt32
    fuente  : Utf8  (siempre 'Humano')

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe o está vacío.
    """
    content = _read_file_safe(filepath)
    if content is None:
        raise FileNotFoundError(
            f"No se pudo leer el archivo humano: {filepath}"
        )

    cleaned = clean_text(content)
    words = cleaned.split()
    return _words_to_freq_df(words, source="Humano")


# ──────────────────────────────────────────────
# 4. Procesamiento de las fuentes IA
# ──────────────────────────────────────────────

# Mapeo nombre → directorio
_IA_SOURCES: dict[str, Path] = {
    "ChatGPT": CHATGPT_DIR,
    "Gemini": GEMINI_DIR,
    "Grok": GROK_DIR,
}


def _process_single_ia(name: str, folder: Path) -> pl.DataFrame:
    """
    Recorre todos los .txt de *folder*, concatena su contenido limpio
    y devuelve el DataFrame de frecuencias para la fuente *name*.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Carpeta IA no encontrada: {folder}")

    txt_files = sorted(folder.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No se encontraron .txt en: {folder}")

    all_words: list[str] = []
    for txt_path in txt_files:
        content = _read_file_safe(txt_path)
        if content is None:
            continue  # saltar archivos problemáticos, ya se advirtió
        cleaned = clean_text(content)
        all_words.extend(cleaned.split())

    return _words_to_freq_df(all_words, source=name)


def process_ia_sources() -> pl.DataFrame:
    """
    Recorre las carpetas ChatGPT, Gemini y Grok, procesa los 12 ensayos
    de cada una y retorna un DataFrame unificado con las frecuencias
    agregadas por fuente.

    Columns
    -------
    palabra : Utf8
    conteo  : UInt32
    fuente  : Utf8  ('ChatGPT' | 'Gemini' | 'Grok')
    """
    frames: list[pl.DataFrame] = []
    for name, folder in _IA_SOURCES.items():
        df = _process_single_ia(name, folder)
        frames.append(df)

    return pl.concat(frames)


# ──────────────────────────────────────────────
# 5. DataFrame consolidado — intersección top 15
# ──────────────────────────────────────────────

def build_consolidated_df(top_n: int = 15) -> pl.DataFrame:
    """
    Construye un DataFrame consolidado con las *top_n* palabras más
    comunes en español que aparecen en las **4 fuentes simultáneamente**
    (intersección).

    Algoritmo
    ---------
    1. Genera los DataFrames de frecuencia de Humano y las 3 IAs.
    2. Concatena todo en un único DataFrame.
    3. Identifica las palabras presentes en las 4 fuentes.
    4. Ordena esas palabras por conteo total descendente y toma las top_n.
    5. Filtra el DataFrame para conservar solo esas top_n palabras.
    6. Calcula la frecuencia relativa por fuente:
       freq_relativa = conteo_palabra / total_palabras_fuente.

    Columns (resultado)
    -------------------
    palabra        : Utf8
    fuente         : Utf8
    conteo         : UInt32
    total_fuente   : UInt32   — total de palabras en la fuente
    freq_relativa  : Float64  — conteo / total_fuente

    Parameters
    ----------
    top_n : int
        Cantidad de palabras de la intersección a conservar (default 15).

    Returns
    -------
    pl.DataFrame
    """
    # 1. Obtener frecuencias individuales
    df_humano = process_humano()
    df_ia = process_ia_sources()
    df_all = pl.concat([df_humano, df_ia])

    # 2. Palabras presentes en las 4 fuentes
    sources_per_word = (
        df_all
        .group_by("palabra")
        .agg(pl.col("fuente").n_unique().alias("n_fuentes"))
    )
    common_words = (
        sources_per_word
        .filter(pl.col("n_fuentes") == 4)
        .select("palabra")
    )

    # 3. Ranking por conteo total para elegir las top_n
    total_counts = (
        df_all
        .join(common_words, on="palabra", how="inner")
        .group_by("palabra")
        .agg(pl.col("conteo").sum().alias("conteo_total"))
        .sort("conteo_total", descending=True)
        .head(top_n)
        .select("palabra")
    )

    # 4. Filtrar df_all a solo las top_n palabras intersectadas
    df_top = df_all.join(total_counts, on="palabra", how="inner")

    # 5. Calcular total de palabras por fuente (sobre el corpus completo)
    total_per_source = (
        df_all
        .group_by("fuente")
        .agg(pl.col("conteo").sum().alias("total_fuente"))
    )

    # 6. Unir y calcular frecuencia relativa
    df_consolidated = (
        df_top
        .join(total_per_source, on="fuente", how="left")
        .with_columns(
            (pl.col("conteo").cast(pl.Float64) / pl.col("total_fuente").cast(pl.Float64))
            .alias("freq_relativa")
        )
        .sort(["palabra", "fuente"])
    )

    return df_consolidated


# ──────────────────────────────────────────────
# 6. Ejecución directa → demo rápida
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  text_processor.py — Demostración")
    print("=" * 70)

    # --- Humano ---
    print("\n▸ Procesando Humano moby dick.txt …")
    df_h = process_humano()
    print(f"  Palabras únicas: {df_h.height:,}")
    print(f"  Total palabras:  {df_h['conteo'].sum():,}")
    print("  Top 10:")
    print(df_h.head(10))

    # --- IAs ---
    print("\n▸ Procesando fuentes IA …")
    df_ia = process_ia_sources()
    for src in ["ChatGPT", "Gemini", "Grok"]:
        sub = df_ia.filter(pl.col("fuente") == src)
        print(f"\n  {src}: {sub.height:,} palabras únicas | "
              f"{sub['conteo'].sum():,} palabras totales")
        print(sub.head(5))

    # --- Consolidado ---
    print("\n▸ Construyendo DataFrame consolidado (top 15 intersección) …")
    df_cons = build_consolidated_df(top_n=15)
    print(f"\n  Filas resultado: {df_cons.height}")
    print(df_cons)

    print("\n" + "=" * 70)
