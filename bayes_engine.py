# -*- coding: utf-8 -*-
"""
bayes_engine.py — Motor de Teorema de Bayes con Diagrama de Árbol de 3 niveles.

Diagrama de Árbol
=================
    Nivel 1 (Tipo de Autor)
    ├── Humano   →  P(Humano)  = palabras_humano / total_corpus
    └── IA       →  P(IA)      = palabras_ia / total_corpus

    Nivel 2 (Fuente / Herramienta)
    ├── Humano  →  'Humano'              P(Humano | Humano) = 1
    └── IA      ├── ChatGPT              P(ChatGPT | IA) = palabras_chatgpt / palabras_ia
                ├── Gemini               P(Gemini  | IA) = palabras_gemini  / palabras_ia
                └── Grok                 P(Grok    | IA) = palabras_grok    / palabras_ia

    Nivel 3 (Palabra)
    └── Selección de una de las 15 palabras comunes
        P(palabra | fuente) = conteo(palabra, fuente) / total(fuente)

Funciones de Bayes
==================
    bayes_fuente_dado_ia_y_palabra:   P(Fuente | IA ∩ Palabra)
    bayes_tipo_dado_palabra:          P(Tipo | Palabra)     {Tipo ∈ {Humano, IA}}
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from text_processor import build_consolidated_df, process_humano, process_ia_sources
from contingency_analysis import build_contingency_table, SOURCE_ORDER


# ──────────────────────────────────────────────
# 1. Resultado detallado de Bayes
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class BayesResult:
    """
    Resultado documentado de un cálculo con el Teorema de Bayes.

    Attributes
    ----------
    titulo : str
        Nombre corto del ejercicio.
    hipotesis : str
        Notación de la probabilidad buscada, ej. P(Gemini | IA ∩ 'la').
    formula_general : str
        Fórmula genérica de Bayes aplicada.
    numerador_desc : str
        Descripción del numerador (probabilidad conjunta).
    numerador_calculo : str
        Cálculo numérico del numerador.
    numerador_valor : float
        Valor numérico del numerador.
    denominador_desc : str
        Descripción del denominador (probabilidad total).
    denominador_calculo : str
        Cálculo numérico del denominador.
    denominador_valor : float
        Valor numérico del denominador.
    resultado : float
        Valor final P(H|E).
    interpretacion : str
        Explicación verbal del resultado.
    """
    titulo: str
    hipotesis: str
    formula_general: str
    numerador_desc: str
    numerador_calculo: str
    numerador_valor: float
    denominador_desc: str
    denominador_calculo: str
    denominador_valor: float
    resultado: float
    interpretacion: str

    def __str__(self) -> str:
        width = 72
        lines = [
            f"╔{'═' * width}╗",
            f"║  {self.titulo:<{width - 2}}║",
            f"╠{'═' * width}╣",
            f"║  Hipótesis:  {self.hipotesis:<{width - 14}}║",
            f"╠{'─' * width}╣",
            f"║  Fórmula General:                                                      ║",
            f"║    {self.formula_general:<{width - 4}}║",
            f"╠{'─' * width}╣",
            f"║  NUMERADOR (probabilidad conjunta):                                    ║",
            f"║    {self.numerador_desc:<{width - 4}}║",
            f"║    = {self.numerador_calculo:<{width - 5}}║",
            f"║    = {self.numerador_valor:<{width - 5}.10f}║",
            f"╠{'─' * width}╣",
            f"║  DENOMINADOR (probabilidad total):                                     ║",
            f"║    {self.denominador_desc:<{width - 4}}║",
            f"║    = {self.denominador_calculo:<{width - 5}}║",
            f"║    = {self.denominador_valor:<{width - 5}.10f}║",
            f"╠{'─' * width}╣",
            f"║  RESULTADO:  {self.resultado:<10.6f}  ({self.resultado * 100:.4f}%){' ' * (width - 42)}║",
            f"║  → {self.interpretacion:<{width - 4}}║",
            f"╚{'═' * width}╝",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 2. Clase BayesTree — Diagrama de Árbol
# ──────────────────────────────────────────────

_IA_SOURCES_NAMES: list[str] = ["ChatGPT", "Gemini", "Grok"]


@dataclass
class BayesTree:
    """
    Encapsula el diagrama de árbol de 3 niveles y expone métodos
    para calcular probabilidades bayesianas.

    Los datos se calculan a partir del corpus completo (no solo las 15 palabras)
    para los priors de Nivel 1 y Nivel 2, y de la tabla de contingencia
    para las verosimilitudes de Nivel 3.
    """

    # --- Nivel 1: priors Humano / IA ---
    total_corpus: int = 0
    total_humano: int = 0
    total_ia: int = 0
    p_humano: float = 0.0
    p_ia: float = 0.0

    # --- Nivel 2: distribución dentro de IA ---
    total_chatgpt: int = 0
    total_gemini: int = 0
    total_grok: int = 0
    p_chatgpt_dado_ia: float = 0.0
    p_gemini_dado_ia: float = 0.0
    p_grok_dado_ia: float = 0.0

    # --- Nivel 3: tabla de contingencia ---
    contingency: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- Palabras disponibles ---
    palabras: list[str] = field(default_factory=list)

    @classmethod
    def from_corpus(cls) -> BayesTree:
        """
        Construye el árbol a partir de los datos reales del corpus.

        Nivel 1 priors: proporciones por volumen total de palabras.
        Nivel 2 priors: proporciones de cada IA sobre el total IA.
        Nivel 3 likelihoods: desde la tabla de contingencia (top 15).
        """
        # Obtener totales de palabras por fuente (corpus completo)
        df_humano = process_humano()
        df_ia = process_ia_sources()

        total_humano = int(df_humano["conteo"].sum())

        totals_ia: dict[str, int] = {}
        for src in _IA_SOURCES_NAMES:
            sub = df_ia.filter(pl.col("fuente") == src)
            totals_ia[src] = int(sub["conteo"].sum())

        total_ia = sum(totals_ia.values())
        total_corpus = total_humano + total_ia

        # Construir tabla de contingencia
        ct = build_contingency_table()

        # Extraer lista de palabras (sin la fila Total)
        palabras = (
            ct.filter(pl.col("palabra") != "Total")
            .select("palabra")
            .to_series()
            .to_list()
        )

        return cls(
            total_corpus=total_corpus,
            total_humano=total_humano,
            total_ia=total_ia,
            p_humano=total_humano / total_corpus,
            p_ia=total_ia / total_corpus,
            total_chatgpt=totals_ia["ChatGPT"],
            total_gemini=totals_ia["Gemini"],
            total_grok=totals_ia["Grok"],
            p_chatgpt_dado_ia=totals_ia["ChatGPT"] / total_ia,
            p_gemini_dado_ia=totals_ia["Gemini"] / total_ia,
            p_grok_dado_ia=totals_ia["Grok"] / total_ia,
            contingency=ct,
            palabras=palabras,
        )

    # ── Utilidades internas ──────────────────

    def _validate_palabra(self, palabra: str) -> None:
        if palabra not in self.palabras:
            raise ValueError(
                f"'{palabra}' no está en las 15 palabras. "
                f"Disponibles: {self.palabras}"
            )

    def _validate_fuente_ia(self, fuente: str) -> None:
        if fuente not in _IA_SOURCES_NAMES:
            raise ValueError(
                f"'{fuente}' no es fuente IA válida. "
                f"Opciones: {_IA_SOURCES_NAMES}"
            )

    def _freq(self, palabra: str, fuente: str) -> int:
        """Frecuencia absoluta de (palabra, fuente) desde la tabla de contingencia."""
        row = self.contingency.filter(pl.col("palabra") == palabra)
        return int(row.select(fuente).item())

    def _total_fuente_ct(self, fuente: str) -> int:
        """Total marginal de una fuente en la tabla de contingencia."""
        return int(
            self.contingency
            .filter(pl.col("palabra") == "Total")
            .select(fuente)
            .item()
        )

    def _p_palabra_dado_fuente(self, palabra: str, fuente: str) -> float:
        """P(palabra | fuente) = freq(palabra, fuente) / total_fuente_ct."""
        total = self._total_fuente_ct(fuente)
        if total == 0:
            return 0.0
        return self._freq(palabra, fuente) / total

    def _p_fuente_dado_ia(self, fuente: str) -> float:
        """P(fuente | IA) — proporción de la fuente dentro de las IAs."""
        mapping = {
            "ChatGPT": self.p_chatgpt_dado_ia,
            "Gemini": self.p_gemini_dado_ia,
            "Grok": self.p_grok_dado_ia,
        }
        return mapping[fuente]

    # ── Diagrama de Árbol (texto) ────────────

    def print_tree(self) -> str:
        """Representación textual del diagrama de árbol de 3 niveles."""
        lines = [
            "Diagrama de Árbol — 3 Niveles",
            "=" * 60,
            "",
            "Nivel 1: Tipo de Autor",
            f"  ├── Humano  P(H) = {self.total_humano:,} / {self.total_corpus:,} = {self.p_humano:.6f}",
            f"  └── IA      P(I) = {self.total_ia:,} / {self.total_corpus:,} = {self.p_ia:.6f}",
            "",
            "Nivel 2: Fuente (condicionado al tipo)",
            f"  Humano → Humano      P(Humano|H) = 1.000000",
            f"  IA ─┬── ChatGPT      P(ChatGPT|IA) = {self.total_chatgpt:,} / {self.total_ia:,} = {self.p_chatgpt_dado_ia:.6f}",
            f"      ├── Gemini       P(Gemini|IA)  = {self.total_gemini:,} / {self.total_ia:,} = {self.p_gemini_dado_ia:.6f}",
            f"      └── Grok         P(Grok|IA)    = {self.total_grok:,} / {self.total_ia:,} = {self.p_grok_dado_ia:.6f}",
            "",
            "Nivel 3: Palabra (condicionado a la fuente)",
            "  P(palabra | fuente) = freq(palabra, fuente) / total(fuente)",
            f"  Palabras disponibles ({len(self.palabras)}): {', '.join(self.palabras)}",
        ]
        return "\n".join(lines)

    # ── Extensiones: Red Bayesiana y Cadena de Markov ──



    def simulate_markov_chain(self, steps: int = 15, inertia: float = 0.7) -> None:
        """
        Simula una Cadena de Markov de 3 estados independientes.
        """
        # Probabilidades estacionarias (priors del corpus)
        pi_1 = self.p_humano
        pi_2 = self.p_ia * self.p_gemini_dado_ia
        pi_3 = self.p_ia * (self.p_chatgpt_dado_ia + self.p_grok_dado_ia)

        pi = np.array([pi_1, pi_2, pi_3])

        # Matriz estocástica
        T = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                T[i, j] = (1.0 - inertia) * pi[j]
                if i == j:
                    T[i, j] += inertia

        print("\n" + "═" * 76)
        print("  CADENA DE MARKOV — Simulación de Lectura de Corpus")
        print("═" * 76)
        print("  Estados Definidos:")
        print("    S1 = Procesando Texto Humano")
        print("    S2 = Procesando Texto IA-Gemini")
        print("    S3 = Procesando Texto IA-Otros (ChatGPT + Grok)")

        print("\n  Vector Estacionario Teórico (π):")
        print(f"    π = [ {pi[0]:.4f},  {pi[1]:.4f},  {pi[2]:.4f} ]")

        print("\n  Matriz de Transición T (Inercia = 0.7):")
        for i, row in enumerate(T, 1):
            print(f"    S{i} -> [ {row[0]:.4f},  {row[1]:.4f},  {row[2]:.4f} ]")

        v = np.array([0.0, 1.0, 0.0])
        print("\n  Simulación (Estado Inicial: 100% S2):")
        print(f"    Paso  0 -> v = [ {v[0]:.6f},  {v[1]:.6f},  {v[2]:.6f} ]")

        for step in range(1, steps + 1):
            v = v @ T
            print(f"    Paso {step:2d} -> v = [ {v[0]:.6f},  {v[1]:.6f},  {v[2]:.6f} ]")

        print("\n  -> El vector de estado converge al vector estacionario π.")

    # ──────────────────────────────────────────
    # 3. BAYES: P(Fuente | IA ∩ Palabra)
    # ──────────────────────────────────────────

    def bayes_fuente_dado_ia_y_palabra(
        self, fuente: str, palabra: str
    ) -> BayesResult:
        """
        Teorema de Bayes — Tipo A:
        P(Fuente | IA ∩ Palabra)

        Dado que se sabe que la palabra proviene de una IA y que la
        palabra observada es *palabra*, ¿cuál es la probabilidad de
        que haya sido generada por *fuente*?

        Fórmula
        -------
                          P(palabra | fuente) · P(fuente | IA)
        P(F | IA ∩ W) = ──────────────────────────────────────────
                         Σ_j  P(palabra | fuente_j) · P(fuente_j | IA)

        donde j ∈ {ChatGPT, Gemini, Grok}
        """
        palabra = palabra.lower()
        self._validate_palabra(palabra)
        self._validate_fuente_ia(fuente)

        # Numerador: P(W|F) · P(F|IA)
        p_w_f = self._p_palabra_dado_fuente(palabra, fuente)
        p_f_ia = self._p_fuente_dado_ia(fuente)
        numerador = p_w_f * p_f_ia

        # Denominador: Σ P(W|F_j) · P(F_j|IA)  para cada IA
        partes_denom: list[tuple[str, float, float]] = []
        denominador = 0.0
        for src in _IA_SOURCES_NAMES:
            p_w_src = self._p_palabra_dado_fuente(palabra, src)
            p_src_ia = self._p_fuente_dado_ia(src)
            partes_denom.append((src, p_w_src, p_src_ia))
            denominador += p_w_src * p_src_ia

        resultado = numerador / denominador if denominador > 0 else 0.0

        # Construcción de cadenas descriptivas
        num_calc = (
            f"P('{palabra}'|{fuente}) × P({fuente}|IA) = "
            f"{p_w_f:.6f} × {p_f_ia:.6f}"
        )

        denom_parts_str = " + ".join(
            f"P('{palabra}'|{s}) × P({s}|IA) [{pw:.6f}×{ps:.6f}]"
            for s, pw, ps in partes_denom
        )

        return BayesResult(
            titulo=f"Bayes — P({fuente} | IA ∩ '{palabra}')",
            hipotesis=f"P({fuente} | IA ∩ '{palabra}')",
            formula_general=(
                f"P({fuente}|IA∩'{palabra}') = "
                f"P('{palabra}'|{fuente})·P({fuente}|IA) / "
                f"Σ P('{palabra}'|Fj)·P(Fj|IA)"
            ),
            numerador_desc=f"P('{palabra}'|{fuente}) · P({fuente}|IA)",
            numerador_calculo=num_calc,
            numerador_valor=numerador,
            denominador_desc=f"Σ sobre {{ChatGPT, Gemini, Grok}}",
            denominador_calculo=denom_parts_str,
            denominador_valor=denominador,
            resultado=resultado,
            interpretacion=(
                f"Dado que una IA escribió '{palabra}', hay un "
                f"{resultado * 100:.4f}% de prob. de que fuera {fuente}."
            ),
        )

    # ──────────────────────────────────────────
    # 4. BAYES: P(Tipo | Palabra)
    # ──────────────────────────────────────────

    def bayes_tipo_dado_palabra(
        self, tipo: str, palabra: str
    ) -> BayesResult:
        """
        Teorema de Bayes — Tipo B:
        P(Tipo | Palabra)    donde Tipo ∈ {'Humano', 'IA'}

        Dado que se observó *palabra*, ¿cuál es la probabilidad de
        que provenga de un *tipo* de autor?

        Fórmula
        -------
                         P(palabra | tipo) · P(tipo)
        P(T | W) = ──────────────────────────────────────
                   P(palabra|Humano)·P(Humano) + P(palabra|IA)·P(IA)

        Nota: P(palabra | IA) se calcula como probabilidad total:
        P(W|IA) = Σ_j P(W|Fj) · P(Fj|IA)
        """
        palabra = palabra.lower()
        self._validate_palabra(palabra)

        if tipo not in ("Humano", "IA"):
            raise ValueError(f"Tipo debe ser 'Humano' o 'IA', se recibió: '{tipo}'")

        # P(W | Humano)
        p_w_humano = self._p_palabra_dado_fuente(palabra, "Humano")

        # P(W | IA) = Σ P(W|Fj) · P(Fj|IA)
        p_w_ia = 0.0
        ia_parts: list[tuple[str, float, float]] = []
        for src in _IA_SOURCES_NAMES:
            p_w_src = self._p_palabra_dado_fuente(palabra, src)
            p_src_ia = self._p_fuente_dado_ia(src)
            ia_parts.append((src, p_w_src, p_src_ia))
            p_w_ia += p_w_src * p_src_ia

        # Denominador (prob. total de la palabra)
        denominador = p_w_humano * self.p_humano + p_w_ia * self.p_ia

        if tipo == "Humano":
            numerador = p_w_humano * self.p_humano
            num_desc = f"P('{palabra}'|Humano) · P(Humano)"
            num_calc = f"{p_w_humano:.6f} × {self.p_humano:.6f}"
        else:  # IA
            numerador = p_w_ia * self.p_ia
            num_desc = f"P('{palabra}'|IA) · P(IA)"
            num_calc = (
                f"[Σ P('{palabra}'|Fj)·P(Fj|IA)] × P(IA) = "
                f"{p_w_ia:.6f} × {self.p_ia:.6f}"
            )

        resultado = numerador / denominador if denominador > 0 else 0.0

        denom_calc = (
            f"P('{palabra}'|Humano)·P(Humano) + P('{palabra}'|IA)·P(IA) = "
            f"{p_w_humano:.6f}×{self.p_humano:.6f} + "
            f"{p_w_ia:.6f}×{self.p_ia:.6f}"
        )

        return BayesResult(
            titulo=f"Bayes — P({tipo} | '{palabra}')",
            hipotesis=f"P({tipo} | '{palabra}')",
            formula_general=(
                f"P({tipo}|'{palabra}') = "
                f"P('{palabra}'|{tipo})·P({tipo}) / "
                f"[P('{palabra}'|H)·P(H) + P('{palabra}'|IA)·P(IA)]"
            ),
            numerador_desc=num_desc,
            numerador_calculo=num_calc,
            numerador_valor=numerador,
            denominador_desc="P(palabra) = P(W|Humano)·P(Humano) + P(W|IA)·P(IA)",
            denominador_calculo=denom_calc,
            denominador_valor=denominador,
            resultado=resultado,
            interpretacion=(
                f"Dado que se observó '{palabra}', hay un "
                f"{resultado * 100:.4f}% de prob. de que sea de tipo {tipo}."
            ),
        )


# ──────────────────────────────────────────────
# 5. Ejecución directa — 6 ejercicios de Bayes
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 76)
    print("  bayes_engine.py — Teorema de Bayes con Diagrama de Árbol (3 niveles)")
    print("=" * 76)

    # Construir el árbol
    print("\n▸ Construyendo Diagrama de Árbol desde el corpus …\n")
    tree = BayesTree.from_corpus()
    print(tree.print_tree())

    # ──────────────────────────────────────────
    # 6 EJERCICIOS DE BAYES
    # ──────────────────────────────────────────

    ejercicios: list[BayesResult] = []

    # ── Ejercicio 1 ──
    # P(Gemini | IA ∩ 'la')
    # Si se sabe que una IA escribió la palabra 'la',
    # ¿cuál es la probabilidad de que haya sido Gemini?
    print("\n" + "━" * 76)
    print("  EJERCICIO 1: P(Gemini | IA ∩ 'la')")
    print("━" * 76 + "\n")
    r1 = tree.bayes_fuente_dado_ia_y_palabra("Gemini", "la")
    print(r1)
    ejercicios.append(r1)

    # ── Ejercicio 2 ──
    # P(ChatGPT | IA ∩ 'de')
    # Si una IA generó la palabra 'de', ¿cuál es la
    # probabilidad de que fuera ChatGPT?
    print("\n" + "━" * 76)
    print("  EJERCICIO 2: P(ChatGPT | IA ∩ 'de')")
    print("━" * 76 + "\n")
    r2 = tree.bayes_fuente_dado_ia_y_palabra("ChatGPT", "de")
    print(r2)
    ejercicios.append(r2)

    # ── Ejercicio 3 ──
    # P(Grok | IA ∩ 'y')
    # Si una IA escribió 'y', ¿fue Grok?
    print("\n" + "━" * 76)
    print("  EJERCICIO 3: P(Grok | IA ∩ 'y')")
    print("━" * 76 + "\n")
    r3 = tree.bayes_fuente_dado_ia_y_palabra("Grok", "y")
    print(r3)
    ejercicios.append(r3)

    # ── Ejercicio 4 ──
    # P(Humano | 'a')
    # Sabiendo que la palabra es 'a', ¿cuál es la
    # probabilidad de que provenga del texto humano?
    print("\n" + "━" * 76)
    print("  EJERCICIO 4: P(Humano | 'a')")
    print("━" * 76 + "\n")
    r4 = tree.bayes_tipo_dado_palabra("Humano", "a")
    print(r4)
    ejercicios.append(r4)

    # ── Ejercicio 5 ──
    # P(IA | 'no')
    # Si se observa la palabra 'no', ¿cuál es la
    # probabilidad de que provenga de una IA?
    print("\n" + "━" * 76)
    print("  EJERCICIO 5: P(IA | 'no')")
    print("━" * 76 + "\n")
    r5 = tree.bayes_tipo_dado_palabra("IA", "no")
    print(r5)
    ejercicios.append(r5)

    # ── Ejercicio 6 ──
    # P(Humano | 'sea')
    # Si la palabra observada es 'sea', ¿proviene
    # del texto humano (Moby Dick)?
    print("\n" + "━" * 76)
    print("  EJERCICIO 6: P(Humano | 'sea')")
    print("━" * 76 + "\n")
    r6 = tree.bayes_tipo_dado_palabra("Humano", "sea")
    print(r6)
    ejercicios.append(r6)

    # ── Tabla resumen ──
    print("\n" + "═" * 76)
    print("  RESUMEN — 6 Ejercicios de Bayes")
    print("═" * 76)
    print(f"  {'#':<4} {'Hipótesis':<40} {'Resultado':>10}")
    print("  " + "─" * 56)
    for i, r in enumerate(ejercicios, 1):
        print(f"  {i:<4} {r.hipotesis:<40} {r.resultado:>10.6f}  ({r.resultado*100:.2f}%)")
    print("═" * 76)

    # ── Extensiones: Cadena de Markov ──
    
    tree.simulate_markov_chain(steps=15, inertia=0.7)
    
    print("\n" + "=" * 76)
