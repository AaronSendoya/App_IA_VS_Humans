# -*- coding: utf-8 -*-
"""
config.py — Configuración central del proyecto.

Define rutas absolutas y relativas a los directorios de datos
y valida que la estructura esperada exista en disco.
"""

from pathlib import Path
import sys
import os

# Forzar salida UTF-8 en Windows para evitar errores de codificación
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────
# 1. Raíz del proyecto (directorio donde vive este archivo)
# ──────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ──────────────────────────────────────────────
# 2. Directorio base de datos ("Estadistica libros")
# ──────────────────────────────────────────────
DATA_DIR_NAME: str = "Estadistica libros"
DATA_DIR: Path = PROJECT_ROOT / DATA_DIR_NAME

# ──────────────────────────────────────────────
# 3. Sub-carpetas de cada fuente IA
# ──────────────────────────────────────────────
CHATGPT_DIR: Path = DATA_DIR / "chatgpt"
GEMINI_DIR: Path = DATA_DIR / "Gemini"
GROK_DIR: Path = DATA_DIR / "Grok"

# ──────────────────────────────────────────────
# 4. Archivo del texto humano de referencia
# ──────────────────────────────────────────────
HUMANO_FILE: Path = DATA_DIR / "Humano moby dick.txt"

# ──────────────────────────────────────────────
# 5. Diccionario resumen de todas las fuentes
#    Facilita la iteración programática.
# ──────────────────────────────────────────────
SOURCES: dict[str, Path] = {
    "ChatGPT": CHATGPT_DIR,
    "Gemini": GEMINI_DIR,
    "Grok": GROK_DIR,
    "Humano": HUMANO_FILE,
}

# ──────────────────────────────────────────────
# 6. Rutas relativas (respecto a PROJECT_ROOT)
# ──────────────────────────────────────────────
DATA_DIR_REL: Path = DATA_DIR.relative_to(PROJECT_ROOT)
CHATGPT_DIR_REL: Path = CHATGPT_DIR.relative_to(PROJECT_ROOT)
GEMINI_DIR_REL: Path = GEMINI_DIR.relative_to(PROJECT_ROOT)
GROK_DIR_REL: Path = GROK_DIR.relative_to(PROJECT_ROOT)
HUMANO_FILE_REL: Path = HUMANO_FILE.relative_to(PROJECT_ROOT)

# Número esperado de archivos .txt por carpeta IA
EXPECTED_TXT_COUNT: int = 12


# ──────────────────────────────────────────────
# 7. Funciones de validación
# ──────────────────────────────────────────────
def validate_structure(*, verbose: bool = True) -> bool:
    """
    Verifica que todos los directorios y archivos esperados existan.

    Parameters
    ----------
    verbose : bool
        Si True, imprime el estado de cada verificación.

    Returns
    -------
    bool
        True si toda la estructura es válida; False en caso contrario.
    """
    all_ok = True
    checks: list[tuple[str, Path, bool]] = [
        ("Directorio de datos", DATA_DIR, True),
        ("Carpeta ChatGPT", CHATGPT_DIR, True),
        ("Carpeta Gemini", GEMINI_DIR, True),
        ("Carpeta Grok", GROK_DIR, True),
        ("Archivo Humano", HUMANO_FILE, False),
    ]

    for label, path, is_dir in checks:
        exists = path.is_dir() if is_dir else path.is_file()
        status = "✔" if exists else "✘"
        if verbose:
            print(f"  {status}  {label}: {path}")
        if not exists:
            all_ok = False

    # Verificar conteo de .txt por cada carpeta IA
    ia_dirs = {"ChatGPT": CHATGPT_DIR, "Gemini": GEMINI_DIR, "Grok": GROK_DIR}
    for name, folder in ia_dirs.items():
        if folder.is_dir():
            txt_files = list(folder.glob("*.txt"))
            count = len(txt_files)
            ok = count == EXPECTED_TXT_COUNT
            status = "✔" if ok else "✘"
            if verbose:
                print(f"  {status}  {name}: {count}/{EXPECTED_TXT_COUNT} archivos .txt")
            if not ok:
                all_ok = False

    return all_ok


def get_all_txt_files() -> dict[str, list[Path]]:
    """
    Retorna un diccionario {fuente: [lista de archivos .txt]}
    para las tres carpetas IA.
    """
    result: dict[str, list[Path]] = {}
    for name, folder in [("ChatGPT", CHATGPT_DIR), ("Gemini", GEMINI_DIR), ("Grok", GROK_DIR)]:
        if folder.is_dir():
            result[name] = sorted(folder.glob("*.txt"))
        else:
            result[name] = []
    return result


# ──────────────────────────────────────────────
# 8. Ejecución directa → diagnóstico rápido
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Configuración del proyecto — IA vs Humanos")
    print("=" * 60)
    print(f"\n  Raíz del proyecto : {PROJECT_ROOT}")
    print(f"  Directorio de datos: {DATA_DIR}\n")
    print("  Verificación de estructura:\n")

    ok = validate_structure(verbose=True)

    print()
    if ok:
        print("  ✅ Estructura completa y válida.")
    else:
        print("  ❌ Faltan elementos. Revisa las rutas anteriores.")
        sys.exit(1)

    # Mostrar resumen de archivos por fuente
    print("\n  Archivos detectados por fuente:")
    print("  " + "-" * 40)
    all_files = get_all_txt_files()
    for source, files in all_files.items():
        print(f"  {source} ({len(files)} archivos):")
        for f in files:
            print(f"    • {f.name}")
    print(f"\n  Humano: {HUMANO_FILE.name} ({HUMANO_FILE.stat().st_size:,} bytes)")
    print("=" * 60)
