# APP IA vs Humanos — Dashboard Estadístico

Análisis comparativo de texto generado por IA (ChatGPT, Gemini, Grok) frente a texto humano (Moby Dick).  
Incluye frecuencias, tablas de contingencia, inferencia bayesiana y simulación de cadenas de Markov.

---

## Requisitos previos

- **Python 3.11 o superior** instalado y en el PATH  
  > Este proyecto usa Python 3.14. Verifica tu versión con `python --version`
- **Git** (solo si clonas el repositorio)
- Windows 10/11 (los comandos están en PowerShell)

---

## Levantar el proyecto desde cero

### Paso 1 — Clonar o descargar el repositorio

```powershell
git clone <URL-del-repositorio>
cd app_ia_vs_humans
```

> Si ya tienes la carpeta descargada, solo navega hasta ella:
> ```powershell
> cd C:\ruta\a\app_ia_vs_humans
> ```

---

### Paso 2 — Crear el entorno virtual

```powershell
python -m venv .venv
```

> Si tienes varias versiones de Python instaladas, especifica la ruta completa:
> ```powershell
> C:\Python314\python.exe -m venv .venv
> ```

---

### Paso 3 — Activar el entorno virtual

```powershell
.\.venv\Scripts\Activate.ps1
```

Deberías ver `(.venv)` al inicio de la línea del terminal.

> **Error de permisos en PowerShell:**  
> Si aparece `"execution of scripts is disabled"`, ejecuta primero:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

### Paso 4 — Instalar las dependencias

```powershell
pip install -e .
```

Esto instala todo lo declarado en `pyproject.toml`:

| Librería | Uso |
|---|---|
| `streamlit` | Dashboard web |
| `polars` | Procesamiento de DataFrames |
| `matplotlib` | Gráficos de convergencia Markov |
| `networkx` | Red bayesiana |

---

### Paso 5 — Verificar la estructura de datos (opcional)

```powershell
python config.py
```

Confirma que los 36 archivos `.txt` de las IAs y el texto humano están en su lugar:

```
Estadistica libros/
├── Humano moby dick.txt
├── chatgpt/        ← 12 ensayos
├── Gemini/         ← 12 ensayos
└── Grok/           ← 12 ensayos
```

---

### Paso 6 — Levantar el dashboard

```powershell
streamlit run app.py
```

El navegador abre automáticamente en `http://localhost:8501`.  
Si no abre solo, cópialo y pégalo manualmente.

---

## Reiniciar el servidor (sesiones siguientes)

Desde la raíz del proyecto, solo necesitas los pasos 3 y 6:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

---

## Estructura del proyecto

```
app_ia_vs_humans/
├── app.py                    # Dashboard principal (Streamlit)
├── text_processor.py         # Limpieza y frecuencias del corpus
├── contingency_analysis.py   # Tablas de contingencia y probabilidades
├── bayes_engine.py           # Motor de inferencia bayesiana
├── config.py                 # Rutas y validación de estructura
├── pyproject.toml            # Dependencias del proyecto
├── .streamlit/
│   └── config.toml           # Tema visual Cyber-Dark
└── Estadistica libros/
    ├── Humano moby dick.txt
    ├── chatgpt/
    ├── Gemini/
    └── Grok/
```

---

## Solución de problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: streamlit` | El venv no está activado | Ejecuta el Paso 3 |
| `FileNotFoundError` al iniciar | Faltan archivos `.txt` | Ejecuta `python config.py` para diagnosticar |
| El navegador no abre | Puerto ocupado | Cambia el puerto: `streamlit run app.py --server.port 8502` |
| Texto invisible / blanco | Conflicto de tema del navegador | El tema está forzado en `.streamlit/config.toml`, recarga la página con `Ctrl+Shift+R` |
| `execution of scripts is disabled` | Política de PowerShell restrictiva | Ver nota del Paso 3 |
