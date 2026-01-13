# proyectos_actuariales_varios

Suite de mini-proyectos actuariales en **Python + Streamlit** para valoración de **SEGUROS (muerte)** y **PENSIONES (vida/supervivencia)**, basada en **tablas de mortalidad** y **funciones conmutadas** (D, N, M).  
Este repositorio incluye un Excel con **datos ficticios** para que la app corra tanto en local como en deploy.

*Quickstart in cmd:* python -m streamlit run app.py
## ✅ Requisitos
- **Python 3.10+** (recomendado)
- **pip**
- Conexión a internet solo para instalar dependencias la primera vez
## 📁 Estructura esperada del proyecto
Asegúrate de que en la carpeta del proyecto existan estos archivos:
- `app.py`
- `requirements.txt`
- `data/tabla_mortalidad.xlsx` *(recomendado)*

> Si no quieres usar carpeta `data/`, también puedes dejar el Excel en la raíz como `tabla_mortalidad.xlsx`, pero lo ideal es mantenerlo en `data/`.

## 📌 Especificaciones del Excel (OBLIGATORIO)
El archivo `tabla_mortalidad.xlsx` debe tener:
- Hojas (sheets): `Hombres` y `Mujeres`
- Columnas (exactamente así): `x`, `q(x)`, `l(x)`, `d(x)`

> Nota: el programa construye los conmutados a partir de estas columnas. No debes traer `Dx` en el Excel.

## 🖥️ Cómo correr la calculadora actuarial (PASO A PASO)

### PASO 1) Abrir la terminal en la carpeta del proyecto
**Windows (modo rápido):**
1. Abre la carpeta del proyecto en el explorador.
2. Haz clic en la barra superior donde aparece la ruta (ej: `C:\Users\...\proyectos_actuariales_varios`).
3. Escribe `cmd` y presiona **Enter**.

**Mac/Linux o alternativa general:**
- Abre una terminal y entra a la carpeta con :

### PASO 2) (Recomendado) Crear y activar entorno virtual

Windows: python -m venv .venv
.venv\Scripts\activate

Mac\Linux: python -m venv .venv
source .venv/bin/activate

### PASO 3) Instalar dependencias (solo la primera vez)

pip install -r requirements.txt

Si pip no funciona:

python -m pip install -r requirements.txt

### PASO 4) Ejecutar la app (cada vez que la uses)

python -m streamlit run app.py


