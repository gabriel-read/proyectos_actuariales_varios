import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Calculadora Actuarial", layout="wide")
st.title("📠 Calculadora Actuarial")

# =======================================================
# Auto-detect del Excel (sin tener que escribirlo)
# =======================================================
def detectar_archivo_por_defecto():
    folder = Path(__file__).parent

    # 1) rutas preferidas (deploy-friendly)
    preferidos = [
        folder / "data" / "tabla_mortalidad.xlsx",
        folder / "tabla_mortalidad.xlsx",
    ]
    for p in preferidos:
        if p.exists():
            return str(p)

    # 2) búsqueda por patrón (por si lo renombraste)
    for ext in ("xlsx", "xlsm", "xls"):
        candidatos = list(folder.rglob(f"*mortalidad*.{ext}"))
        if candidatos:
            return str(candidatos[0])

    # 3) fallback
    return str(folder / "tabla_mortalidad.xlsx")


ARCHIVO_DEFAULT = detectar_archivo_por_defecto()

st.sidebar.header("Archivo y parámetros")
archivo = st.sidebar.text_input("Archivo Excel", value=ARCHIVO_DEFAULT)

# =======================================================
# Leer Excel base: x, q(x), l(x), d(x)
# =======================================================
@st.cache_data
def cargar_base(path: str):
    hombres = pd.read_excel(path, sheet_name="Hombres")
    mujeres = pd.read_excel(path, sheet_name="Mujeres")

    for df, nombre in [(hombres, "Hombres"), (mujeres, "Mujeres")]:
        df.columns = [c.strip() for c in df.columns]
        req = {"x", "q(x)", "l(x)", "d(x)"}
        faltan = req - set(df.columns)
        if faltan:
            raise ValueError(f"En la hoja '{nombre}' faltan columnas: {faltan}")

        df["x"] = df["x"].astype(int)
        df["q(x)"] = df["q(x)"].astype(float)
        df["l(x)"] = df["l(x)"].astype(float)
        df["d(x)"] = df["d(x)"].astype(float)

    return {"Hombres": hombres, "Mujeres": mujeres}

def construir_conmutados(df_base: pd.DataFrame, i: float):
    """
    Convención:
      D_x = v^x * l_x
      C_x = v^(x+1) * d_x  (muerte pagadera al final del año de muerte)
      N_x = sum_{k>=x} D_k
      M_x = sum_{k>=x} C_k
    """
    df = df_base.sort_values("x").copy().set_index("x")

    v = 1.0 / (1.0 + i)
    x_vals = df.index.to_numpy(dtype=int)
    lx = df["l(x)"].to_numpy(dtype=float)
    dx = df["d(x)"].to_numpy(dtype=float)

    vx = v ** x_vals
    Dx = vx * lx
    Cx = (v ** (x_vals + 1)) * dx

    Nx = np.cumsum(Dx[::-1])[::-1]
    Mx = np.cumsum(Cx[::-1])[::-1]

    out = df.copy()
    out["v^x"] = vx
    out["Dx"] = Dx
    out["Cx"] = Cx
    out["Nx"] = Nx
    out["Mx"] = Mx
    return out

def check_age(tabla, *ages):
    omega = int(tabla.index.max())
    for a in ages:
        if a not in tabla.index:
            raise KeyError(f"La edad {a} no existe en la tabla. Rango: 0 a {omega}.")

def D(tabla, x): check_age(tabla, x); return float(tabla.loc[x, "Dx"])
def N(tabla, x): check_age(tabla, x); return float(tabla.loc[x, "Nx"])
def M(tabla, x): check_age(tabla, x); return float(tabla.loc[x, "Mx"])
def l_(tabla, x): check_age(tabla, x); return float(tabla.loc[x, "l(x)"])
def d_(tabla, x): check_age(tabla, x); return float(tabla.loc[x, "d(x)"])

# =======================================================
# Probabilidades útiles (desde l(x), d(x))
# =======================================================
def p_xt(tabla, x, t):
    """p(x;t) = l_{x+t} / l_x"""
    check_age(tabla, x, x + t)
    return l_(tabla, x + t) / l_(tabla, x)

def q_x_t1(tabla, x, t):
    """q(x;t;1) = d_{x+t} / l_x  (equivale a _t p_x q_{x+t})"""
    check_age(tabla, x, x + t)
    return d_(tabla, x + t) / l_(tabla, x)

def curtate_expectation(tabla, x):
    """e_x = sum_{t>=1} p(x;t) (curtata) hasta omega."""
    omega = int(tabla.index.max())
    T = omega - x
    if T <= 0:
        return 0.0
    return sum(p_xt(tabla, x, t) for t in range(1, T + 1))

def top_death_years(tabla, x, max_rows=10):
    """Top años de muerte más probables: P(muerte en (t,t+1]) = d_{x+t}/l_x."""
    omega = int(tabla.index.max())
    T = omega - x
    if T <= 0:
        return pd.DataFrame(columns=["Año", "Edad al inicio", "Probabilidad"])
    denom = l_(tabla, x)
    filas = []
    for t in range(0, T):
        filas.append((t + 1, x + t, d_(tabla, x + t) / denom))
    df = pd.DataFrame(filas, columns=["Año", "Edad al inicio", "Probabilidad"])
    return df.sort_values("Probabilidad", ascending=False).head(max_rows)

# =======================================================
# FUNCIONES BASE exigidas: E(x,t) y A(x,t,1)
# =======================================================
def v_factor(i: float) -> float:
    return 1.0 / (1.0 + i)

def E_xt(tabla, x, t, i: float):
    """
    E(x;t) = p(x;t) * v^t
    """
    v = v_factor(i)
    return p_xt(tabla, x, t) * (v ** t)

def A_x_t1(tabla, x, t, i: float):
    """
    A(x;t;1) = v * E(x;t) - E(x;t+1)
    OJO: en tu latex aparece A(x;t-1;1)= vE(x;t-1)-E(x;t); esto es lo mismo reindexando.
    Aquí lo implementamos como A(x;t;1) (muerte en (t, t+1]).
    """
    v = v_factor(i)
    return v * E_xt(tabla, x, t, i) - E_xt(tabla, x, t + 1, i)

# =======================================================
# Fórmulas clásicas (por unidad) usando conmutados
# =======================================================
def nE(tabla, x, n): return D(tabla, x + n) / D(tabla, x)

def A_vida_entera(tabla, x): return M(tabla, x) / D(tabla, x)
def A_temporal(tabla, x, n): return (M(tabla, x) - M(tabla, x + n)) / D(tabla, x)
def A_diferido_vida_entera(tabla, x, h): return M(tabla, x + h) / D(tabla, x)
def A_diferido_temporal(tabla, x, h, n): return (M(tabla, x + h) - M(tabla, x + h + n)) / D(tabla, x)

# Dotal mixto (endowment): A temporal + pure endowment
def A_dotal_mixto(tabla, x, n):
    """
    PPU por unidad (endowment n): (M_x - M_{x+n} + D_{x+n}) / D_x
    """
    return (M(tabla, x) - M(tabla, x + n) + D(tabla, x + n)) / D(tabla, x)

def a_due_vitalicia(tabla, x): return N(tabla, x) / D(tabla, x)
def a_due_temporal(tabla, x, n): return (N(tabla, x) - N(tabla, x + n)) / D(tabla, x)
def a_due_diferida_vitalicia(tabla, x, h): return N(tabla, x + h) / D(tabla, x)
def a_due_diferida_temporal(tabla, x, h, n): return (N(tabla, x + h) - N(tabla, x + h + n)) / D(tabla, x)

def a_immediate_vitalicia(tabla, x): return N(tabla, x + 1) / D(tabla, x)
def a_immediate_temporal(tabla, x, n): return (N(tabla, x + 1) - N(tabla, x + n + 1)) / D(tabla, x)

# =======================================================
# Fraccionarios (Woolhouse) para RENTAS
# =======================================================
def adj_k(k): return (k - 1) / (2 * k)

def a_due_temporal_fracc(tabla, x, n, k):
    return a_due_temporal(tabla, x, n) - adj_k(k) * (1 - nE(tabla, x, n))

def a_due_vitalicia_fracc(tabla, x, k):
    return a_due_vitalicia(tabla, x) - adj_k(k)

def a_due_diferida_temporal_fracc(tabla, x, h, n, k):
    hE = nE(tabla, x, h)
    hnE = nE(tabla, x, h + n)
    return a_due_diferida_temporal(tabla, x, h, n) - adj_k(k) * (hE - hnE)

# =======================================================
# Fraccionarios de MUERTE (como tu latex)
# A(x;...;k)= (sobre base E y a, con f(k)=i/j^(k))
# =======================================================
def j_nominal_k(i: float, k: int) -> float:
    """j^(k) = k((1+i)^(1/k)-1)"""
    if k <= 0:
        raise ValueError("k debe ser >= 1.")
    return k * ((1.0 + i) ** (1.0 / k) - 1.0)

def f_k(i: float, k: int) -> float:
    """f(k)= i / j^(k)"""
    j = j_nominal_k(i, k)
    if j <= 0:
        raise ValueError("j^(k) <= 0. Revisa i y k.")
    return i / j

def a_due_fracc(tabla, x, h, n, k):
    """ä^{(k)}(x;h;n) (anticipada) con Woolhouse según tus fórmulas de rentas."""
    if h == 0:
        if n is None:
            return a_due_vitalicia_fracc(tabla, x, k)
        return a_due_temporal_fracc(tabla, x, n, k)
    else:
        if n is None:
            # tu latex: ä(x;h;ω-x-h;k)= ä(x;h;ω-x-h) - (k-1)/(2k) E(x;h)
            # Aquí usamos: _h ä_x = N_{x+h}/D_x (anticipada diferida vitalicia)
            return a_due_diferida_vitalicia(tabla, x, h) - adj_k(k) * nE(tabla, x, h)
        return a_due_diferida_temporal_fracc(tabla, x, h, n, k)

def A_muerte_fracc(tabla, x, i, k, h=0, n=None):
    """
    Implementa literalmente tus fórmulas:

    Inmediato limitado:      A(x;0;n;k)=1-E(x;n)-f(k)a(x;0;n;k)
    Inmediato sin límite:    A(x;0;ω-x;k)=1-f(k)a(x;0;ω-x;k)
    Diferido limitado:       A(x;h;n;k)=E(x;h)-E(x;h+n)-f(k)a(x;h;n;k)
    Diferido sin límite:     A(x;h;ω-x-h;k)=E(x;h)-f(k)a(x;h;ω-x-h;k)

    Nota: aquí 'a' interpretado como ä (anticipada) fraccionaria conforme a tu bloque de rentas fraccionarias.
    """
    fk = f_k(i, k)
    if h < 0:
        raise ValueError("h debe ser >= 0.")
    if n is not None and n < 0:
        raise ValueError("n debe ser >= 0.")

    if h == 0 and n is not None:
        return 1.0 - E_xt(tabla, x, n, i) - fk * a_due_fracc(tabla, x, 0, n, k)

    if h == 0 and n is None:
        # whole life fracc
        return 1.0 - fk * a_due_fracc(tabla, x, 0, None, k)

    if h > 0 and n is not None:
        return E_xt(tabla, x, h, i) - E_xt(tabla, x, h + n, i) - fk * a_due_fracc(tabla, x, h, n, k)

    if h > 0 and n is None:
        return E_xt(tabla, x, h, i) - fk * a_due_fracc(tabla, x, h, None, k)

    raise ValueError("Caso fraccionario no reconocido.")

# =======================================================
# Seguros variables en progresión geométrica (por unidad)
# Ahora usando E(x,t) y A(x,t,1) como en tu latex
# =======================================================
def vida_geom_inmediato_limitado(tabla, x, n, i, r):
    return sum(((1 + r) ** t) * E_xt(tabla, x, t, i) for t in range(0, n))

def vida_geom_inmediato_sin_limite(tabla, x, i, r):
    omega = int(tabla.index.max())
    T = omega - x
    return sum(((1 + r) ** t) * E_xt(tabla, x, t, i) for t in range(0, T))

def vida_geom_diferido_limitado(tabla, x, h, n, i, r):
    return sum(((1 + r) ** (t - h)) * E_xt(tabla, x, t, i) for t in range(h, h + n))

def vida_geom_diferido_sin_limite(tabla, x, h, i, r):
    omega = int(tabla.index.max())
    T = omega - x
    return sum(((1 + r) ** (t - h)) * E_xt(tabla, x, t, i) for t in range(h, T))

def muerte_geom_inmediato_limitado(tabla, x, n, i, r):
    return sum(((1 + r) ** t) * A_x_t1(tabla, x, t, i) for t in range(0, n))

def muerte_geom_inmediato_sin_limite(tabla, x, i, r):
    omega = int(tabla.index.max())
    T = omega - x
    return sum(((1 + r) ** t) * A_x_t1(tabla, x, t, i) for t in range(0, T))

def muerte_geom_diferido_limitado(tabla, x, h, n, i, r):
    return sum(((1 + r) ** t) * A_x_t1(tabla, x, t + h, i) for t in range(0, n))

def muerte_geom_diferido_sin_limite(tabla, x, h, i, r):
    omega = int(tabla.index.max())
    T = omega - x - h
    return sum(((1 + r) ** t) * A_x_t1(tabla, x, t + h, i) for t in range(0, T))

# =======================================================
# Prima de Tarifa (PT) (re-hecha) + Loading
# Incluye lambda (liquidación) y separa % vs dinero
# =======================================================
def prima_tarifa_PV(
    B_unit: float,         # valor actuarial por unidad (A, nE, endowment, etc.)
    capital: float,        # C
    a_prem: float,         # ä para primas (pago anual anticipado)
    alpha: float, beta: float, gamma: float, delta: float, lamb: float
) -> float:
    """
    Ecuación PV (consistente y auditable):
      PT = [ C*B*(1+λ) + α*C + β*C*a_prem ] / [ (1-δ)*a_prem - γ ]

    Interpretación de inputs (por tus definiciones):
      α: gasto inicial sobre capital (proporción de C)
      β: gasto periódico sobre capital (proporción de C por cada pago; PV = β*C*a_prem)
      γ: gasto inicial sobre la prima de tarifa (proporción del primer PT)
      δ: gasto periódico sobre PT (proporción de cada pago; PV = δ*PT*a_prem)
      λ: liquidación del siniestro (proporción del beneficio; PV = λ*C*B_unit)
    """
    if a_prem <= 0:
        raise ValueError("ä de primas debe ser > 0.")
    denom = (1.0 - delta) * a_prem - gamma
    if denom <= 0:
        raise ValueError("Denominador <= 0 en PT. Revisa γ, δ y el plazo de primas.")
    num = capital * B_unit * (1.0 + lamb) + alpha * capital + beta * capital * a_prem
    return num / denom

def loading_from_PT_PP(PT: float, PP: float) -> float:
    if PT <= 0:
        raise ValueError("PT debe ser > 0 para calcular loading.")
    return (PT - PP) / PT

# =======================================================
# Cargar base y construir tabla conmutada
# =======================================================
try:
    base = cargar_base(archivo)
except Exception as e:
    st.error(
        f"No pude leer el Excel '{archivo}'. "
        f"Verifica que esté en la misma carpeta y que tenga hojas Hombres/Mujeres. Error: {e}"
    )
    st.stop()

sexo = st.sidebar.selectbox("Sexo", ["Hombres", "Mujeres"])
i_pct = st.sidebar.number_input("Tasa de interés i (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.25)
i = i_pct / 100.0

tabla = construir_conmutados(base[sexo], i=i)
omega = int(tabla.index.max())
st.caption(f"Tabla cargada: {sexo}. Edades: 0 a {omega}. Convención: Cx = v^(x+1)·d(x).")

# =======================================================
# Menú por categorías
# =======================================================
st.sidebar.markdown("---")
categoria = st.sidebar.selectbox(
    "Tipo de producto",
    [
        "Pensiones (Supervivencia)",
        "Seguros de Muerte",
        "Rentas / Anualidades",
        "Seguros variables (Progresión geométrica)",
        "Fraccionarios (según fórmulas)"
    ]
)

opciones = {
    "Pensiones (Supervivencia)": {
        "Dotal puro (capital diferido) E(x;t)": "SOB_DOTAL",
        "Dotal mixto (endowment)": "SOB_DOTAL_MIXTO",
    },
    "Seguros (Muerte)": {
        "Vida entera A(x,x,w)": "MUE_VIDA_ENTERA",
        "Temporal (plazo limitado) A(x,,x,x+n)": "MUE_TEMPORAL",
        "Diferido vida entera A(x,x+h,w)": "MUE_DIF_VIDA_ENTERA",
        "Diferido y temporal A(x,x+h,x+h+n)": "MUE_DIF_TEMPORAL",
    },
    "Pensiones (Rentas / Anualidades)": {
        "Anticipada vitalicia a(x,0,w-x)": "REN_AA_VITAL",
        "Anticipada temporal a(x,0,n)": "REN_AA_TEMP",
        "Anticipada diferida vitalicia a(x,x+h,w)": "REN_AA_DIF_VITAL",
        "Anticipada diferida temporal a(x,x+h,x+h+n-1)": "REN_AA_DIF_TEMP",
        "Vencida vitalicia a(x,1,w-x)": "REN_VEN_VITAL",
        "Vencida temporal a(x,1,n)": "REN_VEN_TEMP",
    },
    "Capitales múltiples variables (Progresión geométrica)": {
        "Vida variable: inmediato y plazo limitado a^{v}(x,0,n,r)": "VG_VIDA_IM_LIM",
        "Vida variable: inmediato sin límite a^{v}(x,0,w-x,r)": "VG_VIDA_IM_WL",
        "Vida variable: diferido y plazo limitado a^{v}(x,h,n,r)": "VG_VIDA_DIF_LIM",
        "Vida variable: diferido sin límite a^{v}(x,h,w-x-h,r)": "VG_VIDA_DIF_WL",
        "Muerte variable: inmediato y plazo limitado A^{v}(x,0,n,r)": "VG_MUE_IM_LIM",
        "Muerte variable: inmediato sin límite A^{v}(x,0,w-x,r)": "VG_MUE_IM_WL",
        "Muerte variable: diferido y plazo limitado A^{v}(x,h,n,r)": "VG_MUE_DIF_LIM",
        "Muerte variable: diferido sin límite A^{v}(x,h,w-x-h,r)": "VG_MUE_DIF_WL",
    },
    "Fraccionarios (según fórmulas)": {
        "Renta anticipada temporal fraccionaria a(x,0,n,k)": "FRAC_REN_AA_TEMP",
        "Renta anticipada vitalicia fraccionaria a(x,0,w-x,h)": "FRAC_REN_AA_VITAL",
        "Renta diferida temporal fraccionaria a(x,h,n,k)": "FRAC_REN_DIF_TEMP",
        "Renta diferida sin límite fraccionaria a(x,h,w-x-h,k)": "PEN_FRAC_REN_DIF_WL",
        "Seguro de muerte fraccionario: inmediato limitado A^{v}(x,0,n,k)": "FRAC_MUE_IM_LIM",
        "Seguro de muerte fraccionario: inmediato sin límite A^{v}(x,0,w-x,k)": "FRAC_MUE_IM_WL",
        "Seguro de muerte fraccionario: diferido limitado A^{v}(x,h,n,k)": "FRAC_MUE_DIF_LIM",
        "Seguro de muerte fraccionario: diferido sin límite A^{v}(x,h,w-x-h,k)": "FRAC_MUE_DIF_WL",
    }
}

nombre_producto = st.sidebar.selectbox("Producto", list(opciones[categoria].keys()))
producto = opciones[categoria][nombre_producto]

# =======================================================
# Inputs: separar CAPITAL vs RENTA
# =======================================================
st.sidebar.markdown("---")
x = int(st.sidebar.number_input("Edad x", min_value=0, max_value=omega, value=min(35, omega), step=1))

if categoria == "Rentas / Anualidades":
    R = float(st.sidebar.number_input("Renta por periodo (R)", min_value=0.0, value=1.0, step=1000.0))
    capital = None
else:
    capital = float(st.sidebar.number_input("Capital asegurado (C)", min_value=0.0, value=1.0, step=1000.0))
    R = None

# Inputs condicionales
n = None
h = None
k = None
r_pct = None

requiere_n = producto in {
    "SOB_DOTAL", "SOB_DOTAL_MIXTO",
    "MUE_TEMPORAL", "MUE_DIF_TEMPORAL",
    "REN_AA_TEMP", "REN_AA_DIF_TEMP", "REN_VEN_TEMP",
    "FRAC_REN_AA_TEMP", "FRAC_REN_DIF_TEMP",
    "VG_VIDA_IM_LIM", "VG_VIDA_DIF_LIM", "VG_MUE_IM_LIM", "VG_MUE_DIF_LIM",
    "FRAC_MUE_IM_LIM", "FRAC_MUE_DIF_LIM"
}
if requiere_n:
    n = int(st.sidebar.number_input("Plazo n (años)", min_value=1, max_value=omega, value=20 if omega >= 20 else omega, step=1))

requiere_h = producto in {
    "MUE_DIF_VIDA_ENTERA", "MUE_DIF_TEMPORAL",
    "REN_AA_DIF_VITAL", "REN_AA_DIF_TEMP",
    "PEN_FRAC_REN_DIF_WL"
    "FRAC_REN_DIF_TEMP",
    "VG_VIDA_DIF_LIM", "VG_VIDA_DIF_WL", "VG_MUE_DIF_LIM", "VG_MUE_DIF_WL",
    "FRAC_MUE_DIF_LIM", "FRAC_MUE_DIF_WL"
}
if requiere_h:
    h = int(st.sidebar.number_input("Diferimiento h (años)", min_value=0, max_value=omega, value=10 if omega >= 10 else omega, step=1))

requiere_k = producto in {
    "FRAC_REN_AA_TEMP", "FRAC_REN_AA_VITAL", "FRAC_REN_DIF_TEMP", "PEN_FRAC_REN_DIF_WL"
    "FRAC_MUE_IM_LIM", "FRAC_MUE_IM_WL", "FRAC_MUE_DIF_LIM", "FRAC_MUE_DIF_WL"
}
if requiere_k:
    k = int(st.sidebar.number_input("Frecuencia k (12 mensual, 2 semestral, etc.)", min_value=1, max_value=365, value=12, step=1))

if producto.startswith("VG_"):
    r_pct = float(st.sidebar.number_input("Razón de crecimiento r (%)", min_value=-99.0, max_value=500.0, value=2.0, step=0.25))

# ¿Tiene sentido hablar de primas?
es_seguro = categoria in {"Seguros de Sobrevivencia", "Seguros de Muerte", "Seguros variables (Progresión geométrica)", "Fraccionarios (según fórmulas)"}
es_renta = categoria == "Rentas / Anualidades"

# =======================================================
# m = plazo de pago (validado contra n = cobertura)
# =======================================================
st.sidebar.markdown("---")
m = None
m_efectivo = None

if es_seguro and not es_renta:
    st.sidebar.subheader("Primas netas (PPU → PPA)")
    usar_m = st.sidebar.checkbox("Pagar primas por m años (plazo de pago)", value=True)
    if usar_m:
        m = int(st.sidebar.number_input("m (años) de pago", min_value=1, max_value=omega, value=10, step=1))
    else:
        m = None  # vitalicio

# =======================================================
# Prima de Tarifa (PT) y loading (con lambda)
# =======================================================
st.sidebar.markdown("---")
calcular_PT = False
alpha = beta = gamma = delta = lamb = 0.0
umbral_loading = 30.0

if es_seguro and not es_renta:
    st.sidebar.subheader("Prima de tarifa (PT) y loading")
    calcular_PT = st.sidebar.checkbox("Calcular PT + loading", value=False)
    if calcular_PT:
        st.sidebar.caption("α,β,γ,δ,λ en %, según tu definición (sobre C, PT, beneficio).")

        alpha_pct = float(st.sidebar.number_input("α (% sobre C)", min_value=0.0, max_value=1000.0, value=0.0, step=0.5))
        beta_pct  = float(st.sidebar.number_input("β (% sobre C por pago)", min_value=0.0, max_value=1000.0, value=0.0, step=0.5))
        gamma_pct = float(st.sidebar.number_input("γ (% del primer PT)", min_value=0.0, max_value=1000.0, value=0.0, step=0.5))
        delta_pct = float(st.sidebar.number_input("δ (% de cada PT)", min_value=0.0, max_value=1000.0, value=0.0, step=0.5))
        lamb_pct  = float(st.sidebar.number_input("λ (% sobre el siniestro/beneficio)", min_value=0.0, max_value=1000.0, value=0.0, step=0.5))

        alpha = alpha_pct / 100.0
        beta  = beta_pct / 100.0
        gamma = gamma_pct / 100.0
        delta = delta_pct / 100.0
        lamb  = lamb_pct / 100.0

        umbral_loading = float(st.sidebar.number_input("Umbral de loading (%) para alertar", min_value=0.0, max_value=100.0, value=30.0, step=1.0))

# =======================================================
# Botón calcular
# =======================================================
if st.sidebar.button("Calcular"):
    try:
        # ------------------- Validaciones de rango -------------------
        if n is not None and x + n > omega:
            raise ValueError("x + n se sale del rango de la tabla.")
        if h is not None and x + h > omega:
            raise ValueError("x + h se sale del rango de la tabla.")
        if h is not None and n is not None and x + h + n > omega:
            raise ValueError("x + h + n se sale del rango de la tabla.")
        if m is not None and x + m > omega:
            raise ValueError("x + m se sale del rango de la tabla.")
        if h is not None and h >= (omega - x):
            raise ValueError("El diferimiento h deja sin años de vida en tabla (h >= ω - x).")


        # ------------------- Validación exigida: consistencia m vs n -------------------
        # Regla: si hay cobertura limitada n, el plazo de pago m no puede exceder n (para no "pagar después de que se acaba el seguro").
        # Si el usuario insiste, ajustamos y avisamos (sin reventar UX).
        m_efectivo = m
        if (m is not None) and (n is not None) and (m > n):
            m_efectivo = n
            st.warning(f"⚠️ Inconsistencia: m={m} > n={n}. Se ajustó automáticamente a m={m_efectivo} (no se pagan primas después de que termina la cobertura).")

        # ------------------- Valor por unidad + fórmula -------------------
        formula = ""
        val_unit = None  # por unidad monetaria del beneficio (C=1 o R=1)
        tipo_monto = "CAPITAL" if (capital is not None) else "RENTA"

        # === Seguros de sobrevivencia ===
        if producto == "SOB_DOTAL":
            val_unit = nE(tabla, x, n)
            formula = r"_nE_x = D_{x+n}/D_x"

        elif producto == "SOB_DOTAL_MIXTO":
            val_unit = A_dotal_mixto(tabla, x, n)
            formula = r"\text{Endowment: } (M_x - M_{x+n} + D_{x+n})/D_x"

        # === Seguros de muerte ===
        elif producto == "MUE_VIDA_ENTERA":
            val_unit = A_vida_entera(tabla, x)
            formula = r"A_x = M_x/D_x"

        elif producto == "MUE_TEMPORAL":
            val_unit = A_temporal(tabla, x, n)
            formula = r"A^1_{x:\overline{n}} = (M_x - M_{x+n})/D_x"

        elif producto == "MUE_DIF_VIDA_ENTERA":
            val_unit = A_diferido_vida_entera(tabla, x, h)
            formula = r"_{h}A_x = M_{x+h}/D_x"

        elif producto == "MUE_DIF_TEMPORAL":
            val_unit = A_diferido_temporal(tabla, x, h, n)
            formula = r"_{h}A^1_{x:\overline{n}} = (M_{x+h}-M_{x+h+n})/D_x"

        # === Rentas / anualidades ===
        elif producto == "REN_AA_VITAL":
            val_unit = a_due_vitalicia(tabla, x)
            formula = r"\ddot{a}_x = N_x/D_x"

        elif producto == "REN_AA_TEMP":
            val_unit = a_due_temporal(tabla, x, n)
            formula = r"\ddot{a}_{x:\overline{n}} = (N_x-N_{x+n})/D_x"

        elif producto == "REN_AA_DIF_VITAL":
            val_unit = a_due_diferida_vitalicia(tabla, x, h)
            formula = r"_{h}\ddot{a}_x = N_{x+h}/D_x"

        elif producto == "REN_AA_DIF_TEMP":
            val_unit = a_due_diferida_temporal(tabla, x, h, n)
            formula = r"_{h}\ddot{a}_{x:\overline{n}} = (N_{x+h}-N_{x+h+n})/D_x"

        elif producto == "REN_VEN_VITAL":
            val_unit = a_immediate_vitalicia(tabla, x)
            formula = r"a_x = N_{x+1}/D_x"

        elif producto == "REN_VEN_TEMP":
            val_unit = a_immediate_temporal(tabla, x, n)
            formula = r"a_{x:\overline{n}} = (N_{x+1}-N_{x+n+1})/D_x"

        # === Fraccionarios: rentas ===
        elif producto == "FRAC_REN_AA_TEMP":
            val_unit = a_due_temporal_fracc(tabla, x, n, k)
            formula = r"\ddot{a}^{(k)}_{x:\overline{n}} \approx \ddot{a}_{x:\overline{n}} - \frac{k-1}{2k}(1-_nE_x)"

        elif producto == "FRAC_REN_AA_VITAL":
            val_unit = a_due_vitalicia_fracc(tabla, x, k)
            formula = r"\ddot{a}^{(k)}_x \approx \ddot{a}_x - \frac{k-1}{2k}"

        elif producto == "FRAC_REN_DIF_TEMP":
            val_unit = a_due_diferida_temporal_fracc(tabla, x, h, n, k)
            formula = r"_{h}\ddot{a}^{(k)}_{x:\overline{n}} \approx _{h}\ddot{a}_{x:\overline{n}} - \frac{k-1}{2k}(_hE_x-_{h+n}E_x)"
            
        elif producto == "PEN_FRAC_REN_DIF_WL":
            val_unit = a_due_fracc(tabla, x, h, None, k)
            formula = r"a(x;h;\omega-x-h;k)=a(x;h;\omega-x-h)-\frac{k-1}{2k}\,E(x;h)"

        # === Fraccionarios: muerte (según tu latex) ===
        elif producto == "FRAC_MUE_IM_LIM":
            val_unit = A_muerte_fracc(tabla, x, i=i, k=k, h=0, n=n)
            formula = r"A(x;0;n;k)=1-E(x;n)-f(k)\,a(x;0;n;k)"

        elif producto == "FRAC_MUE_IM_WL":
            val_unit = A_muerte_fracc(tabla, x, i=i, k=k, h=0, n=None)
            formula = r"A(x;0;\omega-x;k)=1-f(k)\,a(x;0;\omega-x;k)"

        elif producto == "FRAC_MUE_DIF_LIM":
            val_unit = A_muerte_fracc(tabla, x, i=i, k=k, h=h, n=n)
            formula = r"A(x;h;n;k)=E(x;h)-E(x;h+n)-f(k)\,a(x;h;n;k)"

        elif producto == "FRAC_MUE_DIF_WL":
            val_unit = A_muerte_fracc(tabla, x, i=i, k=k, h=h, n=None)
            formula = r"A(x;h;\omega-x-h;k)=E(x;h)-f(k)\,a(x;h;\omega-x-h;k)"

        # === Variables geométricos ===
        elif producto.startswith("VG_"):
            r = r_pct / 100.0
            if producto == "VG_VIDA_IM_LIM":
                val_unit = vida_geom_inmediato_limitado(tabla, x, n, i, r)
                formula = r"\sum_{t=0}^{n-1} (1+r)^t\,E(x;t)"
            elif producto == "VG_VIDA_IM_WL":
                val_unit = vida_geom_inmediato_sin_limite(tabla, x, i, r)
                formula = r"\sum_{t=0}^{\omega-x-1} (1+r)^t\,E(x;t)"
            elif producto == "VG_VIDA_DIF_LIM":
                val_unit = vida_geom_diferido_limitado(tabla, x, h, n, i, r)
                formula = r"\sum_{t=h}^{h+n-1}(1+r)^{t-h}E(x;t)"
            elif producto == "VG_VIDA_DIF_WL":
                val_unit = vida_geom_diferido_sin_limite(tabla, x, h, i, r)
                formula = r"\sum_{t=h}^{\omega-x-1}(1+r)^{t-h}E(x;t)"
            elif producto == "VG_MUE_IM_LIM":
                val_unit = muerte_geom_inmediato_limitado(tabla, x, n, i, r)
                formula = r"\sum_{t=0}^{n-1}(1+r)^tA(x;t;1)"
            elif producto == "VG_MUE_IM_WL":
                val_unit = muerte_geom_inmediato_sin_limite(tabla, x, i, r)
                formula = r"\sum_{t=0}^{\omega-x-1}(1+r)^tA(x;t;1)"
            elif producto == "VG_MUE_DIF_LIM":
                val_unit = muerte_geom_diferido_limitado(tabla, x, h, n, i, r)
                formula = r"\sum_{t=0}^{n-1}(1+r)^tA(x;t+h;1)"
            elif producto == "VG_MUE_DIF_WL":
                val_unit = muerte_geom_diferido_sin_limite(tabla, x, h, i, r)
                formula = r"\sum_{t=0}^{\omega-x-h-1}(1+r)^tA(x;t+h;1)"
            else:
                raise ValueError("Producto geométrico no reconocido.")

        else:
            raise ValueError("Producto no reconocido.")

        # ------------------- Mostrar resultados base -------------------
        val_unit = float(val_unit)

        st.success("Resultado ✅")
        st.write(f"**Tipo:** {categoria}")
        st.write(f"**Producto:** {nombre_producto}")
        st.write(f"**Sexo/Hoja:** {sexo}")
        st.write(f"**i:** {i_pct:.4f}%  →  v = {1/(1+i):.10f}")

        if producto.startswith("VG_"):
            r = r_pct / 100.0
            st.write(f"**r:** {r_pct:.4f}%  →  1+r = {1+r:.10f}")

        # ------------------- Separación de monto (capital vs renta) -------------------
        if tipo_monto == "CAPITAL":
            total = float(capital * val_unit)
            st.markdown("### 🏢 Costo técnico (seguro de capital)")
            st.write(f"**Valor actuarial por unidad (C=1):** {val_unit:.10f}")
            st.write(f"**VPA total (C = {capital:,.2f}):** {total:,.6f}")
            st.caption("Interpretación: valor presente esperado del beneficio (según tabla e i).")
        else:
            total = float(R * val_unit)
            st.markdown("### 🏢 Costo técnico (renta / anualidad)")
            st.write(f"**Factor por unidad de renta (R=1):** {val_unit:.10f}")
            st.write(f"**VPA total (R = {R:,.2f} por periodo):** {total:,.6f}")
            st.caption("Interpretación: valor presente esperado de la renta contingente.")

        # ------------------- PRIMAS netas y PT (solo seguros de capital) -------------------
        if tipo_monto == "CAPITAL":
            PPU = total  # prima pura única
            st.markdown("### 💰 Primas netas")
            st.write(f"**PPU (prima pura única):** {PPU:,.6f}")

            # Factor de primas ä para pagar por m años o vitalicio
            if m_efectivo is not None:
                a_prem_anual = float(a_due_temporal(tabla, x, m_efectivo))
                a_prem_mensual = float(a_due_temporal_fracc(tabla, x, m_efectivo, 12))
                m_desc = f"{m_efectivo} años"
            else:
                a_prem_anual = float(a_due_vitalicia(tabla, x))
                a_prem_mensual = float(a_due_vitalicia_fracc(tabla, x, 12))
                m_desc = "de por vida"

            PPA = PPU / a_prem_anual if a_prem_anual > 0 else float("nan")
            PPM = PPU / (12.0 * a_prem_mensual) if a_prem_mensual > 0 else float("nan")

            st.write(f"**PPA (prima pura anual, pago anticipado, {m_desc}):** {PPA:,.6f}")
            st.write(f"**PPM (prima pura mensual aprox., {m_desc}):** {PPM:,.6f}")

            with st.expander("Detalles de factores de primas usados"):
                st.write(f"ä (anual anticipada) = {a_prem_anual:.10f}")
                st.write(f"ä^(12) (mensual, Woolhouse) = {a_prem_mensual:.10f}")

            # ---- PT + loading ----
            if calcular_PT:
                PT_anual = float(prima_tarifa_PV(
                    B_unit=val_unit,
                    capital=capital,
                    a_prem=a_prem_anual,
                    alpha=alpha, beta=beta, gamma=gamma, delta=delta, lamb=lamb
                ))
                L = float(loading_from_PT_PP(PT_anual, PPA))

                st.markdown("### 🧾 Prima de tarifa (PT) y carga (loading)")
                st.write(f"**PP (prima pura anual):** {PPA:,.6f}")
                st.write(f"**PT (prima de tarifa anual):** {PT_anual:,.6f}")
                st.write(f"**PT mensual (referencia, PT/12):** {(PT_anual/12.0):,.6f}")
                st.write(f"**Loading:** {100*L:.2f}%  ( (PT - PP) / PT )")

                if PT_anual < PPA:
                    st.error("⚠️ PT < PP: se vendería por debajo de la prima pura → pérdida esperada.")
                if (100*L) > umbral_loading:
                    st.warning(f"⚠️ Loading = {100*L:.2f}% supera el umbral de {umbral_loading:.0f}%.")

                with st.expander("Desglose de PT (PV)"):
                    st.write(f"Beneficio unitario B = {val_unit:.10f}")
                    st.write(f"λ = {lamb:.6f}, α = {alpha:.6f}, β = {beta:.6f}, γ = {gamma:.6f}, δ = {delta:.6f}")
                    st.write(f"ä primas = {a_prem_anual:.10f}")
                    st.latex(r"PT=\frac{C\cdot B\cdot(1+\lambda)+\alpha C+\beta C\cdot \ddot a}{(1-\delta)\ddot a-\gamma}")

        # ------------------- Extras (riesgo) -------------------
        st.markdown("### 🔎 Extras (para entender el riesgo)")
        col1, col2, col3 = st.columns(3)

        with col1:
            if n is not None and n > 0:
                npx = p_xt(tabla, x, n)
                st.write(f"**Prob. sobrevivir {n} años** (_{n}p_x): {npx:.6f}")
                st.write(f"**Prob. morir en {n} años** (_{n}q_x): {1 - npx:.6f}")
            else:
                st.write("**Plazo n:** (no aplica en este producto)")

        with col2:
            if h is not None and h > 0:
                hpx = p_xt(tabla, x, h)
                st.write(f"**Prob. sobrevivir al diferimiento {h}** (_{h}p_x): {hpx:.6f}")
            else:
                st.write("**Diferimiento h:** (no aplica en este producto)")

        with col3:
            ex = curtate_expectation(tabla, x)
            st.write(f"**Esperanza de vida curtata eₓ**: {ex:.4f} años")

        with st.expander("Top años más probables de muerte (según tabla)"):
            st.dataframe(top_death_years(tabla, x, max_rows=10), use_container_width=True)

        with st.expander("Ver fórmula usada"):
            st.latex(formula)

        with st.expander("Ver funciones base E y A(x;t;1) (ejemplo t=0,1,2)"):
            for t in [0, 1, 2]:
                if x + t + 1 <= omega:
                    st.write(f"t={t}: E(x;t)={E_xt(tabla, x, t, i):.10f} | A(x;t;1)={A_x_t1(tabla, x, t, i):.10f}")

        with st.expander("Ver conmutados en la edad x"):
            st.write(f"D_{x} = {D(tabla, x)}")
            st.write(f"N_{x} = {N(tabla, x)}")
            st.write(f"M_{x} = {M(tabla, x)}")

    except Exception as e:
        st.error(str(e))

# =======================================================
# Tabla preview
# =======================================================
st.subheader("Tabla (preview)")
st.dataframe(tabla[["q(x)", "l(x)", "d(x)", "v^x", "Dx", "Cx", "Nx", "Mx"]].head(25), use_container_width=True)
