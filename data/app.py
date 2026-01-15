import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Calculadora Actuarial", layout="wide")
st.title("📠 Calculadora Actuarial")

# =======================================================
# Utils: normalización para evitar KeyError por strings raros
# =======================================================
def norm(s: str) -> str:
    return " ".join(str(s).strip().split())

# =======================================================
# Auto-detect del Excel (deploy-friendly)
# =======================================================
def detectar_archivo_por_defecto():
    folder = Path(__file__).parent

    preferidos = [
        folder / "data" / "tabla_mortalidad.xlsx",
        folder / "tabla_mortalidad.xlsx",
    ]
    for p in preferidos:
        if p.exists():
            return str(p)

    for ext in ("xlsx", "xlsm", "xls"):
        candidatos = list(folder.rglob(f"*mortalidad*.{ext}"))
        if candidatos:
            return str(candidatos[0])

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
    """E(x;t) = p(x;t) * v^t"""
    v = v_factor(i)
    return p_xt(tabla, x, t) * (v ** t)

def A_x_t1(tabla, x, t, i: float):
    """
    A(x;t;1) = v * E(x;t) - E(x;t+1)
    (muerte en (t, t+1])
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

def A_dotal_mixto(tabla, x, n):
    """Endowment n: (M_x - M_{x+n} + D_{x+n}) / D_x"""
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
    """
    ä^{(k)}(x;h;n) (anticipada) con Woolhouse y tu caso 'sin límite' diferido.
    n=None representa ω-x-h.
    """
    if h == 0:
        if n is None:
            return a_due_vitalicia_fracc(tabla, x, k)
        return a_due_temporal_fracc(tabla, x, n, k)

    # h > 0
    if n is None:
        return a_due_diferida_vitalicia(tabla, x, h) - adj_k(k) * nE(tabla, x, h)
    return a_due_diferida_temporal_fracc(tabla, x, h, n, k)

def A_muerte_fracc(tabla, x, i, k, h=0, n=None):
    """
    Inmediato limitado:      A(x;0;n;k)=1-E(x;n)-f(k)a(x;0;n;k)
    Inmediato sin límite:    A(x;0;ω-x;k)=1-f(k)a(x;0;ω-x;k)
    Diferido limitado:       A(x;h;n;k)=E(x;h)-E(x;h+n)-f(k)a(x;h;n;k)
    Diferido sin límite:     A(x;h;ω-x-h;k)=E(x;h)-f(k)a(x;h;ω-x-h;k)
    """
    fk = f_k(i, k)
    if h < 0:
        raise ValueError("h debe ser >= 0.")
    if n is not None and n < 0:
        raise ValueError("n debe ser >= 0.")

    if h == 0 and n is not None:
        return 1.0 - E_xt(tabla, x, n, i) - fk * a_due_fracc(tabla, x, 0, n, k)

    if h == 0 and n is None:
        return 1.0 - fk * a_due_fracc(tabla, x, 0, None, k)

    if h > 0 and n is not None:
        return E_xt(tabla, x, h, i) - E_xt(tabla, x, h + n, i) - fk * a_due_fracc(tabla, x, h, n, k)

    if h > 0 and n is None:
        return E_xt(tabla, x, h, i) - fk * a_due_fracc(tabla, x, h, None, k)

    raise ValueError("Caso fraccionario no reconocido.")

# =======================================================
# Capitales múltiples variables (geométrica) con E y A(x;t;1)
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
# Prima de Tarifa (PT) con lambda
# =======================================================
def prima_tarifa_PV(B_unit, capital, a_prem, alpha, beta, gamma, delta, lamb):
    """
    PT = [ C*B*(1+λ) + α*C + β*C*a_prem ] / [ (1-δ)*a_prem - γ ]
    α,β,γ,δ,λ se pasan ya en decimales (p.ej. 0.05 = 5%).
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
# NUEVA FUNCIÓN: MOSTRAR RESUMEN (Pegada aquí)
# =======================================================
def mostrar_resumen_poliza(
    *,
    producto: str,
    x: int,
    n: int | None,
    h: int | None,
    m: int | None,
    k: int | None,
    capital: float | None,
    R: float | None,
    prima_anual: float,
    r_pct: float | None,
) -> None:
    # =========================
    # Helpers
    # =========================
    def money(v: float | None) -> str:
        if v is None:
            return "**$0.00**"
        return f"**${v:,.2f}**"

    def years(t: int | None, *, default_lifetime: str = "de por vida") -> str:
        if t is None:
            return f"**{default_lifetime}**"
        if t == 0:
            return "**0 años**"
        return f"**{t} años**"

    def edad_futura(base: int, delta: int | None) -> str:
        if delta is None:
            return f"**{base}**"
        return f"**{base + delta}**"

    def invers_text() -> str:
        # m controla pago único vs periódico
        if m is None or m == 0:
            return f"💰 **Inversión:** Usted realiza un **pago único** de {money(prima_anual)} hoy."
        return f"💰 **Inversión:** Usted realiza **pagos anuales** de {money(prima_anual)} durante {years(m)}."

    def nota_var() -> str:
        if producto.startswith("VG_") and r_pct is not None:
            return f"\n\n📈 **Nota de indexación:** Este producto es indexado. El beneficio crece un **{r_pct:.2f}% anual** para proteger su poder adquisitivo."
        return ""

    def nota_fracc() -> str:
        if ("FRAC_" in producto) or (k is not None):
            return f"\n\n🧾 **Nota de frecuencia:** Cálculos ajustados a frecuencia **k={k}** (pagos fraccionados)."
        return ""

    # =========================
    # Normalización de inputs
    # =========================
    n_eff = n
    h_eff = 0 if (h is None) else h

    # =========================
    # Plantillas por grupo
    # =========================
    # GRUPO 1: AHORRO / SUPERVIVENCIA (PEN_SUP)
    if producto == "SOB_DOTAL":
        # Dotal puro: si vive cobra; si muere pierde
        txt = (
            "📄 **Nota Legal Simplificada — Propuesta de Seguro**\n\n"
            "🎯 **Concepto:** Plan de Ahorro Objetivo (Sin devolución por fallecimiento).\n\n"
            f"{invers_text()}\n\n"
            f"✅ **Garantía de Vida:** Si usted vive hasta terminar el plazo de {years(n_eff)} recibirá el capital total de {money(capital)} para cumplir su meta.\n\n"
            "🛡️ **Garantía de Fallecimiento:** ⚠️ Atención: Este es un producto de ahorro puro. "
            "Si usted fallece antes de finalizar el plazo, **no hay devolución de primas ni pago de capital**. "
            "Todo el fondo queda en la mutualidad para beneficiar a los sobrevivientes."
        )
        st.success(txt + nota_var() + nota_fracc())
        return

    if producto == "SOB_DOTAL_MIXTO":
        # Endowment: paga si vive al final o si muere antes
        txt = (
            "📄 **Nota Legal Simplificada — Propuesta de Seguro**\n\n"
            "🧩 **Concepto:** Plan Integral: Ahorro Garantizado + Protección Familiar.\n\n"
            f"{invers_text()}\n\n"
            f"✅ **Garantía de Vida:** Al finalizar el plazo de {years(n_eff)}, si usted vive, recibirá íntegramente {money(capital)}.\n\n"
            f"🛡️ **Garantía de Fallecimiento:** Si llegara a faltar en cualquier momento antes de finalizar el plazo, "
            f"sus beneficiarios recibirán la suma de {money(capital)} de forma inmediata. "
            "La meta financiera se cumple **sí o sí**."
        )
        st.success(txt + nota_var() + nota_fracc())
        return

    # GRUPO 2: SEGUROS DE MUERTE (SEG_MUE)
    if producto == "MUE_VIDA_ENTERA":
        txt = (
            "📄 **Nota Legal Simplificada — Propuesta de Seguro**\n\n"
            "🏛️ **Concepto:** Legado Patrimonial Garantizado.\n\n"
            f"{invers_text()}\n\n"
            "✅ **Garantía de Vida:** Este seguro no tiene vencimiento. Mientras usted viva, mantiene la cobertura vigente.\n\n"
            f"🛡️ **Garantía de Fallecimiento:** En el momento que ocurra su fallecimiento (hoy o dentro de muchos años), "
            f"pagaremos {money(capital)} a sus herederos."
        )
        st.success(txt + nota_var() + nota_fracc())
        return

    if producto == "MUE_TEMPORAL":
        txt = (
            "📄 **Nota Legal Simplificada — Propuesta de Seguro**\n\n"
            "🛡️ **Concepto:** Protección Pura por Plazo Definido.\n\n"
            f"{invers_text()}\n\n"
            f"✅ **Garantía de Vida:** Si usted sobrevive a {years(n_eff)}, la póliza termina **sin valor de rescate** "
            "(funciona como un seguro de auto: usted pagó por la tranquilidad durante el periodo).\n\n"
            f"🛡️ **Garantía de Fallecimiento:** Si fallece dentro de los próximos {years(n_eff)}, "
            f"sus beneficiarios cobran {money(capital)}."
        )
        st.success(txt + nota_var() + nota_fracc())
        return

    if producto.startswith("MUE_DIF_"):
        # Diferidos: vida entera diferido / temporal diferido
        # Para temporal diferido, usa n; para vida entera diferido, n puede ser None
        txt = (
            "📄 **Nota Legal Simplificada — Propuesta de Seguro**\n\n"
            "⏳ **Concepto:** Protección a Futuro (con periodo de espera).\n\n"
            f"{invers_text()}\n\n"
            f"⏳ **Garantía de Vida:** Usted cuenta con un periodo de espera de {years(h_eff)} donde **no hay cobertura**.\n\n"
            f"🛡️ **Garantía de Fallecimiento:** La cobertura de {money(capital)} se activa únicamente si el fallecimiento "
            f"ocurre **después** de los primeros {years(h_eff)} de espera."
        )
        st.info(txt + nota_var() + nota_fracc())
        return

    # GRUPO 3: PENSIONES / RENTAS (PEN_REN)
    # Nota: prima_anual aquí es el capital/fondo para comprar la renta
    if producto.endswith("_VITAL") and producto.startswith("REN_"):
        # Vitalicias: anticipada o vencida
        inicio = "comenzando **hoy**" if "AA" in producto else "comenzando **al final del primer periodo**"
        txt = (
            "📄 **Nota Legal Simplificada — Propuesta de Pensión**\n\n"
            "🧓 **Concepto:** Jubilación Vitalicia Garantizada.\n\n"
            f"💰 **Aporte / Capital (CCI):** A cambio de su capital acumulado de {money(prima_anual)}.\n\n"
            f"✅ **Beneficio (Renta):** Nos comprometemos a pagarle una pensión de {money(R)} {inicio}.\n\n"
            "🛡️ **Duración:** Estos pagos están garantizados **de por vida**, sin importar cuánto tiempo viva usted."
        )
        st.success(txt + nota_fracc())
        return

    if producto.endswith("_TEMP") and producto.startswith("REN_"):
        inicio = "comenzando **hoy**" if "AA" in producto else "comenzando **al final del primer periodo**"
        txt = (
            "📄 **Nota Legal Simplificada — Propuesta de Pensión**\n\n"
            "📆 **Concepto:** Renta por Plazo Fijo.\n\n"
            f"💰 **Aporte / Capital (CCI):** A cambio de su capital acumulado de {money(prima_anual)}.\n\n"
            f"✅ **Beneficio:** Recibirá una pensión de {money(R)} {inicio} durante los próximos {years(n_eff)}.\n\n"
            f"⛔ **Condición:** Los pagos cesan si usted fallece o cuando se cumple el plazo de {years(n_eff)}, "
            "lo que ocurra primero."
        )
        st.info(txt + nota_fracc())
        return

    if "_DIF_" in producto and producto.startswith("REN_"):
        # Rentas diferidas (vital o temp, anticipada)
        inicio = "desde el inicio del periodo" if "AA" in producto else "al final del periodo"
        txt = (
            "📄 **Nota Legal Simplificada — Propuesta de Pensión**\n\n"
            "⏳ **Concepto:** Jubilación Planificada a Futuro.\n\n"
            f"💰 **Aporte / Capital (CCI):** Usted aporta el capital hoy: {money(prima_anual)}.\n\n"
            f"⏳ **Mecánica:** Los pagos de la pensión {money(R)} comenzarán dentro de {years(h_eff)} "
            f"(cuando usted tenga {edad_futura(x, h_eff)} años), pagando {inicio}.\n\n"
            "🛡️ **Nota:** Para que exista el pago, usted debe estar con vida al inicio de la renta."
        )
        st.info(txt + nota_fracc())
        return

    # GRUPO 4: CASOS ESPECIALES
    # CAP_VAR (VG_) y FRACC ya añaden nota; si llega aquí, damos resumen genérico.
    txt = (
        "📄 **Nota Legal Simplificada — Resumen del Producto**\n\n"
        f"💼 **Producto:** **{producto}**\n\n"
        f"{invers_text()}\n\n"
        f"⏳ **Plazo (n):** {years(n_eff)}\n"
        f"⏳ **Diferimiento (h):** {years(h_eff)}\n\n"
        "✅ **Nota:** Este resumen es una traducción amigable de resultados técnicos. "
        "Para condiciones contractuales exactas, se debe emitir la póliza formal."
    )
    st.info(txt + nota_var() + nota_fracc())

# =======================================================
# Cargar base y construir conmutados
# =======================================================
try:
    base = cargar_base(archivo)
except Exception as e:
    st.error(
        f"No pude leer el Excel '{archivo}'. "
        f"Verifica que el archivo exista en el repo (root o /data) y tenga hojas Hombres/Mujeres. Error: {e}"
    )
    st.stop()

sexo = st.sidebar.selectbox("Sexo", ["Hombres", "Mujeres"], key="sexo")
i_pct = st.sidebar.number_input("Tasa de interés i (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.25, key="i_pct")
i = i_pct / 100.0

tabla = construir_conmutados(base[sexo], i=i)
omega = int(tabla.index.max())
st.caption(f"Tabla cargada: {sexo}. Edades: 0 a {omega}. Convención: Cx = v^(x+1)·d(x).")

# =======================================================
# Menú robusto (sin KeyError): claves internas estables
# =======================================================
st.sidebar.markdown("---")

CATS = {
    "PEN_SUP": "Ahorro y Supervivencia (Dotales)",
    "SEG_MUE": "Seguros de Vida (Muerte)",
    "PEN_REN": "Pensiones y Rentas (Anualidades)",
    "CAP_VAR": "Productos con Aumento Anual (Indexados/Crecientes)",
    "FRACC":   "Pagos Frecuentes (Mensuales, Semestrales...)",
}

OPCIONES = {
    "PEN_SUP": {
        "Dotal puro (Si vive cobra, si muere pierde)  E(x,n)": "SOB_DOTAL",
        "Dotal mixto (Ahorro garantizado + Seguro)    Endow(x,n)": "SOB_DOTAL_MIXTO",
    },
    "SEG_MUE": {
        "Vida entera (Cubre siempre)             A(x)": "MUE_VIDA_ENTERA",
        "Temporal (Cubre n años)                 A(x, n)": "MUE_TEMPORAL",
        "Diferido vida entera (Espera h años)    A(x, h -> vida)": "MUE_DIF_VIDA_ENTERA",
        "Diferido y temporal (Espera h, cubre n) A(x, h, n)": "MUE_DIF_TEMPORAL",
    },
    "PEN_REN": {
        "Anticipada vitalicia (De por vida)      ä(x)": "REN_AA_VITAL",
        "Anticipada temporal (Por n años)        ä(x, n)": "REN_AA_TEMP",
        "Anticipada diferida vitalicia           ä(x, h -> vida)": "REN_AA_DIF_VITAL",
        "Anticipada diferida temporal            ä(x, h, n)": "REN_AA_DIF_TEMP",
        "Vencida vitalicia (Pago a fin de año)   a(x)": "REN_VEN_VITAL",
        "Vencida temporal (Pago a fin de año)    a(x, n)": "REN_VEN_TEMP",
    },
    "CAP_VAR": {
        "Vida Creciente: Inmediato por n años": "VG_VIDA_IM_LIM",
        "Vida Creciente: Inmediato Vitalicio": "VG_VIDA_IM_WL",
        "Vida Creciente: Diferido por n años": "VG_VIDA_DIF_LIM",
        "Vida Creciente: Diferido Vitalicio": "VG_VIDA_DIF_WL",
        "Muerte Creciente: Inmediato por n años": "VG_MUE_IM_LIM",
        "Muerte Creciente: Inmediato Vitalicio": "VG_MUE_IM_WL",
        "Muerte Creciente: Diferido por n años": "VG_MUE_DIF_LIM",
        "Muerte Creciente: Diferido Vitalicio": "VG_MUE_DIF_WL",
    },
    "FRACC": {
        "Renta Anticipada Temporal (Mensual/k)   ä(x, n, k)": "FRAC_REN_AA_TEMP",
        "Renta Anticipada Vitalicia (Mensual/k)  ä(x, k)": "FRAC_REN_AA_VITAL",
        "Renta Diferida Temporal (Mensual/k)     ä(x, h, n, k)": "FRAC_REN_DIF_TEMP",
        "Renta Diferida Vitalicia (Mensual/k)    ä(x, h, k)": "PEN_FRAC_REN_DIF_WL",
        "Seguro Muerte: Inmediato Temporal       A(x, n, k)": "FRAC_MUE_IM_LIM",
        "Seguro Muerte: Inmediato Vitalicio      A(x, k)": "FRAC_MUE_IM_WL",
        "Seguro Muerte: Diferido Temporal        A(x, h, n, k)": "FRAC_MUE_DIF_LIM",
        "Seguro Muerte: Diferido Vitalicio       A(x, h, k)": "FRAC_MUE_DIF_WL",
    }
}

# Normaliza (por si acaso)
CATS = {k: norm(v) for k, v in CATS.items()}
OPCIONES = {k: {norm(nm): code for nm, code in v.items()} for k, v in OPCIONES.items()}

cat_keys = list(CATS.keys())

if "cat_key" not in st.session_state:
    st.session_state["cat_key"] = cat_keys[0]

# Si quedó una clave vieja, resetea
if st.session_state["cat_key"] not in OPCIONES:
    st.session_state["cat_key"] = cat_keys[0]

cat_key = st.sidebar.selectbox(
    "Tipo de producto",
    cat_keys,
    format_func=lambda k: CATS[k],
    key="cat_key",
)

prod_names = list(OPCIONES[cat_key].keys())

if "prod_name" not in st.session_state:
    st.session_state["prod_name"] = prod_names[0]

if st.session_state["prod_name"] not in OPCIONES[cat_key]:
    st.session_state["prod_name"] = prod_names[0]

prod_name = st.sidebar.selectbox(
    "Producto",
    prod_names,
    key="prod_name",
)

producto = OPCIONES[cat_key][prod_name]
categoria_label = CATS[cat_key]
nombre_producto = prod_name

# =======================================================
# Inputs: separar CAPITAL vs RENTA (según categoría interna)
# =======================================================
st.sidebar.markdown("---")
x = int(st.sidebar.number_input("Edad x", min_value=0, max_value=omega, value=min(35, omega), step=1, key="x"))

es_renta = (cat_key == "PEN_REN")
es_capital = not es_renta

if es_renta:
    R = float(st.sidebar.number_input("Renta por periodo (R)", min_value=0.0, value=1.0, step=1000.0, key="R"))
    capital = None
else:
    capital = float(st.sidebar.number_input("Capital asegurado (C)", min_value=0.0, value=1.0, step=1000.0, key="C"))
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
    "FRAC_MUE_IM_LIM", "FRAC_MUE_DIF_LIM",
}
if requiere_n:
    n = int(st.sidebar.number_input("Plazo n (años)", min_value=1, max_value=omega, value=20 if omega >= 20 else omega, step=1, key="n"))

requiere_h = producto in {
    "MUE_DIF_VIDA_ENTERA", "MUE_DIF_TEMPORAL",
    "REN_AA_DIF_VITAL", "REN_AA_DIF_TEMP",
    "PEN_FRAC_REN_DIF_WL",
    "FRAC_REN_DIF_TEMP",
    "VG_VIDA_DIF_LIM", "VG_VIDA_DIF_WL", "VG_MUE_DIF_LIM", "VG_MUE_DIF_WL",
    "FRAC_MUE_DIF_LIM", "FRAC_MUE_DIF_WL",
}
if requiere_h:
    h = int(st.sidebar.number_input("Diferimiento h (años)", min_value=0, max_value=omega, value=10 if omega >= 10 else omega, step=1, key="h"))

requiere_k = producto in {
    "FRAC_REN_AA_TEMP", "FRAC_REN_AA_VITAL", "FRAC_REN_DIF_TEMP", "PEN_FRAC_REN_DIF_WL",
    "FRAC_MUE_IM_LIM", "FRAC_MUE_IM_WL", "FRAC_MUE_DIF_LIM", "FRAC_MUE_DIF_WL",
}
if requiere_k:
    k = int(st.sidebar.number_input("Frecuencia k (12 mensual, 2 semestral, etc.)", min_value=1, max_value=365, value=12, step=1, key="k"))

if producto.startswith("VG_"):
    r_pct = float(st.sidebar.number_input("Razón de crecimiento r (%)", min_value=-99.0, max_value=500.0, value=2.0, step=0.25, key="r_pct"))

# ¿Tiene sentido hablar de primas?
es_seguro_capital = (cat_key in {"PEN_SUP", "SEG_MUE", "CAP_VAR", "FRACC"}) and (not es_renta)

# =======================================================
# m = plazo de pago (validado contra n = cobertura)
# =======================================================
st.sidebar.markdown("---")
m = None
m_efectivo = None

if es_seguro_capital:
    st.sidebar.subheader("Primas netas (PPU → PPA)")
    usar_m = st.sidebar.checkbox("Pagar primas por m años (plazo de pago)", value=True, key="usar_m")
    if usar_m:
        m = int(st.sidebar.number_input("m (años) de pago", min_value=1, max_value=omega, value=10, step=1, key="m"))
    else:
        m = None  # vitalicio

# =======================================================
# Prima de Tarifa (PT) y loading (con lambda)
# =======================================================
# =======================================================
# Prima de Tarifa (PT) y loading (Teoría de Gastos)
# =======================================================
st.sidebar.markdown("---")
calcular_PT = False
alpha = beta = gamma = delta = lamb = 0.0
umbral_loading = 30.0

if es_seguro_capital:
    st.sidebar.subheader("Prima de Tarifa (Gastos y Utilidad)")
    calcular_PT = st.sidebar.checkbox("Incluir Gastos (Loading)", value=False, key="calcular_PT")
    
    if calcular_PT:
        st.sidebar.info("Configure los gastos según la matriz de carga (Iniciales vs. Periódicos).")

        # 1. Gastos Iniciales (Alpha y Gamma)
        st.sidebar.markdown("##### 1. Gastos INICIALES (Sólo al inicio)")
        st.sidebar.caption("Gastos de colocación, emisión, adquisición.")
        alpha_pct = float(st.sidebar.number_input("α (% s/ Capital Asegurado)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, help="Gasto inicial sobre la Suma Asegurada (ej. Gastos de emisión)."))
        gamma_pct = float(st.sidebar.number_input("γ (% s/ Prima de Tarifa)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, help="Gasto inicial sobre la prima (ej. Comisión inicial agente)."))

        # 2. Gastos Periódicos (Beta y Delta)
        st.sidebar.markdown("##### 2. Gastos PERIÓDICOS (Durante el pago)")
        st.sidebar.caption("Gastos de administración, cobro, mantenimiento.")
        beta_pct  = float(st.sidebar.number_input("β (% s/ Capital Asegurado)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, help="Gasto anual sobre la Suma Asegurada (ej. Administración)."))
        delta_pct = float(st.sidebar.number_input("δ (% s/ Prima de Tarifa)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, help="Gasto anual sobre la prima (ej. Gastos de cobranza)."))

        # 3. Utilidad / Margen de seguridad (Lambda)
        st.sidebar.markdown("##### 3. Margen de Seguridad")
        lamb_pct  = float(st.sidebar.number_input("λ (% s/ Siniestro)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, help="Recargo de seguridad sobre la probabilidad de muerte."))

        # Conversión a decimales
        alpha = alpha_pct / 100.0
        beta  = beta_pct / 100.0
        gamma = gamma_pct / 100.0
        delta = delta_pct / 100.0
        lamb  = lamb_pct / 100.0

        st.sidebar.markdown("---")
        umbral_loading = float(st.sidebar.number_input("Alerta de Loading excesivo (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0))

# =======================================================
# Botón calcular
# =======================================================
if st.sidebar.button("Calcular", key="btn_calc"):
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

        # ------------------- Consistencia m vs n -------------------
        m_efectivo = m
        if (m is not None) and (n is not None) and (m > n):
            m_efectivo = n
            st.warning(f"⚠️ Inconsistencia: m={m} > n={n}. Ajustado a m={m_efectivo} (no se paga después de terminar la cobertura).")

        # ------------------- Valor por unidad + fórmula -------------------
        formula = ""
        val_unit = None  # por unidad (C=1 o R=1)

        # Pensiones (supervivencia)
        if producto == "SOB_DOTAL":
            val_unit = nE(tabla, x, n)
            formula = r"_nE_x = D_{x+n}/D_x"

        elif producto == "SOB_DOTAL_MIXTO":
            val_unit = A_dotal_mixto(tabla, x, n)
            formula = r"\text{Endowment: } (M_x - M_{x+n} + D_{x+n})/D_x"

        # Seguros (muerte)
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

        # Pensiones (rentas/anualidades)
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

        # Fraccionarios: rentas
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
            formula = r"_{h}\ddot{a}^{(k)}_x \approx _{h}\ddot{a}_x - \frac{k-1}{2k}\,_{h}E_x"

        # Fraccionarios: muerte
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

        # Variables geométricos
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

        val_unit = float(val_unit)

        # ------------------- Resultados base -------------------
        st.success("Resultado ✅")
        st.write(f"**Categoría:** {categoria_label}")
        st.write(f"**Producto:** {nombre_producto}")
        st.write(f"**Sexo/Hoja:** {sexo}")
        st.write(f"**i:** {i_pct:.4f}%  →  v = {1/(1+i):.10f}")

        if producto.startswith("VG_"):
            r = r_pct / 100.0
            st.write(f"**r:** {r_pct:.4f}%  →  1+r = {1+r:.10f}")

        # ------------------- Separación: capital vs renta -------------------
        if es_capital:
            total = float(capital * val_unit)
            st.markdown("### 🏢 Costo técnico (capital)")
            st.write(f"**Valor actuarial por unidad (C=1):** {val_unit:.10f}")
            # AQUÍ ESTÁ EL CAMBIO DE TEXTO:
            st.write(f"**VPA (Valor Presente Actuarial) total (C = {capital:,.2f}):** {total:,.6f}")
        else:
            total = float(R * val_unit)
            st.markdown("### 🏢 Costo técnico (renta / anualidad)")
            st.write(f"**Factor por unidad de renta (R=1):** {val_unit:.10f}")
            # AQUÍ ESTÁ EL CAMBIO DE TEXTO:
            st.write(f"**VPA (Valor Presente Actuarial) total (R = {R:,.2f} por periodo):** {total:,.6f}")

        # ------------------- Primas netas y PT (solo capital) -------------------
        if es_capital:
            PPU = total
            st.markdown("### 💰 Primas netas")
            st.write(f"**PPU (prima pura única):** {PPU:,.6f}")

            if es_seguro_capital:
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

                st.write(f"**PPA (prima pura anual, anticipada, {m_desc}):** {PPA:,.6f}")
                st.write(f"**PPM (prima pura mensual aprox., {m_desc}):** {PPM:,.6f}")

                with st.expander("Detalles de factores de primas usados"):
                    st.write(f"ä (anual anticipada) = {a_prem_anual:.10f}")
                    st.write(f"ä^(12) (mensual, Woolhouse) = {a_prem_mensual:.10f}")

                if calcular_PT:
                    # Cálculo matemático (Usamos tu función existente, que es matemáticamente equivalente)
                    PT_anual = float(prima_tarifa_PV(
                        B_unit=val_unit,
                        capital=capital,
                        a_prem=a_prem_anual,
                        alpha=alpha, beta=beta, gamma=gamma, delta=delta, lamb=lamb
                    ))
                    L = float(loading_from_PT_PP(PT_anual, PPA))

                    st.markdown("### 🧾 Prima de Tarifa (PT) y Desglose de Gastos")
                    
                    # Métricas principales
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Prima Pura (PPA)", f"{PPA:,.2f}")
                    c2.metric("Prima Tarifa (PT)", f"{PT_anual:,.2f}")
                    c3.metric("Loading (Carga)", f"{L*100:.2f}%", delta_color="inverse")

                    if PT_anual < PPA:
                        st.error("⚠️ Error Crítico: La Prima de Tarifa es menor que la Prima Pura. Revise los parámetros (delta muy alto).")
                    if (100*L) > umbral_loading:
                        st.warning(f"⚠️ Alerta: Los gastos representan el {100*L:.1f}% de la prima. Verifique si es competitivo.")

                    # =======================================================
                    # VISUALIZACIÓN TEÓRICA (EXACTA A TU CLASE)
                    # =======================================================
                    with st.expander("📘 Ver Fórmula de Cálculo y Matriz de Gastos (Teoría)", expanded=False):
                        st.markdown("#### 1. Matriz de Gastos Definida")
                        # Creamos un DataFrame para mostrar la matriz tal cual la imagen
                        df_gastos = pd.DataFrame(
                            [
                                [f"α = {alpha_pct}%", f"β = {beta_pct}%"],
                                [f"γ = {gamma_pct}%", f"δ = {delta_pct}%"]
                            ],
                            columns=["Iniciales (Una vez)", "Periódicos (Anuales)"],
                            index=["Sobre Capital (C)", "Sobre Prima (PT)"]
                        )
                        st.table(df_gastos)

                        st.markdown("#### 2. Fórmula de Cálculo")
                        st.write("Aplicando la fórmula de amortización de gastos:")
                        
                        # Preparamos los valores para mostrar en la fórmula
                        # a^-1 es 1 / a_prem_anual
                        inv_a = 1.0 / a_prem_anual if a_prem_anual > 0 else 0
                        
                        # LaTeX exacto a tu imagen
                        st.latex(r"""
                        PT_{x:n} = \frac{P(x;n) + \alpha \cdot \ddot{a}^{-1}_{x:0:n} + \beta}{1 - \gamma \cdot \ddot{a}^{-1}_{x:0:n} - \delta}
                        """)
                        
                        st.markdown("**Donde:**")
                        st.markdown(f"* **P(x;n):** Prima Pura Anual = `{PPA:,.4f}`")
                        st.markdown(f"* **Cuota de amortización** ($\\ddot{{a}}^{{-1}}$): $1 / {a_prem_anual:.4f} = $ `{inv_a:.6f}`")
                        st.markdown(f"* **Gastos s/Capital:** $\\alpha={alpha}, \\beta={beta}$")
                        st.markdown(f"* **Gastos s/Prima:** $\\gamma={gamma}, \\delta={delta}$")
                        
                        st.markdown("---")
                        st.markdown(f"**Cálculo del Loading:** $[ PT - PP ] / PT = $ `{L*100:.4f}%`")

        # ------------------- Extras -------------------
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

# CAMBIO EN EL TÍTULO DEL EXPANDER Y EXPLICACIÓN INTERNA
        with st.expander("Ver Factores Base: Riesgo Anual de Vida (E) vs Muerte (A)"):
            st.info(
                """
                **Explicación de los factores:**
                * **E(x, t):** Es el valor hoy de pagar $1 en el futuro (año t), **solo si la persona VIVE**. (Es la esencia de las Pensiones).
                * **A(x, t):** Es el costo hoy de asegurar $1 si la persona **MUERE** en ese año específico. (Es la esencia de los Seguros).
                """
            )
            for t in [0, 1, 2]:
                if x + t + 1 <= omega:
                    st.write(f"**Año {t} (Edad {x+t}):** E = {E_xt(tabla, x, t, i):.6f} (Si vive) | A = {A_x_t1(tabla, x, t, i):.6f} (Si muere)")

        with st.expander("Ver conmutados en la edad x"):
            st.write(f"D_{x} = {D(tabla, x)}")
            st.write(f"N_{x} = {N(tabla, x)}")
            st.write(f"M_{x} = {M(tabla, x)}")

        # =======================================================
        # ✅ BLOQUE DE LLAMADA CORREGIDO (NUEVO)
        # =======================================================
        st.markdown("---")
        st.subheader("📝 Resumen de su Póliza (Explicación Cliente)")

        if es_capital:
            # Regla: si hay plazo de pago (m_efectivo), la narrativa debe usar PPA (anual).
            # Si NO hay plazo de pago (m_efectivo es None o 0), usa PPU (pago único).
            prima_para_cliente = float(PPA) if (m_efectivo is not None and m_efectivo > 0) else float(PPU)

            mostrar_resumen_poliza(
                producto=producto,
                x=x,
                n=n,
                h=h,
                m=m_efectivo,   # esto controla el texto "pago único" vs "pagos anuales"
                k=k,
                capital=capital,
                R=None,
                prima_anual=prima_para_cliente,  # 👈 aquí va PPA o PPU según corresponda
                r_pct=r_pct,
            )
        else:
            # En rentas: prima_anual representa el capital/fondo necesario para comprar la pensión.
            mostrar_resumen_poliza(
                producto=producto,
                x=x,
                n=n,
                h=h,
                m=None,
                k=k,
                capital=None,
                R=R,
                prima_anual=float(total),  # total = capital/fondo requerido (según tu script)
                r_pct=None,
            )

    except Exception as e:
        st.error(str(e))

# =======================================================
# Tabla preview
# =======================================================
st.subheader("Tabla (preview)")
st.dataframe(tabla[["q(x)", "l(x)", "d(x)", "v^x", "Dx", "Cx", "Nx", "Mx"]].head(25), use_container_width=True)
