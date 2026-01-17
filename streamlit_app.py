import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="Riesgo de crédito — Probabilidad de impago", layout="wide")

@st.cache_data
def simular_datos(n=4000, seed=42):
    rng=np.random.default_rng(seed)
    edad=rng.integers(18,75,n)
    ingresos=rng.normal(32000,14000,n).clip(6000,180000)
    importe=rng.normal(18000,12000,n).clip(500,120000)
    plazo=rng.choice([12,24,36,48,60,72],n,p=[0.12,0.18,0.25,0.18,0.20,0.07])
    tin=rng.normal(7.5,3.0,n).clip(1.5,22)
    empleo=rng.choice(["Asalariado","Autónomo","Funcionario","Desempleado"],n,p=[0.56,0.25,0.12,0.07])
    provincia=rng.choice(["Girona","Barcelona","Tarragona","Lleida","Otro"],n,p=[0.18,0.40,0.12,0.08,0.22])
    impagos_prev=np.clip(rng.poisson(0.25,n),0,5)
    dti=np.clip((importe/(ingresos+1))*rng.normal(0.65,0.15,n),0.02,2.5)
    score=rng.normal(660,80,n).clip(300,850)

    logit=(-4.2+0.012*(importe/1000)+1.1*np.maximum(dti-0.45,0)+0.35*impagos_prev
           -0.004*(score-650)-0.00001*(ingresos-30000)
           +np.where(empleo=="Desempleado",1.2,0)
           +np.where(empleo=="Autónomo",0.25,0)
           +np.where(plazo>=72,0.3,0))
    prob=1/(1+np.exp(-logit))
    y=rng.binomial(1,prob)

    df=pd.DataFrame({
        "edad":edad,
        "ingresos_anuales":ingresos.round(0),
        "importe_prestamo":importe.round(0),
        "plazo_meses":plazo,
        "tipo_interes":tin.round(2),
        "situacion_laboral":empleo,
        "provincia":provincia,
        "impagos_previos":impagos_prev,
        "dti":dti.round(3),
        "credit_score":score.round(0),
        "default":y
    })
    for col in ["ingresos_anuales","credit_score","situacion_laboral"]:
        df.loc[rng.choice(n,int(0.03*n),replace=False),col]=np.nan
    return df

@st.cache_resource
def entrenar_modelo(df: pd.DataFrame):
    X=df.drop(columns=["default"])
    y=df["default"]

    num=["edad","ingresos_anuales","importe_prestamo","plazo_meses","tipo_interes","impagos_previos","dti","credit_score"]
    cat=["situacion_laboral","provincia"]

    pre=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median"))]),num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                         ("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat)
    ])

    modelo=Pipeline([("pre",pre),("clf",LogisticRegression(max_iter=900))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    modelo.fit(Xtr,ytr)

    auc=roc_auc_score(yte, modelo.predict_proba(Xte)[:,1])
    return modelo, float(auc)

st.title("Riesgo de crédito — Probabilidad de impago (Demo)")
st.caption("Portafolio: modelo entrenado con datos sintéticos (sustituible por datos reales).")

df = simular_datos()
modelo, auc = entrenar_modelo(df)

colA, colB = st.columns([1,1])

with colA:
    st.subheader("Datos del solicitante")
    edad = st.slider("Edad", 18, 75, 34)
    ingresos = st.number_input("Ingresos anuales (€)", min_value=0, max_value=300000, value=42000, step=1000)
    importe = st.number_input("Importe del préstamo (€)", min_value=0, max_value=200000, value=15000, step=500)
    plazo = st.selectbox("Plazo (meses)", [12,24,36,48,60,72], index=3)
    tin = st.slider("Tipo de interés (%)", 1.5, 22.0, 7.2)
    impagos_prev = st.slider("Impagos previos", 0, 5, 0)
    dti = st.slider("DTI (deuda/ingresos)", 0.0, 2.5, 0.29, step=0.01)
    score = st.slider("Credit score", 300, 850, 710)
    empleo = st.selectbox("Situación laboral", ["Asalariado","Autónomo","Funcionario","Desempleado"], index=0)
    provincia = st.selectbox("Provincia", ["Girona","Barcelona","Tarragona","Lleida","Otro"], index=1)

    fila = pd.DataFrame([{
        "edad": edad,
        "ingresos_anuales": ingresos,
        "importe_prestamo": importe,
        "plazo_meses": plazo,
        "tipo_interes": tin,
        "impagos_previos": impagos_prev,
        "dti": dti,
        "credit_score": score,
        "situacion_laboral": empleo,
        "provincia": provincia
    }])

    if st.button("Calcular probabilidad de impago", type="primary"):
        p = modelo.predict_proba(fila)[:,1][0]
        st.metric("Probabilidad de impago", f"{p:.2%}")
        st.write("Calidad del modelo (ROC-AUC en test):", f"**{auc:.3f}**")

with colB:
    st.subheader("Vista previa de datos (sintéticos)")
    st.dataframe(df.sample(20, random_state=1), height=420)
    st.info("En un caso real, se sustituye el generador sintético por datos reales manteniendo el pipeline.")
