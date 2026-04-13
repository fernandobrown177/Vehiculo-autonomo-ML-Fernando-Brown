"""
API de Operacionalización - Vehículo Autónomo Industrial
Integra los modelos de Fase 1 (SVD), Fase 2 (Random Forest) y Fase 3 (K-Means)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import joblib
import os

app = FastAPI(
    title="API Vehículo Autónomo Industrial",
    description="Sistema de ML para clasificación y optimización de rutas en planta de manufactura",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Carga de modelos ───────────────────────────────────────────────────────
# Se cargan al iniciar la app; si no existen se omiten con advertencia
models = {}

def cargar_modelos():
    archivos = {
        "kmeans":     "models/kmeans_model.joblib",
        "pca":        "models/pca_model.joblib",
        "rf":         "models/rf_model.joblib",
        "preprocess": "models/preprocessor.joblib",
        "preprocess_cluster": "models/preprocessor_cluster.joblib",
    }
    for nombre, ruta in archivos.items():
        if os.path.exists(ruta):
            models[nombre] = joblib.load(ruta)
            print(f"✅ Modelo cargado: {nombre}")
        else:
            print(f"⚠️  Modelo no encontrado: {ruta}")

cargar_modelos()

# ─── Schemas ────────────────────────────────────────────────────────────────

class ProductoInput(BaseModel):
    embalaje: int                  # 1-4
    ancho_cm: float
    largo_cm: float
    alto_cm: float
    peso_kg: float
    procedencia: str               # "A", "B", "C", "D"
    manipulacion: str              # "fragil", "normal"
    temperatura: str               # "ambiente", "refrigerado"
    protocolo: Optional[str] = None  # "Protocolo_1", "Protocolo_2"

class ProductosBatch(BaseModel):
    productos: List[ProductoInput]

DEPOSITO_MAP = {0: "Deposito_1", 1: "Deposito_2", 2: "Deposito_3", 3: "Deposito_4"}
PROTOCOLO_MAP = {0: "Protocolo_1", 1: "Protocolo_2"}

# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", tags=["Estado"])
def root():
    return {
        "status": "activo",
        "modelos_disponibles": list(models.keys()),
        "endpoints": ["/predecir-deposito", "/predecir-protocolo", "/ruta-optima", "/health"]
    }

@app.get("/health", tags=["Estado"])
def health():
    return {"status": "ok", "modelos": {k: "cargado" for k in models}}


@app.post("/predecir-deposito", tags=["Fase 3 - Clustering"])
def predecir_deposito(producto: ProductoInput):
    """
    Usa K-Means para asignar el producto a uno de los 4 depósitos del área.
    Corresponde al modelo no supervisado de Fase 3.
    """
    if "kmeans" not in models or "preprocess_cluster" not in models or "pca" not in models:
        raise HTTPException(status_code=503, detail="Modelos de clustering no disponibles")

    import pandas as pd
    df = pd.DataFrame([producto.dict()])

    X_prep = models["preprocess_cluster"].transform(df)
    if hasattr(X_prep, "toarray"):
        X_prep = X_prep.toarray()

    X_pca = models["pca"].transform(X_prep)
    cluster = int(models["kmeans"].predict(X_pca)[0])
    deposito = DEPOSITO_MAP[cluster]

    return {
        "deposito_asignado": deposito,
        "cluster_id": cluster,
        "producto": producto.dict()
    }


@app.post("/predecir-protocolo", tags=["Fase 2 - Supervisado"])
def predecir_protocolo(producto: ProductoInput):
    """
    Usa Random Forest para predecir el protocolo de manejo (supervisado, Fase 2).
    """
    if "rf" not in models or "preprocess" not in models:
        raise HTTPException(status_code=503, detail="Modelos supervisados no disponibles")

    import pandas as pd
    df = pd.DataFrame([producto.dict()])
    df = df.drop(columns=["protocolo"], errors="ignore")

    X_prep = models["preprocess"].transform(df)
    pred = int(models["rf"].predict(X_prep)[0])
    proba = models["rf"].predict_proba(X_prep)[0].tolist()

    return {
        "protocolo_predicho": PROTOCOLO_MAP.get(pred, str(pred)),
        "probabilidades": {PROTOCOLO_MAP[i]: round(p, 4) for i, p in enumerate(proba)},
        "producto": producto.dict()
    }


@app.post("/predecir-batch", tags=["Batch"])
def predecir_batch(batch: ProductosBatch):
    """
    Procesa múltiples productos en una sola llamada (depósito + protocolo).
    """
    resultados = []
    for p in batch.productos:
        try:
            dep = predecir_deposito(p)
            prot = predecir_protocolo(p)
            resultados.append({
                "deposito": dep["deposito_asignado"],
                "protocolo": prot["protocolo_predicho"],
                "producto": p.dict()
            })
        except Exception as e:
            resultados.append({"error": str(e), "producto": p.dict()})
    return {"total": len(resultados), "resultados": resultados}


@app.get("/ruta-optima", tags=["Fase 2 - Optimización"])
def ruta_optima_demo():
    """
    Ejemplo de la lógica de ruta óptima Manhattan de Fase 2.
    """
    DESTINOS = {
        ("A", "Protocolo_1"): (1, 0),
        ("A", "Protocolo_2"): (4, 0),
        ("B", "Protocolo_1"): (4, 5),
        ("B", "Protocolo_2"): (7, 5),
        ("C", "Protocolo_1"): (10, 5),
        ("C", "Protocolo_2"): (10, 3),
    }

    def manhattan(p, q):
        return abs(p[0]-q[0]) + abs(p[1]-q[1])

    origen = (0, 3)
    ejemplo = [("A","Protocolo_1"), ("B","Protocolo_2"), ("C","Protocolo_1")]
    ruta = [DESTINOS[d] for d in ejemplo]
    distancia = sum(manhattan(ruta[i], ruta[i+1]) for i in range(len(ruta)-1))
    distancia += manhattan(origen, ruta[0]) + manhattan(ruta[-1], origen)

    return {
        "origen": origen,
        "destinos": ejemplo,
        "coordenadas": ruta,
        "distancia_total_metros": distancia * 100,
        "nota": "Cada movimiento equivale a 1 metro (100cm)"
    }
