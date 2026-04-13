"""
tests/test_api.py
Tests automáticos para el pipeline CI/CD.
Se ejecutan con: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import app

client = TestClient(app)

PRODUCTO_EJEMPLO = {
    "embalaje": 2,
    "ancho_cm": 45.0,
    "largo_cm": 60.0,
    "alto_cm": 30.0,
    "peso_kg": 12.5,
    "procedencia": "B",
    "manipulacion": "normal",
    "temperatura": "ambiente",
    "protocolo": "Protocolo_2"
}

# ── Tests básicos ────────────────────────────────────────────────────────────

def test_root():
    """El endpoint raíz debe responder con status activo"""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "activo"
    assert "endpoints" in data

def test_health():
    """El endpoint de salud debe responder 200"""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

def test_ruta_optima():
    """El cálculo de ruta Manhattan debe retornar distancia positiva"""
    res = client.get("/ruta-optima")
    assert res.status_code == 200
    data = res.json()
    assert "distancia_total_metros" in data
    assert data["distancia_total_metros"] > 0

# ── Tests de modelos (solo si están cargados) ────────────────────────────────

def test_predecir_deposito_estructura():
    """Si el modelo existe, debe retornar deposito_asignado válido"""
    from main import models
    if "kmeans" not in models:
        pytest.skip("Modelo K-Means no disponible en este entorno")

    res = client.post("/predecir-deposito", json=PRODUCTO_EJEMPLO)
    assert res.status_code == 200
    data = res.json()
    assert "deposito_asignado" in data
    assert data["deposito_asignado"] in ["Deposito_1","Deposito_2","Deposito_3","Deposito_4"]
    assert "cluster_id" in data
    assert 0 <= data["cluster_id"] <= 3

def test_predecir_protocolo_estructura():
    """Si el modelo existe, debe retornar protocolo_predicho válido"""
    from main import models
    if "rf" not in models:
        pytest.skip("Modelo RF no disponible en este entorno")

    res = client.post("/predecir-protocolo", json=PRODUCTO_EJEMPLO)
    assert res.status_code == 200
    data = res.json()
    assert "protocolo_predicho" in data
    assert data["protocolo_predicho"] in ["Protocolo_1", "Protocolo_2"]
    assert "probabilidades" in data

def test_predecir_batch():
    """El endpoint batch debe procesar múltiples productos"""
    from main import models
    if "kmeans" not in models or "rf" not in models:
        pytest.skip("Modelos no disponibles")

    payload = {"productos": [PRODUCTO_EJEMPLO, PRODUCTO_EJEMPLO]}
    res = client.post("/predecir-batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 2
    assert len(data["resultados"]) == 2

def test_producto_invalido():
    """Campos faltantes deben retornar error 422"""
    res = client.post("/predecir-deposito", json={"embalaje": 1})
    assert res.status_code == 422
