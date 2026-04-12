import pytest
from fastapi.testclient import TestClient
from server.app import app

@pytest.fixture
def client():
    return TestClient(app)

def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "ARGUS" in response.text

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_docs(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "Swagger UI" in response.text
