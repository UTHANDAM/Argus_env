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


def test_http_reset_step_round_trip(client):
    reset_response = client.post("/reset", json={"task": "easy", "seed": 0})
    assert reset_response.status_code == 200

    reset_body = reset_response.json()
    episode_id = reset_body["observation"]["metadata"]["episode_id"]
    assert episode_id

    step_response_1 = client.post(
        "/step",
        json={"action": {"missing_baseline": "BEiT"}},
    )
    assert step_response_1.status_code == 200
    assert step_response_1.json()["observation"]["metadata"]["episode_id"] == episode_id
    assert step_response_1.json()["reward"] == pytest.approx(0.35)

    step_response_2 = client.post(
        "/step",
        json={"action": {"missing_baseline": "BEiT-B/16"}},
    )
    assert step_response_2.status_code == 200
    assert step_response_2.json()["observation"]["metadata"]["episode_id"] == episode_id
    assert step_response_2.json()["done"] is True
    assert step_response_2.json()["reward"] == pytest.approx(0.65)

    state_response = client.get("/state")
    assert state_response.status_code == 200
    assert state_response.json()["episode_id"] == episode_id
    assert state_response.json()["step_count"] == 2

def test_docs(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "Swagger UI" in response.text
