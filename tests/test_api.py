import json
import pytest
from flask import Flask
from scripts.run_api import app 

@pytest.fixture
def client():
    """
    Fixture that provides a test client for the Flask app.
    The app is imported from scripts/run_api.py
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_root_route(client):
    """
    Tests the root route (GET /) to ensure it returns 200 and a welcome message.
    """
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Welcome to the Multi-Modal Retrieval API" in resp.data

def test_search_route_valid(client):
    """
    Tests POST /search with a valid JSON query. We assume the API
    will return a 200 status and a JSON 'results' key.
    """
    payload = {
        "query": "a futuristic cityscape",
        "k": 3
    }
    resp = client.post("/search",
                       data=json.dumps(payload),
                       content_type="application/json")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.get_json()

    # We expect data to have a 'results' key that is a list
    assert "results" in data, "Response JSON missing 'results' key"
    assert isinstance(data["results"], list), "'results' should be a list"

    # Check the first result structure
    if data["results"]:
        first_result = data["results"][0]
        assert "image_path" in first_result
        assert "similarity" in first_result

def test_search_route_invalid_json(client):
    """
    Tests POST /search with no body or invalid JSON.
    We expect the server to handle it gracefully (likely a 400 or a 200 with an error message).
    """
    resp = client.post("/search",
                       data="not valid json",
                       content_type="application/json")
    # This might be a 400 or some form of error handling
    assert resp.status_code in (200, 400), f"Unexpected status code: {resp.status_code}"

