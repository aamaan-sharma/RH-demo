def test_health_ok(client):
    res = client.get("/health")
    assert res.status_code == 200
    payload = res.get_json()
    assert payload["status"] == "ok"
    assert payload["service"] == "contract-pdf-qna"
