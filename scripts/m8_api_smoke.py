from pathlib import Path

from fastapi.testclient import TestClient

from spam_filter.api import create_app


def main() -> int:
    client = TestClient(create_app(Path.cwd()))
    health = client.get("/health")
    ready = client.get("/ready")
    info = client.get("/model-info")
    pred = client.post(
        "/predict",
        json={"text": "WIN a FREE prize now. Claim your reward today!", "channel": "sms"},
    )
    for response in [health, ready, info, pred]:
        response.raise_for_status()
    body = pred.json()
    if not 0.0 <= body["spam_probability"] <= 1.0:
        raise RuntimeError("API probability outside [0, 1]")
    print(f"health={health.json()['status']}")
    print(f"ready={ready.json()['status']}")
    print(f"model_version={info.json()['model_version']}")
    print(f"predict_label={body['label']}")
    print("milestone=M8")
    print("status=ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
