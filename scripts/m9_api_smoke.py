from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from spam_filter.api import create_app


def main() -> int:
    client = TestClient(create_app(Path.cwd()))
    assert client.get('/health').status_code == 200
    assert client.get('/ready').status_code == 200
    prediction = client.post(
        '/v1/predict', json={'text': 'FREE prize claim now!', 'channel': 'sms'}
    )
    assert prediction.status_code == 200
    metrics = client.get('/metrics')
    assert metrics.status_code == 200 and 'spam_filter_requests_total' in metrics.text
    snapshot = client.get('/metrics.json').json()
    assert snapshot['requests_total'] >= 1
    drift = client.get('/drift').json()
    assert drift['status'] in {'ok', 'alert', 'insufficient_observations'}
    print('health=ok')
    print('ready=ready')
    print(f"requests_total={snapshot['requests_total']}")
    print(f"drift_status={drift['status']}")
    print('milestone=M9')
    print('status=ready')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
