from spam_filter import __version__


def test_version_contract() -> None:
    assert __version__ == "1.0.0"
