from spam_filter.preprocess import PreprocessConfig, TextPreprocessor


def test_placeholders_are_preserved() -> None:
    preprocessor = TextPreprocessor(
        PreprocessConfig(
            name="test",
            replace_urls=True,
            replace_emails=True,
            replace_numbers=True,
        )
    )
    tokens = preprocessor.tokenize("WIN 500 now: https://x.com mail A@b.com")
    assert "<num>" in tokens
    assert "<url>" in tokens
    assert "<email>" in tokens
    assert "win" in tokens


def test_stopwords_and_stemming() -> None:
    preprocessor = TextPreprocessor(
        PreprocessConfig(name="test", remove_stopwords=True, stemming=True)
    )
    tokens = preprocessor.tokenize("the prizes are winning quickly")
    assert "the" not in tokens
    assert "prize" in tokens
    assert "win" in tokens
