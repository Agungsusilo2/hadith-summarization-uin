from src.preprocessing import clean_indonesian_text


def test_preprocessing_empty_text():
    assert clean_indonesian_text("") == ""
    assert clean_indonesian_text(None) == ""


def test_preprocessing_whitespace_and_html():
    text = "  <b>Rasulullah</b>\n\nbersabda:   jangan  marah.  "
    assert clean_indonesian_text(text) == "Rasulullah bersabda: jangan marah."

