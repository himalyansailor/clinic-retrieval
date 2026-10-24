from clinic_retrieval.lang import latin_key, skeleton, words


def test_devanagari_survives_tokenisation():
    # With a plain \w regex this returns आय / ष / म / न and every Hindi
    # question abstains.
    got = words("आयुष्मान भारत योजना में कवर कितना है?")
    assert "आयुष्मान" in got
    assert "योजना" in got


def test_romanised_hindi_meets_devanagari():
    for deva, roman in [("आयुष्मान", "ayushman"), ("योजना", "yojana"),
                        ("दवा", "dawa"), ("सुरक्षा", "suraksha")]:
        assert skeleton(deva) == skeleton(roman), (deva, roman)


def test_spelling_variants_collapse():
    assert skeleton("yojana") == skeleton("yojna")
    assert latin_key("aayushmaan") == latin_key("ayushman")


def test_function_words_dropped_in_all_three_forms():
    assert "kitna" not in words("ayushman yojana mein kitna cover milta hai")
    assert "कितना" not in words("आयुष्मान योजना में कितना कवर मिलता है")
    assert "what" not in words("what is the cover")
