from clinic_retrieval.generation import SYSTEM, prompt


# A 3B model told to "answer in the language of the question" answered four of
# nine Devanagari questions in English. Naming the language per request, from
# the detected input form, is what it follows.
def test_a_devanagari_question_names_hindi_as_the_answer_language():
    assert "Answer in: Hindi, in Devanagari script" in prompt(
        "दवा स्टॉक में न हो तो क्या करना चाहिए?", [])


def test_english_and_romanised_hindi_are_answered_in_english():
    assert "Answer in: English" in prompt("What is the PM-JAY cover?", [])
    assert "Answer in: English" in prompt("dawa stock mein nahi hai to kya karna chahiye", [])


def test_a_stated_negative_is_an_answer_not_an_abstention():
    # "Jharkhand does not run a state top-up" answers "does Jharkhand run one".
    assert "does not" in SYSTEM and "insufficient" in SYSTEM
