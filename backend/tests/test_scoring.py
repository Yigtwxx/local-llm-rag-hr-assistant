"""Unit tests for the benchmark's answer scoring. No Ollama required."""

from bench.run_bench import contains, looks_like_refusal, normalize, score


class TestTurkishNormalization:
    def test_dotted_capital_i_matches_lowercase(self) -> None:
        """`İzmir` and `izmir` must compare equal despite Turkish casing rules."""
        assert contains("En kalabalık şehirler: İstanbul, Ankara, İzmir", "izmir")
        assert contains("ANKARA", "Ankara")

    def test_accents_are_folded(self) -> None:
        assert contains("mükemmel sayılar", "mukemmel")

    def test_whitespace_is_collapsed(self) -> None:
        assert normalize("iki   satır\nbir") == "iki satır bir"


class TestScoring:
    def test_expect_all_requires_every_keyword(self) -> None:
        case = {"expect_all": ["16", "iş günü"]}
        assert score("Yıllık izniniz 16 iş günüdür.", case)[0] is True

        passed, note = score("Yıllık izniniz 16 gündür.", case)
        assert passed is False
        assert "iş günü" in note

    def test_expect_any_requires_one_keyword(self) -> None:
        case = {"expect_any": ["2 kat", "iki kat"]}
        assert score("Hafta sonu mesaisi iki kat ödenir.", case)[0] is True
        assert score("Hafta sonu mesaisi zamlı ödenir.", case)[0] is False

    def test_expect_none_catches_forbidden_content(self) -> None:
        case = {"expect_none": ["stock option"]}
        assert score("Bu bilgi dokümanlarda yok.", case)[0] is True
        assert score("Evet, stock option veriliyor.", case)[0] is False

    def test_empty_case_passes(self) -> None:
        assert score("herhangi bir cevap", {})[0] is True


class TestRefusalDetection:
    def test_canonical_refusal_is_detected(self) -> None:
        answer = (
            "Bu bilgi elimdeki İK dokümanlarında yer almıyor. "
            "İK ekibine ik@novatek.example adresinden ulaşabilirsiniz."
        )
        assert looks_like_refusal(answer) is True

    def test_model_phrased_refusals_are_detected(self) -> None:
        assert looks_like_refusal("Bu konuda dokümanlarda bilgi yok.") is True
        assert looks_like_refusal("Kaynaklarda bu konu belirtilmemiş.") is True

    def test_a_real_answer_is_not_a_refusal(self) -> None:
        assert looks_like_refusal("Yıllık izin hakkınız 16 iş günüdür.") is False

    def test_answer_with_a_trailing_caveat_is_not_a_refusal(self) -> None:
        """Regression: a real answer that ends with a caveat was scored as a refusal.

        qwen3.5 answered the paternity-leave question correctly (10 iş günü) and
        then noted that further detail was not specified. Matching the refusal
        phrase anywhere in the text marked that correct answer as a failure.
        """
        answer = (
            "Eşinizin doğumu nedeniyle kullanabileceğiniz babalık izni 10 iş "
            "günüdür. Ücretli mazeret iznidir ve yıllık izin bakiyesinden "
            "düşülmez. Bunun dışında bir ayrıntı kaynaklarda bulunmuyor."
        )
        assert looks_like_refusal(answer) is False

    def test_refusal_at_the_start_is_still_detected(self) -> None:
        answer = (
            "Bu konuda dokümanlarda bilgi yok. Dilerseniz İK ekibine "
            "danışabilirsiniz; başka bir sorunuz olursa yardımcı olurum."
        )
        assert looks_like_refusal(answer) is True
