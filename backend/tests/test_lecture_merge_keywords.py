from app.services.lecture_loader import merge_keywords


def test_merge_keywords_curated_first_then_auto():
    out = merge_keywords(
        ["curated1"],
        ["extra1"],
        ["auto1", "auto2"],
        cap=10,
    )
    assert out[0] == "curated1"
    assert "extra1" in out
    assert "auto1" in out


def test_merge_keywords_respects_cap():
    out = merge_keywords(["aaa", "bbb"], None, ["ccc", "ddd", "eee"], cap=3)
    assert len(out) == 3
