"""Regression tests for offline demo behavior."""

from src.m3_rerank import _version_adjustment
from src.pipeline import _extractive_answer


def test_current_policy_is_preferred_over_superseded_version():
    old = {"text": "Trạng thái: ĐÃ THAY THẾ", "metadata": {"source": "mat_khau_v1.md"}}
    current = {"text": "Phiên bản hiện hành", "metadata": {"source": "mat_khau_v2.md"}}

    assert _version_adjustment(current) > _version_adjustment(old)


def test_extractive_answer_combines_multi_hop_contexts():
    answer = _extractive_answer(
        "Senior 9 năm thâm niên có bao nhiêu ngày phép và lương trong khoảng nào?",
        [
            "Nhân viên Senior 9 năm thâm niên được 18 ngày phép.",
            "Lương Senior nằm trong khoảng 20 đến 35 triệu đồng mỗi tháng.",
        ],
    )

    assert "18" in answer
    assert "35" in answer
