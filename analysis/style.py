"""Bảng màu + kiểu vẽ dùng chung cho analysis/ (một hệ màu nhất quán ở mọi hình của báo cáo).

Màu phân loại theo THỨ TỰ CỐ ĐỊNH (không xoay vòng); nhãn tấn công luôn sắp theo tên nên mỗi loại tấn công
giữ cùng một màu ở mọi hình. Benign: xám trung tính. DL vs ML: xanh (slot 1) vs xám.
"""
from __future__ import annotations

BENIGN_COLOR = "#9ca3af"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAMILY_COLORS = {"DL": "#2a78d6", "ML": BENIGN_COLOR}
TEXT_MUTED = "#6b7280"
DPI = 150


def apply_style() -> None:
    """Trục/lưới nhạt, nét mảnh, không khung trên/phải."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#d1d5db", "axes.labelcolor": "#374151", "axes.titlesize": 11,
        "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED, "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False, "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


def label_colors(labels, benign: str = "Benign") -> dict:
    """{nhãn: màu}; tấn công sắp theo tên -> màu cố định; quá 8 loại thì gộp màu xám đậm (không sinh màu mới)."""
    attacks = sorted(set(labels) - {benign})
    colors = {a: (CATEGORICAL[i] if i < len(CATEGORICAL) else "#4b5563") for i, a in enumerate(attacks)}
    colors[benign] = BENIGN_COLOR
    return colors
