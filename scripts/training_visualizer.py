"""
training_visualizer.py
=======================
Đọc file JSON log training và vẽ các biểu đồ phân tích.

Định dạng log hỗ trợ (mỗi dòng 1 JSON object):
  {"epoch": 1, "global_step": 403, "loss_total": 26.0,
   "loss_conf": 4.37, "loss_landm": 14.52, "loss_loc": 3.56,
   "lr": 0.0004, "epoch_time_sec": 689.13}

Cách dùng:
  python training_visualizer.py --log train.log
  python training_visualizer.py --log train.log --x global_step --save ./charts
"""

import json
import argparse
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.axes import Axes


# ──────────────────────────────────────────────
# 1. ĐỌC DỮ LIỆU
# ──────────────────────────────────────────────


def load_jsonl(filepath: str) -> list[dict]:
    """Đọc file JSONL (mỗi dòng là 1 JSON object).
    Bỏ qua dòng trống hoặc comment (#).
    """
    records = []
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [!] Bỏ qua dòng {i} (lỗi JSON): {e}")

    if not records:
        raise ValueError("File không chứa dữ liệu hợp lệ.")

    print(f"  Đọc được {len(records)} bản ghi từ '{filepath}'")
    return records


def load_json_array(filepath: str) -> list[dict]:
    """Đọc file JSON dạng mảng: [{...}, {...}, ...]"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("File phải là mảng JSON.")
    return data


def load_log(filepath: str) -> list[dict]:
    """Tự động nhận diện định dạng file (JSONL hoặc JSON array)."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if raw.startswith("["):
        return load_json_array(filepath)
    return load_jsonl(filepath)


def extract_field(records: list[dict], key: str) -> list:
    """Lấy danh sách giá trị của một trường, bỏ qua None."""
    return [r.get(key) for r in records]


def _as_float(value: Any) -> float | None:
    """Ép kiểu an toàn về float, trả về None nếu không hợp lệ."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_x_axis(records: list[dict], x: str = "epoch") -> tuple[list, str]:
    """Trả về (values, label) cho trục X."""
    values = extract_field(records, x)
    if all(v is None for v in values):
        # fallback về index
        return list(range(1, len(records) + 1)), "Iteration"
    label_map = {
        "epoch": "Epoch",
        "global_step": "Global Step",
        "step": "Step",
    }
    return values, label_map.get(x, x.replace("_", " ").title())


# ──────────────────────────────────────────────
# 2. STYLE CHUNG
# ──────────────────────────────────────────────

PALETTE = {
    "total": "#E24B4A",
    "conf": "#378ADD",
    "landm": "#1D9E75",
    "loc": "#EF9F27",
    "lr": "#7F77DD",
    "time": "#888780",
    "delta+": "#1D9E75",
    "delta-": "#E24B4A",
}


def _base_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAF9",
            "axes.grid": True,
            "grid.color": "#E0DED8",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.labelcolor": "#5F5E5A",
            "axes.labelsize": 10,
            "xtick.color": "#888780",
            "ytick.color": "#888780",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "sans-serif",
        }
    )


def _finish(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(
        title, fontsize=12, fontweight="bold", color="#2C2C2A", pad=10, loc="left"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))


# ──────────────────────────────────────────────
# 3. CÁC HÀM VẼ BIỂU ĐỒ
# ──────────────────────────────────────────────


def plot_total_loss(
    records: list[dict],
    x: str = "epoch",
    ax: Axes | None = None,
    save_path: str | None = None,
):
    """Biểu đồ tổng loss theo quá trình training."""
    _base_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 4))

    xs, xlabel = get_x_axis(records, x)
    ys = extract_field(records, "loss_total")

    ax.plot(
        xs,
        ys,
        color=PALETTE["total"],
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="Total Loss",
    )
    ax.fill_between(xs, ys, alpha=0.12, color=PALETTE["total"])

    # annotate first & last
    for idx in [0, -1]:
        if ys[idx] is not None:
            ax.annotate(
                f"{ys[idx]:.3f}",
                xy=(xs[idx], ys[idx]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color=PALETTE["total"],
            )

    _finish(ax, "Tổng loss theo quá trình training", xlabel, "Loss")
    ax.legend()

    if standalone:
        plt.tight_layout()
        _save_or_show(save_path, "total_loss.png")
    return ax


def plot_loss_components(
    records: list[dict],
    x: str = "epoch",
    ax: Axes | None = None,
    save_path: str | None = None,
):
    """Biểu đồ 3 thành phần loss riêng biệt."""
    _base_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 4))

    xs, xlabel = get_x_axis(records, x)
    components = [
        ("loss_conf", "Confidence Loss", PALETTE["conf"], "-"),
        ("loss_landm", "Landmark Loss", PALETTE["landm"], "--"),
        ("loss_loc", "Localization Loss", PALETTE["loc"], "-."),
    ]
    for key, label, color, ls in components:
        ys = extract_field(records, key)
        if any(v is not None for v in ys):
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2,
                linestyle=ls,
                marker="o",
                markersize=3.5,
                label=label,
            )

    _finish(ax, "Phân tách các thành phần loss", xlabel, "Loss")
    ax.legend()

    if standalone:
        plt.tight_layout()
        _save_or_show(save_path, "loss_components.png")
    return ax


def plot_loss_ratio(
    records: list[dict],
    x: str = "epoch",
    ax: Axes | None = None,
    save_path: str | None = None,
):
    """Stacked bar: tỉ lệ % đóng góp từng thành phần loss."""
    _base_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 4))

    xs, xlabel = get_x_axis(records, x)
    conf = np.array([r.get("loss_conf", 0) or 0 for r in records])
    landm = np.array([r.get("loss_landm", 0) or 0 for r in records])
    loc = np.array([r.get("loss_loc", 0) or 0 for r in records])
    total = conf + landm + loc
    total[total == 0] = 1  # tránh chia 0

    pct_conf = conf / total * 100
    pct_landm = landm / total * 100
    pct_loc = loc / total * 100

    width = 0.6
    ax.bar(xs, pct_conf, width, label="Confidence", color=PALETTE["conf"] + "CC")
    ax.bar(
        xs,
        pct_landm,
        width,
        bottom=pct_conf,
        label="Landmark",
        color=PALETTE["landm"] + "CC",
    )
    ax.bar(
        xs,
        pct_loc,
        width,
        bottom=pct_conf + pct_landm,
        label="Localization",
        color=PALETTE["loc"] + "CC",
    )

    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    _finish(ax, "Tỉ lệ đóng góp từng thành phần loss (%)", xlabel, "Tỉ lệ (%)")
    ax.legend()

    if standalone:
        plt.tight_layout()
        _save_or_show(save_path, "loss_ratio.png")
    return ax


def plot_learning_rate(
    records: list[dict],
    x: str = "epoch",
    ax: Axes | None = None,
    save_path: str | None = None,
):
    """Biểu đồ learning rate schedule."""
    _base_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 3.5))

    xs, xlabel = get_x_axis(records, x)
    ys = extract_field(records, "lr")
    if all(v is None for v in ys):
        print("  [!] Không tìm thấy trường 'lr' trong dữ liệu.")
        return ax

    ax.plot(
        xs,
        ys,
        color=PALETTE["lr"],
        linewidth=2,
        marker="o",
        markersize=4,
        label="Learning Rate",
    )
    ax.fill_between(xs, ys, alpha=0.1, color=PALETTE["lr"])
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    _finish(ax, "Learning rate schedule", xlabel, "LR")
    ax.legend()

    if standalone:
        plt.tight_layout()
        _save_or_show(save_path, "learning_rate.png")
    return ax


def plot_loss_delta(
    records: list[dict],
    x: str = "epoch",
    ax: Axes | None = None,
    save_path: str | None = None,
):
    """Biểu đồ biến thiên loss (Δ loss) so với epoch trước — xanh giảm, đỏ tăng."""
    _base_style()
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 3.5))

    xs, xlabel = get_x_axis(records, x)
    totals = extract_field(records, "loss_total")
    deltas = [None] + [
        (
            (totals[i] - totals[i - 1])
            if (totals[i] is not None and totals[i - 1] is not None)
            else None
        )
        for i in range(1, len(totals))
    ]

    colors = []
    for d in deltas:
        if d is None:
            colors.append("#CCCCCC")
        elif d < 0:
            colors.append(PALETTE["delta+"])
        else:
            colors.append(PALETTE["delta-"])

    bar_values = [0.0 if d is None else float(d) for d in deltas]
    ax.bar(xs, bar_values, color=colors, width=0.6)
    ax.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")

    # legend thủ công
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=PALETTE["delta+"], label="Loss giảm (tốt)"),
            Patch(color=PALETTE["delta-"], label="Loss tăng (xấu)"),
        ]
    )

    _finish(ax, "Biến thiên tổng loss (Δ so epoch trước)", xlabel, "Δ Loss")

    if standalone:
        plt.tight_layout()
        _save_or_show(save_path, "loss_delta.png")
    return ax


def plot_epoch_time(
    records: list[dict],
    x: str = "epoch",
    ax: Axes | None = None,
    save_path: str | None = None,
):
    """Biểu đồ thời gian huấn luyện mỗi epoch (giây)."""
    _base_style()
    ys = [r.get("epoch_time_sec") for r in records]
    if all(v is None for v in ys):
        print("  [!] Không tìm thấy 'epoch_time_sec', bỏ qua biểu đồ thời gian.")
        return ax

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 3.5))

    xs, xlabel = get_x_axis(records, x)
    bar_values = [0.0 if v is None else float(v) for v in ys]
    ax.bar(xs, bar_values, color=PALETTE["time"] + "AA", width=0.6)

    # đường trung bình
    valid = [float(v) for v in ys if v is not None]
    if valid:
        mean_val = float(np.mean(valid))
        ax.axhline(
            mean_val,
            color=PALETTE["time"],
            linewidth=1.4,
            linestyle="--",
            label=f"TB: {mean_val:.0f}s",
        )
        ax.legend()

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}s"))
    _finish(ax, "Thời gian mỗi epoch", xlabel, "Thời gian (giây)")

    if standalone:
        plt.tight_layout()
        _save_or_show(save_path, "epoch_time.png")
    return ax


def plot_all(
    records: list[dict],
    x: str = "epoch",
    save_path: str | None = None,
    figsize: tuple = (14, 20),
):
    """Vẽ toàn bộ biểu đồ trên một figure duy nhất (dashboard)."""
    _base_style()
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle(
        "Training Dashboard", fontsize=15, fontweight="bold", color="#2C2C2A", y=1.01
    )

    plot_total_loss(records, x, ax=axes[0, 0])
    plot_loss_components(records, x, ax=axes[0, 1])
    plot_loss_ratio(records, x, ax=axes[1, 0])
    plot_learning_rate(records, x, ax=axes[1, 1])
    plot_loss_delta(records, x, ax=axes[2, 0])
    plot_epoch_time(records, x, ax=axes[2, 1])

    # Biểu đồ tổng loss + LR trên cùng 1 trục (dual-axis)
    _plot_dual_axis(records, x, axes[3, 0])

    # Summary text box
    _plot_summary_text(records, axes[3, 1])

    plt.tight_layout()
    _save_or_show(save_path, "dashboard.png")
    return fig


# ──────────────────────────────────────────────
# 4. BIỂU ĐỒ NÂNG CAO
# ──────────────────────────────────────────────


def _plot_dual_axis(records: list[dict], x: str, ax: Axes):
    """Tổng loss và learning rate trên cùng một biểu đồ, hai trục Y."""
    xs, xlabel = get_x_axis(records, x)
    loss = [
        np.nan if v is None else float(v) for v in extract_field(records, "loss_total")
    ]
    lr = [np.nan if v is None else float(v) for v in extract_field(records, "lr")]

    ax2 = ax.twinx()
    ax.plot(
        xs,
        loss,
        color=PALETTE["total"],
        linewidth=2,
        marker="o",
        markersize=3.5,
        label="Total Loss",
    )
    ax2.plot(
        xs,
        lr,
        color=PALETTE["lr"],
        linewidth=1.8,
        linestyle="--",
        marker="s",
        markersize=3,
        label="LR",
    )

    ax.set_ylabel("Loss", color=PALETTE["total"])
    ax.set_xlabel(xlabel)
    ax2.set_ylabel("Learning Rate", color=PALETTE["lr"])
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_title(
        "Loss & Learning Rate",
        fontsize=12,
        fontweight="bold",
        color="#2C2C2A",
        pad=10,
        loc="left",
    )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)


def _plot_summary_text(records: list[dict], ax: Axes):
    """Hộp tóm tắt các chỉ số cuối training."""
    ax.axis("off")
    if not records:
        return
    last = records[-1]
    first = records[0]

    def pct(a, b):
        a_f = _as_float(a)
        b_f = _as_float(b)
        if a_f is not None and b_f not in (None, 0.0):
            return f"{(a_f - b_f) / b_f * 100:+.1f}%"
        return "N/A"

    def fmt_fixed(value: object, digits: int = 4) -> str:
        value_f = _as_float(value)
        return f"{value_f:.{digits}f}" if value_f is not None else "—"

    def fmt_sci(value: object) -> str:
        value_f = _as_float(value)
        return f"{value_f:.2e}" if value_f is not None else "—"

    lines = [
        ("Tổng số epoch", str(len(records))),
        ("Global step cuối", str(last.get("global_step", "—"))),
        ("Loss đầu", fmt_fixed(first.get("loss_total"))),
        ("Loss cuối", fmt_fixed(last.get("loss_total"))),
        ("Giảm loss", pct(last.get("loss_total"), first.get("loss_total"))),
        ("LR cuối", fmt_sci(last.get("lr"))),
        (
            "Thời gian tổng",
            _fmt_time(sum(r.get("epoch_time_sec", 0) or 0 for r in records)),
        ),
    ]

    y = 0.92
    ax.text(
        0.05,
        y + 0.05,
        "Tóm tắt training",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color="#2C2C2A",
    )
    for label, value in lines:
        ax.text(
            0.05, y, f"{label}:", transform=ax.transAxes, fontsize=9, color="#5F5E5A"
        )
        ax.text(
            0.62,
            y,
            value,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color="#2C2C2A",
        )
        y -= 0.12

    ax.set_facecolor("#F8F7F4")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def plot_loss_heatmap(records: list[dict], save_path: str | None = None):
    """Heatmap tỉ lệ từng loss component theo epoch."""
    _base_style()
    keys = ["loss_conf", "loss_landm", "loss_loc"]
    labels = ["Confidence", "Landmark", "Localization"]
    matrix = []
    for key in keys:
        row = [r.get(key, 0) or 0 for r in records]
        matrix.append(row)

    matrix = np.array(matrix, dtype=float)
    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    matrix_pct = matrix / col_sums * 100

    fig, ax = plt.subplots(figsize=(max(8, len(records) * 0.7), 3))
    im = ax.imshow(matrix_pct, aspect="auto", cmap="Blues", vmin=0, vmax=100)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    epochs = [r.get("epoch", i + 1) for i, r in enumerate(records)]
    ax.set_xticks(range(len(records)))
    ax.set_xticklabels(epochs, fontsize=8)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_title(
        "Heatmap tỉ lệ loss component theo epoch (%)",
        fontsize=12,
        fontweight="bold",
        color="#2C2C2A",
        loc="left",
    )

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, format=mticker.PercentFormatter())
    plt.tight_layout()
    _save_or_show(save_path, "loss_heatmap.png")
    return fig


# ──────────────────────────────────────────────
# 5. TIỆN ÍCH
# ──────────────────────────────────────────────


def _save_or_show(save_path: str | None, filename: str):
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, filename)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Đã lưu: {out}")
        plt.close()
    else:
        plt.show()


def print_summary(records: list[dict]):
    """In tóm tắt số liệu ra terminal."""
    if not records:
        print("Không có dữ liệu.")
        return
    first, last = records[0], records[-1]

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    print("\n" + "=" * 48)
    print("  TRAINING SUMMARY")
    print("=" * 48)
    fields = [
        "epoch",
        "global_step",
        "loss_total",
        "loss_conf",
        "loss_landm",
        "loss_loc",
        "lr",
    ]
    header = f"{'Field':<20} {'Đầu':>12} {'Cuối':>12}"
    print(header)
    print("-" * 48)
    for f in fields:
        a = first.get(f)
        b = last.get(f)
        if a is not None or b is not None:
            print(
                f"  {f:<18} {fmt(a) if a is not None else '—':>12} "
                f"{fmt(b) if b is not None else '—':>12}"
            )

    total_time = sum(r.get("epoch_time_sec", 0) or 0 for r in records)
    if total_time:
        print(f"  {'epoch_time_sec':<18} {'':>12} {_fmt_time(total_time):>12}")

    if first.get("loss_total") and last.get("loss_total"):
        drop = (last["loss_total"] - first["loss_total"]) / first["loss_total"] * 100
        print(f"\n  Loss giảm: {drop:+.2f}%")
    print("=" * 48 + "\n")


# ──────────────────────────────────────────────
# 6. CLI
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Trực quan hóa log training (JSONL / JSON array)"
    )
    parser.add_argument(
        "--log", required=True, help="Đường dẫn file log (.log / .json / .jsonl)"
    )
    parser.add_argument(
        "--x",
        default="epoch",
        choices=["epoch", "global_step", "step"],
        help="Trục X (mặc định: epoch)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Thư mục lưu ảnh (nếu không truyền → hiển thị trực tiếp)",
    )
    parser.add_argument(
        "--charts",
        nargs="*",
        choices=[
            "all",
            "total",
            "components",
            "ratio",
            "lr",
            "delta",
            "time",
            "heatmap",
            "dashboard",
        ],
        default=["dashboard"],
        help="Biểu đồ cần vẽ (mặc định: dashboard)",
    )
    args = parser.parse_args()

    records = load_log(args.log)
    print_summary(records)

    chart_fns = {
        "total": lambda: plot_total_loss(records, args.x, save_path=args.save),
        "components": lambda: plot_loss_components(
            records, args.x, save_path=args.save
        ),
        "ratio": lambda: plot_loss_ratio(records, args.x, save_path=args.save),
        "lr": lambda: plot_learning_rate(records, args.x, save_path=args.save),
        "delta": lambda: plot_loss_delta(records, args.x, save_path=args.save),
        "time": lambda: plot_epoch_time(records, args.x, save_path=args.save),
        "heatmap": lambda: plot_loss_heatmap(records, save_path=args.save),
        "dashboard": lambda: plot_all(records, args.x, save_path=args.save),
    }

    selected = args.charts or ["dashboard"]
    if "all" in selected:
        selected = list(chart_fns.keys())

    for name in selected:
        print(f"  Vẽ biểu đồ: {name} ...")
        chart_fns[name]()

    print("Hoàn tất!")


if __name__ == "__main__":
    main()
