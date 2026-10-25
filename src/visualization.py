import matplotlib.pyplot as plt
import numpy as np


def plot_segments(y, sr, segments, output_path):
    times = np.arange(len(y)) / sr

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, y, color="gray", linewidth=0.5)

    peak = max(float(np.max(np.abs(y))), 0.01)

    for segment in segments:
        ax.axvspan(segment["start"], segment["end"], alpha=0.2)
        ax.text(
            (segment["start"] + segment["end"]) / 2,
            peak * 0.85,
            str(segment["word_id"]),
            ha="center",
            fontsize=8,
        )

    ax.set_xlabel("Zaman (s)")
    ax.set_ylabel("Genlik")
    ax.set_title("Otomatik Ses Segmentasyonu")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_pitch_results(results, output_path):
    fig, ax = plt.subplots(figsize=(14, 5))

    for item in results:
        times = item["start"] + item["times"]
        ax.plot(times, item["f0"], linewidth=1)

    ax.set_xlabel("Zaman (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title("Segment Bazlı Pitch Analizi")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_similarity_matrix(df, segment_ids, output_path):
    n = len(segment_ids)
    index = {sid: i for i, sid in enumerate(segment_ids)}

    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 1.0)

    for row in df.itertuples(index=False):
        i = index[row.segment_a]
        j = index[row.segment_b]

        matrix[i, j] = row.correlation
        matrix[j, i] = row.correlation

    fig, ax = plt.subplots(figsize=(max(7, n * 0.6), max(6, n * 0.6)))

    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="viridis", vmin=-1, vmax=1)

    ax.set_xticks(range(n), segment_ids)
    ax.set_yticks(range(n), segment_ids)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Segment")
    ax.set_title("Pearson Korelasyon Matrisi")

    fig.colorbar(image, ax=ax, label="Korelasyon")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
