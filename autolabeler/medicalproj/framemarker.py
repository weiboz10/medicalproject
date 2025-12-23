"""
Find alternating brightest and dimmest frames in heart videos and save results.

For every .avi in autolabeler/data:
- compute per-frame brightness
- detect extrema in time order, alternating bright -> dim -> bright (or the reverse)
- save the selected frames as images
- write a summary file that groups results per source video
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


# Paths and thresholds for inputs/outputs and white-frame filtering
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_frames"
RESULTS_FILE = Path(__file__).resolve().parent / "frame_results.txt"
LIST_FILE = Path(__file__).resolve().parent / "selected_frames.txt"
WHITE_THRESHOLD = 250.0  # mean brightness above this is treated as white/ignored


def compute_brightness_series(video_path: Path) -> List[float]:
    """Return mean grayscale brightness for every frame in the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    brightness: List[float] = []
    success, frame = cap.read()
    while success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(np.mean(gray)))
        success, frame = cap.read()
    cap.release()
    return brightness


def detect_local_extrema(brightness: Sequence[float]) -> Tuple[List[int], List[int]]:
    """
    Detect local maxima/minima using direct neighbor comparison:
    - brightest if current > previous and current > next
    - dimmest if current < previous and current < next
    """
    maxima: List[int] = []
    minima: List[int] = []
    white_mask = np.asarray(brightness) >= WHITE_THRESHOLD
    for i in range(1, len(brightness) - 1):
        if white_mask[i] or white_mask[i - 1] or white_mask[i + 1]:
            continue  # skip frames that are white or adjacent to white
        prev_val, cur_val, next_val = brightness[i - 1], brightness[i], brightness[i + 1]
        if cur_val > prev_val and cur_val > next_val:
            maxima.append(i)
        if cur_val < prev_val and cur_val < next_val:
            minima.append(i)
    return maxima, minima


def dedup_adjacent_extrema(indices: Sequence[int], brightness: Sequence[float], pick_max: bool) -> List[int]:
    """
    Collapse consecutive extrema (e.g., 22,23 both marked brightest) into a single index.
    Keeps the most extreme brightness within each contiguous run.
    """
    if not indices:
        return []

    deduped: List[int] = []
    run: List[int] = [indices[0]]

    def select_run(run_indices: List[int]) -> int:
        if pick_max:
            return max(run_indices, key=lambda idx: brightness[idx])
        return min(run_indices, key=lambda idx: brightness[idx])

    for idx in indices[1:]:
        if idx == run[-1] + 1:
            run.append(idx)
        else:
            deduped.append(select_run(run))
            run = [idx]
    deduped.append(select_run(run))
    return deduped


def enforce_alternation(
    sequence: Sequence[Tuple[int, str]], brightness: Sequence[float]
) -> List[Tuple[int, str]]:
    """
    Ensure sequence does not contain two identical labels in a row.
    If it happens, keep the more extreme brightness of the pair/run.
    """
    if not sequence:
        return []

    collapsed: List[Tuple[int, str]] = []
    for idx, label in sequence:
        if not collapsed:
            collapsed.append((idx, label))
            continue

        prev_idx, prev_label = collapsed[-1]
        if label != prev_label:
            collapsed.append((idx, label))
            continue

        # Same label: keep the more extreme brightness
        if label == "brightest":
            chosen = (idx, label) if brightness[idx] > brightness[prev_idx] else (prev_idx, prev_label)
        else:
            chosen = (idx, label) if brightness[idx] < brightness[prev_idx] else (prev_idx, prev_label)
        collapsed[-1] = chosen

    return collapsed


def find_white_runs(brightness: Sequence[float]) -> List[Tuple[int, int]]:
    """Return list of (start, end) inclusive indices where brightness >= WHITE_THRESHOLD."""
    runs: List[Tuple[int, int]] = []
    white_mask = np.asarray(brightness) >= WHITE_THRESHOLD
    start = None
    for i, is_white in enumerate(white_mask):
        if is_white and start is None:
            start = i
        elif not is_white and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(white_mask) - 1))
    return runs


def build_alternating_sequence(
    maxima: Sequence[int], minima: Sequence[int]
) -> List[Tuple[int, str]]:
    """
    Merge maxima/minima into an alternating sequence, starting with whichever comes first.
    Returns list of (frame_index, label) where label is "brightest" or "dimmest".
    """
    maxima_sorted = sorted(maxima)
    minima_sorted = sorted(minima)

    if not maxima_sorted and not minima_sorted:
        return []
    if not maxima_sorted:
        return [(idx, "dimmest") for idx in minima_sorted]
    if not minima_sorted:
        return [(idx, "brightest") for idx in maxima_sorted]

    next_max = 0
    next_min = 0
    sequence: List[Tuple[int, str]] = []

    start_with_max = maxima_sorted[0] <= minima_sorted[0]
    expecting_max = start_with_max

    while next_max < len(maxima_sorted) or next_min < len(minima_sorted):
        if expecting_max:
            while next_max < len(maxima_sorted) and (
                sequence and maxima_sorted[next_max] <= sequence[-1][0]
            ):
                next_max += 1
            if next_max >= len(maxima_sorted):
                break
            sequence.append((maxima_sorted[next_max], "brightest"))
            next_max += 1
        else:
            while next_min < len(minima_sorted) and (
                sequence and minima_sorted[next_min] <= sequence[-1][0]
            ):
                next_min += 1
            if next_min >= len(minima_sorted):
                break
            sequence.append((minima_sorted[next_min], "dimmest"))
            next_min += 1
        expecting_max = not expecting_max

    return sequence


def save_target_frames(
    video_path: Path, targets: Iterable[Tuple[int, str]], dest_dir: Path
) -> List[Path]:
    """
    Save the selected frames as PNG files.
    Returns list of saved paths in the order they were written.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_map = {idx: label for idx, label in targets}
    if not target_map:
        return []

    saved_paths: List[Path] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video for saving frames: {video_path}")

    frame_idx = 0
    success, frame = cap.read()
    while success:
        if frame_idx in target_map:
            label = target_map[frame_idx]
            filename = f"frame_{frame_idx:05d}_{label}.png"
            out_path = dest_dir / filename
            cv2.imwrite(str(out_path), frame)
            saved_paths.append(out_path)
        success, frame = cap.read()
        frame_idx += 1
    cap.release()
    return saved_paths


def write_results(
    results: List[Tuple[Path, List[Tuple[int, str, float]], List[Path], List[Tuple[int, int]]]]
) -> None:
    """
    Write a text summary grouped by video.
    Each entry contains (video_path, [(frame_idx, label, brightness)], [saved_paths], white_runs).
    """
    lines: List[str] = []
    list_lines: List[str] = []
    for video_path, extrema, saved, white_runs in results:
        lines.append(f"=== {video_path.name} ===")
        if white_runs:
            for start, end in white_runs:
                lines.append(f"frame_{start:05d}_white_start -> frame_{end:05d}_white_end")
        if extrema:
            for frame_idx, label, brightness in extrema:
                lines.append(f"{label:9} frame {frame_idx:5d} brightness {brightness:8.3f}")
        else:
            lines.append("No extrema detected")
        if saved:
            lines.append("Saved frames:")
            for p in saved:
                lines.append(f"  {p}")
        lines.append("")  # blank line between videos

        list_lines.append(f"=== {video_path.name} ===")
        if white_runs:
            for start, end in white_runs:
                list_lines.append(f"frame_{start:05d}_white_start")
                list_lines.append(f"frame_{end:05d}_white_end")
        if saved:
            for idx, label, brightness in extrema:
                matching = [p for p in saved if f"frame_{idx:05d}_" in p.name]
                saved_path = matching[0] if matching else "not saved"
                list_lines.append(
                    f"{label:9} frame {idx:5d} brightness {brightness:8.3f} -> {saved_path}"
                )
        else:
            list_lines.append("No frames saved")
        list_lines.append("")

    RESULTS_FILE.write_text("\n".join(lines))
    LIST_FILE.write_text("\n".join(list_lines))


def process_video(
    video_path: Path,
) -> Tuple[Path, List[Tuple[int, str, float]], List[Path], List[Tuple[int, int]]]:
    """Handle one video: brightness series, extrema sequence, saved frames, white runs."""
    # Measure brightness per frame
    brightness = compute_brightness_series(video_path)
    # Identify white stretches to skip in extrema picking
    white_runs = find_white_runs(brightness)
    # Find candidate peaks/troughs, then collapse adjacent duplicates
    maxima, minima = detect_local_extrema(brightness)
    maxima = dedup_adjacent_extrema(maxima, brightness, pick_max=True)
    minima = dedup_adjacent_extrema(minima, brightness, pick_max=False)
    # Build alternating bright/dim sequence and enforce no repeats
    sequence = build_alternating_sequence(maxima, minima)
    sequence = enforce_alternation(sequence, brightness)
    extrema_with_values = [(idx, label, brightness[idx]) for idx, label in sequence]

    # Save selected frames and return metadata
    dest_dir = OUTPUT_DIR / video_path.stem
    saved_paths = save_target_frames(video_path, sequence, dest_dir)
    return video_path, extrema_with_values, saved_paths, white_runs


def find_videos(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.iterdir() if p.suffix.lower() == ".avi"])


def main() -> None:
    """Walk all videos, collect results, and write reports."""
    videos = find_videos(DATA_DIR)
    if not videos:
        print(f"No .avi files found in {DATA_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: List[
        Tuple[Path, List[Tuple[int, str, float]], List[Path], List[Tuple[int, int]]]
    ] = []
    for video in videos:
        print(f"Processing {video.name}...")
        all_results.append(process_video(video))

    write_results(all_results)
    print(f"Summary written to {RESULTS_FILE}")
    print(f"Frame list written to {LIST_FILE}")
    print(f"Extracted frames under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
