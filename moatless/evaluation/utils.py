import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from moatless.codeblocks.module import Module
from moatless.repository import FileRepository
from moatless.schema import FileWithSpans

IGNORED_SPANS = ["docstring", "imports"]

logger = logging.getLogger(__name__)
_moatless_instances = {}

_moatless_datasets = {}


def load_moatless_datasets(split: str | None = None):
    """
    Load all instances from the dataset index.
    The split parameter is kept for backward compatibility but is ignored.
    """
    global _moatless_instances

    # Path to the dataset index file
    index_path = os.path.join(os.path.dirname(__file__), "datasets", "avr_dataset_index.json")

    try:
        if os.path.exists(index_path):
            with open(index_path) as f:
                dataset_index = json.load(f)

                # Process all instances from the index
                for instance in dataset_index.get("instances", []):
                    instance_id = instance["instance_id"]

                    # Add dataset information
                    if "datasets" in instance and instance["datasets"]:
                        # Use the highest priority dataset (last in the list based on our priority order)
                        top_dataset = instance["datasets"][-1]
                        instance["dataset"] = top_dataset["dataset"]

                    _moatless_instances[instance_id] = instance

                logger.info(f"Loaded {len(_moatless_instances)} instances from dataset index")
        else:
            logger.warning(f"Dataset index file not found: {index_path}")
            # Fall back to the old loading method
            _load_legacy_datasets(split)
    except Exception as e:
        logger.error(f"Error loading dataset index: {e}")
        # Fall back to the old loading method
        _load_legacy_datasets(split)


def _load_legacy_datasets(split: str | None = None):
    """Legacy method to load datasets from individual files."""
    if split:
        _load_moatless_dataset(split)
    else:
        # _load_moatless_dataset("lite")
        # _load_moatless_dataset("verified")
        _load_moatless_dataset("custom")

    logger.info(f"Loaded {len(_moatless_instances)} instances using legacy method")


def _load_moatless_dataset(split: str):
    """Legacy method to load a single dataset file."""
    global _moatless_instances

    file_path = os.path.join(os.path.dirname(__file__), f"swebench_{split}_all_evaluations.json")
    with open(file_path) as f:
        dataset = json.load(f)
        for d in dataset:
            d["dataset"] = split
            _moatless_instances[d["instance_id"]] = d


def get_moatless_instances(split: str | None = None):
    global _moatless_instances
    if not _moatless_instances:
        load_moatless_datasets(split)
    return dict(_moatless_instances)  # Return a copy to prevent external modifications


def get_moatless_instance(instance_id: str, split: str | None = None):
    return get_swebench_instance(instance_id, split)


def get_swebench_instance(instance_id: str, split: str | None = None):
    global _moatless_instances
    if not _moatless_instances:
        load_moatless_datasets(split)

    instance = _moatless_instances.get(instance_id)
    if not instance:
        raise ValueError(f"Instance {instance_id} not found.")

    return dict(instance)  # Return a copy to prevent external modifications


def find_relevant_spans(original_block: Module, updated_block: Module):
    """Find relevant spans in test content. Used for finding the "perfect" context in benchmark instances."""

    relevant_spans = set()

    for span in updated_block.spans_by_id.values():
        if span.span_id in relevant_spans:
            continue

        if original_block.has_span(span.span_id):
            updated_content = updated_block.to_prompt(span_ids=set(span.span_id), show_outcommented_code=False).strip()
            original_content = original_block.to_prompt(
                span_ids=set(span.span_id), show_outcommented_code=False
            ).strip()
            if original_content != updated_content:
                relevant_spans.add(span.span_id)

            # TODO: Second prio after token count
            related_span_ids = original_block.find_related_span_ids(span.span_id)
            relevant_spans.update(related_span_ids)
        else:
            parent_block = updated_block.find_first_by_span_id(span.span_id).parent
            original_parent_block = original_block.find_by_path(parent_block.full_path())
            span_ids = list(original_parent_block.belongs_to_span.span_id)

            related_span_ids = updated_block.find_related_span_ids(span.span_id)
            for related_span_id in related_span_ids:
                if original_block.has_span(related_span_id):
                    span_ids.append(related_span_id)

    return relevant_spans


def get_diff_lines(diff_input):
    if not diff_input:
        return []
    file_name_re = re.compile(r"diff --git a/(.+) b/.+")
    file_name_no_git_re = re.compile(r"--- a/(.+)")

    line_change_re = re.compile(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@")

    changes = []

    current_file = None
    for line in diff_input.splitlines():
        file_match = file_name_re.match(line)
        if file_match:
            current_file = file_match.group(1)
            continue

        if not current_file:
            file_match = file_name_no_git_re.match(line)
            if file_match:
                current_file = file_match.group(1)

            continue

        line_change_match = line_change_re.match(line)
        if line_change_match:
            old_start, old_length, new_start, new_length = map(int, line_change_match.groups())

            adjustment_start = max(1, min(3, old_start - 3))
            adjusted_start = old_start + adjustment_start

            relevant_diff_lines = max(0, old_length - 7)
            adjusted_end = adjusted_start + relevant_diff_lines

            if old_length == 0:
                change_type = "addition"
            elif new_length == 0:
                change_type = "deletion"
            else:
                change_type = "modification"

            changes.append((current_file, adjusted_start, adjusted_end, change_type))

    return changes


def compare_patches(expected_patch, actual_patch):
    expected_diffs = get_diff_lines(expected_patch)
    actual_diffs = get_diff_lines(actual_patch)

    expected_files = set()
    file_hits = set()
    line_hits = 0

    for patch_diff in expected_diffs:
        change_file, change_start, change_end, change_type = patch_diff

        for actual_diff in actual_diffs:
            (
                actual_change_file,
                actual_change_start,
                actual_change_end,
                actual_change_type,
            ) = actual_diff
            expected_files.add(change_file)
            if change_file == actual_change_file:
                file_hits.add(change_file)
                if change_start >= actual_change_start and change_end <= actual_change_end:
                    line_hits += 1
                    continue

    return len(expected_files) - len(file_hits), len(expected_diffs) - line_hits


def create_file_spans_from_patch(repo_dir: str, patch: str) -> list[FileWithSpans]:
    repository = FileRepository(repo_dir)
    files_with_spans = []
    for file_path, span_ids in get_file_spans_from_patch(repository, patch).items():
        file_with_spans = FileWithSpans(
            file_path=file_path,
            span_ids=span_ids,
        )
        files_with_spans.append(file_with_spans)

    return files_with_spans


def get_file_spans_from_patch(repository: FileRepository, patch: str) -> dict[str, list[str]]:
    expected_diff_lines = get_diff_lines(patch)
    expected_files_with_spans = {}

    for diff_line in expected_diff_lines:
        change_file, change_start, change_end, change_type = diff_line
        file = repository.get_file(change_file)

        if file is None or file.module is None:
            continue

        if file.file_path not in expected_files_with_spans:
            expected_files_with_spans[file.file_path] = []
        spans = file.module.find_spans_by_line_numbers(change_start, change_end)
        for span in spans:
            if span.span_id not in expected_files_with_spans[file.file_path]:
                expected_files_with_spans[file.file_path].append(span.span_id)
    return expected_files_with_spans


def get_files_from_patch(patch: str) -> list[str]:
    diff_lines = get_diff_lines(patch)
    return [diff_line[0] for diff_line in diff_lines]


def file_spans_to_dict(files_with_spans: list[FileWithSpans]) -> dict[str, list[str]]:
    span_dict = {}
    if not files_with_spans:
        return span_dict

    for file_with_spans in files_with_spans:
        if file_with_spans.file_path not in span_dict:
            span_dict[file_with_spans.file_path] = []

        for span_id in file_with_spans.span_ids:
            if span_id not in span_dict[file_with_spans.file_path]:
                span_dict[file_with_spans.file_path].append(span_id)
    return span_dict


def get_missing_files(
    expected_files_with_spans: dict[str, list[str]],
    files: list[str],
) -> list[str]:
    misses = list(expected_files_with_spans.keys())
    for actual_file in files:
        if actual_file in misses:
            misses.remove(actual_file)
    return misses


def get_missing_spans(
    expected_files_with_spans: dict[str, list[str]],
    actual_files_with_spans: dict[str, list[str]],
) -> dict[str, list[str]]:
    misses = {}
    for expected_file, expected_span_ids in expected_files_with_spans.items():
        if expected_file not in actual_files_with_spans:
            actual_span_ids = []
        else:
            actual_span_ids = actual_files_with_spans[expected_file]

        missing_span_ids = [
            span_id for span_id in expected_span_ids if span_id not in actual_span_ids and span_id not in IGNORED_SPANS
        ]

        if missing_span_ids:
            misses[expected_file] = missing_span_ids
    return misses


def count_identified_spans(
    expected_files_with_spans: dict[str, list[str]],
    actual_files_with_spans: dict[str, list[str]],
) -> int:
    count = 0
    for actual_file, actual_span_ids in actual_files_with_spans.items():
        if expected_files_with_spans.get(actual_file, []):
            for actual_span_id in actual_span_ids:
                if actual_span_id in expected_files_with_spans[actual_file]:
                    count += 1
    return count


def count_identified_files(
    expected_files_with_spans: dict[str, list[str]],
    actual_files_with_spans: dict[str, list[str]],
) -> int:
    count = 0
    for actual_file, actual_span_ids in actual_files_with_spans.items():
        if expected_files_with_spans.get(actual_file, []):
            count += 1
    return count


def has_identified_spans(
    expected_solutions: list[dict[str, list[str]]],
    actual_files_with_spans: dict[str, list[str]],
) -> bool:
    for expected_file_with_spans in expected_solutions:
        missing_spans = get_missing_spans(expected_file_with_spans, actual_files_with_spans)
        if not missing_spans or missing_spans == ["docstring"]:
            return True
    return False


def find_identified_spans(
    expected_solutions: list[dict[str, list[str]]],
    actual_files_with_spans: dict[str, list[str]],
) -> dict[str, list[str]] | None:
    for expected_file_with_spans in expected_solutions:
        missing_spans = get_missing_spans(expected_file_with_spans, actual_files_with_spans)
        if not missing_spans or missing_spans == ["docstring"]:
            return expected_file_with_spans
    return None


def has_identified_files(
    expected_solutions: list[dict[str, list[str]]],
    actual_files_with_spans: dict[str, list[str]] | list[str],
) -> bool:
    if isinstance(actual_files_with_spans, dict):
        actual_files = list(actual_files_with_spans.keys())
    else:
        actual_files = actual_files_with_spans

    for expected_file_with_spans in expected_solutions:
        if not get_missing_files(expected_file_with_spans, actual_files):
            return True
    return False


def find_identified_files(
    expected_solutions: list[dict[str, list[str]]],
    actual_files_with_spans: dict[str, list[str]],
) -> dict[str, list[str]] | None:
    if not actual_files_with_spans:
        return None

    for expected_file_with_spans in expected_solutions:
        if not get_missing_files(expected_file_with_spans, actual_files_with_spans):
            return expected_file_with_spans
    return None


def missing_expected_test_files(
    expected_test_files: list[str], actual_files_with_spans: dict[str, list[str]]
) -> list[str]:
    if not actual_files_with_spans:
        return []

    missing_files = []
    for expected_file in expected_test_files:
        if expected_file not in actual_files_with_spans:
            missing_files.append(expected_file)
    return missing_files


def calculate_estimated_context_window(instance, results):
    patch = instance.get("patch") or instance.get("golden_patch")
    patch_diffs = get_diff_lines(patch)
    expected_changes = []

    for patch_diff in patch_diffs:
        change_file, change_start, change_end, change_type = patch_diff
        expected_changes.append(
            {
                "file_path": change_file,
                "start_line": change_start,
                "end_line": change_end,
                "closest_match_context_window": None,
                "closest_match_lines": None,
                "position": None,
                "distance": None,
                "context_window": None,
            }
        )

    sum_tokens = 0

    for i, result in enumerate(results):
        sum_tokens += result.tokens
        for change in expected_changes:
            if result.file_path == change["file_path"]:
                logger.info(
                    f"Found result for {change['file_path']} ({change['start_line']}-{change['end_line']}) at {result.start_line}-{result.end_line} with distance {result.distance}"
                )
                if result.start_line - 1 <= change["start_line"] and result.end_line + 1 >= change["end_line"]:
                    change["distance"] = result.distance
                    change["context_window"] = sum_tokens
                    change["position"] = i

                    if all(context["context_window"] is not None for context in expected_changes):
                        return expected_changes, sum_tokens
                else:
                    closest_match_lines = change.get("closest_match_lines")
                    if (
                        not closest_match_lines
                        or abs(result.start_line - change["start_line"])
                        < abs(closest_match_lines[0] - change["start_line"])
                    ) or (
                        abs(result.end_line - change["end_line"]) == abs(closest_match_lines[0] - change["end_line"])
                    ):
                        change["closest_match_lines"] = (
                            result.start_line,
                            result.end_line,
                        )
                        change["closest_match_context_window"] = sum_tokens

    return expected_changes, sum_tokens


def trace_metadata(instance_id: str, session_id: str, trace_name: str):
    date_time_str = time.strftime("%Y%m%d-%H%M%S")
    trace_id = f"coder_{instance_id}_{date_time_str}"
    return {
        "session_id": session_id,
        "name": trace_name,
        "trace": trace_name,
        "trace_id": trace_id,
        "tags": [instance_id],
    }


def get_moatless_dataset_splits() -> dict[str, dict]:
    """Get all available datasets with their metadata."""
    global _moatless_datasets
    if not _moatless_datasets:
        datasets = {}
        dataset_dir = Path(__file__).parent / "datasets"

        for dataset_file in dataset_dir.glob("*_dataset.json"):
            try:
                with open(dataset_file) as f:
                    data = json.load(f)
                    if "name" in data and "instance_ids" in data:
                        datasets[data.get("name", "")] = {
                            "name": data.get("name", ""),
                            "description": data.get("description", ""),
                            "instance_count": len(data.get("instance_ids", [])),
                            "instance_ids": data.get("instance_ids", []),
                        }
            except Exception as e:
                logger.exception(f"Failed to load dataset {dataset_file}: {e}")
                continue

        _moatless_datasets = datasets

    return _moatless_datasets


def get_moatless_dataset_split(name: str) -> dict[str, Any] | None:
    """Get a dataset by name."""
    return get_moatless_dataset_splits().get(name, None)
