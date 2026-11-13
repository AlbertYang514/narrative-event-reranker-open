"""Dependency-light tests for the data and scoring interfaces.

Heavy ML tests are intentionally not run here. The training/evaluation scripts
are validated by a separate optional smoke run with a local model.
"""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SYNTHETIC = ROOT / "examples" / "synthetic"


def load_script(name):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SyntheticSplitTests(unittest.TestCase):
    def test_splits_exist_and_have_no_pair_overlap(self):
        splits = {name: read_jsonl(SYNTHETIC / f"{name}.jsonl") for name in ("train", "valid", "test")}
        seen = {}
        for name, rows in splits.items():
            self.assertGreater(len(rows), 0)
            for row in rows:
                self.assertIn("current_text", row)
                self.assertIn("candidate_text", row)
                self.assertIn("final_score", row)
                self.assertEqual(row["source_id"], "synthetic_001")
                self.assertNotIn(row["pair_id"], seen)
                seen[row["pair_id"]] = name


class ChunkTests(unittest.TestCase):
    def test_chunking_respects_max_length_and_keeps_text(self):
        mod = load_script("build_chunks.py")
        text = "第一句。" * 40
        chunks = mod.build_chunks_from_text(text, min_chars=20, max_chars=60)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 70)
        self.assertEqual("".join(chunks), text)

    def test_sentence_split_keeps_punctuation(self):
        mod = load_script("build_chunks.py")
        self.assertEqual(mod.split_sentences("甲。乙！丙？"), ["甲。", "乙！", "丙？"])


class PairCandidateTests(unittest.TestCase):
    def test_pair_id_and_make_pair(self):
        mod = load_script("build_pair_candidates.py")
        a = {"chapter_index": 1, "chapter_title": "一", "chunk_index": 1, "global_chunk_index": 1, "text": "旧"}
        b = {"chapter_index": 2, "chapter_title": "二", "chunk_index": 1, "global_chunk_index": 8, "text": "新"}
        self.assertEqual(mod.pair_id("s", a, b), "s:c00001-c00008")
        pair = mod.make_pair("s", a, b, "chapter_boundary")
        self.assertEqual(pair["current"]["text"], "新")
        self.assertEqual(pair["candidate"]["text"], "旧")
        self.assertEqual(pair["metadata"]["chunk_distance"], 7)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_script("rank_candidates.py")

    def test_chunk_gap_priority(self):
        current = {"pair_id": "s:c00001-c00002", "global_chunk_index": 2}
        candidate = {"pair_id": "s:c00010-c00011", "global_chunk_index": 10, "metadata": {"chunk_distance": 8}}
        self.assertEqual(self.mod.compute_chunk_gap(current, candidate), 8)
        candidate_no_meta = {"pair_id": "s:c00010-c00011", "global_chunk_index": 10}
        self.assertEqual(self.mod.compute_chunk_gap(current, candidate_no_meta), 8)
        self.assertEqual(self.mod.compute_chunk_gap({}, {"pair_id": "s:c00003-c00019"}), None)
        self.assertEqual(
            self.mod.compute_chunk_gap(
                {"pair_id": "s:c00010-c00011"},
                {"pair_id": "s:c00026-c00027"},
            ),
            15,
        )

    def test_distance_penalty_thresholds(self):
        self.assertEqual(self.mod.compute_distance_penalty(99), 0.0)
        self.assertEqual(self.mod.compute_distance_penalty(100), 0.08)
        self.assertEqual(self.mod.compute_distance_penalty(140), 0.12)
        self.assertEqual(self.mod.compute_distance_penalty(250), 0.16)

    def test_v14_hybrid_only_penalises_same_source(self):
        current = {"source_id": "a", "pair_id": "a:c00001-c00002", "global_chunk_index": 2}
        scored = [
            {
                "text": "near",
                "source_id": "a",
                "pair_id": "a:c00003-c00004",
                "global_chunk_index": 4,
                "score": 0.80,
            },
            {
                "text": "far",
                "source_id": "a",
                "pair_id": "a:c00001-c00200",
                "global_chunk_index": 200,
                "score": 0.79,
            },
            {
                "text": "other source",
                "source_id": "b",
                "pair_id": "b:c00001-c00200",
                "global_chunk_index": 200,
                "score": 0.70,
            },
        ]
        result = self.mod.apply_scoring_strategy(scored, "v14_hybrid", current)
        by_text = {r["text"]: r for r in result}
        self.assertEqual(by_text["near"]["distance_penalty"], 0.0)
        self.assertEqual(by_text["far"]["distance_penalty"], 0.12)
        self.assertEqual(by_text["far"]["score"], 0.67)
        self.assertEqual(by_text["other source"]["distance_penalty"], 0.0)
        self.assertEqual(result[0]["text"], "near")

    def test_raw_strategy_is_a_noop(self):
        scored = [{"text": "a", "score": 0.5}, {"text": "b", "score": 0.7}]
        self.assertIs(self.mod.apply_scoring_strategy(scored, "raw", {}), scored)


class SoftLabelTests(unittest.TestCase):
    def test_expected_score_normalises_probabilities(self):
        mod = load_script("build_soft_labels.py")
        self.assertAlmostEqual(mod.expected_score({"A": 0.5, "D": 0.5}), 0.5)
        self.assertEqual(mod.expected_score({"A": 0.0}), 0.0)

    def test_convert_file_is_input_driven(self):
        mod = load_script("build_soft_labels.py")
        source = {
            "pair_id": "s:c00001-c00002",
            "source_id": "s",
            "strategy": "adjacent_chunk",
            "current": {"text": "后文"},
            "candidate": {"text": "前文"},
            "teacher_label": {"choice": "C", "choice_probs": {"C": 1.0}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.jsonl"
            out = Path(tmp) / "out.jsonl"
            inp.write_text(json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8")
            count = mod.convert_file(inp, out, training=False, usage="local_eval_only", teacher_weight=0.0)
            self.assertEqual(count, 1)
            row = read_jsonl(out)[0]
            self.assertEqual(row["current_text"], "后文")
            self.assertFalse(row["training"])
            self.assertEqual(row["usage"], "local_eval_only")


class ManifestTests(unittest.TestCase):
    def test_example_and_public_manifests_validate(self):
        mod = load_script("validate_manifest.py")
        self.assertEqual(mod.validate_file(ROOT / "configs" / "source_manifest.example.jsonl"), 0)
        self.assertEqual(mod.validate_file(ROOT / "data" / "manifests" / "sources.public_domain.jsonl"), 0)


class RepoPolicyTests(unittest.TestCase):
    @staticmethod
    def repository_files():
        """Return tracked files when git is available, else a safe fallback."""
        try:
            result = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files"],
                capture_output=True,
                text=True,
                check=True,
            )
            return [ROOT / line for line in result.stdout.splitlines() if line]
        except (OSError, subprocess.CalledProcessError):
            ignored = {".git", ".venv", "venv", "__pycache__", "outputs", "node_modules", "hf-cache"}
            return [
                path for path in ROOT.rglob("*")
                if path.is_file() and not any(part in ignored for part in path.parts)
            ]

    def test_no_private_names_or_local_paths_in_text_files(self):
        banned = ["\u788e\u51b0", "\u5eff\u56db\u5473", "".join(["sui", "bing"]), "".join(["nian", "shiwei"]), "/Users/" + "".join(["albert", "yang"]), "D:\\" + "narrative-event-reranker"]
        for path in self.repository_files():
            if path.suffix.lower() not in {".py", ".md", ".json", ".jsonl", ".txt", ".yml", ".yaml"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for needle in banned:
                self.assertNotIn(needle, text, f"{needle!r} found in {path}")

    def test_no_model_weights_are_committed(self):
        forbidden = {".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".onnx"}
        for path in self.repository_files():
            self.assertNotIn(path.suffix.lower(), forbidden, str(path))

    def test_no_large_files(self):
        for path in self.repository_files():
            self.assertLess(path.stat().st_size, 2 * 1024 * 1024, str(path))


if __name__ == "__main__":
    unittest.main()
