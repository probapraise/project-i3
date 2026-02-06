import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class L5ScriptSessionRunnerTests(unittest.TestCase):
    def _base_session_paths(self) -> tuple[Path, Path]:
        return (
            ROOT / "examples" / "artifacts" / "sample_session_input_l5.json",
            ROOT / "examples" / "artifacts" / "sample_session_objective_l5_script.json",
        )

    def _base_session_ids(self) -> tuple[str, str]:
        session_input_path, _ = self._base_session_paths()
        session_input = json.loads(session_input_path.read_text(encoding="utf-8"))
        return session_input["session_id"], session_input["objective_id"]

    def _write_aligned_topic_bundle(self, tmpdir: Path) -> Path:
        session_id, objective_id = self._base_session_ids()
        payload = json.loads(
            (ROOT / "examples" / "artifacts" / "sample_topic_research_bundle.json").read_text(
                encoding="utf-8"
            )
        )
        payload["session_id"] = session_id
        payload["objective_id"] = objective_id
        path = tmpdir / "aligned_topic_research_bundle.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _write_aligned_entity_profile(
        self,
        tmpdir: Path,
        payload_override: dict[str, Any] | None = None,
    ) -> Path:
        session_id, objective_id = self._base_session_ids()
        payload = payload_override
        if payload is None:
            payload = json.loads(
                (ROOT / "examples" / "artifacts" / "sample_entity_profile_pack.json").read_text(
                    encoding="utf-8"
                )
            )
        payload["session_id"] = session_id
        payload["objective_id"] = objective_id
        path = tmpdir / "aligned_entity_profile_pack.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _write_script_seed(
        self,
        tmpdir: Path,
        *,
        session_id: str | None = None,
        objective_id: str | None = None,
        payload_override: dict[str, Any] | None = None,
    ) -> Path:
        payload = payload_override
        if payload is None:
            payload = json.loads(
                (ROOT / "examples" / "artifacts" / "sample_script_seed_pack.json").read_text(
                    encoding="utf-8"
                )
            )

        if session_id is not None:
            payload["session_id"] = session_id
        if objective_id is not None:
            payload["objective_id"] = objective_id

        path = tmpdir / "script_seed_pack.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _run_runner(
        self,
        *,
        out_dir: Path,
        topic_bundle_path: Path,
        entity_profile_path: Path,
        script_seed_path: Path,
        extra_args: list[str] | None = None,
        score_spec_path: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        session_input_path, session_objective_path = self._base_session_paths()
        cmd = [
            sys.executable,
            str(ROOT / "l5_script_session_runner.py"),
            "--session-input",
            str(session_input_path),
            "--session-objective",
            str(session_objective_path),
            "--topic-research-bundle",
            str(topic_bundle_path),
            "--entity-profile-pack",
            str(entity_profile_path),
            "--script-seed-pack",
            str(script_seed_path),
        ]
        if score_spec_path is not None:
            cmd.extend(["--score-spec", str(score_spec_path)])
        if extra_args:
            cmd.extend(extra_args)
        cmd.extend(["--out-dir", str(out_dir)])
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_runner_completes_and_outputs_script_session_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            out_dir = tmpdir / "out"
            topic = self._write_aligned_topic_bundle(tmpdir)
            profile = self._write_aligned_entity_profile(tmpdir)
            seed = self._write_script_seed(tmpdir)

            proc = self._run_runner(
                out_dir=out_dir,
                topic_bundle_path=topic,
                entity_profile_path=profile,
                script_seed_path=seed,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

            required_files = [
                "gate_in_report.json",
                "candidates_script.json",
                "evaluation_report_script.json",
                "selection_trace_script.json",
                "script_session_pack.json",
                "runtime_log_l5.json",
                "session_output.json",
                "gate_out_report.json",
            ]
            for name in required_files:
                self.assertTrue((out_dir / name).exists(), f"missing {name}")

            gate_out = json.loads((out_dir / "gate_out_report.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_out["overall_status"], "PASS")

            script_pack = json.loads((out_dir / "script_session_pack.json").read_text(encoding="utf-8"))
            self.assertIn(
                script_pack["quality_status"],
                {"READY_FOR_SCENE_PLANNING", "NEEDS_SCRIPT_REWORK"},
            )
            self.assertIn("script_text_en", script_pack["selected_script"])
            self.assertGreaterEqual(len(script_pack["translation_outputs"]["structured_doc"]), 1)

    def test_selected_script_exists_in_script_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            out_dir = tmpdir / "out"
            topic = self._write_aligned_topic_bundle(tmpdir)
            profile = self._write_aligned_entity_profile(tmpdir)
            seed = self._write_script_seed(tmpdir)

            proc = self._run_runner(
                out_dir=out_dir,
                topic_bundle_path=topic,
                entity_profile_path=profile,
                script_seed_path=seed,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

            script_pack = json.loads((out_dir / "script_session_pack.json").read_text(encoding="utf-8"))
            selected_id = script_pack["selected_script"]["script_id"]
            script_ids = {row["script_id"] for row in script_pack.get("script_bank", [])}
            self.assertIn(selected_id, script_ids)

    def test_real_lock_priority_is_respected_in_fact_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            out_dir = tmpdir / "out"

            profile_pack = json.loads(
                (ROOT / "examples" / "artifacts" / "sample_entity_profile_pack.json").read_text(
                    encoding="utf-8"
                )
            )
            victim = next(e for e in profile_pack["entities"] if e["entity_id"] == "person_victim")
            victim["fields"].append(
                {
                    "field": "occupation",
                    "value": "delivery driver",
                    "source_tag": "INFERRED",
                    "confidence": 0.99,
                    "reason": "test override",
                }
            )

            topic = self._write_aligned_topic_bundle(tmpdir)
            profile = self._write_aligned_entity_profile(tmpdir, payload_override=profile_pack)
            seed = self._write_script_seed(tmpdir)
            proc = self._run_runner(
                out_dir=out_dir,
                topic_bundle_path=topic,
                entity_profile_path=profile,
                script_seed_path=seed,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

            script_pack = json.loads((out_dir / "script_session_pack.json").read_text(encoding="utf-8"))
            victim_registry = next(
                e for e in script_pack["character_fact_registry"] if e["entity_id"] == "person_victim"
            )
            occupation = next(f for f in victim_registry["fields"] if f["field"] == "occupation")
            self.assertEqual(occupation["source_tag"], "REAL")
            self.assertEqual(occupation["value"], "taxi driver")

    def test_runner_fails_on_cross_session_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            out_dir = tmpdir / "out"
            topic = self._write_aligned_topic_bundle(tmpdir)
            profile = self._write_aligned_entity_profile(tmpdir)
            seed = self._write_script_seed(tmpdir, session_id="sess_other_9999")

            proc = self._run_runner(
                out_dir=out_dir,
                topic_bundle_path=topic,
                entity_profile_path=profile,
                script_seed_path=seed,
            )
            self.assertEqual(proc.returncode, 2)

    def test_runner_allows_cross_session_with_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            out_dir = tmpdir / "out"
            topic = self._write_aligned_topic_bundle(tmpdir)
            profile = self._write_aligned_entity_profile(tmpdir)
            seed = self._write_script_seed(tmpdir, session_id="sess_other_9999")

            proc = self._run_runner(
                out_dir=out_dir,
                topic_bundle_path=topic,
                entity_profile_path=profile,
                script_seed_path=seed,
                extra_args=["--allow-cross-session"],
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            gate_out = json.loads((out_dir / "gate_out_report.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_out["overall_status"], "PASS")

    def test_runner_fails_on_score_spec_mismatch_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            alt_score_spec = tmpdir / "score_spec_script_alt.json"
            alt_score_spec.write_text(
                (ROOT / "score_specs" / "score_spec_script.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            out_dir = tmpdir / "out"
            topic = self._write_aligned_topic_bundle(tmpdir)
            profile = self._write_aligned_entity_profile(tmpdir)
            seed = self._write_script_seed(tmpdir)

            proc = self._run_runner(
                out_dir=out_dir,
                topic_bundle_path=topic,
                entity_profile_path=profile,
                script_seed_path=seed,
                score_spec_path=alt_score_spec,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("CLI --score-spec does not match", proc.stderr)

    def test_runner_allows_score_spec_override_in_dev_with_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            alt_score_spec = tmpdir / "score_spec_script_alt.json"
            alt_score_spec.write_text(
                (ROOT / "score_specs" / "score_spec_script.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            out_dir = tmpdir / "out"
            topic = self._write_aligned_topic_bundle(tmpdir)
            profile = self._write_aligned_entity_profile(tmpdir)
            seed = self._write_script_seed(tmpdir)

            proc = self._run_runner(
                out_dir=out_dir,
                topic_bundle_path=topic,
                entity_profile_path=profile,
                script_seed_path=seed,
                score_spec_path=alt_score_spec,
                extra_args=[
                    "--allow-score-spec-override",
                    "--override-reason",
                    "recovery rerun after upstream failure",
                    "--pipeline-env",
                    "dev",
                ],
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            runtime_log = json.loads((out_dir / "runtime_log_l5.json").read_text(encoding="utf-8"))
            self.assertTrue(runtime_log["policy"]["score_spec_override_used"])
            self.assertEqual(
                runtime_log["policy"]["score_spec_override_reason"],
                "recovery rerun after upstream failure",
            )

    def test_runner_rejects_score_spec_override_in_prod(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            alt_score_spec = tmpdir / "score_spec_script_alt.json"
            alt_score_spec.write_text(
                (ROOT / "score_specs" / "score_spec_script.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            out_dir = tmpdir / "out"
            topic = self._write_aligned_topic_bundle(tmpdir)
            profile = self._write_aligned_entity_profile(tmpdir)
            seed = self._write_script_seed(tmpdir)

            proc = self._run_runner(
                out_dir=out_dir,
                topic_bundle_path=topic,
                entity_profile_path=profile,
                script_seed_path=seed,
                score_spec_path=alt_score_spec,
                extra_args=[
                    "--allow-score-spec-override",
                    "--override-reason",
                    "recovery rerun after upstream failure",
                    "--pipeline-env",
                    "prod",
                ],
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("--pipeline-env=prod", proc.stderr)

    def test_runner_rejects_score_spec_override_without_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            alt_score_spec = tmpdir / "score_spec_script_alt.json"
            alt_score_spec.write_text(
                (ROOT / "score_specs" / "score_spec_script.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            out_dir = tmpdir / "out"
            topic = self._write_aligned_topic_bundle(tmpdir)
            profile = self._write_aligned_entity_profile(tmpdir)
            seed = self._write_script_seed(tmpdir)

            proc = self._run_runner(
                out_dir=out_dir,
                topic_bundle_path=topic,
                entity_profile_path=profile,
                script_seed_path=seed,
                score_spec_path=alt_score_spec,
                extra_args=[
                    "--allow-score-spec-override",
                    "--pipeline-env",
                    "dev",
                ],
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("--override-reason must be non-empty", proc.stderr)

    def test_runner_fails_on_invalid_gate_in_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            bad_script_seed = tmpdir / "bad_script_seed_pack.json"
            bad_script_seed.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "session_id": "sess_l5_0001",
                        "objective_id": "obj_l5_script",
                        "created_at": "2026-02-06T13:30:00Z",
                        "quality_status": "READY_FOR_SCRIPT",
                    }
                ),
                encoding="utf-8",
            )

            out_dir = tmpdir / "out"
            topic = self._write_aligned_topic_bundle(tmpdir)
            profile = self._write_aligned_entity_profile(tmpdir)
            proc = self._run_runner(
                out_dir=out_dir,
                topic_bundle_path=topic,
                entity_profile_path=profile,
                script_seed_path=bad_script_seed,
            )
            self.assertEqual(proc.returncode, 1)
            gate_in = json.loads((out_dir / "gate_in_report.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_in["overall_status"], "FAIL")

    def test_runner_fails_on_topic_research_contract_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            out_dir = tmpdir / "out"
            profile = self._write_aligned_entity_profile(tmpdir)
            seed = self._write_script_seed(tmpdir)
            topic = ROOT / "examples" / "artifacts" / "sample_topic_research_bundle.json"

            proc = self._run_runner(
                out_dir=out_dir,
                topic_bundle_path=topic,
                entity_profile_path=profile,
                script_seed_path=seed,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("topic_research_bundle.session_id must match", proc.stderr)


if __name__ == "__main__":
    unittest.main()
