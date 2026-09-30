"""Job orchestration and HTTP boundaries, runnable without ML dependencies."""

import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tn_compression.ui.jobs import JobManager, read_json, validate_request
from tn_compression.ui.server import PRESETS, ThreadingHTTPServer, handler_for


class UIJobsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.config = read_json(PRESETS)[0]["config"]

    def payload(self, **overrides):
        return {"config": self.config, "mode": "compress", "device": "cpu", **overrides}

    def manager(self, runner=None):
        manager = JobManager(self.root / "runs", self.data, command_runner=runner)
        self.addCleanup(lambda: manager.pool.shutdown(wait=True))
        return manager

    def test_paths_and_untrained_real_vision_rejected(self):
        with self.assertRaisesRegex(ValueError, "inside"):
            validate_request(self.payload(checkpoint="../outside"), self.data)
        self.config["data"] = {"kind": "oxford_pet", "root": "."}
        with self.assertRaisesRegex(ValueError, "trained checkpoint"):
            validate_request(self.payload(), self.data)

    def test_own_vision_starting_choice_requires_trained_weights(self):
        config = next(preset["config"] for preset in read_json(PRESETS) if preset["id"] == "own-vision")
        with self.assertRaisesRegex(ValueError, "trained checkpoint"):
            validate_request(self.payload(config=config), self.data)

    def test_invalid_source_and_nonfinite_target_rejected(self):
        self.config["model"]["source"] = "torch_hub"
        with self.assertRaisesRegex(ValueError, "TorchVision"):
            validate_request(self.payload(), self.data)
        self.config["model"]["source"] = "torchvision"
        self.config["analysis"]["target_size_mb"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_request(self.payload(), self.data)

    def test_checkpoint_recipe_must_match_supported_model(self):
        bundle = self.data / "bundle"
        bundle.mkdir()
        (bundle / "weights.pt").write_bytes(b"not real weights")
        (bundle / "manifest.json").write_text(json.dumps({"schema_version": 1,
            "model": {"source": "torch_hub", "name": "untrusted", "kwargs": {}}}))
        with self.assertRaisesRegex(ValueError, "supported"):
            validate_request(self.payload(checkpoint="bundle"), self.data)
        (bundle / "manifest.json").write_text(json.dumps({"schema_version": 1,
            "model": {"source": "torchvision", "name": "resnet34", "kwargs": {"num_classes": 3}}}))
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_request(self.payload(checkpoint="bundle"), self.data)

    def test_remote_name_traversal_and_code_request_rejected(self):
        self.config["model"]["name"] = "x/../../private"
        with self.assertRaisesRegex(ValueError, "Invalid model"):
            validate_request(self.payload(), self.data)
        self.config["model"]["name"] = "resnet18"
        self.config["model"]["kwargs"]["trust_remote_code"] = True
        with self.assertRaisesRegex(ValueError, "Remote model code"):
            validate_request(self.payload(), self.data)

    def test_infeasible_plan_never_compresses(self):
        commands = []
        def runner(argv, **kwargs):
            command = argv[3]
            commands.append(command)
            output = Path(argv[argv.index("--output-dir") + 1])
            output.mkdir(parents=True)
            if command == "analyze":
                (output / "compression_plan.json").write_text(json.dumps({"status": "infeasible", "transformations": []}))
            return subprocess.CompletedProcess(argv, 0)
        manager = self.manager(runner)
        job = manager.submit(self.payload())
        manager.pool.shutdown(wait=True)
        self.assertEqual(manager.get(job["id"])["status"], "infeasible")
        self.assertEqual(commands, ["snapshot", "analyze"])

    def test_failed_stage_persists_actionable_error(self):
        def runner(argv, **kwargs):
            kwargs["stdout"].write(b"Missing model cache\n")
            return subprocess.CompletedProcess(argv, 2)
        manager = self.manager(runner)
        job = manager.submit(self.payload())
        manager.pool.shutdown(wait=True)
        result = manager.get(job["id"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("exit 2", result["error"])
        self.assertIn("Missing model cache", result["logs"])

    def test_stale_runs_marked_interrupted(self):
        run = self.root / "runs" / ("a" * 32)
        run.mkdir(parents=True)
        (run / "job.json").write_text(json.dumps({"id": run.name, "status": "running", "created_at": "today"}))
        (run / "config.json").write_text(json.dumps({"model": {"name": "resnet18"}}))
        manager = self.manager()
        self.assertEqual(manager.list()[0]["status"], "interrupted")
        self.assertEqual(manager.list()[0]["model_name"], "resnet18")

    def test_http_same_origin_and_artifact_traversal(self):
        manager = self.manager()
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(manager))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/api/health") as response:
            self.assertEqual(json.load(response)["status"], "ok")
        with urlopen(base + "/api/presets") as response:
            self.assertGreaterEqual(len(json.load(response)["presets"]), 3)
        for headers in ({"Origin": "https://malicious.example"}, {"Host": "malicious.example"}):
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + "/api/jobs", headers=headers))
            self.assertEqual(error.exception.code, 403)
        with self.assertRaises(HTTPError) as error:
            urlopen(base + "/api/jobs/" + "a" * 32 + "/artifacts/../../secret")
        self.assertEqual(error.exception.code, 404)

    def test_upload_endpoint_and_vision_recipe_validation(self):
        manager = self.manager()
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(manager))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        request = Request(base + "/api/uploads", data=b"tiny tensor file", method="POST",
                          headers={"Content-Type": "application/octet-stream", "X-Filename": "weights.pt"})
        with urlopen(request) as response:
            upload = json.load(response)
            self.assertEqual(response.status, 201)
        self.assertTrue(Path(upload["path"]).is_file())
        self.config["model"]["state_dict_path"] = upload["path"]
        self.config["recipe"] = {"methods": ["tensor_decomposition", "quantization"],
                                 "svd_energy": 0.9, "quantization_bits": 8}
        validated = validate_request(self.payload(), self.data, manager.upload_root)
        self.assertEqual(validated["config"]["model"]["state_dict_path"], upload["path"])
        bundle = self.data / "bundle"
        bundle.mkdir()
        (bundle / "weights.pt").write_bytes(b"not real weights")
        (bundle / "manifest.json").write_text(json.dumps({"schema_version": 1,
            "model": {"source": "torchvision", "name": "resnet18", "task": "classification",
                      "kwargs": {"num_classes": 3}}}))
        with self.assertRaisesRegex(ValueError, "either an uploaded"):
            validate_request(self.payload(checkpoint="bundle"), self.data, manager.upload_root)
        self.config["recipe"]["methods"].append("pruning")
        with self.assertRaisesRegex(ValueError, "Llama/Qwen"):
            validate_request(self.payload(), self.data, manager.upload_root)

    def test_relevance_mode_does_not_run_candidate_search(self):
        commands = []
        def runner(argv, **kwargs):
            commands.append(argv[3])
            return subprocess.CompletedProcess(argv, 0)
        manager = self.manager(runner)
        job = manager.submit(self.payload(mode="analyze"))
        manager.pool.shutdown(wait=True)
        self.assertEqual(manager.get(job["id"])["status"], "completed")
        self.assertEqual(commands, ["snapshot", "relevance"])

    def test_download_opt_in_changes_child_environment(self):
        environments = []
        def runner(argv, **kwargs):
            environments.append(kwargs["env"])
            return subprocess.CompletedProcess(argv, 0)
        manager = self.manager(runner)
        job = {"id": "a" * 32, "device": "cpu", "allow_download": True, "status": "queued"}
        directory = manager.root / job["id"]
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        manager._stage(job, "Test", "snapshot", "test")
        self.assertEqual(environments[0]["HF_HUB_OFFLINE"], "0")
        self.assertEqual(environments[0]["HF_DATASETS_OFFLINE"], "0")
        self.assertEqual(environments[0]["TN_ALLOW_HF_DOWNLOAD"], "1")


if __name__ == "__main__":
    unittest.main()
