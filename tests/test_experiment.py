"""Independent mathematical checks and an end-to-end persisted-artifact test."""
import ast
import copy
from dataclasses import replace
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib
import numpy as np

from shortcut_learning.artifacts import report, snapshot, write_png
from shortcut_learning.config import Config
from shortcut_learning.data import balanced_backgrounds, generate_objects, load_dataset, make_dataset, shift_right, validate_dataset
from shortcut_learning.experiment import evaluate, file_hash, generate, train
from shortcut_learning.metrics import classification_metrics
from shortcut_learning.model import LogisticRegression, loss_and_gradients, sigmoid


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(seeds=(7,), primary_seed=7, n_train=40, n_test=40, steps=20)
        self.data = make_dataset(7, self.config)

    def test_generation_is_deterministic(self):
        repeated = make_dataset(7, self.config)
        for key in self.data:
            np.testing.assert_array_equal(self.data[key], repeated[key])

    def test_new_seed_changes_noise(self):
        other = make_dataset(19, self.config)
        self.assertFalse(np.array_equal(self.data["train_noise"], other["train_noise"]))

    def test_shape_labels_and_background_counts(self):
        y = self.data["train_labels"]
        np.testing.assert_array_equal(np.bincount(y), [20, 20])
        np.testing.assert_array_equal(self.data["train_biased_backgrounds"], y)
        for label in [0, 1]:
            bg = self.data["train_varied_backgrounds"][y == label]
            np.testing.assert_array_equal(np.bincount(bg), [10, 10])

    def test_paired_foreground_is_identical(self):
        masks = self.data["test_masks"]
        for condition in ["background_reversed", "background_independent"]:
            np.testing.assert_array_equal(self.data["test_same_rule"][masks], self.data[f"test_{condition}"][masks])

    def test_position_shift_preserves_area_without_wrap(self):
        masks, shifted = self.data["test_masks"], self.data["shifted_masks"]
        np.testing.assert_array_equal(masks.sum(axis=(1, 2)), shifted.sum(axis=(1, 2)))
        self.assertFalse(shifted[:, :, :6].any())
        np.testing.assert_array_equal(shift_right(masks, 0), masks)

    def test_cropping_shift_is_rejected(self):
        masks = np.zeros((1, 24, 24), dtype=bool)
        masks[0, 12, 23] = True
        with self.assertRaisesRegex(ValueError, "crop or wrap"):
            shift_right(masks, 1)

    def test_altered_foreground_is_detected(self):
        broken = copy.deepcopy(self.data)
        index = tuple(np.argwhere(broken["test_masks"])[0])
        broken["test_background_reversed"][index] = .91
        with self.assertRaisesRegex(ValueError, "foreground"):
            validate_dataset(broken, self.config)

    def test_invalid_sample_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.config, n_train=42)
        with self.assertRaises(ValueError):
            generate_objects(1, 3, self.config)

    def test_invalid_optimizer_configuration_is_rejected(self):
        for change in [{"learning_rate": -1}, {"l2": -1}, {"steps": 0}, {"seeds": (7, 7)}, {"noise_std": float("nan")}]:
            with self.assertRaises(ValueError):
                replace(self.config, **change)


class MathematicsTests(unittest.TestCase):
    def test_gradients_match_central_finite_differences(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(7, 4))
        y = np.array([0., 1., 1., 0., 1., 0., 1.])
        w = rng.normal(size=4)*.1
        b, l2, epsilon = .03, .07, 1e-6
        _, dw, db = loss_and_gradients(x, y, w, b, l2)
        numerical = np.zeros_like(w)
        for i in range(len(w)):
            plus, minus = w.copy(), w.copy()
            plus[i] += epsilon; minus[i] -= epsilon
            numerical[i] = (loss_and_gradients(x, y, plus, b, l2)[0] - loss_and_gradients(x, y, minus, b, l2)[0])/(2*epsilon)
        numerical_b = (loss_and_gradients(x, y, w, b+epsilon, l2)[0] - loss_and_gradients(x, y, w, b-epsilon, l2)[0])/(2*epsilon)
        np.testing.assert_allclose(dw, numerical, atol=1e-8, rtol=1e-6)
        self.assertAlmostEqual(db, numerical_b, places=8)

    def test_sigmoid_handles_extreme_logits(self):
        values = sigmoid(np.array([-1000., 0., 1000.]))
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(values, [0, .5, 1], atol=1e-15)

    def test_metrics_match_hand_computed_confusion(self):
        metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([.1, .8, .7, .9]))
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(metrics["accuracy"], .75)
        self.assertAlmostEqual(metrics["precision_macro"], (1+2/3)/2)
        self.assertAlmostEqual(metrics["recall_macro"], .75)
        self.assertAlmostEqual(metrics["f1_macro"], (2/3+.8)/2)

    def test_zero_denominators_have_explicit_zero_policy(self):
        metrics = classification_metrics(np.array([0, 1]), np.array([.1, .2]))
        self.assertEqual(metrics["per_class"][1]["precision"], 0)
        self.assertEqual(metrics["per_class"][1]["f1"], 0)
        self.assertEqual(metrics["zero_division"], 0)

    def test_log_loss_is_stable_with_extreme_logits(self):
        z = np.array([1000., -1000.])
        metrics = classification_metrics(np.array([0, 1]), sigmoid(z), z)
        self.assertEqual(metrics["log_loss"], 1000.)

    def test_invalid_metrics_input_is_rejected(self):
        for y, p in [(np.array([]), np.array([])), (np.array([2]), np.array([.5])),
                     (np.array([0]), np.array([float("nan")])), (np.array([0]), np.array([1.1]))]:
            with self.assertRaises(ValueError):
                classification_metrics(y, p)


class ModelTests(unittest.TestCase):
    def test_first_update_matches_finite_differences(self):
        images = np.array([[0., .2], [.8, .9], [.6, .7]], dtype=np.float32)
        labels = np.array([0, 1, 1])
        x = images.astype(np.float64)-.5
        initial = np.zeros(2, dtype=np.float64)
        epsilon, rate = 1e-6, .025
        numerical = np.zeros(2)
        for i in range(2):
            plus, minus = initial.copy(), initial.copy()
            plus[i] += epsilon; minus[i] -= epsilon
            numerical[i] = (loss_and_gradients(x, labels, plus, 0., .001)[0] -
                            loss_and_gradients(x, labels, minus, 0., .001)[0])/(2*epsilon)
        numerical_b = (loss_and_gradients(x, labels, initial, epsilon, .001)[0] -
                       loss_and_gradients(x, labels, initial, -epsilon, .001)[0])/(2*epsilon)
        model = LogisticRegression(steps=1, learning_rate=rate).fit(images, labels)
        np.testing.assert_allclose(model.weights, -rate*numerical, atol=1e-8)
        self.assertAlmostEqual(model.bias, -rate*numerical_b, places=8)

    def test_model_learns_a_separable_problem(self):
        x = np.array([[0., .1], [.1, 0.], [.9, 1.], [1., .9]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression(steps=100, learning_rate=.3, l2=0, history_every=10).fit(x, y)
        np.testing.assert_array_equal(model.predict(x), y)
        self.assertLess(model.history[-1]["objective"], model.history[0]["objective"])

    def test_unfitted_model_rejects_prediction(self):
        with self.assertRaisesRegex(ValueError, "not fitted"):
            LogisticRegression().predict(np.zeros((2, 2)))

    def test_invalid_pixels_and_labels_are_rejected(self):
        for x, y in [(np.array([[float("nan")], [0.]]), [0, 1]),
                     (np.array([[2.], [0.]]), [0, 1]), (np.zeros((2, 1)), [0, 0])]:
            with self.assertRaises(ValueError):
                LogisticRegression().fit(x, np.array(y))

    def test_serialized_model_preserves_predictions(self):
        x, y = np.array([[0.], [1.]], dtype=np.float32), np.array([0, 1])
        model = LogisticRegression(steps=20).fit(x, y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.npz"
            model.save(path, {"test": "serialization"})
            loaded, meta = LogisticRegression.load(path)
        self.assertEqual(meta, {"test": "serialization"})
        np.testing.assert_array_equal(model.predict_proba(x), loaded.predict_proba(x))


class IntegrationTests(unittest.TestCase):
    def test_full_pipeline_keeps_models_fixed_and_exports_artifacts(self):
        cfg = Config(seeds=(7,), primary_seed=7, n_train=40, n_test=40, steps=30, history_every=10)
        with tempfile.TemporaryDirectory() as directory:
            output, destination = Path(directory)/"run", Path(directory)/"snapshot"
            generate(output, cfg)
            train(output, cfg)
            paths = list((output/"models").glob("*.npz"))
            before = {p.name: file_hash(p) for p in paths}
            result = evaluate(output, cfg)
            self.assertEqual(len(result["rows"]), 10)
            self.assertEqual(before, {p.name: file_hash(p) for p in paths})
            self.assertTrue(all(r["weights_unchanged"] for r in result["model_integrity"]))
            for row in result["rows"]:
                self.assertEqual(sum(map(sum, row["confusion_matrix"])), 40)
            data = load_dataset(output/"data/seed-7.npz", 7, cfg)
            self.assertEqual(data["train_biased"].shape, (40, 24, 24))
            report(output, cfg)
            snapshot(output, destination)
            self.assertEqual(len(list((destination/"examples").glob("*.png"))), 14)
            self.assertEqual(len(list((destination/"results/models").glob("*.npz"))), 2)
            self.assertTrue((destination/"results/confusion-matrices.svg").is_file())

    def test_changing_config_in_existing_run_is_rejected(self):
        cfg = Config(seeds=(7,), primary_seed=7, n_train=40, n_test=40)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            generate(output, cfg)
            with self.assertRaisesRegex(ValueError, "different configuration"):
                train(output, replace(cfg, l2=.02))

    def test_png_export_can_be_decoded_without_image_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"example.png"
            write_png(path, np.array([[0., 1.], [.5, .25]]), scale=1)
            content = path.read_bytes()
        self.assertEqual(content[:8], b"\x89PNG\r\n\x1a\n")
        chunks, cursor = {}, 8
        while cursor < len(content):
            size = struct.unpack(">I", content[cursor:cursor+4])[0]
            name = content[cursor+4:cursor+8]
            chunks[name] = content[cursor+8:cursor+8+size]
            cursor += 12+size
        self.assertEqual(zlib.decompress(chunks[b"IDAT"]), bytes([0, 0, 255, 0, 128, 64]))

    def test_runtime_imports_are_numpy_or_standard_library(self):
        root = Path(__file__).parents[1]/"src/shortcut_learning"
        allowed = set(sys.stdlib_module_names) | {"numpy", "shortcut_learning", "__future__"}
        imported = set()
        for path in root.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    imported.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
        self.assertEqual(imported-allowed, set())


if __name__ == "__main__":
    unittest.main()
