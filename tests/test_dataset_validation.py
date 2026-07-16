import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.validate_dataset import validate_label_file


class DatasetValidationTests(unittest.TestCase):
    def test_valid_yolo_label(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
            self.assertEqual(validate_label_file(label), [])

    def test_invalid_coordinate_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("0 1.5 0.5 0.2 0.3\n", encoding="utf-8")
            errors = validate_label_file(label)
            self.assertEqual(len(errors), 1)

    def test_wrong_number_of_values_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("0 0.5 0.5\n", encoding="utf-8")
            self.assertEqual(len(validate_label_file(label)), 1)


if __name__ == "__main__":
    unittest.main()
