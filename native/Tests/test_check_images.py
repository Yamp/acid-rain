"""Behavioral tests for capture diagnostics, using tiny known PNG fixtures."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from check_images import analyze_image, compare_images


class ImageDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def image(self, name, colors, size=(2, 2)):
        path = Path(self.folder.name) / name
        picture = Image.new("RGB", size)
        picture.putdata(colors)
        picture.save(path)
        return path

    def test_primary_colors_have_coverage_and_do_not_fail(self):
        path = self.image("primaries.png", [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)])
        result = analyze_image(path)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["size"], [2, 2])
        self.assertEqual(result["metrics"]["black_fraction"], .25)
        self.assertEqual(result["metrics"]["saturated_fraction"], .75)
        self.assertEqual(result["metrics"]["hue_coverage"]["red"], .25)
        self.assertEqual(result["metrics"]["hue_coverage"]["green"], .25)
        self.assertEqual(result["metrics"]["hue_coverage"]["blue"], .25)

    def test_white_and_gray_are_diagnostics_only_but_total_black_fails(self):
        for color, status, warning in [(255, "PASS", "white"), (128, "PASS", "gray"), (0, "FAIL", "black")]:
            result = analyze_image(self.image(f"{color}.png", [(color,) * 3] * 4))
            self.assertEqual(result["status"], status)
            self.assertTrue(any(warning in text for text in result["diagnostics"] + result["errors"]))

    def test_invalid_and_requested_wrong_dimensions_fail(self):
        broken = Path(self.folder.name) / "broken.png"
        broken.write_bytes(b"not an image")
        self.assertEqual(analyze_image(broken)["status"], "FAIL")
        path = self.image("valid.png", [(100, 0, 200)] * 4)
        self.assertEqual(analyze_image(path, expected_size=(3, 2))["status"], "FAIL")

    def test_frame_difference_is_exact_without_resizing_or_fps_inference(self):
        red = self.image("red.png", [(255, 0, 0)] * 4)
        blue = self.image("blue.png", [(0, 0, 255)] * 4)
        unchanged = compare_images(red, red)
        changed = compare_images(red, blue)
        self.assertEqual(unchanged.get("mean_absolute_channel_delta"), 0)
        self.assertEqual(changed.get("changed_pixel_fraction"), 1)
        self.assertAlmostEqual(changed["mean_absolute_channel_delta"], 170)
        self.assertNotIn("fps", changed)


if __name__ == "__main__":
    unittest.main()
