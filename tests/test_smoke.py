import sys
import unittest
import zipfile
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import choreography
from choreography import io as chore_io


class SmokeTest(unittest.TestCase):
    def test_package_imports(self):
        self.assertTrue(hasattr(choreography, "Choreography"))
        self.assertTrue(hasattr(choreography, "Dance"))
        self.assertTrue(hasattr(choreography, "MeasureReversal"))
        self.assertTrue(hasattr(choreography, "DataMapper"))
        self.assertIsInstance(choreography.ViewRequest().dot_painter, choreography.DotPainter)
        self.assertEqual(choreography.DotPainter().alpha, 0)
        self.assertFalse(choreography.ViewRequest().show_crosshairs)

    def test_basic_workflow(self):
        dance = choreography.Dance(worm_id=1)
        dance.times = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        dance.frames = np.array([0, 1, 2], dtype=np.int32)
        dance.area = np.array([10.0, 10.5, 11.0], dtype=np.float32)
        dance.centroid = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=np.float32,
        )
        dance.bearing = np.array(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        )
        dance.extent = np.array(
            [[5.0, 1.0], [5.0, 1.0], [5.0, 1.0]],
            dtype=np.float32,
        )
        dance.spine = [None, None, None]
        dance.outline = [None, None, None]

        chore = choreography.Choreography(mm_per_pixel=0.1, speed_window=0.5)
        chore.load_from_dict({1: dance})

        speed = chore.get_quantity("speed")[1]
        length = chore.get_quantity("length")[1]
        summary = chore.summarise("speed")[1]
        frame_df = chore.to_dataframe(["speed", "length"])

        self.assertEqual(speed.shape, (3,))
        self.assertEqual(length.shape, (3,))
        self.assertEqual(summary.n, 3)
        self.assertEqual(list(frame_df.columns), ["worm_id", "frame", "time", "speed", "length"])

    def test_choreography_datamap_wrappers(self):
        dance = choreography.Dance(worm_id=1)
        dance.times = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        dance.frames = np.array([0, 1, 2], dtype=np.int32)
        dance.area = np.array([10.0, 10.5, 11.0], dtype=np.float32)
        dance.centroid = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=np.float32,
        )
        dance.bearing = np.array(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        )
        dance.extent = np.array(
            [[5.0, 1.0], [5.0, 1.0], [5.0, 1.0]],
            dtype=np.float32,
        )
        dance.spine = [None, None, None]
        dance.outline = [None, None, None]

        chore = choreography.Choreography(mm_per_pixel=0.1, speed_window=0.5)
        chore.load_from_dict({1: dance})

        try:
            mapper = chore.get_datamapper()
            img = chore.render_map(
                quantity="speed",
                color_mapper="rainbow",
                backgrounder="green",
                dot_painter="line",
                width_px=64,
                height_px=48,
            )
            img_t = chore.render_map_at_time(
                0.5,
                trail_s=0.5,
                quantity="speed",
                color_mapper="rainbow",
                backgrounder="green",
                dot_painter="line",
                width_px=64,
                height_px=48,
            )
            img_bar = chore.add_map_colorbar(
                img,
                "rainbow",
                v_min=0.0,
                v_max=1.0,
                label="speed",
            )
        except RuntimeError as exc:
            if "Pillow is required" in str(exc):
                self.skipTest("Pillow not installed")
            raise

        self.assertIs(mapper, chore.get_datamapper())
        self.assertEqual(img.size, (64, 48))
        self.assertEqual(img_t.size, (64, 48))
        self.assertGreater(img_bar.size[0], img.size[0])

    def test_datamap_absolute_spine_overlay(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")

        dance = choreography.Dance(worm_id=1)
        dance.times = np.array([0.0], dtype=np.float32)
        dance.frames = np.array([0], dtype=np.int32)
        dance.area = np.array([10.0], dtype=np.float32)
        dance.centroid = np.array([[100.0, 100.0]], dtype=np.float32)
        dance.bearing = np.array([[1.0, 0.0]], dtype=np.float32)
        dance.extent = np.array([[5.0, 1.0]], dtype=np.float32)
        dance.spine = [
            choreography.SpineData(
                np.array([[100.0, 100.0], [110.0, 100.0]], dtype=np.float32),
                absolute=True,
            )
        ]
        dance.outline = [None]

        dm = choreography.DataMapper(mm_per_pixel=0.1)
        img = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        view = choreography.ViewRequest(
            center_x=105.0,
            center_y=100.0,
            width_px=40,
            height_px=40,
            pixel_size=1.0,
        )

        dm.overlay_spines(img, {1: dance}, view, t=0.0, color=(255, 0, 0, 255))
        self.assertEqual(img.getpixel((20, 20)), (255, 0, 0, 255))

    def test_datamap_infers_spine_widths_from_extent(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")

        dance = choreography.Dance(worm_id=1)
        dance.times = np.array([0.0], dtype=np.float32)
        dance.frames = np.array([0], dtype=np.int32)
        dance.area = np.array([200.0], dtype=np.float32)
        dance.centroid = np.array([[20.0, 20.0]], dtype=np.float32)
        dance.bearing = np.array([[1.0, 0.0]], dtype=np.float32)
        dance.extent = np.array([[12.0, 5.0]], dtype=np.float32)
        dance.spine = [
            choreography.SpineData(
                np.array([[12.0, 20.0], [20.0, 20.0], [28.0, 20.0]], dtype=np.float32),
                widths=np.zeros(3, dtype=np.float32),
                absolute=True,
            )
        ]
        dance.outline = [None]

        dm = choreography.DataMapper(mm_per_pixel=0.1)
        img = dm.render(
            {1: dance},
            quantity="speed",
            width_px=40,
            height_px=40,
            show_paths=False,
            show_history_morphology=True,
        )
        # Filled body should cover pixels above/below the centerline.
        self.assertNotEqual(img.getpixel((20, 16))[3], 0)

    def test_datamap_default_hides_fallback_glyphs_without_morphology(self):
        dance = choreography.Dance(worm_id=1)
        dance.times = np.array([0.0], dtype=np.float32)
        dance.frames = np.array([0], dtype=np.int32)
        dance.area = np.array([200.0], dtype=np.float32)
        dance.centroid = np.array([[20.0, 20.0]], dtype=np.float32)
        dance.bearing = np.array([[1.0, 0.0]], dtype=np.float32)
        dance.extent = np.array([[12.0, 5.0]], dtype=np.float32)
        dance.spine = [None]
        dance.outline = [None]

        dm = choreography.DataMapper(mm_per_pixel=0.1)
        img = dm.render(
            {1: dance},
            quantity="speed",
            width_px=40,
            height_px=40,
            t_at=0.0,
        )
        self.assertEqual(img.getpixel((20, 20)), (0, 0, 0, 255))

    def test_zip_summary_and_load_directory(self):
        summary_text = "\n".join(
            [
                "1 0 0.0 10.0 20.0 30.0 0 0 0.0 5.0 1.0",
                "1 1 0.5 11.0 20.0 31.0 0 0 0.0 5.0 1.0",
            ]
        )
        blob_text = "\n".join(
            [
                "% 0 0.0",
                "10.0 20.0 30.0 0 0 0.0 5.0 1.0",
                "% 1 0.5",
                "11.0 20.0 31.0 0 0 0.0 5.0 1.0",
            ]
        )

        with TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "sample_mwt.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("dataset/sample.summary", summary_text)
                zf.writestr("dataset/sample_00001.blob", blob_text)

            dances = choreography.read_summary(zip_path)
            self.assertIn(1, dances)
            self.assertEqual(dances[1].n_frames, 2)

            loaded = choreography.load_directory(zip_path, quiet=True)
            self.assertIn(1, loaded)
            self.assertEqual(loaded[1].n_frames, 2)
            self.assertEqual(len(loaded[1].spine), 2)
            self.assertEqual(len(loaded[1].outline), 2)

    def test_zip_multiworm_blobs(self):
        summary_text = "\n".join(
            [
                "1 0.018  62 0 0.0  0.00 0.000  0.0 0.000  0.0 0.000",
                "2 0.194  65 0 0.0  0.00 0.000  0.0 0.000  0.0 0.000",
            ]
        )
        blobs_text = "\n".join(
            [
                "% 43",
                "1 0.018 1013.082 1632.025 257 6.630 9.978 6.162 38.3 31.6 % -1 21 4 18 5 12 4 4",
                "2 0.194 1012.729 1634.125 247 6.128 11.114 4.627 37.4 20.3 % -3 19 2 19 7 15 7 9",
                "% 12",
                "1 0.018 371.854 1874.376 125 -2.256 6.689 2.310 26.6 16.0 % -9 10 -5 9 -1 7 1 2",
                "2 0.194 371.210 1875.466 138 -2.581 6.561 2.374 27.8 14.1 % -10 9 -7 8 -3 8 0 4",
            ]
        )

        with TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "sample_mwt.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("dataset/sample.summary", summary_text)
                zf.writestr("dataset/sample_00000k.blobs", blobs_text)

            dances = choreography.read_summary(zip_path)
            self.assertEqual(sorted(dances.keys()), [12, 43])
            self.assertEqual(dances[43].n_frames, 2)
            self.assertEqual(dances[12].n_frames, 2)
            self.assertIsNotNone(dances[43].spine[0])

            loaded = choreography.load_directory(zip_path, quiet=True)
            self.assertEqual(sorted(loaded.keys()), [12, 43])
            self.assertEqual(loaded[43].centroid.shape, (2, 2))

    def test_parse_blob_record_format(self):
        line = (
            "1 0.019 711.038 947.976 171 -7.466 1.584 2.306 28.6 13.9 "
            "% -12 11 -14 8 -14 4 -10 1 -7 1 -2 0 1 0 6 1 7 -1 7 -5 5 -7 "
            "%% 695 952 117 moMf:ZYUUEMEMIFGGGHRZX:88aOLc<0S20033c<"
        )
        rec = chore_io._parse_blob_record(line)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["frame"], 1)
        self.assertAlmostEqual(rec["time"], 0.019, places=3)
        self.assertEqual(rec["spine"].size(), 11)
        self.assertFalse(rec["spine"].absolute)
        self.assertIsNotNone(rec["outline"])
        self.assertTrue(rec["outline"].absolute)
        self.assertEqual(rec["outline"].size(), 117)

    def test_read_summary_frame_table_format(self):
        summary_text = (
            "1 0.019 62 0 0.0 0.00 0.000 0.0 0.000 0.0 0.000 0.000 0.000 0.000 0.000 "
            "% 0xB %% 24 0 25 40 %%% 24 3.145 25 3.201\n"
            "2 0.199 65 1 0.4 7.75 0.976 29.2 1.019 12.9 1.019 0.456 1.000 1.380 143.701\n"
        )
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.summary"
            path.write_text(summary_text)
            df = choreography.read_summary(path)
        self.assertTrue(hasattr(df, "columns"))
        self.assertEqual(list(df["image_number"]), [1, 2])
        self.assertEqual(df.loc[0, "event_flags"], [0xB])
        self.assertEqual(df.loc[0, "lineage_pairs"], [(24, 0), (25, 40)])
        self.assertEqual(df.loc[0, "blob_refs"], [(24, 3, 145), (25, 3, 201)])

    def test_read_summary_accepts_legacy_nonstandard_float_tokens(self):
        summary_text = (
            "10874 1961.778 0 0 -1.#IO 0.00 0.000 0.0 0.000 0.0 0.000 0.000 0.000 0.000 0.000\n"
        )
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.summary"
            path.write_text(summary_text)
            df = choreography.read_summary(path)
        self.assertEqual(list(df["image_number"]), [10874])
        self.assertTrue(math.isnan(df.loc[0, "mean_duration_s"]))


if __name__ == "__main__":
    unittest.main()
