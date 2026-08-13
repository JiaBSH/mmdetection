import os
import unittest
from unittest.mock import patch

from postprocess.sliding_window_infer import (
    _context_margin_pixels,
    _edge_touch_margin_pixels,
    _effective_overlap_pixels,
)


class SlidingWindowOverlapTest(unittest.TestCase):
    def test_requested_overlap_is_not_overridden_by_context_margin(self):
        env = {
            "BL_SLIDING_CONTEXT_MARGIN_RATIO": "0.25",
            "BL_SLIDING_EDGE_TOUCH_MARGIN_RATIO": "0.02",
        }
        with patch.dict(os.environ, env, clear=True):
            patch_size = 400
            context_margin = _context_margin_pixels(patch_size)
            requested, effective = _effective_overlap_pixels(
                patch_size,
                patch_overlap_ratio=0.15,
                context_margin=context_margin,
            )

            self.assertEqual(context_margin, 100)
            self.assertEqual(requested, 60)
            self.assertEqual(effective, requested)
            self.assertEqual(
                _edge_touch_margin_pixels(patch_size, context_margin),
                8,
            )


if __name__ == "__main__":
    unittest.main()
