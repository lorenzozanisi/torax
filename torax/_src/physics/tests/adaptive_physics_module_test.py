# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for domain-agnostic adaptive physics module."""

import shutil
import tempfile
from absl.testing import absltest
import numpy as np
from torax._src import data_harvesting
from torax._src.physics import adaptive_physics_module


class AdaptivePhysicsModuleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_acquisition_function_full_profile(self):
    engine = adaptive_physics_module.AdaptivePhysicsEngine(
        uncertainty_threshold=0.25,
        fallback_mode=adaptive_physics_module.FallbackMode.FULL_PROFILE,
    )

    # Case 1: All uncertainties low
    rel_unc_low = np.array([0.1, 0.05, 0.20, 0.15])
    needs_fallback, mask = engine.acquisition_function(rel_unc_low)
    self.assertFalse(needs_fallback)
    self.assertFalse(np.any(mask))

    # Case 2: One point exceeds threshold
    rel_unc_high = np.array([0.1, 0.05, 0.30, 0.15])
    needs_fallback, mask = engine.acquisition_function(rel_unc_high)
    self.assertTrue(needs_fallback)
    # In full_profile mode, entire profile is flagged
    self.assertTrue(np.all(mask))

  def test_acquisition_function_per_face(self):
    engine = adaptive_physics_module.AdaptivePhysicsEngine(
        uncertainty_threshold=0.25,
        fallback_mode=adaptive_physics_module.FallbackMode.PER_FACE,
    )

    rel_unc = np.array([0.1, 0.35, 0.15, 0.40])
    needs_fallback, mask = engine.acquisition_function(rel_unc)
    self.assertTrue(needs_fallback)
    np.testing.assert_array_equal(mask, [False, True, False, True])

  def test_fuse(self):
    engine = adaptive_physics_module.AdaptivePhysicsEngine(
        uncertainty_threshold=0.25,
        fallback_mode=adaptive_physics_module.FallbackMode.PER_FACE,
    )

    surr = np.ones(5) * 1.0
    hi_fi = np.ones(5) * 5.0
    mask = np.array([False, False, True, False, False])

    fused = engine.fuse(surr, hi_fi, mask)
    self.assertEqual(fused.shape, surr.shape)
    np.testing.assert_allclose(fused, [1.0, 1.0, 5.0, 1.0, 1.0])

  def test_fuse_full_profile(self):
    engine = adaptive_physics_module.AdaptivePhysicsEngine(
        uncertainty_threshold=0.25,
        fallback_mode=adaptive_physics_module.FallbackMode.FULL_PROFILE,
    )

    surr = np.ones(5) * 1.0
    hi_fi = np.ones(5) * 5.0
    mask = np.ones(5, dtype=bool)

    fused = engine.fuse(surr, hi_fi, mask)
    np.testing.assert_allclose(fused, hi_fi)

  def test_harvest(self):
    sink = data_harvesting.StagingSink(
        output_dir=self.test_dir, run_id="harvest_test"
    )
    engine = adaptive_physics_module.AdaptivePhysicsEngine(sink=sink)

    engine.harvest(
        fingerprint="tglf_fingerprint123",
        inputs={"RLTS_1": np.ones(5)},
        high_fidelity_outputs={"efi_gb": np.ones(5) * 2.0},
    )
    flushed_path = sink.flush()
    self.assertIsNotNone(flushed_path)


if __name__ == "__main__":
  absltest.main()
