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
"""Domain-agnostic Active Learning and Adaptive Physics Module architecture."""

from collections.abc import Mapping
import enum
from typing import Any

import numpy as np
from torax._src import data_harvesting


@enum.unique
class FallbackMode(enum.Enum):
  """Fallback mode for active learning high-fidelity query."""

  FULL_PROFILE = "full_profile"
  PER_FACE = "per_face"


class AdaptivePhysicsEngine:
  """Coordinates surrogate uncertainty gating, fallback execution, and harvesting."""

  def __init__(
      self,
      uncertainty_threshold: float = 0.20,
      fallback_mode: FallbackMode | str = FallbackMode.FULL_PROFILE,
      sink: data_harvesting.StagingSink | None = None,
  ):
    self.uncertainty_threshold = uncertainty_threshold
    if isinstance(fallback_mode, str):
      fallback_mode = FallbackMode(fallback_mode)
    self.fallback_mode = fallback_mode
    self.sink = sink

  def acquisition_function(
      self,
      relative_uncertainty: np.ndarray,
  ) -> tuple[bool, np.ndarray]:
    """Active learning acquisition function deciding high-fidelity queries.

    Args:
      relative_uncertainty: 1D array of uncertainty estimates (e.g. sigma / |mu|).

    Returns:
      needs_fallback: True if any face requires high-fidelity solver execution.
      run_mask: Boolean 1D array indicating which radial points require high-fidelity solver.
    """
    exceeds = relative_uncertainty > self.uncertainty_threshold
    any_exceeds = bool(np.any(exceeds))

    if not any_exceeds:
      return False, np.zeros_like(relative_uncertainty, dtype=bool)

    if self.fallback_mode == FallbackMode.FULL_PROFILE:
      return True, np.ones_like(relative_uncertainty, dtype=bool)
    elif self.fallback_mode == FallbackMode.PER_FACE:
      return True, exceeds
    else:
      raise ValueError(f"Unknown fallback mode: {self.fallback_mode}")

  decide_fallback = acquisition_function  # Alias for backward compatibility

  def fuse(
      self,
      surrogate_vals: np.ndarray,
      high_fidelity_vals: np.ndarray,
      run_mask: np.ndarray,
  ) -> np.ndarray:
    """Combines surrogate and high-fidelity evaluations across radial faces."""
    if self.fallback_mode == FallbackMode.FULL_PROFILE:
      return high_fidelity_vals

    # Splicing in per_face mode without redundant spatial smoothing
    return np.where(run_mask, high_fidelity_vals, surrogate_vals)

  def fuse_and_smooth(
      self,
      surrogate_vals: np.ndarray,
      high_fidelity_vals: np.ndarray,
      run_mask: np.ndarray,
      coords: np.ndarray | None = None,
  ) -> np.ndarray:
    """Backward-compatible alias for fuse."""
    del coords
    return self.fuse(surrogate_vals, high_fidelity_vals, run_mask)

  def harvest(
      self,
      fingerprint: str,
      inputs: Mapping[str, np.ndarray],
      high_fidelity_outputs: Mapping[str, np.ndarray],
      uncertainties: Mapping[str, np.ndarray] | None = None,
      metadata: Mapping[str, Any] | None = None,
  ) -> None:
    """Dispatches evaluated input-output pairs to the data harvesting sink."""
    if self.sink is None:
      return

    sample = data_harvesting.HarvestSample(
        fingerprint=fingerprint,
        inputs=inputs,
        outputs=high_fidelity_outputs,
        uncertainties=uncertainties,
        metadata=metadata,
    )
    self.sink.record(sample)

  harvest_if_enabled = harvest  # Alias for backward compatibility
