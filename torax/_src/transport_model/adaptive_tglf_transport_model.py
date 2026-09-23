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
"""Active Learning Adaptive TGLF transport model coupling surrogate with TGLF."""

from collections.abc import Callable, Sequence
from concurrent import futures
import dataclasses
from typing import Any

import jax
import numpy as np
from torax._src import acquisition_function
from torax._src import array_typing
from torax._src import data_harvesting
from torax._src.geometry import geometry
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import tglfnn_ukaea_transport_model
from torax._src.transport_model import transport_coeffs
from torax._src.transport_model.tglf import defaults as tglf_defaults
from torax._src.transport_model.tglf import tglf_transport_model
from typing_extensions import override

# Type alias for high-fidelity evaluation worker
HighFidelitySolverFn = Callable[..., Any]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(tglf_based_transport_model.RuntimeParams):
  """Runtime parameters for AdaptiveTGLFTransportModel."""

  tglf_settings: tuple[tuple[str, Any], ...] = ()
  output_directory: str = "/tmp/torax_tglf_runs"
  uncertainty_threshold: float = 0.20
  fallback_mode: acquisition_function.FallbackMode = (
      acquisition_function.FallbackMode.FULL_PROFILE
  )
  enable_data_harvesting: bool = True
  harvest_output_dir: str = "/tmp/torax_harvest"


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class AdaptiveTGLFTransportModel(
    tglf_based_transport_model.TGLFBasedTransportModel
):
  """Adaptive TGLF transport model coupling surrogate NN inference with TGLF fallback.

  Evaluates fast neural network surrogate models (such as TGLFNN) and calculates
  predictive uncertainty. If the uncertainty exceeds a user-configured threshold,
  it falls back to the high-fidelity TGLF solver and optionally stages the
  evaluation pairs into a local Parquet dataset for active learning surrogate retraining.
  """

  surrogate_model: tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel
  executor: futures.Executor = dataclasses.field(metadata={"hash_by_id": True})
  sink: data_harvesting.StagingSink | None = dataclasses.field(
      default=None, metadata={"hash_by_id": True}
  )
  high_fidelity_solver_fn: HighFidelitySolverFn = dataclasses.field(
      default=tglf_transport_model._run_single_tglf,
      metadata={"hash_by_id": True},
  )
  flux_channels: tuple[str, ...] = ("pfi_gb", "efe_gb", "efi_gb")

  @override
  def call_implementation(
      self,
      transport_runtime_params: transport_runtime_params_lib.ComponentRuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
  ) -> transport_coeffs.TransportCoeffs:
    assert isinstance(transport_runtime_params, RuntimeParams)

    tglf_inputs = self._prepare_tglf_inputs(
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        poloidal_velocity_multiplier=runtime_params.neoclassical.poloidal_velocity_multiplier,
        two_point_mask=two_point_mask,
    )
    n_faces = len(geo.rho_face_norm)

    # 1. Fast surrogate prediction with uncertainty (single forward pass)
    surrogate_means, surrogate_variances = (
        self.surrogate_model.predict_with_uncertainty(tglf_inputs)
    )
    relative_uncertainty = self.surrogate_model.compute_relative_uncertainty(
        means=surrogate_means,
        variances=surrogate_variances,
        flux_names=self.flux_channels,
    )

    # 2. Extract settings for TGLF solver fallback
    local_settings_dict = dataclasses.asdict(tglf_inputs)
    valid_keys = tglf_defaults.TGLF_DEFAULTS.keys()
    filtered_local_settings = {
        k: v for k, v in local_settings_dict.items() if k in valid_keys
    }
    global_settings_dict = dict(transport_runtime_params.tglf_settings)

    # 3. Solver fingerprint for data harvesting
    fingerprint = data_harvesting.compute_solver_fingerprint(
        model_name="tglf",
        settings=global_settings_dict,
    )

    # 4. Online data acquisition configured from runtime params
    sink = self.sink or (
        data_harvesting.StagingSink(
            output_dir=transport_runtime_params.harvest_output_dir
        )
        if transport_runtime_params.enable_data_harvesting
        else None
    )
    acquisition = acquisition_function.OnlineDataAcquisition(
        uncertainty_threshold=float(
            transport_runtime_params.uncertainty_threshold
        ),
        fallback_mode=transport_runtime_params.fallback_mode,
        sink=sink,
    )

    def callback(
        local_dict: dict[str, np.ndarray],
        rel_unc: np.ndarray,
        *surr_flux_args: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
      needs_fallback, run_mask = acquisition.acquisition_function(rel_unc)

      if not needs_fallback:
        # Zero expensive TGLF runs executed
        return surr_flux_args

      # Fallback required on indicated faces
      faces_to_run = [i for i in range(n_faces) if run_mask[i]]

      # Generalize over arbitrary surrogate flux channels
      tglf_fluxes = {
          channel: np.copy(arr)
          for channel, arr in zip(self.flux_channels, surr_flux_args)
      }

      submitted_futures = [
          self.executor.submit(
              self.high_fidelity_solver_fn,
              i,
              local_dict,
              global_settings_dict,
          )
          for i in faces_to_run
      ]

      for fut in futures.as_completed(submitted_futures):
        res = fut.result()
        if isinstance(res, dict):
          face_idx = res["face_index"]
          for channel in self.flux_channels:
            if channel in res:
              tglf_fluxes[channel][face_idx] = res[channel]
        elif isinstance(res, (tuple, list)):
          face_idx = res[0]
          for channel, val in zip(self.flux_channels, res[1:]):
            tglf_fluxes[channel][face_idx] = val
        else:
          raise TypeError(f"Unexpected solver result type: {type(res)}")

      # Fuse predictions across all channels
      final_fluxes = [
          acquisition.fuse(surr_arr, tglf_fluxes[channel], run_mask)
          for channel, surr_arr in zip(self.flux_channels, surr_flux_args)
      ]

      # Data harvesting dispatch
      if transport_runtime_params.enable_data_harvesting:
        acquisition.harvest(
            fingerprint=fingerprint,
            inputs=local_dict,
            high_fidelity_outputs=tglf_fluxes,
            uncertainties={"rel_unc": rel_unc},
        )

      return tuple(final_fluxes)

    face_struct = jax.ShapeDtypeStruct(
        shape=(n_faces,), dtype=jax_utils.get_dtype()
    )
    result_shape_dtypes = tuple(face_struct for _ in self.flux_channels)
    surr_flux_list = [surrogate_means[ch] for ch in self.flux_channels]

    final_results = jax.pure_callback(
        callback,
        result_shape_dtypes,
        filtered_local_settings,
        relative_uncertainty,
        *surr_flux_list,
    )
    final_flux_dict = dict(zip(self.flux_channels, final_results))

    return self._make_core_transport(
        electron_heat_flux_GB=final_flux_dict.get(
            "efe_gb", surrogate_means.get("efe_gb")
        ),
        ion_heat_flux_GB=final_flux_dict.get(
            "efi_gb", surrogate_means.get("efi_gb")
        ),
        electron_particle_flux_GB=final_flux_dict.get(
            "pfi_gb", surrogate_means.get("pfi_gb")
        ),
        tglf_inputs=tglf_inputs,
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )
