import numpy as np
import pytest

from analysis.anatomical_compartment_validation.analyze_synthetic_compartments import (
    classify_compartment_points,
    local_lollipop_coordinates,
)


pytestmark = pytest.mark.fast


def test_local_lollipop_coordinates_default_c_axis_matches_generator_convention():
    coords = np.asarray(
        [
            [5, 5, 4],
            [5, 5, 5],
            [5, 5, 6],
        ]
    )

    x_rel, perp1, perp2 = local_lollipop_coordinates(
        coords_ijk=coords,
        shape=(11, 11, 11),
        rotation_zyx_deg=(0.0, 0.0, 0.0),
        canal_axis="c",
    )

    assert np.allclose(x_rel, [1.0, -0.0, -1.0])
    assert np.allclose(perp1, [0.0, 0.0, 0.0])
    assert np.allclose(perp2, [0.0, 0.0, 0.0])


def test_classify_compartment_points_separates_stem_transition_and_bulb():
    x_rel = np.asarray([2.0, -1.5, -2.4, -6.0])
    perp1 = np.zeros_like(x_rel)
    perp2 = np.zeros_like(x_rel)
    geom = {
        "canal_base_radius_init": 2.0,
        "canal_apex_radius_init": 1.0,
        "canal_length_init": 4.0,
        "bulb_radius_init": 3.0,
    }
    phase = {
        "cpa_lob_amp": 0.0,
        "cpa_lob_phase_1": 0.0,
        "cpa_lob_phase_2": 0.0,
        "cpa_bias_1": 0.0,
        "cpa_bias_2": 0.0,
    }

    compartments = classify_compartment_points(x_rel, perp1, perp2, geom, phase)

    assert compartments["stem"][0]
    assert compartments["transition"][1]
    assert compartments["bulb"][2]
    assert not compartments["union"][3]
