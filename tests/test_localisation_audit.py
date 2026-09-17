"""Hardware-free algorithm audit. Strict xfails are unresolved defects, not fixes.

Run with --runxfail to see the assertions fail against the current implementation.
The actual Local/PF methods are imported; no copied algorithm is tested.
"""
import json
import math
from multiprocessing import Array, Value
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from Soccer.Localisation.class_Local import Local
from Soccer.Localisation.PF.ParticleFilter import Agent, ParticleFilter, gaussian, weight_calculation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def local():
    instance = Local.__new__(Local)  # Skip hardware/visualisation initialization only.
    instance.glob = NS(use_particle_filter=True, pf_coord=[0., 0., 0.], monitor_is_on=False, SIMULATION=5)
    instance.motion = NS(imu_body_yaw=Mock(return_value=0.2), refresh_Orientation=Mock())
    instance.pf_return_coord = Array('f', 4)
    instance.pf_odometry = Array('f', 31)
    instance.pf_command = Value('i', 0)
    instance.coord_for_PF_global = Array('f', 3)
    instance.coord_odometry = [0., 0., 0.]
    instance.coord_odometry_old = [0., 0., 0.]
    instance.coord_shift = [0., 0., 0.]
    instance.cross_points = []
    instance.penalty_points = []
    instance.landmarks_for_PF = {}
    for flag in ('USE_LANDMARKS_FOR_LOCALISATION', 'USE_LINES_FOR_LOCALISATION',
                 'USE_PENALTY_MARKS_FOR_LOCALISATION', 'USE_SINGLE_POST_MEASUREMENT',
                 'DIRECT_COORD_MEASUREMENT_BY_PAIRS_OF_POST'):
        setattr(instance, flag, False)
    return instance


def queued_motion(local, axis):
    # Same count/stride contract used by particle_filter_as_process.
    return sum(local.pf_odometry[i * 3 + 1 + axis]
               for i in range(int(local.pf_odometry[0])))


@pytest.mark.parametrize('sigma', [0.1, 0.4, 2.0])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason='Wrong Gaussian exponent denominator')
def test_gaussian_one_sigma_ratio(sigma):
    assert gaussian(sigma, sigma) / gaussian(0, sigma) == pytest.approx(math.exp(-0.5))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='Particle weights use the same incorrect Gaussian')
def test_landmark_weights_one_sigma():
    particles = [32768, 32768, 32768, 0, 33168, 32768, 32768, 0]
    weights, _ = weight_calculation(2, [0, 0], {'post1': [[0, 0, 1]], 'lines': []},
                                    {'post1': [[0, 0]], 'lines': {}}, 0.4, 0.4, particles)
    assert weights[1] / weights[0] == pytest.approx(math.exp(-0.5), abs=1e-4)


@pytest.mark.xfail(strict=True, raises=ValueError, reason='Empty pose assigned to fixed-length shared Array')
def test_pf_without_direct_visual_pose(local):
    local.pf_update(False)
    assert local.coord_for_PF_global[:] == [1000., 1000., 1000.]
    assert local.pf_command.value == 3


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='Same absolute yaw error is queued repeatedly')
def test_unchanged_imu_is_not_integrated_twice(local):
    local.correct_yaw_in_pf()
    local.correct_yaw_in_pf()
    assert queued_motion(local, 2) == pytest.approx(0.2)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='Saturated yaw queue writes past published count')
def test_saturated_queue_preserves_yaw(local):
    local.pf_odometry[0] = 9
    local.correct_yaw_in_pf()
    assert queued_motion(local, 2) == pytest.approx(0.2)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='Saturated translation queue writes past published count')
def test_saturated_queue_preserves_translation(local):
    local.motion.imu_body_yaw.return_value = 0
    local.pf_odometry[0] = 9
    local.coord_shift = [0.1, 0., 0.]
    local.coordinate_record(odometry=True, shift=True)
    assert queued_motion(local, 0) == pytest.approx(0.1)


@pytest.mark.xfail(strict=True, raises=KeyError, reason='Real defaults lack mandatory USE_PARTICLE_FILTER')
def test_real_defaults_define_pf_switch():
    directory = ROOT / 'Init_params/defaults/Real'
    params = json.loads((directory / 'Real_params.json').read_text())
    params.update(json.loads((directory / 'Real_params_2.json').read_text()))
    assert isinstance(params['USE_PARTICLE_FILTER'], bool)


@pytest.mark.parametrize('yaw', [-9 * math.pi, -math.pi, -0.1, 0., math.pi, 9 * math.pi])
def test_angle_normalization_preserves_direction(local, yaw):
    normalized = local.normalize_yaw(yaw)
    assert -math.pi <= normalized <= math.pi
    assert math.sin(normalized) == pytest.approx(math.sin(yaw), abs=1e-12)
    assert math.cos(normalized) == pytest.approx(math.cos(yaw), abs=1e-12)


@pytest.mark.parametrize('yaw', [0., math.pi / 2, math.pi, -math.pi / 2])
def test_forward_odometry_rotates_into_field(local, yaw):
    local.motion.imu_body_yaw.return_value = yaw
    local.coord_odometry = [0.3, -0.2, 0.]
    local.coord_shift = [0.1, 0., 0.]
    local.refresh_odometry()
    assert local.coord_odometry == pytest.approx([0.3 + 0.1 * math.cos(yaw),
                                                -0.2 + 0.1 * math.sin(yaw), yaw])


def test_agent_closed_square():
    agent = Agent()
    for _ in range(4):
        agent.move(1., 0., math.pi / 2)
    assert (agent.x, agent.y, agent.yaw) == pytest.approx((0., 0., 0.), abs=1e-12)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='Negative previous yaw chooses incorrect unwrap around pi')
def test_particle_heading_mean_across_pi():
    pf = ParticleFilter.__new__(ParticleFilter)
    pf.myrobot = Agent(yaw=-math.pi + 0.1)
    particles = [32768, 32768, 32768 + int((math.pi - 0.1) * 1000), 10000,
                 32768, 32768, 32768 + int((math.pi + 0.1) * 1000), 10000]
    pf.update_coord(particles, 2)
    assert abs(pf.norm_yaw(pf.myrobot.yaw - math.pi)) < 0.002


@pytest.mark.parametrize('position', [(0., 0.), (0.8, 0.4), (-0.8, -0.4)])
@pytest.mark.parametrize('post_ids', [(1, 2), (3, 4), (1, 2, 3, 4)])
def test_goal_geometry_on_exact_measurements(local, position, post_ids):
    local.glob.landmarks = json.loads((ROOT / 'Init_params/defaults/Real/Real_landmarks.json').read_text())
    x, y = position
    posts = []
    for number in post_ids:
        px, py = local.glob.landmarks[f'post{number}'][0]
        posts.append([math.atan2(py - y, px - x), math.hypot(px - x, py - y), number])
    assert local.coord_calculation(posts)
    assert local.coord_visible == pytest.approx([x, y, 0.2], abs=1e-10)


def test_no_posts_produce_no_visual_fix(local):
    assert local.coord_calculation([]) is False
