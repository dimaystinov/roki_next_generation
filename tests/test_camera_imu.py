"""Camera IDs must select the matching STM sample, including frame zero."""
import importlib
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def channel():
    with patch.dict(sys.modules, {'Roki': ModuleType('Roki')}):
        module = importlib.import_module('Soccer.Motion.class_stm_channel')
    result = module.STM_channel.__new__(module.STM_channel)
    result.glob = NS(camera_streaming=False, camera_down_Flag=False)
    frame = NS(Orientation=NS(X=0, Y=0, Z=0, W=1), Timestamp=NS(TimeS=1, TimeNS=2))
    result.mb = NS(GetIMUFrame=Mock(return_value=(True, frame)),
                   GetIMULatest=Mock(return_value=(True, frame)),
                   GetError=Mock(return_value='test error'), GetStrobeWidth=Mock(return_value=0),
                   GetIMUContainerInfo=Mock(return_value=(True, NS(First=10, NumAv=20, MaxFrames=100))))
    return result


@pytest.mark.parametrize('sequence', [0, 17, 0xffffffff])
def test_image_number_selects_imu_even_without_background_vision(channel, sequence):
    assert channel.read_quaternion_from_imu_in_head(sequence) == (0, 0, 0, 1, 1, 2)
    channel.mb.GetIMUFrame.assert_called_once_with(sequence)
    channel.mb.GetIMULatest.assert_not_called()


def test_no_image_number_uses_latest_imu(channel):
    channel.read_quaternion_from_imu_in_head()
    channel.mb.GetIMULatest.assert_called_once()
    channel.mb.GetIMUFrame.assert_not_called()


def test_failed_latest_read_does_not_compare_none_with_int(channel):
    channel.mb.GetIMULatest.return_value = (False, None)
    assert channel.read_quaternion_from_imu_in_head() == (0, 0, 0, 0, 0, 0)


def test_expired_image_sample_is_not_replaced_by_latest_imu(channel):
    channel.mb.GetIMUFrame.return_value = (False, None)
    assert channel.read_quaternion_from_imu_in_head(3) == (0, 0, 0, 0, 0, 0)
    assert channel.glob.camera_down_Flag
    channel.mb.GetIMULatest.assert_not_called()


def test_failed_imu_info_does_not_dereference_missing_info(channel):
    channel.mb.GetIMUFrame.return_value = (False, None)
    channel.mb.GetIMUContainerInfo.return_value = (False, None)
    assert channel.read_quaternion_from_imu_in_head(3) == (0, 0, 0, 0, 0, 0)
