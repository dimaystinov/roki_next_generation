"""Registry behavior with pre-registered and explicitly selected MYRIAD plugins."""
import sys
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET
import pytest


@pytest.mark.parametrize('builtin', [False, True])
def test_default_detector_accepts_builtin_or_external_plugin(monkeypatch, builtin):
    import Soccer.Vision.neural as module
    class Core:
        def __init__(self, config=None):
            self.registry = {'MYRIAD': 'builtin'} if builtin else {}
        def get_versions(self, name):
            return {name: NS(description='NCS2 native blob plugin')} if name in self.registry else {}
        def register_plugin(self, path, name):
            if name in self.registry:
                raise RuntimeError('already registered')
            self.registry[name] = path
    monkeypatch.setitem(sys.modules, 'openvino', NS(Core=Core))
    core = module.create_ncs2_core()
    assert core.get_versions('MYRIAD')['MYRIAD'].description == 'NCS2 native blob plugin'


def test_explicit_plugin_overrides_builtin_without_dispatch_group(monkeypatch, tmp_path):
    import Soccer.Vision.neural as module
    selected = str(tmp_path / 'a & b' / 'custom.so')
    class Core:
        def __init__(self, config=None):
            self.selected = 'builtin'
            if config:
                plugin = ET.parse(config).find('./plugins/plugin')
                assert plugin.attrib['name'] == 'MYRIAD'
                self.selected = plugin.attrib['location']
        def register_plugin(self, *args):
            raise RuntimeError('must not append a conflicting MYRIAD candidate')
    monkeypatch.setitem(sys.modules, 'openvino', NS(Core=Core))
    assert module.create_ncs2_core(selected).selected == selected
