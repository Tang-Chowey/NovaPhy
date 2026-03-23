import importlib.util
import json
import sys
from pathlib import Path


def _load_demo_module():
    root = Path(__file__).resolve().parents[2]
    demo_path = root / 'demos' / 'demo_basic_urdf.py'
    spec = importlib.util.spec_from_file_location('demo_basic_urdf', demo_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_basic_urdf_demo_headless_outputs(tmp_path):
    module = _load_demo_module()
    cfg = module.DemoConfig()
    cfg.frames = 40
    cfg.output_dir = str(tmp_path / 'basic_urdf_outputs')
    outputs = module.run_demo(cfg)

    for key in ['summary_json', 'joint_trace_csv', 'contact_trace_csv', 'config_json']:
        assert Path(outputs[key]).exists()

    summary = json.loads(Path(outputs['summary_json']).read_text(encoding='utf-8'))
    assert summary['joint_names'] == list(module.DEFAULT_STANDING_POSE.keys())
    assert summary['filtered_link_pair_count'] > 0


def test_basic_urdf_demo_reaches_terminal_height_and_slows_down(tmp_path):
    module = _load_demo_module()
    cfg = module.DemoConfig()
    cfg.frames = 400
    cfg.output_dir = str(tmp_path / 'basic_urdf_terminal')
    outputs = module.run_demo(cfg)

    summary = json.loads(Path(outputs['summary_json']).read_text(encoding='utf-8'))
    assert abs(summary['final_root_height'] - 0.46) < 0.02
    assert summary['final_max_abs_qd'] < 0.15


def test_basic_urdf_demo_avoids_robot_self_contacts_when_disabled(tmp_path):
    module = _load_demo_module()
    cfg = module.DemoConfig()
    cfg.frames = 220
    cfg.output_dir = str(tmp_path / 'basic_urdf_contacts')
    outputs = module.run_demo(cfg)

    summary = json.loads(Path(outputs['summary_json']).read_text(encoding='utf-8'))
    assert summary['contact_steps'] > 0
    assert summary['ground_only_contacts'] is True
