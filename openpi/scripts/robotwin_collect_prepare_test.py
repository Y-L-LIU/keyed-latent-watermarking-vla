import pathlib

from . import robotwin_collect_prepare


def test_build_plan_places_robotwin_collection_under_data_sdh():
    plan = robotwin_collect_prepare.build_plan(
        data_root=pathlib.Path("/tmp/data_sdh"),
        task_name="beat_block_hammer",
        task_config="demo_clean",
        gpu_id="0",
    )

    assert plan.repo_dir == pathlib.Path("/tmp/data_sdh/robotwin/RoboTwin")
    assert plan.data_dir == pathlib.Path("/tmp/data_sdh/robotwin/data")
    assert plan.project_data_path == pathlib.Path("/tmp/data_sdh/robotwin/RoboTwin/data")
    assert plan.raw_output_dir == pathlib.Path("/tmp/data_sdh/robotwin/data/beat_block_hammer/demo_clean")
    assert "collect_data.sh beat_block_hammer demo_clean 0" in " ".join(plan.collect_command)
    assert plan.should_start_collection is False
