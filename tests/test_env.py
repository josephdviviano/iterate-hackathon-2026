import json
import subprocess

from committee.committee import Committee, Member
from committee.env import contract, signature
from committee.loader import Transition
from committee.verify import canonical, run_program, train_report

ENGINE = ("def extract_objects(g):\n    return [{'name': 'p', 'x': x, 'y': y} for y in range(len(g)) "
          "for x in range(len(g[0])) if g[y][x] == 9]\n")
MOVE_OBJ = "def transition_function(state, action):\n    return [dict(o, x=o['x'] + 1) for o in state]\n"
READ_FRAME = "def transition_function(state, action, frame):\n    return [{'name': 'p', 'x': frame[1].index(9) + 1, 'y': 1}]\n"
MOVE_FRAME = ("def transition_function(state, action, frame):\n    f = [r[:] for r in frame]\n"
              "    x = f[1].index(9)\n    f[1][x] = 0\n    f[1][x + 1] = 9\n    return f\n")


def _grid(x):
    g = [[0] * 4 for _ in range(4)]
    g[1][x] = 9
    return g


def _split(n_train=2, n_test=1):
    ts = [Transition(i + 1, 1, 1, None, False, _grid(i), _grid(i + 1), [{"name": "p", "x": i, "y": 1}],
                     [{"name": "p", "x": i + 1, "y": 1}]) for i in range(n_train + n_test)]
    return ts[:n_train], ts[n_train:]


def test_runner_compares_objects_or_frames_by_mode_and_keys_keep_frame_rows_in_order():
    import pytest

    train, test = _split()
    assert run_program(MOVE_OBJ, train, test).test_pass == [True]
    assert run_program(READ_FRAME, train, test, mode="frame").test_pass == [True]
    v = run_program(MOVE_FRAME, train, test, mode="frame_out", engine_src=ENGINE)
    assert v.consistent and v.test_pass == [True] and v.test_preds == [test[0].after_grid]
    assert v.test_objs == [test[0].after_objs] and v.mode == "frame_out"
    stray = MOVE_FRAME.replace("    return f", "    f[3][3] = 7\n    return f")
    v = run_program(stray, train, test, mode="frame_out", engine_src=ENGINE)
    assert v.test_pass == [False] and v.test_objs == [test[0].after_objs]  # the extractor's view is advisory
    v = run_program(MOVE_FRAME.replace("= 9", "= 9.0"), train, test, mode="frame_out")
    assert v.test_pass == [False] and v.test_preds == [None] and v.test_objs is None
    with pytest.raises(ValueError):
        run_program(MOVE_FRAME, train, test, mode="frame_out", open_loop=True)
    a, b = [[1, 2], [3, 4]], [[3, 4], [1, 2]]
    assert canonical(a) != canonical(b) and canonical(a) == canonical([[1, 2], [3, 4]])
    m = Member("m", MOVE_FRAME, [test[0].after_grid], 10, mode="frame_out", test_objs=[test[0].after_objs])
    wrong = Member("w", MOVE_FRAME, [b], 10, mode="frame_out", test_objs=[[]])
    ev = Committee([m, wrong]).evaluate_on(test)
    assert ev["member_accuracy"] == [1.0, 0.0] and ev["n_distinct_behaviours"] == 2 and m.objects(0) == test[0].after_objs
    assert canonical({"a": 1}) == ['{"a": 1}'] and Member("d", "", [{"a": 1}], 1).keys == ['["{\\"a\\": 1}"]']
    with pytest.raises(ValueError):
        Committee([m, Member("o", MOVE_OBJ, [test[0].after_objs], 10)])
    passed, report = train_report(MOVE_FRAME, train, mode="frame_out")
    assert passed and report.startswith("2/2 frames replay exactly")
    passed, report = train_report(stray, train, mode="frame_out")
    assert not passed and "1 cells differ: (3,3) want 0 got 7" in report


def test_workspace_checker_agrees_with_the_verifier_in_every_mode(tmp_path):
    from committee.synth import write_workspace

    train, _ = _split()
    for mode, good in (("objects", MOVE_OBJ), ("frame", READ_FRAME), ("frame_out", MOVE_FRAME)):
        ws = tmp_path / mode
        ws.mkdir()
        write_workspace(ws, train, "seed text", mode)
        task = (ws / "TASK.md").read_text()
        assert signature(mode) in task and "seed text" in task and task.count("transition_function(") == 1
        assert ("`frame`: the 64x64 before frame" in task) == (mode != "objects")
        assert ("list of object dicts with the same" in task) == (mode != "frame_out") and ("after frame in the same form" in task) == (mode == "frame_out")
        rows = json.loads((ws / "buffer.json").read_text())
        assert ("frame" in rows[0]) == (mode != "objects") and ("after_frame" in rows[0]) == (mode == "frame_out")
        stub_out = subprocess.run(["python3", "check.py"], cwd=ws, capture_output=True, text=True).stdout
        assert "ALL PASS" not in stub_out
        (ws / "program.py").write_text(good)
        out = subprocess.run(["python3", "check.py"], cwd=ws, capture_output=True, text=True).stdout
        assert "ALL PASS" in out and out.startswith("2/2 frames" if mode == "frame_out" else "2/2 transitions")
        (ws / "program.py").write_text("# exact replay of the data\n" + good)
        out = subprocess.run(["python3", "check.py"], cwd=ws, capture_output=True, text=True).stdout
        assert out.startswith("FORBIDDEN") and "ALL PASS" not in out
        assert run_program("# exact replay of the data\n" + good, train, [], mode=mode).error.startswith("forbidden")
    assert "cells differ" in stub_out  # the frame_out stub returns the before frame


def test_run_record_carries_the_mode_and_the_object_view_through_evaluation(tmp_path, monkeypatch):
    import pytest

    from committee import experiment
    from committee.cegis import counterexample_text
    from committee.evaluate import members_from
    from committee.experiment import check_mode, run_mode, run_split
    from committee.synth import SynthResult

    train, test = _split()
    monkeypatch.setattr(experiment, "synthesize_any", lambda tr, seed, cfg: SynthResult(MOVE_FRAME, seed))
    base = tmp_path / "cond"
    run_split(train, test, base, [None, None], {"mode": "frame_out"}, "t", engine_src=ENGINE)
    assert run_mode(base) == "frame_out" and (base / "run0" / "test_objs.json").exists()
    members = members_from(base, train, test)
    assert len(members) == 2 and members[0].mode == "frame_out" and members[0].objects(0) == test[0].after_objs
    assert Committee(members).evaluate_on(test)["vote_accuracy"] == 1.0
    with pytest.raises(SystemExit):
        check_mode(base, "objects")
    with pytest.raises(SystemExit):
        experiment.main(["ar25", "--level", "3", "--frame", "--frame-out"])
    from committee import evaluate as evaluate_module
    monkeypatch.setattr(evaluate_module, "engine_source", lambda game, mode: ENGINE)
    for d in base.glob("run*"):
        (d / "test_objs.json").unlink()
    with pytest.raises(ValueError):
        members_from(base, train, test)[0].objects(0)  # no game: no object view
    assert members_from(base, train, test, "g")[0].objects(0) == test[0].after_objs  # rebuilt from the stored frames
    assert (base / "run0" / "test_objs.json").exists()
    for d in base.glob("run*"):
        (d / "test_preds.json").unlink()
    with pytest.raises(ValueError):
        members_from(base, train, test)  # a frame_out replay needs the game
    rebuilt = members_from(base, train, test, "g")
    assert rebuilt[0].test_preds == [test[0].after_grid] and rebuilt[0].objects(0) == test[0].after_objs
    refuted = [Member(f"m{k}", MOVE_FRAME, [_grid(0)], 10, "frame_out", [[{"name": "p", "x": 0, "y": 1}]]) for k in range(2)]
    text = counterexample_text(refuted, test, [0])
    assert "2 program(s) predicted" in text and "2 cells differ" in text and "observed but not predicted" in text


def test_live_member_process_and_predict_pass_the_frame():
    from committee.live import MemberProcess, predict

    before, grid = [{"name": "p", "x": 0, "y": 1}], _grid(0)
    proc = MemberProcess(MOVE_FRAME, mode="frame_out")
    try:
        assert proc.step(before, 1, grid) == _grid(1)
    finally:
        proc.close()
    proc = MemberProcess(MOVE_FRAME.replace("= 9", "= 9.0"), mode="frame_out")
    try:
        assert proc.step(before, 1, grid) is None  # a malformed frame is an error, as under run_program
    finally:
        proc.close()
    assert predict(MOVE_FRAME, before, [1, 1], grid, "frame_out") == [_grid(1), _grid(1)]
    assert predict(MOVE_OBJ, before, [1]) == [[{"name": "p", "x": 1, "y": 1}]]


def test_frame_out_gate_reloads_the_extractor_per_frame_and_refuses_a_view_that_differs(monkeypatch):
    import pytest

    from committee import env
    from committee.loader import run_extractor

    # module state shifts every object after the first call; a reload per frame removes the shift
    stateful = "N = [0]\n" + ENGINE.replace("'x': x,", "'x': x + N[0] - 1,").replace("def extract_objects(g):\n", "def extract_objects(g):\n    N[0] += 1\n")
    train, test = _split()
    grids = [t.after_grid for t in train + test]
    assert run_extractor(stateful, grids, fresh=True) == run_extractor(ENGINE, grids) != run_extractor(stateful, grids)
    monkeypatch.setattr(env, "load_bundle", lambda game: game)
    monkeypatch.setattr(env, "final_engine_source", lambda bundle: stateful)
    monkeypatch.setattr(env, "build_buffer", lambda game: train + test)
    assert env.engine_source("reloaded", "frame_out") == stateful and env.engine_source("reloaded", "objects") is None
    stale = [Transition(t.step, t.level, t.action_id, None, False, t.before_grid, t.after_grid, t.before_objs, []) for t in train]
    monkeypatch.setattr(env, "build_buffer", lambda game: stale)
    with pytest.raises(ValueError, match="2 of 2 transitions"):
        env.engine_source("stale", "frame_out")
    monkeypatch.setattr(env, "build_buffer", lambda game: train + test)
    with pytest.raises(ValueError):
        env.engine_source("stale", "frame_out")  # the refusal is remembered, the pass is not repeated
