from committee.loader import Transition, completion_steps, parse_action, settled_steps, temporal_split
from committee.matrix import effect_signature, pair_objects


def _bundle():
    g0 = [[0] * 64 for _ in range(64)]
    return {
        # step 0: initial frame. step 1: one tick frame then the settled frame.
        "frames": [
            {"s": 0, "g": g0, "a": "RESET"},
            {"s": 1, "d": [[3, 4, 7]], "a": "ACTION2"},
            {"s": 1, "d": [[3, 4, 9], [5, 5, 1]]},
            {"s": 2, "a": "click(10,20)"},
        ],
        "level_steps": [3],
    }


def test_settled_steps_keep_last_frame_of_each_step_and_its_action():
    steps = settled_steps(_bundle())
    assert [s.action_label for s in steps] == ["RESET", "ACTION2", "click(10,20)"]
    assert steps[1].grid[4][3] == 9 and steps[1].grid[5][5] == 1
    assert steps[0].grid[4][3] == 0
    assert steps[2].grid == steps[1].grid


def test_parse_action_and_completion_steps():
    assert parse_action("RESET") == (0, None)
    assert parse_action("ACTION4") == (4, None)
    assert parse_action("click(10,20)") == (6, (10, 20))
    assert completion_steps(_bundle()) == {2}


def test_pairing_prefers_same_position_and_signature_lists_changed_fields():
    a = {"name": "p", "x": 1, "y": 1, "type": "player"}
    b = {"name": "p", "x": 5, "y": 1, "type": "player"}
    after = [{"name": "p", "x": 5, "y": 1, "type": "player"}, {"name": "p", "x": 1, "y": 1, "type": "player"}]
    pairs = pair_objects([a, b], after)
    assert pairs[0][1] is after[1] and pairs[1][1] is after[0]
    assert effect_signature(a, {"name": "q", "x": 2, "y": 1, "type": "player"}) == "x"
    assert effect_signature(a, a) == "no_change"
    assert effect_signature(None, a) == "born" and effect_signature(a, None) == "gone"


def test_temporal_split_within_level_and_across_levels():
    def tr(step, level, action_id=1, advance=False):
        return Transition(step, level, action_id, None, advance, [], [], [], [])

    ts = [tr(1, 1), tr(2, 1), tr(3, 1), tr(4, 1, 0), tr(5, 1, advance=True), tr(6, 2), tr(7, 2, advance=True)]
    train, test = temporal_split(ts, 1, 0.4)
    assert [t.step for t in train] == [1] and [t.step for t in test] == [2, 3]
    train, test = temporal_split(ts, 1, 1.0, test_level=2)
    assert [t.step for t in train] == [1, 2, 3] and [t.step for t in test] == [6]


def test_static_terrain_keeps_unchanging_cells_outside_objects_and_drops_moving_ones():
    from committee.loader import Transition
    from committee.terrain import add_terrain, static_terrain

    def grid(player_x, door):
        g = [[1] * 8 for _ in range(6)]
        for x in range(8):
            g[0][x] = 7                      # a wall row that never changes
        g[3][player_x] = 5                   # the player, extracted as an object
        if door:
            for x in range(4):
                g[5][x] = 8                  # a door that is open later: not static
        return g

    def tr(step, x0, x1, door0, door1):
        before = [{"name": "p", "type": "player", "x": x0, "y": 3, "w": 1, "h": 1}]
        after = [{"name": "p", "type": "player", "x": x1, "y": 3, "w": 1, "h": 1}]
        return Transition(step, 1, 1, None, False, grid(x0, door0), grid(x1, door1), before, after)

    train = [tr(1, 2, 3, True, False), tr(2, 3, 4, False, False)]
    terrain = static_terrain(train)
    assert [(o["x"], o["y"], o["w"], o["h"], o["colour"]) for o in terrain] == [(0, 0, 8, 1, 7)]
    out = add_terrain(train, terrain)
    assert [o["type"] for o in out[0].before_objs] == ["player", "terrain"] and out[0].after_objs[-1] == terrain[0]
