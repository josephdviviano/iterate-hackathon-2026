# Mechanics:
# 1. Objects are either boxes (targets) or shapes (players).
# 2. Boxes have a color tag (color_9, color_12, color_8, color_11, color_14, or color_None).
#    They are static and occupy a rectangle of size w×h.
# 3. Shapes are of type X, diamond, raw, or plus.
#    - X: active, moves in the direction of the action (1=up,2=down,3=left,4=right).
#      After moving it disappears and a raw shape of color_9 appears at the new location.
#    - diamond: active, moves similarly. If it collides with a box, the diamond disappears,
#      the box changes its color to the diamond's color, and a raw shape of that color
#      appears at the diamond's previous location.
#    - raw: inactive, stays in place for one step, then turns into a plus shape.
#    - plus: inactive, static. When a new plus is created at a location, its name
#      increments the local counter for that location.
# 4. Action 5 toggles the active flag of all raw and diamond shapes.
# 5. Rendering: background is '5'. Each object paints its rectangle with a single
#    colour code derived from its color tag:
#        color_9   -> 'b'
#        color_12  -> 'c'
#        color_8   -> 'a'
#        color_11  -> 'e'
#        color_14  -> 'd'
#        color_None-> '5'
#    Objects are rendered in the order they appear in the state list; later objects
#    overwrite earlier ones.

import copy

COLOR_MAP = {
    'color_9':  'b',
    'color_12': 'c',
    'color_8':  'a',
    'color_11': 'e',
    'color_14': 'd',
    'color_None': '5'
}

class TransitionModel:
    def __init__(self):
        # mapping from (x,y) to next plus index
        self.plus_index_map = {}
        self.last_state = None

    def init_plus_index_map(self, state):
        for obj in state:
            if obj.get('type') == 'player' and 'plus' in obj.get('tags', []):
                loc = (obj['x'], obj['y'])
                name = obj['name']
                try:
                    idx = int(name.split('_')[1])
                except:
                    idx = 0
                self.plus_index_map[loc] = max(self.plus_index_map.get(loc, 0), idx + 1)

    def get_next_plus_index(self, loc):
        idx = self.plus_index_map.get(loc, 0)
        self.plus_index_map[loc] = idx + 1
        return idx

    def bbox_intersect(self, a, b):
        return not (a['x'] + a['w'] <= b['x'] or b['x'] + b['w'] <= a['x'] or
                    a['y'] + a['h'] <= b['y'] or b['y'] + b['h'] <= a['y'])

    def transition_function(self, state, action, frame=None):
        new_state = copy.deepcopy(state)

        # init mapping on first call
        if self.last_state is None:
            self.init_plus_index_map(state)

        # if state changed unexpectedly, fallback to stateless default
        if self.last_state is not None and self.last_state != state:
            self.last_state = copy.deepcopy(state)
            return self.render_frame(new_state)

        # toggle active flag for raw and diamond shapes
        if action == 5:
            for obj in new_state:
                if obj.get('type') == 'player' and ('raw' in obj.get('tags', []) or 'diamond' in obj.get('tags', [])):
                    obj['active'] = not obj.get('active', False)
            self.last_state = copy.deepcopy(state)
            return self.render_frame(new_state)

        # movement actions
        if action in (1, 2, 3, 4):
            delta = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            dx, dy = delta[action]
            moving_objs = [o for o in new_state if o.get('type') == 'player' and o.get('active', False) and ('X' in o.get('tags', []) or 'diamond' in o.get('tags', []))]
            for obj in moving_objs:
                new_x = obj['x'] + dx
                new_y = obj['y'] + dy
                new_cx = obj['cx'] + dx
                new_cy = obj['cy'] + dy
                if 'X' in obj.get('tags', []):
                    # X moves: remove X, create raw shape at new location
                    new_state.remove(obj)
                    raw_obj = {
                        'h': obj['h'],
                        'layer': obj['layer'],
                        'name': f"shape_{self.get_next_plus_index((new_x,new_y))}_raw",
                        'tags': ['shape', 'raw', 'color_9', 'active'],
                        'type': 'player',
                        'visible': obj['visible'],
                        'w': obj['w'],
                        'x': new_x,
                        'y': new_y,
                        'cx': new_cx,
                        'cy': new_cy
                    }
                    new_state.append(raw_obj)
                elif 'diamond' in obj.get('tags', []):
                    # diamond moves
                    old_x, old_y = obj['x'], obj['y']
                    old_cx, old_cy = obj['cx'], obj['cy']
                    obj['x'] = new_x
                    obj['y'] = new_y
                    obj['cx'] = new_cx
                    obj['cy'] = new_cy
                    # check collision with boxes
                    for box in new_state:
                        if box.get('type') == 'target':
                            if self.bbox_intersect(obj, box):
                                # collision
                                new_state.remove(obj)
                                # change box color
                                old_color = None
                                for t in box['tags']:
                                    if t.startswith('color_'):
                                        old_color = t
                                new_color = obj['tags'][obj['tags'].index('color_')+1] if 'color_' in obj['tags'] else 'color_None'
                                box['tags'] = [t for t in box['tags'] if not t.startswith('color_')]
                                box['tags'].append(new_color)
                                parts = box['name'].split('_')
                                parts[1] = new_color.split('_')[1]
                                box['name'] = '_'.join(parts)
                                # create raw shape at diamond's previous location
                                raw_obj = {
                                    'h': obj['h'],
                                    'layer': obj['layer'],
                                    'name': f"shape_{self.get_next_plus_index((old_x,old_y))}_raw",
                                    'tags': ['shape', 'raw', new_color, 'active'],
                                    'type': 'player',
                                    'visible': obj['visible'],
                                    'w': obj['w'],
                                    'x': old_x,
                                    'y': old_y,
                                    'cx': old_cx,
                                    'cy': old_cy
                                }
                                new_state.append(raw_obj)
                                break

        # transform raw shapes to plus shapes
        raw_objs = [o for o in new_state if o.get('type') == 'player' and 'raw' in o.get('tags', [])]
        for raw in raw_objs:
            loc = (raw['x'], raw['y'])
            idx = self.get_next_plus_index(loc)
            plus_obj = copy.deepcopy(raw)
            plus_obj['name'] = f"shape_{idx}_plus"
            plus_obj['tags'] = [t for t in plus_obj['tags'] if t != 'raw'] + ['plus']
            new_state.remove(raw)
            new_state.append(plus_obj)

        # render frame
        frame_out = self.render_frame(new_state)
        self.last_state = copy.deepcopy(state)
        return frame_out

    def render_frame(self, state):
        # start with background '5'
        frame = [['5'] * 64 for _ in range(64)]
        for obj in state:
            color_tag = None
            for t in obj.get('tags', []):
                if t.startswith('color_'):
                    color_tag = t
                    break
            if color_tag is None:
                color_tag = 'color_None'
            color = COLOR_MAP.get(color_tag, '5')
            for dy in range(obj['h']):
                y = obj['y'] + dy
                if 0 <= y < 64:
                    for dx in range(obj['w']):
                        x = obj['x'] + dx
                        if 0 <= x < 64:
                            frame[y][x] = color
        return frame

# Singleton model instance
model = TransitionModel()

def transition_function(state, action, frame=None):
    return model.transition_function(state, action, frame)
