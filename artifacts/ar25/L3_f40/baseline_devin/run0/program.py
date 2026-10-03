# Arrow actions move the selected mirror axis or piece by one 3-pixel cell.
# Action 5 cycles selection; reflections mirror pieces across the axis.
# Goals hide axis cells and remain yellow at overlapping piece centers.
# Rendered components determine names, masks, and clipped bounding boxes.
# Undo, clicks, and multiple-axis reflection behavior are unconfirmed.
from copy import deepcopy
import json

_last = None
_model = None
SCALE = 3
LIMIT = 63


def canonical(state):
    return sorted(json.dumps(o, sort_keys=True) for o in state)


def cell_footprint(obj, color):
    return {( (obj['x'] + i) // SCALE, (obj['y'] + j) // SCALE)
            for i, row in enumerate(obj.get('pixels', []))
            for j, value in enumerate(row) if value == color}


def connected(points):
    todo = set(points)
    groups = []
    while todo:
        seed = min(todo)
        todo.remove(seed)
        group = {seed}
        stack = [seed]
        while stack:
            x, y = stack.pop()
            for q in ((x-1,y), (x+1,y), (x,y-1), (x,y+1)):
                if q in todo:
                    todo.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def recover(state):
    players = [o for o in state if o['type'] == 'player']
    players.sort(key=lambda o: (o['x'], o['y']))
    walls = [o for o in state if o['type'] == 'wall']
    axes = sorted({(o['x']//3, 'horizontal') if 'horizontal' in o['tags']
                   else (o['y']//3, 'vertical') for o in walls})
    pieces = [cell_footprint(o, 5) for o in players]
    goals = set().union(*(cell_footprint(o, 11) for o in state
                          if o['type'] == 'target'))
    selected = 0
    for i, o in enumerate(players):
        if any(0 in row for row in o['pixels']):
            selected = len(axes) + i
            break
    return {'axes': axes, 'pieces': pieces, 'goals': goals,
            'selected': selected}


def at_goal(cell, model):
    return cell in model['goals']


def within_board(cells):
    return all(0 <= x < LIMIT//3 and 0 <= y < LIMIT//3 for x,y in cells)


def collides_with_axis(cells, axes):
    return any(any((x if orientation == 'horizontal' else y) == value
                   for x,y in cells) for value,orientation in axes)


def update_player(model, index, delta):
    dx,dy = delta
    cells = {(x+dx,y+dy) for x,y in model['pieces'][index]}
    if within_board(cells) and not collides_with_axis(cells, model['axes']):
        model['pieces'][index] = cells


def update_wall(model, index, delta):
    value,orientation = model['axes'][index]
    shift = delta[0] if orientation == 'horizontal' else delta[1]
    new_value = value + shift
    if 0 <= new_value < LIMIT//3:
        model['axes'][index] = (new_value,orientation)


def update_counter(obj):
    return deepcopy(obj)


def paint_cell(canvas, cell, color, center):
    x,y = cell
    for i in range(3):
        for j in range(3):
            p = (3*x+i, 3*y+j)
            if 0 <= p[0] < LIMIT and 0 <= p[1] < LIMIT:
                canvas[p] = center if i == j == 1 else color


def update_target(model, canvas):
    for cell in model['goals']:
        paint_cell(canvas, cell, 11, 11)


def update_reflection(model, canvas):
    cells = set()
    for value, orientation in model['axes']:
        for piece in model['pieces']:
            for x,y in piece:
                q = (2*value-x,y) if orientation == 'horizontal' else (x,2*value-y)
                cells.add(q)
    for cell in cells:
        paint_cell(canvas, cell, 4, 11 if at_goal(cell, model) else 4)


def render(model):
    canvas = {}
    for i,(value,orientation) in enumerate(model['axes']):
        for t in range(LIMIT//3):
            cell = (value,t) if orientation == 'horizontal' else (t,value)
            paint_cell(canvas,cell,10,0 if model['selected'] == i else -1)
    update_target(model,canvas)
    update_reflection(model,canvas)
    for i,piece in enumerate(model['pieces']):
        for cell in piece:
            center = 0 if model['selected'] == len(model['axes'])+i else -1
            if at_goal(cell,model):
                center = 11
            paint_cell(canvas,cell,5,center)
    return canvas


def make_object(points, canvas, template, name, colors):
    x = min(p[0] for p in points)
    y = min(p[1] for p in points)
    w = max(p[0] for p in points)-x+1
    h = max(p[1] for p in points)-y+1
    obj = deepcopy(template)
    obj.update(name=name,x=x,y=y,w=w,h=h,
               pixels=[[canvas.get((x+i,y+j),-1)
                        if (x+i,y+j) in points and canvas.get((x+i,y+j)) in colors
                        else -1 for j in range(h)] for i in range(w)])
    return obj


def extract(state, model, canvas):
    templates = {o['type']:o for o in state}
    defaults = {
        'reflection': {'type':'reflection','layer':3,'tags':['mirror','gray']},
        'player': {'type':'player','layer':4,'tags':['movable','black']},
        'wall': {'type':'wall','layer':1,'tags':['axis','horizontal']},
        'target': {'type':'target','layer':2,'tags':['goal','yellow']}}
    result = [update_counter(o) for o in state if o['type'] == 'counter']
    result += [deepcopy(o) for o in state if o['type'] not in defaults and o['type'] != 'counter']
    for kind,colors,prefix,main_color in [
            ('wall',{0,10},'wall_h_',10),
            ('player',{0,5},'piece_',5),
            ('reflection',{4},'reflection_',4)]:
        groups = connected({p for p,c in canvas.items() if c in colors})
        if kind == 'player':
            groups = [g for g in groups if any(canvas[p] == main_color for p in g)]
        for i,g in enumerate(groups):
            if not any(canvas[p] == main_color for p in g):
                continue
            result.append(make_object(g,canvas,templates.get(kind,defaults[kind]),prefix+str(i),colors))
    groups = []
    for cells in connected(model['goals']):
        pixels = {p for p,c in canvas.items() if c == 11 and (p[0]//3,p[1]//3) in cells}
        if pixels:
            groups.append(pixels)
    groups.sort(key=lambda g:(min(p[0] for p in g),min(p[1] for p in g)))
    for i,g in enumerate(groups):
        result.append(make_object(g,canvas,templates.get('target',defaults['target']),
                                  'target_'+str(i),{11}))
    return result


def transition_function(state, action):
    global _last,_model
    continuous = _last is not None and canonical(state) == _last
    model = deepcopy(_model) if continuous else recover(state)
    aid = action.get('action_id') if isinstance(action,dict) else action
    deltas = {1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}
    if aid == 5:
        model['selected'] = (model['selected']+1) % max(1,len(model['axes'])+len(model['pieces']))
    elif aid in deltas:
        selected = model['selected']
        if selected < len(model['axes']):
            update_wall(model,selected,deltas[aid])
        elif selected-len(model['axes']) < len(model['pieces']):
            update_player(model,selected-len(model['axes']),deltas[aid])
    canvas = render(model)
    result = extract(state,model,canvas)
    _model = model
    _last = canonical(result)
    return result
