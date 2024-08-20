import yaml, math
from global_vars import STATE_SPLIT

def load_conf(name: str):
    with open(f'./configs/{name}.yml', 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
            return None
        
def distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def check_bounds(pos, limit, velo, g=True):
    if g:
        if pos > limit and velo > 0:
            return -1
    else:
        if pos < limit and velo < 0:
            return -1
    return 1

def collide(obj1, obj2, momentum_factor=1.2, size_speed_factor=1.2, min_speed=2.0, max_speed=15):
    diff = distance(obj1, obj2)

    nx = (obj2.x - obj1.x) / diff
    ny = (obj2.y - obj1.y) / diff

    vel_along_normal = (obj2.vx - obj1.vx) * nx + (obj2.vy - obj1.vy) * ny

    if vel_along_normal > 0:
        return

    size_multiplier1 = max(1, 1 / (obj1.r**2 * size_speed_factor))
    size_multiplier2 = max(1, 1 / (obj2.r**2 * size_speed_factor))

    j = -(1 + momentum_factor) * vel_along_normal
    j /= 1 / obj1.mass + 1 / obj2.mass

    impulse_x = j * nx
    impulse_y = j * ny
    
    obj1.vx -= impulse_x / obj1.mass * size_multiplier1
    obj1.vy -= impulse_y / obj1.mass * size_multiplier1
    obj2.vx += impulse_x / obj2.mass * size_multiplier2
    obj2.vy += impulse_y / obj2.mass * size_multiplier2

def bresenham_line(x0, y0, x1, y1):
    """Generate points between two points using the Bresenham line algorithm."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def interpolate_color(value, min_val, max_val, color1, color2):
    """Interpolate between two colors based on the normalized value."""
    ratio = (value - min_val) / (max_val - min_val) if min_val != max_val else 0.5
    return (
        int(color1[0] * ratio + color2[0] * (1 - ratio)),
        int(color1[1] * ratio + color2[1] * (1 - ratio)),
        int(color1[2] * ratio + color2[2] * (1 - ratio))
    )

def norm_c(v, d, t, neg=False):
    base = (v / d)
    if neg:
        base = 0.5 + base / 2
        base = max(0.0, min(1.0,base))
    return t(base)

def get_obj_state_repr(obj, o_type, max_v, rev=0):
    loc = (0, norm_c(obj.x, STATE_SPLIT, int), norm_c(obj.y, STATE_SPLIT, int))
    vx = obj.vx * (-1 if rev else 1)
    vy = obj.vy * (-1 if rev else 1)
    rep = [norm_c(vx, max_v, float, 1), norm_c(vy, max_v, float, 1), o_type/4]
    return loc, rep