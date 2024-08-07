import yaml, math

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