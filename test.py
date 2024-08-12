import pygame
import numpy as np
from utils import bresenham_line, interpolate_color

def create_dense_plot(values, width, height, point_size=10, colors=[(0, 255, 0), (255,0,0)], bg_color=(0, 0, 0)):
    # Create a 3D numpy array for the pixel data (height, width, RGB)
    pixel_array = np.full((width, height, 3), bg_color, dtype=np.uint8)
    
    if len(values) == 0:
        return pygame.surfarray.make_surface(pixel_array)

    # Normalize the values to fit within the height
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        normalized_values = [height // 2] * len(values)
    else:
        normalized_values = [int((v - min_val) / (max_val - min_val) * (height - point_size)) for v in values]

    min_norm = min(normalized_values)
    max_norm = max(normalized_values)
    # Determine the step size for x-axis
    step = max(1, len(values) // width)

    # Plot the points and draw lines between them
    prev_pt = None
    for i in range(0, len(values), step):
        x = int(i / len(values) * (width - point_size))
        y = max(0, min(height - point_size, height - 1 - normalized_values[i]))  # Ensure y is within bounds

        # Draw line from previous point to current point
        if prev_pt is not None:
            for px, py in bresenham_line(prev_pt[0], prev_pt[1], x, y):
                
                pixel_array[px, py] = interpolate_color(y, min_norm, max_norm, colors[1], colors[0])
        
        prev_pt = (x, y)

    return pygame.surfarray.make_surface(pixel_array)

# Example usage:
if __name__ == "__main__":
    import random
    pygame.init()
    screen = pygame.display.set_mode((1000, 200))
    clock = pygame.time.Clock()

    # Example data
    x = 1
    data = []
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        data.append(x)
        plot_surface = create_dense_plot(data, 1000, 200, point_size=1)
        screen.fill((0, 0, 0))
        screen.blit(plot_surface, (0, 0))
        pygame.display.flip()
        clock.tick(60)
        x = x + (random.random() - 0.5) * 3

    pygame.quit()