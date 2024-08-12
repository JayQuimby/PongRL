import pygame
import numpy as np

def create_dense_plot(values, width, height, point_size=10, color=(0, 255, 0), bg_color=(0, 0, 0)):
    # Create a 3D numpy array for the pixel data (height, width, RGB)
    pixel_array = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    if len(values) == 0:
        return pygame.surfarray.make_surface(pixel_array)

    # Normalize the values to fit within the height
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        normalized_values = [height // 2] * len(values)
    else:
        normalized_values = [int((v - min_val) / (max_val - min_val) * (height - point_size)) for v in values]

    # Determine the step size for x-axis
    step = max(1, len(values) // width)

    # Plot the points
    for i in range(0, len(values), step):
        x = int(i / len(values) * (width - point_size))
        y = max(0, min(height - point_size, height - 1 - normalized_values[i]))  # Ensure y is within bounds
        pixel_array[y, x] = color

    return pygame.transform.rotate(pygame.transform.flip(pygame.surfarray.make_surface(pixel_array),1,0),90)

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
        plot_surface = create_dense_plot(data, 200, 100, point_size=1)
        screen.fill((0, 0, 0))
        screen.blit(plot_surface, (0, 50))
        pygame.display.flip()
        clock.tick(60)
        x = x + random.random() - 0.5

    pygame.quit()