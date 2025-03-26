import numpy as np

def rotate_60_degrees(matrix, direction="left"):
    """
    Rotates a 2D array (list of lists) by 60 degrees.
    - direction: "left" for counterclockwise, "right" for clockwise
    """
    angle = np.radians(60 if direction == "left" else -60)
    
    cos_theta, sin_theta = np.cos(angle), np.sin(angle)
    
    height, width = len(matrix), len(matrix[0])
    
    # Compute center of the matrix for rotation
    center_x, center_y = (width - 1) / 2, (height - 1) / 2
    
    # Create a new empty grid
    new_matrix = [[None for _ in range(width)] for _ in range(height)]
    
    for y in range(height):
        for x in range(width):
            # Translate coordinates to center, rotate, then translate back
            x_new = (x - center_x) * cos_theta - (y - center_y) * sin_theta + center_x
            y_new = (x - center_x) * sin_theta + (y - center_y) * cos_theta + center_y
            
            # Round to nearest grid index
            x_new, y_new = int(round(x_new)), int(round(y_new))
            
            # Place element if within bounds
            if 0 <= x_new < width and 0 <= y_new < height:
                new_matrix[y_new][x_new] = matrix[y][x]

    return new_matrix

# Example usage
array = [
    [[1], [2], [3]],
    [[4], [5], [6]],
    [[7], [8], [9]]
]

rotated_left = rotate_60_degrees(array, "left")
rotated_right = rotate_60_degrees(array, "right")

for row in rotated_left:
    print(row)
