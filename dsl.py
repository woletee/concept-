import numpy as np
from collections import deque, Counter

# --- Grid Transformation Functions ---
def remove_vertical_lines(ctx):
    rows, cols = len(ctx.grid), len(ctx.grid[0])
    
    for obj in ctx.objects:
        columns = {}
        for r, c in obj[2]:
            if c not in columns:
                columns[c] = []
            columns[c].append(r)

        for c, rows_in_col in columns.items():
            if len(rows_in_col) > 1:
                unique_vals = {ctx.grid[r][c] for r in rows_in_col}
                if len(unique_vals) == 1:
                    for r in rows_in_col:
                        ctx.grid[r][c] = 0
    return ctx.grid
def fill_object_interior(ctx):
    """
    Fills the interior of each object in the GPContext grid.
    Assumes ctx.objects has been extracted already.
    """
    rows, cols = len(ctx.grid), len(ctx.grid[0])

    for obj in ctx.objects:
        min_r = min(r for r, c in obj)
        max_r = max(r for r, c in obj)
        min_c = min(c for r, c in obj)
        max_c = max(c for r, c in obj)

        obj_color = ctx.grid[min_r][min_c]
        fill_color = (obj_color + 1) % 9 or 1  # consistent color, avoid zero

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) not in obj and ctx.grid[r][c] == 0:
                    ctx.grid[r][c] = fill_color

    return ctx.grid

def move_right_most_object(ctx):
    if not ctx.objects:
        return ctx.grid

    # Get rightmost object: object with largest column index
    rightmost_object = max(ctx.objects, key=lambda obj: max(y for x, y in obj[2]))
    _, value, block = rightmost_object

    for x, y in block:
        ctx.grid[x][y] = 0

    shift = 0
    cols = len(ctx.grid[0])
    while True:
        can_move = True
        for x, y in block:
            new_y = y + shift + 1
            if new_y >= cols or ctx.grid[x][new_y] != 0:
                can_move = False
                break
        if not can_move:
            break
        shift += 1

    for x, y in block:
        ctx.grid[x][y + shift] = value

    return ctx.grid
def move_left_most_object(ctx):
    if not ctx.objects:
        return ctx.grid

    # Get leftmost object: object with smallest column index
    leftmost_object = min(ctx.objects, key=lambda obj: min(y for x, y in obj[2]))
    _, value, block = leftmost_object

    for x, y in block:
        ctx.grid[x][y] = 0

    shift = 0
    while True:
        can_move = True
        for x, y in block:
            new_y = y - (shift + 1)
            if new_y < 0 or ctx.grid[x][new_y] != 0:
                can_move = False
                break
        if not can_move:
            break
        shift += 1

    for x, y in block:
        ctx.grid[x][y - shift] = value

    return ctx.grid

def move_bottom_most_object(ctx):
    if not ctx.objects:
        return ctx.grid

    # Get bottommost object (last one in the list after sorting by top_row)
    bottom_object = ctx.objects[-1]
    _, value, block = bottom_object

    # Remove it from the grid
    for x, y in block:
        ctx.grid[x][y] = 0

    # Compute shift (same as top, move down)
    shift = 0
    rows = len(ctx.grid)
    while True:
        can_move = True
        for x, y in block:
            new_x = x + shift + 1
            if new_x >= rows or ctx.grid[new_x][y] != 0:
                can_move = False
                break
        if not can_move:
            break
        shift += 1

    # Place object
    for x, y in block:
        ctx.grid[x + shift][y] = value

    return ctx.grid
def detect_objects(grid):
    """
    Detects objects in an ARC grid.
    Objects are contiguous regions of the same color (4-connected).
    Returns a list of objects, where each object is a set of (row, col) coordinates.
    """
    rows, cols = len(grid), len(grid[0])
    visited = set()
    objects = []
    
    def bfs(start_r, start_c, color):
        """ Perform BFS to find all connected pixels of the same color """
        queue = deque([(start_r, start_c)])
        obj_pixels = set()
        
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            
            visited.add((r, c))
            obj_pixels.add((r, c))
            
            # Check 4-connected neighbors (up, down, left, right)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if grid[nr][nc] == color:
                        queue.append((nr, nc))
        
        return obj_pixels
    
    # Iterate over the grid to find objects
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid[r][c] != 0:  # Ignore background (0)
                obj = bfs(r, c, grid[r][c])
                objects.append(obj)
    
    return objects
def highlight_detected_objects(grid):
    objects = detect_objects(grid)
    new_grid = [row[:] for row in grid]
    for idx, obj in enumerate(objects, start=1):
        for r, c in obj:
            new_grid[r][c] = (idx % 9) or 1
    return new_grid

def fill_object_interior(grid):
    """ Modifies the grid by filling the interiors of detected objects with a different color."""
    objects = detect_objects(grid)
    rows, cols = len(grid), len(grid[0])
    new_grid = [row[:] for row in grid]  # Create a copy of the grid
    
    for obj in objects:
        min_r = min(r for r, c in obj)
        max_r = max(r for r, c in obj)
        min_c = min(c for r, c in obj)
        max_c = max(c for r, c in obj)
        
        # Find a new fill color (incrementing the current color modulo 9 for variation)
        obj_color = grid[min_r][min_c]
        fill_color = (obj_color + 1) % 9 if obj_color + 1 != 0 else 1
        
        # Identify and fill the interior pixels of the object
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) not in obj and grid[r][c] == 0:  # Empty space inside the object
                    new_grid[r][c] = fill_color
    
    return new_grid
def diamirror(input_grid):
    return np.transpose(input_grid)


import numpy as np

def get_object_bounds(grid):
    grid = np.array(grid)
    top, bottom = None, None
    for i in range(grid.shape[0]):
        if np.any(grid[i] != 0):
            if top is None:
                top = i
            bottom = i
    return top, bottom
def reverse_object_top_bottom(grid):
    grid = np.array(grid)
    top, bottom = get_object_bounds(grid)
    if top is None or bottom is None:
        return grid
    grid_copy = np.copy(grid)
    grid_copy[top:bottom+1] = np.flipud(grid[top:bottom+1])
    return grid_copy

def hmirror(input_grid: np.ndarray) -> np.ndarray:
    return np.fliplr(input_grid)

def vmirrors(input_grid: np.ndarray) -> np.ndarray:
    return np.flipud(input_grid)

def flip_horizontal(input_grid: np.ndarray) -> np.ndarray:
    return np.fliplr(input_grid)

def flip_vertical(input_grid: np.ndarray) -> np.ndarray:
    return np.flipud(input_grid)

def rotate_90(input_grid: np.ndarray) -> np.ndarray:
    return np.rot90(input_grid, k=-1)

def rotate_180(input_grid: np.ndarray) -> np.ndarray:
    return np.rot90(input_grid, k=2)

def rotate_270(input_grid: np.ndarray) -> np.ndarray:
    return np.rot90(input_grid, k=1)

def identity(input_grid: np.ndarray) -> np.ndarray:
    return input_grid
def find_center_pixel(grid):
    """Finds the center pixel of the input grid and returns it as a 1x1 output grid."""
    center_index = len(grid[0]) // 2  # Get the middle index
    return [[grid[0][center_index]]]  # Return as a 1x1 grid with the center pixel

# --- Object Detection and Manipulation ---
def detect_objects(grid):
    """Detects objects in the grid and returns a list of bounding boxes and pixel coordinates."""
    height, width = len(grid), len(grid[0])
    visited = set()
    objects = []

    def bfs(r, c, color):
        """Finds all pixels belonging to an object using BFS."""
        queue = [(r, c)]
        pixels = []  # Use a list instead of a set
        min_r, max_r, min_c, max_c = r, r, c, c

        while queue:
            x, y = queue.pop(0)
            if (x, y) in visited:
                continue
            visited.add((x, y))
            pixels.append((x, y))  # Append to list
            min_r, max_r = min(min_r, x), max(max_r, x)
            min_c, max_c = min(min_c, y), max(max_c, y)

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Only vertical & horizontal connections
                nx, ny = x + dx, y + dy
                if (0 <= nx < height and 0 <= ny < width and (nx, ny) not in visited and grid[nx][ny] == color):
                    queue.append((nx, ny))

        return (min_r, max_r, min_c, max_c, color, pixels)  # Return tuple, pixels as a list

    for r in range(height):
        for c in range(width):
            if grid[r][c] != 0 and (r, c) not in visited:
                visited.add((r, c))
                objects.append(bfs(r, c, grid[r][c]))  # Append tuple to list

    return objects  # Ensure `objects` is a list, not a set

def extract_bottom_object(grid):
    """Extracts the bottom-most object from the grid, crops it, and returns it as a new grid."""
    objects = detect_objects(grid)
    if not objects:
        return grid  

    bottom_object = max(objects, key=lambda obj: obj[1])  # obj[1] is max_r
    min_r, max_r, min_c, max_c, obj_color, pixels = bottom_object
    cropped_height = max_r - min_r + 1
    cropped_width = max_c - min_c + 1
    cropped_grid = np.zeros((cropped_height, cropped_width), dtype=int)

    for r, c in pixels:
        cropped_grid[r - min_r, c - min_c] = obj_color  

    return cropped_grid.tolist()

def keep_bottom_object(grid):
    """Keeps only the bottom-most object and removes all others."""
    height, width = len(grid), len(grid[0])
    objects = detect_objects(grid)
    output_grid = np.zeros((height, width), dtype=int)

    if not objects:
        return output_grid.tolist()  

    bottom_object = max(objects, key=lambda obj: obj[1])  # obj[1] is max_r

    for r, c in bottom_object[5]:  # obj[5] contains pixels
        output_grid[r][c] = bottom_object[4]  # obj[4] is color

    return output_grid.tolist()

def recolor_to_bottom_object(grid):
    """Recolors all objects to match the color of the bottom-most object."""
    height, width = len(grid), len(grid[0])
    objects = detect_objects(grid)
    output_grid = np.array(grid)

    if not objects:
        return output_grid.tolist()  

    bottom_object = max(objects, key=lambda obj: obj[1])  # obj[1] is max_r
    bottom_color = bottom_object[4]  # obj[4] is color

    for min_r, max_r, min_c, max_c, obj_color, pixels in objects:
        for r, c in pixels:
            output_grid[r][c] = bottom_color  # Change to bottom-most object's color

    return output_grid.tolist()

def remove_top_bottom_objects(grid):
    """Removes objects that touch either the top or bottom of the grid."""
    height, width = len(grid), len(grid[0])
    objects = detect_objects(grid)
    output_grid = np.zeros((height, width), dtype=int)

    if not objects:
        return output_grid.tolist()  

    min_top = min(obj[0] for obj in objects)  
    max_bottom = max(obj[1] for obj in objects)  

    for (min_r, max_r, min_c, max_c, obj_color, pixels) in objects:
        if min_r == min_top or max_r == max_bottom:
            continue
        for r, c in pixels:
            output_grid[r][c] = obj_color

    return output_grid.tolist()

def extract_topmost_object(grid):
    """Extracts the top-most object from the grid, crops it, and returns it as a new grid."""
    objects = detect_objects(grid)
    if not objects:
        return grid  

    topmost_object = min(objects, key=lambda obj: obj[0])  # obj[0] is min_r
    min_r, max_r, min_c, max_c, obj_color, pixels = topmost_object
    cropped_height = max_r - min_r + 1
    cropped_width = max_c - min_c + 1
    cropped_grid = np.zeros((cropped_height, cropped_width), dtype=int)

    for r, c in pixels:
        cropped_grid[r - min_r, c - min_c] = obj_color  

    return cropped_grid.tolist()

def swap_objects(grid):
    """Swaps detected objects in the grid."""
    objects = detect_objects(grid)
    objects = sorted(objects, key=lambda obj: obj[1])  # Sort by vertical position

    object_positions = [obj[5] for obj in objects]  # obj[5] contains pixels
    object_colors = [obj[4] for obj in objects]  # obj[4] is color
    swapped_positions = object_positions[::-1]  

    new_grid = np.zeros_like(grid)
    for color, new_positions in zip(object_colors, swapped_positions):
        for r, c in new_positions:
            new_grid[r][c] = color

    return new_grid.tolist()

# --- Pixel & Color Manipulation ---
def transform_blue_to_red(input_grid):
    """Transforms all blue (1) pixels to red (2)."""
    grid = np.array(input_grid)
    return np.where(grid == 1, 2, grid).tolist()

def fill_downward(grid):
    """Fills non-zero pixels downward, propagating their colors downwards in each column."""
    height, width = len(grid), len(grid[0])
    output_grid = np.array(grid)

    for col in range(width):
        fill_color = 0  
        for row in range(height):
            if grid[row][col] != 0:
                fill_color = grid[row][col]  
            if fill_color != 0:
                output_grid[row][col] = fill_color  
    return output_grid.tolist()

def remove_below_horizontal_line(grid):
    """Detects the first fully connected horizontal line and removes everything below it."""
    height, width = len(grid), len(grid[0])
    output_grid = np.array(grid)

    for row in range(height):
        if np.all(output_grid[row] != 0):  
            output_grid[row + 1:] = 0
            break
    return output_grid.tolist()

def find_center_pixel(grid):
    """Finds the center of the grid and returns it as a 1x1 pixel grid."""
    center_index = len(grid[0]) // 2  
    return [[grid[0][center_index]]]

def extract_largest_row(grid):
    """Finds the row with the most non-zero elements and extracts it."""
    grid = np.array(grid)
    max_length = 0
    longest_row = []

    for row in grid:
        row_values = row[row > 0]  
        if len(row_values) > max_length:
            max_length = len(row_values)
            longest_row = row_values.tolist()  
    return [longest_row]

def extract_dominant_colors(grid):
    """Finds the two most dominant non-zero colors in the grid."""
    flattened = [cell for row in grid for cell in row if cell != 0]
    color_counts = Counter(flattened)

    if not color_counts:
        return [[]]  
    most_common_colors = [color for color, _ in color_counts.most_common(2)]
    return [[color for color in most_common_colors]]

def remove_dominant_color(grid):
    """Removes the most dominant color from the grid."""
    color_counts = Counter(cell for row in grid for cell in row if cell != 0)

    if color_counts:
        dominant_color = max(color_counts, key=color_counts.get)
    else:
        return grid  
    return [[0 if cell == dominant_color else cell for cell in row] for row in grid]

def find_least_dominant_pixel(grid):
    """Finds the least occurring non-zero pixel in the grid."""
    pixel_counts = {}

    for row in grid:
        for value in row:
            if value != 0:
                pixel_counts[value] = pixel_counts.get(value, 0) + 1

    if not pixel_counts:
        return None

    return min(pixel_counts, key=pixel_counts.get)

def remove_least_dominant_pixel(grid):
    """Removes the least dominant pixel from the grid."""
    rows, cols = len(grid), len(grid[0])
    least_dominant_pixel = find_least_dominant_pixel(grid)

    if least_dominant_pixel is None:
        return grid  

    new_grid = np.array(grid)
    for x in range(rows):
        for y in range(cols):
            if grid[x][y] == least_dominant_pixel:
                new_grid[x, y] = 0  
    return new_grid.tolist()

def upscale(input_grid, upscale_factor=3):
    """Upscales the grid by expanding each pixel into a 3x3 block."""
    def expand_pixel_with_grid(pixel, input_grid):
        if pixel == 0:
            return np.zeros((upscale_factor, upscale_factor), dtype=int)
        else:
            return input_grid

    input_rows, input_cols = len(input_grid), len(input_grid[0])
    output_grid = np.zeros((input_rows * upscale_factor, input_cols * upscale_factor), dtype=int)

    for r in range(input_rows):
        for c in range(input_cols):
            expanded_block = expand_pixel_with_grid(input_grid[r][c], input_grid)
            output_grid[r * upscale_factor: (r + 1) * upscale_factor, c * upscale_factor: (c + 1) * upscale_factor] = expanded_block
    
    return output_grid

def remove_center_object(grid):
    """Removes anything located at the center of the grid."""
    height, width = len(grid), len(grid[0])
    center_r, center_c = height // 2, width // 2
    grid = np.array(grid)

    center_value = grid[center_r, center_c]
    if center_value != 0:
        grid[grid == center_value] = 0  
    return grid.tolist()
import numpy as np

def draw_horizontal_vertical(grid):
    """Adds a horizontal or vertical line of 8s based on object orientation."""
    if grid is None or len(grid) == 0 or len(grid[0]) == 0:  
        print("ERROR: Grid is empty. Cannot apply draw_horizontal_vertical.")
        return grid  

    rows, cols = len(grid), len(grid[0])
    print(f"Grid Shape Before Modification: {rows}x{cols}")  # Debugging Info

    new_grid = np.array(grid)

    for r in range(rows):
        new_grid[r][-1] = 8  # Rightmost column

    for c in range(cols):
        new_grid[0][c] = 8  # Topmost row

    return new_grid.tolist()

