import tkinter as tk
import math
import time
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from Pathfinding import create_Commands

# -------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# -------------------------------------------------------------------------
GRID_SIZE = 20
CELL_PIXELS = 30
UPDATE_DELAY = 50
CM_TO_GRID = 0.1

ROBOT_SIZE = 3  # 3x3 squares
OBSTACLE_DIRECTIONS = ["NORTH", "EAST", "SOUTH", "WEST"]

# Instead of bottom-left at (0,0), let's keep the *center* of the robot at (1,1).
ROBOT_INIT_X = 1
ROBOT_INIT_Y = 1
ROBOT_INIT_HEADING_DEG = 0.0


def get_commands(obstacles):
    images = []
    for obstacle in obstacles:
        images.append(obstacles[obstacle])  # these are Obstacle objects

    segments = create_Commands(images)
    commands = []

    for s_i, seg in enumerate(segments, start=1):
        for cmd_text in seg['commands']:
            commands.append(parse_command(cmd_text))
        commands.append(parse_command("STOP: TAKE PHOTO"))

    return commands


def parse_command(cmd_string):
    parts = cmd_string.strip().upper().split()
    if not parts:
        return {"type": "UNKNOWN"}

    if parts[0] == "STOP:":
        return {"type": "STOP", "detail": " ".join(parts[1:])}

    ctype = parts[0]  # ARC or STRAIGHT
    direction = parts[1] if len(parts) > 1 else ""

    radius = 0.0
    angle = 0.0
    distance = 0.0

    for part in parts[2:]:
        if part.startswith("RADIUS="):
            val_str = part.split("=")[1]
            val = float(val_str.replace("CM", ""))
            radius = val * CM_TO_GRID
        elif part.startswith("ANGLE="):
            val_str = part.split("=")[1]
            val = float(val_str.replace("DEG", ""))
            angle = math.radians(val)
        elif part.endswith("CM"):
            val = float(part.replace("CM", ""))
            distance = val * CM_TO_GRID

    return {
        "type": ctype,
        "direction": direction,
        "radius": radius,
        "angle": angle,
        "distance": distance
    }


def rotate_point(px, py, cx, cy, theta):
    """Rotate (px,py) about (cx,cy) by 'theta' radians."""
    tx = px - cx
    ty = py - cy
    rx = tx * math.cos(theta) - ty * math.sin(theta)
    ry = tx * math.sin(theta) + ty * math.cos(theta)
    return (rx + cx, ry + cy)


# ------------------------------------------------------------------------------
# OBSTACLE CLASS
# ------------------------------------------------------------------------------
class Obstacle:
    def __init__(self, x, y, direction="NORTH"):
        self.x = x
        self.y = y
        self.direction = direction
        self.canvas_object = None
        self.direction_line = None

    def getParams(self):
        return self.x, self.y, self.direction

    def draw(self, canvas):
        if self.canvas_object is not None:
            canvas.delete(self.canvas_object)
        if self.direction_line is not None:
            canvas.delete(self.direction_line)

        px = self.x * CELL_PIXELS
        py = (GRID_SIZE - self.y - 1) * CELL_PIXELS
        sz = CELL_PIXELS

        self.canvas_object = canvas.create_rectangle(px, py, px + sz, py + sz,
                                                     fill='red', outline='black')
        midx = px + sz / 2
        midy = py + sz / 2
        offset = sz / 4

        if self.direction == "NORTH":
            self.direction_line = canvas.create_line(
                midx, midy, midx, midy - offset, width=2, fill='white'
            )
        elif self.direction == "EAST":
            self.direction_line = canvas.create_line(
                midx, midy, midx + offset, midy, width=2, fill='white'
            )
        elif self.direction == "SOUTH":
            self.direction_line = canvas.create_line(
                midx, midy, midx, midy + offset, width=2, fill='white'
            )
        elif self.direction == "WEST":
            self.direction_line = canvas.create_line(
                midx, midy, midx - offset, midy, width=2, fill='white'
            )


# ------------------------------------------------------------------------------
# ROBOT CLASS
# ------------------------------------------------------------------------------
class Robot:
    """
    The robot's internal (x,y) is now its CENTER on the grid.
    That means if x=1,y=1, the robot extends from (x-1,y-1) to (x+1,y+1).
    """

    def __init__(self, canvas, start_x=1.0, start_y=1.0, heading_deg=0.0):
        self.canvas = canvas
        # Robot's center
        self.x = float(start_x)
        self.y = float(start_y)
        self.heading = math.radians(heading_deg)
        self.canvas_objects = []
        self.draw_robot()

    def draw_robot(self):
        # Clear old shapes
        for obj in self.canvas_objects:
            self.canvas.delete(obj)
        self.canvas_objects.clear()

        # We'll draw the 3×3 squares around the center.
        # e.g. if center=(1,1), squares go from x in [0..2], y in [0..2].
        # We'll do: for row in [-1, 0, 1], for col in [-1, 0, 1].
        for row in range(-1, 2):
            for col in range(-1, 2):
                gx = self.x + col
                gy = self.y + row

                px = gx * CELL_PIXELS
                py = (GRID_SIZE - gy - 1) * CELL_PIXELS

                rect = self.canvas.create_rectangle(px, py, px + CELL_PIXELS, py + CELL_PIXELS,
                                                    fill='blue', outline='black')
                self.canvas_objects.append(rect)

        # Draw the heading arrow from the center.
        center_x = self.x * CELL_PIXELS
        center_y = (GRID_SIZE - self.y - 1) * CELL_PIXELS

        arrow_len = 20
        # Because the canvas is top-down, we do minus sin(...) for a positive heading angle
        arrow_x = center_x + arrow_len * math.cos(self.heading)
        arrow_y = center_y - arrow_len * math.sin(self.heading)

        arrow = self.canvas.create_line(center_x, center_y, arrow_x, arrow_y,
                                        width=3, fill='yellow', arrow=tk.LAST)
        self.canvas_objects.append(arrow)

    def move_straight(self, distance, forward=True):
        """Move in a straight line by 'distance' (in grid units)."""
        direction_sign = 1 if forward else -1
        dx = distance * math.cos(self.heading) * direction_sign
        dy = distance * math.sin(self.heading) * direction_sign
        self.x += dx
        self.y += dy

    def move_arc(self, radius, angle, forward=True):
        """
        Perform an arc movement in increments, matching the path planner approach:
         - 'angle' sign => left vs. right turn
         - 'forward' => traveling forward or backward along the arc
        """
        STEPS = 5
        alpha_inc = angle / STEPS
        direction_sign = 1 if forward else -1

        for _ in range(STEPS):
            yaw_mid = self.heading + alpha_inc / 2.0
            self.x += radius * alpha_inc * math.cos(yaw_mid) * direction_sign
            self.y += radius * alpha_inc * math.sin(yaw_mid) * direction_sign
            self.heading += alpha_inc

        self.heading %= (2 * math.pi)

    def execute_command(self, command):
        ctype = command["type"]
        if ctype == "STOP":
            print("STOP command received ->", command.get("detail", ""))
            return
        if ctype not in ("ARC", "STRAIGHT"):
            return

        direction = command.get("direction", "")
        forward = (direction == "FORWARD")

        if ctype == "STRAIGHT":
            dist = command["distance"]
            self.move_straight(dist, forward=forward)
            # Debug
            print(f"STRAIGHT {dist:.3f} => x={self.x:.2f}, y={self.y:.2f}, heading={math.degrees(self.heading):.1f}°")
        elif ctype == "ARC":
            rad = command["radius"]
            ang = command["angle"]
            self.move_arc(rad, ang, forward=forward)
            # Debug
            deg_ang = math.degrees(ang)
            print(f"ARC radius={rad:.3f}, angle={deg_ang:.1f}°, forward={forward} => "
                  f"x={self.x:.2f}, y={self.y:.2f}, heading={math.degrees(self.heading):.1f}°")

    def update_canvas(self):
        self.draw_robot()


# ------------------------------------------------------------------------------
# MAIN APPLICATION
# ------------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MDP Robot Simulation (Center-based)")

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#E6F0FA")
        self.style.configure("TLabel", background="#E6F0FA", font=("Helvetica", 12, "bold"))
        self.style.configure("TButton", font=("Helvetica", 10, "bold"), padding=4)

        main_frame = ttk.Frame(self.root, padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(canvas_frame,
                                width=GRID_SIZE * CELL_PIXELS,
                                height=GRID_SIZE * CELL_PIXELS,
                                bg='#FFFFFF',
                                highlightthickness=1,
                                highlightbackground="#AACCEE")
        self.canvas.pack(padx=5, pady=5)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        title_label = ttk.Label(control_frame, text="Control Panel")
        title_label.pack(pady=(0, 10))

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_movement)
        self.start_button.pack(pady=5)

        self.reset_button = ttk.Button(control_frame, text="Reset", command=self.reset_simulation)
        self.reset_button.pack(pady=5)

        self.log_text = ScrolledText(control_frame, width=35, height=25, font=("Consolas", 10))
        self.log_text.pack(pady=(10, 5), fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.draw_grid()

        # Obstacles
        self.obstacles = {}

        # Robot at (1,1), heading=0 => facing east
        self.robot = Robot(self.canvas, start_x=ROBOT_INIT_X, start_y=ROBOT_INIT_Y, heading_deg=ROBOT_INIT_HEADING_DEG)

        self.command_list = []
        self.cmd_index = 0
        self.is_running = False

    def draw_grid(self):
        for i in range(GRID_SIZE + 1):
            self.canvas.create_line(0, i * CELL_PIXELS,
                                    GRID_SIZE * CELL_PIXELS, i * CELL_PIXELS,
                                    fill='#C0C0C0')
            self.canvas.create_line(i * CELL_PIXELS, 0,
                                    i * CELL_PIXELS, GRID_SIZE * CELL_PIXELS,
                                    fill='#C0C0C0')

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                label = f"({x},{y})"
                px = x * CELL_PIXELS + 2
                py = (GRID_SIZE - y - 1) * CELL_PIXELS + CELL_PIXELS - 12
                self.canvas.create_text(px, py, text=label, font=("Arial", 6), fill="#AAAAAA")

    def on_canvas_click(self, event):
        gx = event.x // CELL_PIXELS
        gy = GRID_SIZE - 1 - (event.y // CELL_PIXELS)

        if not (0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE):
            return

        key = (gx, gy)
        if key not in self.obstacles:
            obstacle = Obstacle(gx, gy, direction="NORTH")
            self.obstacles[key] = obstacle
            obstacle.draw(self.canvas)
        else:
            obstacle = self.obstacles[key]
            current_dir = obstacle.direction
            idx = OBSTACLE_DIRECTIONS.index(current_dir)
            idx = (idx + 1) % len(OBSTACLE_DIRECTIONS)
            obstacle.direction = OBSTACLE_DIRECTIONS[idx]
            obstacle.draw(self.canvas)

    def start_movement(self):
        if not self.is_running:
            self.command_list = get_commands(self.obstacles)
            self.is_running = True
            self.cmd_index = 0
            print("Commands to execute:", self.command_list)
            self.process_next_command()

    def process_next_command(self):
        if self.cmd_index >= len(self.command_list):
            self.is_running = False
            return

        cmd = self.command_list[self.cmd_index]
        self.cmd_index += 1

        self.log_text.insert(tk.END, f"Executing: {cmd}\n")
        self.log_text.see(tk.END)

        self.robot.execute_command(cmd)
        self.robot.update_canvas()

        # Next command in 500ms
        self.root.after(500, self.process_next_command)

    def reset_simulation(self):
        self.is_running = False
        self.cmd_index = 0

        for obs in self.obstacles.values():
            if obs.canvas_object:
                self.canvas.delete(obs.canvas_object)
            if obs.direction_line:
                self.canvas.delete(obs.direction_line)
        self.obstacles.clear()

        # Reset robot to center=(1,1), heading=0
        self.robot.x = ROBOT_INIT_X
        self.robot.y = ROBOT_INIT_Y
        self.robot.heading = math.radians(ROBOT_INIT_HEADING_DEG)
        self.robot.update_canvas()

        self.command_list = []
        self.log_text.insert(tk.END, "Simulation reset.\n")
        self.log_text.see(tk.END)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
