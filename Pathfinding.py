import math
import heapq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from keras.src.backend.jax.core import switch
from matplotlib.patches import FancyArrowPatch


##############################################################################
#                          Utility Functions                                 #
##############################################################################

def pi_2_pi(angle):
    """Normalize angle into [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def angle_degrees(rad):
    """Convert radians to degrees in [-180,180)."""
    deg = math.degrees(pi_2_pi(rad))
    if deg > 180:
        deg -= 360
    return deg


def euclid_dist(x1, y1, x2, y2):
    """Euclidean distance."""
    return math.hypot(x2 - x1, y2 - y1)


##############################################################################
#                        Hybrid A* With Arc Commands                         #
##############################################################################

class HybridAStar:
    """
    3D grid A* in (x, y, theta).
    Each expansion is a short arc or straight segment (length=step_size).
    Instead of returning just positions, we also store a "drive command."
    """

    def __init__(
            self,
            grid_w,
            grid_h,
            obstacles,
            n_theta=72,  # 5° increments
            step_size=3.0,  # distance (cm) for each motion
            min_turn_radius=10.0,  # car turning radius
            reverse_ok=True,
            goal_tol_xy=2.0,
            goal_tol_yaw_deg=5.0
    ):
        self.w = grid_w
        self.h = grid_h
        self.obstacles = obstacles
        self.n_theta = n_theta
        self.step_size = step_size
        self.min_turn_radius = min_turn_radius
        self.reverse_ok = reverse_ok
        self.goal_tol_xy = goal_tol_xy
        self.goal_tol_yaw = math.radians(goal_tol_yaw_deg)

        self.motion_primitives = self._create_motion_primitives()

    def _create_motion_primitives(self):
        """
        We define arcs ±15°, ±30°, plus straight (0°).
        Forward/back if reverse_ok is True.
        """
        motions = []
        for deg in [0, 15, -15, 30, -30]:
            for direction in (1, -1):
                if direction == -1 and not self.reverse_ok:
                    continue
                motions.append((math.radians(deg), direction))
        return motions

    def _theta_index(self, theta):
        t = pi_2_pi(theta)
        if t < 0:
            t += 2 * math.pi
        frac = t / (2 * math.pi)
        return int(round(frac * self.n_theta)) % self.n_theta

    def _collision_check(self, px, py):
        """Check each sample for out-of-bounds or obstacle collisions."""
        for x, y in zip(px, py):
            ix, iy = round(x), round(y)
            if ix < 0 or ix >= self.w or iy < 0 or iy >= self.h:
                return True
            if (ix, iy) in self.obstacles:
                return True
        return False

    def _make_command(self, turn_rad, direction):
        """
        Return a textual "ARC FORWARD ..." or "STRAIGHT FORWARD ..."
        that describes this single motion step.
        """
        step_cm = self.step_size
        if abs(turn_rad) < 1e-9:
            # Straight
            if direction > 0:
                return f"STRAIGHT FORWARD {int(step_cm)}cm"
            else:
                return f"STRAIGHT REVERSE {int(step_cm)}cm"
        else:
            # Arc
            radius_cm = self.min_turn_radius
            arc_angle_rad = (step_cm / radius_cm) * (1.0 if direction > 0 else -1.0)
            arc_angle_deg = int(round(math.degrees(arc_angle_rad)))
            if direction > 0:
                return f"ARC FORWARD radius={int(radius_cm)}cm angle={arc_angle_deg}deg"
            else:
                return f"ARC REVERSE radius={int(radius_cm)}cm angle={arc_angle_deg}deg"

    def _motion_step(self, x, y, yaw, turn_rad, direction):
        """
        Move step_size along an arc (turn_rad) or straight (turn_rad=0).
        Return final (nx, ny, nyaw), the sample points, and a single "command."
        """
        step_len = self.step_size
        sub_steps = int(step_len)  # e.g. 3 => 3 increments
        px, py, pyaw = [], [], []

        # Build the command string
        command_str = self._make_command(turn_rad, direction)

        if abs(turn_rad) < 1e-9:
            # Straight
            dsign = 1.0 if direction > 0 else -1.0
            for _ in range(sub_steps):
                x += dsign * math.cos(yaw)
                y += dsign * math.sin(yaw)
                px.append(x)
                py.append(y)
                pyaw.append(yaw)
            return x, y, yaw, px, py, pyaw, command_str

        # Arc
        R = self.min_turn_radius
        alpha = (step_len / R) * (1.0 if direction > 0 else -1.0)
        alpha_inc = alpha / sub_steps
        for _ in range(sub_steps):
            yaw_mid = yaw + alpha_inc / 2.0
            x += R * alpha_inc * math.cos(yaw_mid)
            y += R * alpha_inc * math.sin(yaw_mid)
            yaw = pi_2_pi(yaw + alpha_inc)
            px.append(x)
            py.append(y)
            pyaw.append(yaw)

        return x, y, yaw, px, py, pyaw, command_str

    def search(self, sx, sy, syaw, gx, gy, gyaw):
        """
        Hybrid A* from (sx, sy, syaw) to (gx, gy, gyaw), returning:
          (px, py, pyaw, commands)
        """
        start_thi = self._theta_index(syaw)
        goal_thi = self._theta_index(gyaw)

        def heuristic(cx, cy):
            return math.hypot(cx - gx, cy - gy)

        g_cost = {}
        parent = {}
        start_node = (round(sx), round(sy), start_thi)
        g_cost[start_node] = 0.0

        open_set = []
        heapq.heappush(open_set, (heuristic(sx, sy), 0.0, start_node))

        while open_set:
            f_val, c_val, node = heapq.heappop(open_set)
            cx, cy, cthi = node

            # Check goal
            if math.hypot(cx - gx, cy - gy) < self.goal_tol_xy:
                ctheta = (cthi * 2.0 * math.pi) / self.n_theta
                d_yaw = abs(pi_2_pi(ctheta - gyaw))
                if d_yaw < self.goal_tol_yaw:
                    # reconstruct commands
                    return self._reconstruct_path(parent, node, (sx, sy, syaw))

            ctheta = (cthi * 2.0 * math.pi) / self.n_theta

            # Expand
            for (turn_rad, direction) in self.motion_primitives:
                nx, ny, nyaw, px_list, py_list, pyaw_list, cmd_str = \
                    self._motion_step(cx, cy, ctheta, turn_rad, direction)

                # collision check
                if self._collision_check(px_list, py_list):
                    continue

                nthi = self._theta_index(nyaw)
                new_node = (round(nx), round(ny), nthi)
                new_cost = c_val + self.step_size

                if new_node not in g_cost or new_cost < g_cost[new_node]:
                    g_cost[new_node] = new_cost
                    parent[new_node] = (node, px_list, py_list, pyaw_list, cmd_str)
                    f_score = new_cost + heuristic(nx, ny)
                    heapq.heappush(open_set, (f_score, new_cost, new_node))

        return None  # no path found

    def _reconstruct_path(self, parent, goal_node, start_pose):
        """
        Climb the parent chain. For each step, we have:
          - the sample points px_list, py_list, pyaw_list
          - a single command_str
        We'll build a final (px, py, pyaw) plus a list of commands.
        """
        path_x, path_y, path_yaw = [], [], []
        commands = []

        node = goal_node
        while node in parent:
            pn, segx, segy, segyaw, cmd_str = parent[node]
            path_x.extend(reversed(segx))
            path_y.extend(reversed(segy))
            path_yaw.extend(reversed(segyaw))
            commands.append(cmd_str)
            node = pn

        # Add final node (the start of the chain)
        path_x.append(node[0])
        path_y.append(node[1])
        cth = (node[2] * 2.0 * math.pi) / self.n_theta
        path_yaw.append(cth)

        # Reverse arrays
        path_x.reverse()
        path_y.reverse()
        path_yaw.reverse()
        commands.reverse()

        # Ensure the very first point is exactly the start pose
        path_x[0] = start_pose[0]
        path_y[0] = start_pose[1]
        path_yaw[0] = start_pose[2]

        return (path_x, path_y, path_yaw, commands)


##############################################################################
#                   Snap Final Orientation (in-place)                        #
##############################################################################

def snap_final_orientation(px, py, pyaw, gx, gy, gyaw, obstacles, w, h):
    """
    If within ~2 cm, do a small in-place rotation. We'll skip detailed
    commands for the snap or treat it as a quick rotation.
    """
    fx, fy, fyaw = px[-1], py[-1], pyaw[-1]
    if math.hypot(fx - gx, fy - gy) > 2.0:
        return px, py, pyaw  # not close enough

    yaw_diff = pi_2_pi(gyaw - fyaw)
    deg_diff = abs(math.degrees(yaw_diff))
    if deg_diff < 2.0:
        return px, py, pyaw  # already aligned

    sign = 1.0 if yaw_diff > 0 else -1.0
    steps = int(deg_diff)
    tmp_px, tmp_py, tmp_pyaw = [], [], []
    x, y, yaw = fx, fy, fyaw

    for _ in range(steps):
        alpha = math.radians(1.0) * sign
        R = 0.0001  # tiny radius for basically in-place spin
        yaw_mid = yaw + alpha / 2.0
        x += R * alpha * math.cos(yaw_mid)
        y += R * alpha * math.sin(yaw_mid)
        yaw = pi_2_pi(yaw + alpha)
        tmp_px.append(x)
        tmp_py.append(y)
        tmp_pyaw.append(yaw)

    # collision check
    for xx, yy in zip(tmp_px, tmp_py):
        ix, iy = round(xx), round(yy)
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return px, py, pyaw
        if (ix, iy) in obstacles:
            return px, py, pyaw

    px.extend(tmp_px)
    py.extend(tmp_py)
    pyaw.extend(tmp_pyaw)
    return px, py, pyaw


##############################################################################
#           RobotPathfinding3D - Splitting path & generating commands        #
##############################################################################

class RobotPathfinding3D:
    """
    Plans sub-paths for each image (start->img1->img2->...->start).
    """

    def __init__(
            self,
            grid_size=(200, 200),
            start_pose=(0, 0, 0.0),
            images=None,  # [((x,y), direction), ...]
            user_obstacles=None,
            min_turn_radius=10.0,
            step_size=3.0
    ):
        if images is None:
            images = []
        if user_obstacles is None:
            user_obstacles = set()

        self.grid_size = grid_size
        self.start_pose = start_pose
        self.images = images
        self.obstacles = set(user_obstacles)
        self.min_turn_radius = min_turn_radius
        self.step_size = step_size

        # Mark near-image region as blocked
        self._setup_image_obstacles(radius=15)

        w, h = grid_size
        self.planner = HybridAStar(
            grid_w=w,
            grid_h=h,
            obstacles=self.obstacles,
            n_theta=72,
            step_size=step_size,
            min_turn_radius=min_turn_radius,
            reverse_ok=True,
            goal_tol_xy=2.0,
            goal_tol_yaw_deg=5.0
        )

    def _setup_image_obstacles(self, radius=15):
        for (img_coord, _) in self.images:
            ix, iy = img_coord
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if math.hypot(dx, dy) < radius:
                        ox, oy = ix + dx, iy + dy
                        if 0 <= ox < self.grid_size[0] and 0 <= oy < self.grid_size[1]:
                            self.obstacles.add((ox, oy))

    def approach_pose(self, img_coord, direction, dist=20):
        """
        Return a (gx, gy, gyaw) from which to face the image coordinate.
        """
        x_img, y_img = img_coord
        if direction == 'N':
            return (x_img, y_img - dist, math.pi / 2)
        elif direction == 'E':
            return (x_img - dist, y_img, 0.0)
        elif direction == 'S':
            return (x_img, y_img + dist, -math.pi / 2)
        elif direction == 'W':
            return (x_img + dist, y_img, math.pi)
        # fallback
        return (x_img, y_img, 0.0)

    def visit_images_in_order(self):
        """
        Return a list of sub-routes, each sub-route is:
          { 'poses': [(x, y, yaw), ...],
            'commands': [ "ARC FORWARD ...", "STRAIGHT REVERSE ...", ... ] }
        """
        w, h = self.grid_size
        current_pose = self.start_pose

        segments = []

        # Go to each image approach in order
        for (img_coord, direction) in self.images:
            gx, gy, gyaw = self.approach_pose(img_coord, direction, dist=20)
            result = self.planner.search(
                current_pose[0], current_pose[1], current_pose[2],
                gx, gy, gyaw
            )
            if not result:
                print(f"Skipping unreachable {img_coord}.")
                continue

            px, py, pyaw, commands = result

            # Snap orientation if needed
            px, py, pyaw = snap_final_orientation(
                px, py, pyaw,
                gx, gy, gyaw,
                obstacles=self.obstacles, w=w, h=h
            )

            subpath_poses = [(px[i], py[i], pyaw[i]) for i in range(len(px))]
            segments.append({
                'poses': subpath_poses,
                'commands': commands
            })

            current_pose = subpath_poses[-1]  # update

        # Return to start
        sx, sy, syaw = self.start_pose
        if (
                abs(current_pose[0] - sx) > 1e-9 or
                abs(current_pose[1] - sy) > 1e-9 or
                abs(pi_2_pi(current_pose[2] - syaw)) > 1e-3
        ):
            backres = self.planner.search(
                current_pose[0], current_pose[1], current_pose[2],
                sx, sy, syaw
            )
            if backres:
                bpx, bpy, bpyaw, bcommands = backres
                bpx, bpy, bpyaw = snap_final_orientation(
                    bpx, bpy, bpyaw,
                    sx, sy, syaw,
                    obstacles=self.obstacles, w=w, h=h
                )
                subpath_poses = [(bpx[i], bpy[i], bpyaw[i]) for i in range(len(bpx))]
                segments.append({
                    'poses': subpath_poses,
                    'commands': bcommands
                })
            else:
                print(f"Cannot return from {current_pose} to start.")

        return segments

    def plot_all_segments(self, segments):
        """
        Static plot of the entire path across all sub-routes.
        """
        if not segments:
            print("No path segments to plot.")
            return

        # Merge for plotting
        merged = []
        for i, seg in enumerate(segments):
            if i == 0:
                merged.extend(seg['poses'])
            else:
                # Skip the first pose to avoid duplication
                merged.extend(seg['poses'][1:])

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Hybrid A* Car Commands (split by images)")

        if self.obstacles:
            ox, oy = zip(*self.obstacles)
            ax.scatter(ox, oy, c='red', s=5, label="Obstacles")

        # images
        for idx, ((ix, iy), d) in enumerate(self.images):
            if idx == 0:
                ax.scatter(ix, iy, c='blue', s=50, marker='s', label="Images")
            else:
                ax.scatter(ix, iy, c='blue', s=50, marker='s')
            ax.text(ix + 1, iy + 1, d, color='blue', fontsize=8)

        # entire path
        px = [p[0] for p in merged]
        py = [p[1] for p in merged]
        ax.plot(px, py, '-g', label="Path")

        sx, sy, _ = self.start_pose
        ax.scatter(sx, sy, c='black', s=50, label="Start/End")

        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True)
        plt.show()


##############################################################################
#                         ANIMATION WITH FancyArrowPatch                     #
##############################################################################

def animate_robot_movement(grid_size, obstacles, images, all_poses):
    """
    Animate the robot's movement using a FancyArrowPatch for orientation.
    all_poses: list of (x,y,yaw) across the entire route (flattened).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Robot Car Simulator (FancyArrowPatch Animation)")

    # Obstacles
    if obstacles:
        ox, oy = zip(*obstacles)
        ax.scatter(ox, oy, c='red', s=5, label="Obstacles")

    # Images
    for idx, ((ix, iy), d) in enumerate(images):
        if idx == 0:
            ax.scatter(ix, iy, c='blue', s=50, marker='s', label="Images")
        else:
            ax.scatter(ix, iy, c='blue', s=50, marker='s')
        ax.text(ix + 1, iy + 1, d, color='blue', fontsize=8)

    # Start/End
    sx, sy, _ = all_poses[0]
    ax.scatter(sx, sy, c='black', s=50, label="Start/End")

    # A line to trace the path
    path_line, = ax.plot([], [], '-g', label="Path", linewidth=2)

    # Create a FancyArrowPatch for the robot orientation
    arrow_len = 5.0  # distance from tail to head
    robot_arrow = FancyArrowPatch(
        posA=(0, 0),  # tail
        posB=(arrow_len, 0),  # head
        arrowstyle='-|>',  # arrow shape
        mutation_scale=15,  # size of arrow head
        color='black'
    )
    ax.add_patch(robot_arrow)

    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_aspect('equal', 'box')
    ax.legend()
    ax.grid(True)

    # Extract x, y, yaw from all_poses
    path_x = [p[0] for p in all_poses]
    path_y = [p[1] for p in all_poses]
    path_yaw = [p[2] for p in all_poses]

    def init():
        path_line.set_data([], [])
        # put the arrow somewhere valid initially
        robot_arrow.set_positions((sx, sy), (sx + arrow_len, sy))
        return path_line, robot_arrow

    def update(frame):
        x_current = path_x[frame]
        y_current = path_y[frame]
        yaw_current = path_yaw[frame]

        # Update path line up to current frame
        path_line.set_data(path_x[:frame + 1], path_y[:frame + 1])

        # Head of the arrow in front of the car:
        x_head = x_current + arrow_len * math.cos(yaw_current)
        y_head = y_current + arrow_len * math.sin(yaw_current)

        # Update the arrow from tail=(x_current, y_current) to head=(x_head, y_head)
        robot_arrow.set_positions((x_current, y_current), (x_head, y_head))

        return path_line, robot_arrow

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(all_poses),
        init_func=init,
        interval=200,  # ms
        blit=False,
        repeat=False
    )
    ani.save("animation.gif")
    plt.show()


##############################################################################
#                             MAIN / DEMO                                    #
##############################################################################
def create_Commands(obstacles):
    # Translate coordinates for images
    images = []
    for im in obstacles:

        #x,y, direc = im.getParams()
        images.append(((im.x*10,im.y*10),im.direction[:1]))
    user_obstacles = {}
    grid_size = (200, 200)
    start_pose = (10, 10, 0.0)

    robot = RobotPathfinding3D(
        grid_size=grid_size,
        start_pose=start_pose,
        images=images,
        user_obstacles=user_obstacles,
        min_turn_radius=10.0,
        step_size=3.0
    )

    segments = robot.visit_images_in_order()

    robot.plot_all_segments(segments)

    return segments


if __name__ == "__main__":
    # Example usage
    grid_size = (200, 200)
    start_pose = (10, 10, 0.0)

    images = [
        ((100, 100), 'N'),
        ((150, 150), 'W'),
        ((180, 50), 'E')
    ]
    user_obstacles = {
        (50, 50), (51, 50), (52, 50),
        (100, 85), (100, 86)
    }

    # Create the pathfinding object
    robot = RobotPathfinding3D(
        grid_size=grid_size,
        start_pose=start_pose,
        images=images,
        user_obstacles=user_obstacles,
        min_turn_radius=10.0,
        step_size=3.0
    )

    # 1) Build subpaths
    segments = robot.visit_images_in_order()

    # 2) Optional: static plot
    robot.plot_all_segments(segments)

    # 3) Print out the commands for each segment
    print("\n--- CAR COMMANDS ---")
    for s_i, seg in enumerate(segments, start=1):
        print(f"Segment {s_i}:")
        for cmd in seg['commands']:
            print(f"  {cmd}")
        print("  STOP: TAKE PHOTO\n")

    # 4) Flatten all segments into one list of poses for animation
    all_poses = []
    for i, seg in enumerate(segments):
        if i == 0:
            all_poses.extend(seg['poses'])
        else:
            # skip the first pose of each subsequent segment to avoid duplication
            all_poses.extend(seg['poses'][1:])

    # 5) Animate the path with a FancyArrowPatch
    if all_poses:
        animate_robot_movement(
            grid_size=grid_size,
            obstacles=robot.obstacles,
            images=robot.images,
            all_poses=all_poses
        )
    else:
        print("No valid path found. Nothing to animate.")
