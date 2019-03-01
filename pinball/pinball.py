"""Pinball domain for reinforcement learning
"""
import os
import numpy as np
import itertools
from itertools import tee

from gym import core, spaces
from gym.envs.registration import register

class BallModel:

    """ This class maintains the state of the ball
    in the pinball domain. It takes care of moving
    it according to the current velocity and drag coefficient.

    """
    DRAG = 0.995

    def __init__(self, start_position, radius):
        """
        :param start_position: The initial position
        :type start_position: float
        :param radius: The ball radius
        :type radius: float
        """
        self.position = start_position
        self.radius = radius
        self.xdot = 0.0
        self.ydot = 0.0

    def add_impulse(self, delta_xdot, delta_ydot):
        """ Change the momentum of the ball
        :param delta_xdot: The change in velocity in the x direction
        :type delta_xdot: float
        :param delta_ydot: The change in velocity in the y direction
        :type delta_ydot: float
        """
        self.xdot += delta_xdot / 5.0
        self.ydot += delta_ydot / 5.0
        self.xdot = self._clip(self.xdot)
        self.ydot = self._clip(self.ydot)

    def add_drag(self):
        """ Add a fixed amount of drag to the current velocity """
        self.xdot *= self.DRAG
        self.ydot *= self.DRAG

    def step(self):
        """ Move the ball by one increment """
        self.position[0] += self.xdot * self.radius / 20.0
        self.position[1] += self.ydot * self.radius / 20.0

    def _clip(self, val, low=-2, high=2):
        """ Clip a value in a given range """
        if val > high:
            val = high
        if val < low:
            val = low
        return val


class PinballObstacle:

    """ This class represents a single polygon obstacle in the
    pinball domain and detects when a :class:`BallModel` hits it.

    When a collision is detected, it also provides a way to
    compute the appropriate effect to apply on the ball.
    """

    def __init__(self, points):
        """
        :param points: A list of points defining the polygon
        :type points: list of lists
        """
        self.points = np.array(list(points))
        self.min_x = np.min(self.points[:,0])
        self.max_x = np.max(self.points[:,0])
        self.min_y = np.min(self.points[:,1])
        self.max_y = np.max(self.points[:,1])

        self._double_collision = False
        self._intercept = None

    def collision(self, ball):
        """ Determines if the ball hits this obstacle

    :param ball: An instance of :class:`BallModel`
    :type ball: :class:`BallModel`
        """
        self._double_collision = False

        if ball.position[0] - ball.radius > self.max_x:
            return False
        if ball.position[0] + ball.radius < self.min_x:
            return False
        if ball.position[1] - ball.radius > self.max_y:
            return False
        if ball.position[1] + ball.radius < self.min_y:
            return False

        a, b = tee(np.vstack([np.array(self.points), self.points[0]]))
        next(b, None)
        intercept_found = False
        for pt_pair in zip(a, b):
            if self._intercept_edge(pt_pair, ball):
                if intercept_found:
                    # Ball has hit a corner
                    self._intercept = self._select_edge(
                        pt_pair,
                        self._intercept,
                        ball)
                    self._double_collision = True
                else:
                    self._intercept = pt_pair
                    intercept_found = True

        return intercept_found

    def collision_effect(self, ball):
        """ Based of the collision detection result triggered
    in :func:`PinballObstacle.collision`, compute the
        change in velocity.

    :param ball: An instance of :class:`BallModel`
    :type ball: :class:`BallModel`

        """
        if self._double_collision:
            return [-ball.xdot, -ball.ydot]

        # Normalize direction
        obstacle_vector = self._intercept[1] - self._intercept[0]
        if obstacle_vector[0] < 0:
            obstacle_vector = self._intercept[0] - self._intercept[1]

        velocity_vector = np.array([ball.xdot, ball.ydot])
        theta = self._angle(velocity_vector, obstacle_vector) - np.pi
        if theta < 0:
            theta += 2 * np.pi

        intercept_theta = self._angle([-1, 0], obstacle_vector)
        theta += intercept_theta

        if theta > 2 * np.pi:
            theta -= 2 * np.pi

        velocity = np.linalg.norm([ball.xdot, ball.ydot])

        return [velocity * np.cos(theta), velocity * np.sin(theta)]

    def _select_edge(self, intersect1, intersect2, ball):
        """ If the ball hits a corner, select one of two edges.

    :param intersect1: A pair of points defining an edge of the polygon
    :type intersect1: list of lists
    :param intersect2: A pair of points defining an edge of the polygon
    :type intersect2: list of lists
    :returns: The edge with the smallest angle with the velocity vector
    :rtype: list of lists

        """
        velocity = np.array([ball.xdot, ball.ydot])
        obstacle_vector1 = intersect1[1] - intersect1[0]
        obstacle_vector2 = intersect2[1] - intersect2[0]

        angle1 = self._angle(velocity, obstacle_vector1)
        if angle1 > np.pi:
            angle1 -= np.pi

        angle2 = self._angle(velocity, obstacle_vector2)
        if angle1 > np.pi:
            angle2 -= np.pi

        if np.abs(angle1 - (np.pi / 2.0)) < np.abs(angle2 - (np.pi / 2.0)):
            return intersect1
        return intersect2

    def _angle(self, v1, v2):
        """ Compute the angle difference between two vectors

    :param v1: The x,y coordinates of the vector
    :type: v1: list
    :param v2: The x,y coordinates of the vector
    :type: v2: list
    :rtype: float

    """
        angle_diff = np.arctan2(v1[0], v1[1]) - np.arctan2(v2[0], v2[1])
        if angle_diff < 0:
            angle_diff += 2 * np.pi
        return angle_diff

    def _intercept_edge(self, pt_pair, ball):
        """ Compute the projection on and edge and find out

    if it intercept with the ball.
    :param pt_pair: The pair of points defining an edge
    :type pt_pair: list of lists
    :param ball: An instance of :class:`BallModel`
    :type ball: :class:`BallModel`
    :returns: True if the ball has hit an edge of the polygon
    :rtype: bool

        """
        # Find the projection on an edge
        obstacle_edge = pt_pair[1] - pt_pair[0]
        difference = np.array(ball.position) - pt_pair[0]

        scalar_proj = difference.dot(
            obstacle_edge) / obstacle_edge.dot(obstacle_edge)
        if scalar_proj > 1.0:
            scalar_proj = 1.0
        elif scalar_proj < 0.0:
            scalar_proj = 0.0

        # Compute the distance to the closest point
        closest_pt = pt_pair[0] + obstacle_edge * scalar_proj
        obstacle_to_ball = ball.position - closest_pt
        distance = obstacle_to_ball.dot(obstacle_to_ball)

        if distance <= ball.radius * ball.radius:
            # A collision only if the ball is not already moving away
            velocity = np.array([ball.xdot, ball.ydot])
            ball_to_obstacle = closest_pt - ball.position

            angle = self._angle(ball_to_obstacle, velocity)
            if angle > np.pi:
                angle = 2 * np.pi - angle

            if angle > np.pi / 1.99:
                return False

            return True
        else:
            return False

class PinballModel:

    """ This class is a self-contained model of the pinball
    domain for reinforcement learning.

    It can be used either over RL-Glue through the :class:`PinballRLGlue`
    adapter or interactively with :class:`PinballView`.

    """
    ACC_X = 0
    ACC_Y = 1
    DEC_X = 2
    DEC_Y = 3
    ACC_NONE = 4

    STEP_PENALTY = -1
    THRUST_PENALTY = -5
    END_EPISODE = 10000

    def __init__(self, configuration, random_state=np.random.RandomState()):
        """ Read a configuration file for Pinball and draw the domain to screen

    :param configuration: a configuration file containing the polygons,
        source(s) and target location.
    :type configuration: str

        """

        self.random_state = random_state
        self.action_effects = {self.ACC_X: (1, 0), self.ACC_Y: (
            0, 1), self.DEC_X: (-1, 0), self.DEC_Y: (0, -1), self.ACC_NONE: (0, 0)}

        # Set up the environment according to the configuration
        self.obstacles = []
        self.target_pos = []
        self.target_rad = 0.01

        ball_rad = 0.01
        start_pos = []
        with open(configuration) as fp:
            for line in fp.readlines():
                tokens = line.strip().split()
                if not len(tokens):
                    continue
                elif tokens[0] == 'polygon':
                    self.obstacles.append(
                        PinballObstacle(zip(*[iter(map(float, tokens[1:]))] * 2)))
                elif tokens[0] == 'target':
                    self.target_pos = [float(tokens[1]), float(tokens[2])]
                    self.target_rad = float(tokens[3])
                elif tokens[0] == 'start':
                    start_pos = list(zip(*[iter(map(float, tokens[1:]))] * 2))
                elif tokens[0] == 'ball':
                    ball_rad = float(tokens[1])
        self.start_pos = start_pos[0]
        a = self.random_state.randint(len(start_pos))
        self.ball = BallModel(list(start_pos[a]), ball_rad)

    def get_state(self):
        """ Access the current 4-dimensional state vector

        :returns: a list containing the x position, y position, xdot, ydot
        :rtype: list

        """
        return (
            [self.ball.position[0],
             self.ball.position[1],
             self.ball.xdot,
             self.ball.ydot]
        )

    def take_action(self, action):
        """ Take a step in the environment

        :param action: The action to apply over the ball
        :type action: int

        """
        for i in range(20):
            if i == 0:
                self.ball.add_impulse(*self.action_effects[action])

            self.ball.step()

            # Detect collisions
            ncollision = 0
            dxdy = np.array([0, 0])

            for obs in self.obstacles:
                if obs.collision(self.ball):
                    dxdy = dxdy + obs.collision_effect(self.ball)
                    ncollision += 1

            if ncollision == 1:
                self.ball.xdot = dxdy[0]
                self.ball.ydot = dxdy[1]
                if i == 19:
                    self.ball.step()
            elif ncollision > 1:
                self.ball.xdot = -self.ball.xdot
                self.ball.ydot = -self.ball.ydot

            if self.episode_ended():
                return self.END_EPISODE

        self.ball.add_drag()
        self._check_bounds()

        if action == self.ACC_NONE:
            return self.STEP_PENALTY

        self._check_bounds()
        return self.THRUST_PENALTY

    def episode_ended(self):
        """ Find out if the ball reached the target

        :returns: True if the ball reached the target position
        :rtype: bool

        """
        return (
            np.linalg.norm(np.array(self.ball.position)
                           - np.array(self.target_pos)) < self.target_rad
        )

    def _check_bounds(self):
        """ Make sure that the ball stays within the environment """
        if self.ball.position[0] > 1.0:
            self.ball.position[0] = 0.95
        if self.ball.position[0] < 0.0:
            self.ball.position[0] = 0.05
        if self.ball.position[1] > 1.0:
            self.ball.position[1] = 0.95
        if self.ball.position[1] < 0.0:
            self.ball.position[1] = 0.05

class PinballDomain(core.Env):
    def __init__(self, path='pinball_simple_single.cfg'):
        self.path = path
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=np.array([0, 0, -2, -2]), high=np.array([1, 1, 2, 2]))

    def _reset(self):
        self.model = PinballModel(self.path)
        return self.model.get_state()

    def _step(self, action):
        reward = self.model.take_action(action)
        return self.model.get_state(), reward, self.model.episode_ended(), None

    def _seed(self, seed):
        pass

register(
    id='Pinball-v0',
    entry_point='pinball:PinballDomain',
    timestep_limit=20000,
    reward_threshold=8000,
)
