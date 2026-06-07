"""Central Safety-Gym configuration for supervised switching experiments."""

STATIC_CONFIG = {
    "placements_extents": [-1.5, -1.5, 1.5, 1.5],
    "robot_base": "xmls/point.xml",
    "task": "goal",
    "goal_size": 0.3,
    "goal_keepout": 0.305,
    "goal_locations": [(1.1, 1.1)],
    "observe_goal_lidar": True,
    "observe_hazards": True,
    "constrain_hazards": True,
    "lidar_max_dist": 3,
    "lidar_num_bins": 16,
    "hazards_num": 1,
    "hazards_size": 0.7,
    "hazards_keepout": 0.705,
    "hazards_locations": [(0, 0)],
}
