PARAMS1 = {
    "goal_weight": 100.0,
    "noncollide_weight": 200.0,
    "normal_weight": 200.0,
    "contact_weight": 0.0,
    "fine_voxel_resolution": 32,
    "exclude_object_points": True,
}

PARAMS2 = {
    "goal_weight": 100.0,
    "noncollide_weight": 200.0,
    "normal_weight": 10.0,
    "contact_weight": 0.0,
    "fine_voxel_resolution": 128,
    "exclude_object_points": False,
}

PARAMS3 = {
    "goal_weight": 100.0,
    "noncollide_weight": 500.0,
    "normal_weight": 0.0,
    "contact_weight": 500.0,
    "fine_voxel_resolution": 128,
    "exclude_object_points": True,
}

GUIDANCE_PARAMS_DICT = {
    "open": PARAMS1,
    "close": PARAMS1,
    "pull": PARAMS1,
    "push": PARAMS1,
    "press": PARAMS1,
    "pick": PARAMS2,
    "pickup": PARAMS2,
    "take": PARAMS2,
    "get": PARAMS2,
    "put": PARAMS2,
    "place": PARAMS2,
    "putdown": PARAMS2,
    "drop": PARAMS2,
    "wipe": PARAMS3,
    "move": PARAMS3,
    "other": PARAMS2,

}

COMMON_ACTIONS = [
    "open",
    "close",
    "pull",
    "push",
    "press",
    "pick",
    "pickup",
    "take",
    "get",
    "put",
    "place",
    "putdown",
    "drop",
    "wipe",
    "move",
]
