import random
from cv2 import add

import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy, BrakeVehicle,
                                                                      StopVehicle,
                                                                      WaypointFollower)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToVehicle,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance,
                                                                               StandStill)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import generate_target_waypoint_list, get_waypoint_in_distance


def _drive_for_distance(
    scenario: BasicScenario,
    composite: py_trees.composites.Composite,
    target_speed: float,
    distance: float
) -> None:
  other_actor = scenario.other_actors[0]
  drive = py_trees.composites.Parallel(
      "DrivingForDistance",
      policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
  drive.add_child(WaypointFollower(other_actor, target_speed))
  drive.add_child(DriveDistance(other_actor, distance))
  composite.add_child(drive)


class FollowLeadingVehicleChangingVelocity(BasicScenario):

  timeout = 60            # Timeout of scenario in seconds

  def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
               timeout=60):
    """
    Setup all relevant parameters and create scenario

    If randomize is True, the scenario parameters are randomized
    """

    self._map = CarlaDataProvider.get_map()
    self._ego_vehicle_transform = config.trigger_points[0]
    self._first_vehicle_speed = 8
    self._first_vehicle_x_offset = 20
    self._other_actor_max_brake = 1.0
    self._other_actor_transform = None
    # Timeout of scenario in seconds
    self.timeout = timeout

    super(FollowLeadingVehicleChangingVelocity, self).__init__("FollowVehicleChangingVelocity",
                                                               ego_vehicles,
                                                               config,
                                                               world,
                                                               debug_mode,
                                                               criteria_enable=criteria_enable)

    if randomize:
      self._ego_other_distance_start = random.randint(4, 8)

  def _initialize_actors(self, config):
    """
    Custom initialization
    """

    self._other_actor_transform = carla.Transform(
        carla.Location(self._ego_vehicle_transform.location.x + self._first_vehicle_x_offset,
                       self._ego_vehicle_transform.location.y,
                       self._ego_vehicle_transform.location.z),
        self._ego_vehicle_transform.rotation)
    first_vehicle_transform = carla.Transform(
        carla.Location(self._other_actor_transform.location.x,
                       self._other_actor_transform.location.y,
                       self._other_actor_transform.location.z - 500),
        self._other_actor_transform.rotation)
    first_vehicle = CarlaDataProvider.request_new_actor(
        'vehicle.nissan.patrol', first_vehicle_transform)
    first_vehicle.set_simulate_physics(enabled=False)
    self.other_actors.append(first_vehicle)

  def _create_behavior(self):
    """
    The scenario defined after is a "follow leading vehicle" scenario. After
    invoking this scenario, it will wait for the user controlled vehicle to
    enter the start region, then make the other actor to drive until reaching
    the next intersection. Finally, the user-controlled vehicle has to be close
    enough to the other actor to end the scenario.
    If this does not happen within 60 seconds, a timeout stops the scenario
    """

    # set other_actor to first other actor
    other_actor = self.other_actors[0]

    # to avoid the other actor blocking traffic, it was spawed elsewhere
    # reset its pose to the required one
    start_transform = ActorTransformSetter(
        other_actor, self._other_actor_transform)

    # drive straight while changing velocity
    drive = py_trees.composites.Sequence(
        "Drive straight while changing velocity")
    _drive_for_distance(self, drive, self._first_vehicle_speed, 40)

    _drive_for_distance(self, drive, 10, 20)

    _drive_for_distance(self, drive, 12, 100)
    _drive_for_distance(self, drive, 11, 20)
    _drive_for_distance(self, drive, 10, 20)
    _drive_for_distance(self, drive, 9, 12)
    _drive_for_distance(self, drive, 8, 12)

    _drive_for_distance(self, drive, 12, 10)
    _drive_for_distance(self, drive, 20, 100)
    _drive_for_distance(self, drive, 19, 10)
    _drive_for_distance(self, drive, 18, 10)
    _drive_for_distance(self, drive, 16, 10)
    _drive_for_distance(self, drive, 14, 10)
    _drive_for_distance(self, drive, 12, 10)
    _drive_for_distance(self, drive, 10, 5)

    _drive_for_distance(self, drive, 16, 50)
    _drive_for_distance(self, drive, 15, 20)
    _drive_for_distance(self, drive, 14, 10)
    _drive_for_distance(self, drive, 13, 6)
    _drive_for_distance(self, drive, 12, 6)
    _drive_for_distance(self, drive, 11, 6)
    _drive_for_distance(self, drive, 10, 6)
    _drive_for_distance(self, drive, 9, 6)
    _drive_for_distance(self, drive, 12, 70)

    # stop vehicle
    stop = StopVehicle(other_actor, self._other_actor_max_brake)

    # end condition
    endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
    endcondition_part1 = InTriggerDistanceToVehicle(other_actor,
                                                    self.ego_vehicles[0],
                                                    distance=20,
                                                    name="FinalDistance")
    # endcondition_part2 = StandStill(
    #     self.ego_vehicles[0], name="StandStill", duration=1)
    # endcondition.add_child(endcondition_part1)
    # endcondition.add_child(endcondition_part2)

    # Build behavior tree
    sequence = py_trees.composites.Sequence("Sequence Behavior")
    sequence.add_child(start_transform)
    sequence.add_child(drive)
    sequence.add_child(endcondition)
    sequence.add_child(ActorDestroy(self.other_actors[0]))

    return sequence

  def _create_test_criteria(self):
    """
    A list of all test criteria will be created that is later used
    in parallel behavior tree.
    """
    criteria = []

    collision_criterion = CollisionTest(self.ego_vehicles[0])

    criteria.append(collision_criterion)

    return criteria

  def __del__(self):
    """
    Remove all actors upon deletion
    """
    self.remove_all_actors()


class FollowLeadingVehicleChangingVelocityAndStop(BasicScenario):

  timeout = 60            # Timeout of scenario in seconds

  def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
               timeout=60):
    """
    Setup all relevant parameters and create scenario

    If randomize is True, the scenario parameters are randomized
    """

    self._map = CarlaDataProvider.get_map()
    self._first_vehicle_location = 30
    self._first_vehicle_speed = 10
    self._reference_waypoint = self._map.get_waypoint(
        config.trigger_points[0].location)
    self._other_actor_max_brake = 1.0
    self._other_actor_stop_in_front_intersection = 100
    self._other_actor_transform = None
    # Timeout of scenario in seconds
    self.timeout = timeout

    super(FollowLeadingVehicleChangingVelocityAndStop, self).__init__("FollowVehicleChangingVelocityAndStop",
                                                                      ego_vehicles,
                                                                      config,
                                                                      world,
                                                                      debug_mode,
                                                                      criteria_enable=criteria_enable)

    if randomize:
      self._ego_other_distance_start = random.randint(4, 8)

  def _initialize_actors(self, config):
    """
    Custom initialization
    """

    first_vehicle_waypoint, _ = get_waypoint_in_distance(
        self._reference_waypoint, self._first_vehicle_location)
    self._other_actor_transform = carla.Transform(
        carla.Location(first_vehicle_waypoint.transform.location.x,
                       first_vehicle_waypoint.transform.location.y,
                       first_vehicle_waypoint.transform.location.z + 0.2),
        first_vehicle_waypoint.transform.rotation)
    first_vehicle_transform = carla.Transform(
        carla.Location(self._other_actor_transform.location.x,
                       self._other_actor_transform.location.y,
                       self._other_actor_transform.location.z - 500),
        self._other_actor_transform.rotation)
    first_vehicle = CarlaDataProvider.request_new_actor(
        'vehicle.nissan.patrol', first_vehicle_transform)
    first_vehicle.set_simulate_physics(enabled=False)
    self.other_actors.append(first_vehicle)

  def _create_behavior(self):
    """
    The scenario defined after is a "follow leading vehicle" scenario. After
    invoking this scenario, it will wait for the user controlled vehicle to
    enter the start region, then make the other actor to drive until reaching
    the next intersection. Finally, the user-controlled vehicle has to be close
    enough to the other actor to end the scenario.
    If this does not happen within 60 seconds, a timeout stops the scenario
    """

    # set other_actor to first other actor
    other_actor = self.other_actors[0]

    # to avoid the other actor blocking traffic, it was spawed elsewhere
    # reset its pose to the required one
    start_transform = ActorTransformSetter(
        other_actor, self._other_actor_transform)

    # make plan for going straight at first intersection with desired speed
    first_vehicle_waypoint, _ = get_waypoint_in_distance(
        self._reference_waypoint, self._first_vehicle_location)
    plan, _ = generate_target_waypoint_list(first_vehicle_waypoint, turn=0)
    straight_at_first_intersection = WaypointFollower(
        other_actor, self._first_vehicle_speed, plan=plan)

    driving_to_next_intersection = py_trees.composites.Parallel(
        "DrivingTowardsIntersection",
        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

    driving_to_next_intersection.add_child(WaypointFollower(
        self.other_actors[0], self._first_vehicle_speed))
    driving_to_next_intersection.add_child(InTriggerDistanceToNextIntersection(
        self.other_actors[0], 0))

    # drive straight while changing velocity
    # drive = py_trees.composites.Sequence("Drive straight while changing velocity")
    # _drive_for_distance(drive, 8, 30)
    # drive.add_child(BrakeVehicle(other_actor, 0.2 , 1))
    # _drive_for_distance(drive, 6, 20)
    # _drive_for_distance(drive, 12, 40)
    # drive.add_child(BrakeVehicle(other_actor, 1 , 1))
    # _drive_for_distance(drive, 8, 50)
    # drive.add_child(BrakeVehicle(other_actor, 0.2 , 1))
    # _drive_for_distance(drive, 8, 30)
    # drive.add_child(BrakeVehicle(other_actor, 0.1 , 1))
    # _drive_for_distance(drive, 6, 30)

    # drive straight while changing velocity
    # drive_1 = py_trees.composites.Sequence("Drive straight while changing velocity")
    # _drive_for_distance(drive_1, 8, 30)
    # drive_1.add_child(BrakeVehicle(other_actor, 0.2 , 1))
    # _drive_for_distance(drive_1, 6, 20)
    # _drive_for_distance(drive_1, 12, 40)
    # drive_1.add_child(BrakeVehicle(other_actor, 1 , 1))
    # _drive_for_distance(drive_1, 8, 50)
    # drive_1.add_child(BrakeVehicle(other_actor, 0.2 , 1))
    # _drive_for_distance(drive_1, 8, 30)
    # drive_1.add_child(BrakeVehicle(other_actor, 0.1 , 1))
    # _drive_for_distance(drive_1, 8, 25)

    # drive straight while changing velocity
    drive_1 = py_trees.composites.Sequence(
        "Drive straight while changing velocity")
    _drive_for_distance(self, drive_1, 8, 30)
    drive_1.add_child(BrakeVehicle(other_actor, 0.2, 1))
    _drive_for_distance(self, drive_1, 6, 20)
    _drive_for_distance(self, drive_1, 12, 40)
    drive_1.add_child(BrakeVehicle(other_actor, 1, 1))
    _drive_for_distance(self, drive_1, 8, 50)
    drive_1.add_child(BrakeVehicle(other_actor, 0.2, 1))
    _drive_for_distance(self, drive_1, 8, 30)
    drive_1.add_child(BrakeVehicle(other_actor, 0.1, 1))
    _drive_for_distance(self, drive_1, 8, 25)

    drive_2 = py_trees.composites.Sequence(
        "Drive straight while changing velocity")
    drive_2.add_child(drive_1)
    drive_2.add_child(BrakeVehicle(other_actor, 0.2, 1))
    _drive_for_distance(self, drive_2, 8, 30)
    drive_2.add_child(BrakeVehicle(other_actor, 0.2, 1))
    _drive_for_distance(self, drive_2, 8, 20)

    # stop vehicle
    # stop = StopVehicle(other_actor, self._other_actor_max_brake)
    stop = StopVehicle(other_actor, 0.005)

    # end condition
    endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
    endcondition_part1 = InTriggerDistanceToVehicle(other_actor,
                                                    self.ego_vehicles[0],
                                                    distance=20,
                                                    name="FinalDistance")
    endcondition_part2 = StandStill(
        self.ego_vehicles[0], name="StandStill", duration=1)
    endcondition.add_child(endcondition_part1)
    endcondition.add_child(endcondition_part2)

    # Build behavior tree
    sequence = py_trees.composites.Sequence("Sequence Behavior")
    sequence.add_child(start_transform)
    sequence.add_child(straight_at_first_intersection)
    sequence.add_child(drive_2)
    sequence.add_child(stop)
    sequence.add_child(endcondition)
    sequence.add_child(ActorDestroy(self.other_actors[0]))

    return sequence

  def _create_test_criteria(self):
    """
    A list of all test criteria will be created that is later used
    in parallel behavior tree.
    """
    criteria = []

    collision_criterion = CollisionTest(self.ego_vehicles[0])

    criteria.append(collision_criterion)

    return criteria

  def __del__(self):
    """
    Remove all actors upon deletion
    """
    self.remove_all_actors()


class FollowLeadingVehicleTailgating(BasicScenario):

  timeout = 60            # Timeout of scenario in seconds

  def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
               timeout=60):
    """
    Setup all relevant parameters and create scenario

    If randomize is True, the scenario parameters are randomized
    """

    self._map = CarlaDataProvider.get_map()
    self._ego_vehicle_transform = config.trigger_points[0]
    self._first_vehicle_speed = 16
    self._first_vehicle_x_offset = 50
    self._other_actor_max_brake = 1.0
    self._other_actor_transform = None
    # Timeout of scenario in seconds
    self.timeout = timeout

    super(FollowLeadingVehicleTailgating, self).__init__("FollowVehicleTailgating",
                                                               ego_vehicles,
                                                               config,
                                                               world,
                                                               debug_mode,
                                                               criteria_enable=criteria_enable)

    if randomize:
      self._ego_other_distance_start = random.randint(4, 8)

  def _initialize_actors(self, config):
    """
    Custom initialization
    """

    self._other_actor_transform = carla.Transform(
        carla.Location(self._ego_vehicle_transform.location.x + self._first_vehicle_x_offset,
                       self._ego_vehicle_transform.location.y,
                       self._ego_vehicle_transform.location.z),
        self._ego_vehicle_transform.rotation)
    first_vehicle_transform = carla.Transform(
        carla.Location(self._other_actor_transform.location.x,
                       self._other_actor_transform.location.y,
                       self._other_actor_transform.location.z - 500),
        self._other_actor_transform.rotation)
    first_vehicle = CarlaDataProvider.request_new_actor(
        'vehicle.nissan.patrol', first_vehicle_transform)
    first_vehicle.set_simulate_physics(enabled=False)
    self.other_actors.append(first_vehicle)

  def _create_behavior(self):
    """
    The scenario defined after is a "follow leading vehicle" scenario. After
    invoking this scenario, it will wait for the user controlled vehicle to
    enter the start region, then make the other actor to drive until reaching
    the next intersection. Finally, the user-controlled vehicle has to be close
    enough to the other actor to end the scenario.
    If this does not happen within 60 seconds, a timeout stops the scenario
    """

    # set other_actor to first other actor
    other_actor = self.other_actors[0]

    # to avoid the other actor blocking traffic, it was spawed elsewhere
    # reset its pose to the required one
    start_transform = ActorTransformSetter(
        other_actor, self._other_actor_transform)

    # drive straight while changing velocity
    drive = py_trees.composites.Sequence(
        "Drive straight while changing velocity")
    _drive_for_distance(self, drive, self._first_vehicle_speed, 540)

    # stop vehicle
    stop = StopVehicle(other_actor, self._other_actor_max_brake)

    # end condition
    endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
    endcondition_part1 = InTriggerDistanceToVehicle(other_actor,
                                                    self.ego_vehicles[0],
                                                    distance=20,
                                                    name="FinalDistance")

    # Build behavior tree
    sequence = py_trees.composites.Sequence("Sequence Behavior")
    sequence.add_child(start_transform)
    sequence.add_child(drive)
    sequence.add_child(endcondition)
    sequence.add_child(ActorDestroy(self.other_actors[0]))

    return sequence

  def _create_test_criteria(self):
    """
    A list of all test criteria will be created that is later used
    in parallel behavior tree.
    """
    criteria = []

    collision_criterion = CollisionTest(self.ego_vehicles[0])

    criteria.append(collision_criterion)

    return criteria

  def __del__(self):
    """
    Remove all actors upon deletion
    """
    self.remove_all_actors()