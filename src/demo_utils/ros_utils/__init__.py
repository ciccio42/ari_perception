from typing import List
from rclpy.parameter import Parameter, ParameterType
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.node import Node
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, IntegerRange, FloatingPointRange
from dataclasses import dataclass
from typing import Optional, Any, Dict, Iterable
from collections import OrderedDict
from copy import deepcopy

@dataclass
class ROS2Param:
    name: str
    dest: str = "_"
    description: Optional[str] = None
    value: Optional[Any] = None
    min: Optional[Any] = None
    max: Optional[Any] = None
    step: Optional[int] = None

    def __post_init__(self):
        if self.value is None:
            raise ValueError(f"Parameter {self.name} must be set.")
        self.template()
    
    def template(self):
        param_desc = ParameterDescriptor()
        param_desc = self._add_description(param_desc)
        param_desc = self._add_min_max(param_desc)
        self.description = param_desc

    def _add_description(self, param_description: ParameterDescriptor):
        if self.description is not None:
            param_description.description = self.description
        return param_description

    def _add_min_max(self, param_description: ParameterDescriptor):
        if self.min is None and self.max is not None:
            self.min = 1e-6
        elif self.min is not None and self.max is None:
            self.max = 1e6
        elif self.min is None and self.max is None:
            return param_description
        
        if self.step is None:
            self.step = 0
        param_range = FloatingPointRange(
            from_value=float(self.min),
            to_value=float(self.max),
            step=float(self.step)
        )
        param_description.floating_point_range.append(param_range)
        return param_description


class ROS2ParamList(list):
    def __init__(self, *args: Iterable[ROS2Param]):
        check = map(lambda x: isinstance(x, ROS2Param), args)
        assert all(check), "All params must a RO2Param instance"
        super().__init__(args)
        self.map = OrderedDict()
        e: ROS2Param
        for e in args:
            self.map[e.name] = e
    
    def __contains__(self, param: Parameter) -> bool:
        return param.name in self.map.keys()

    def get(self, param_name: str) -> ROS2Param:
        return self.map[param_name]
        
    def change_dest(self, param_name, nw_dest):
        param_names = list(self.map.keys())
        idx_param = param_names.index(param_name)
        param_value = self.get(param_name)
        param_value.dest = nw_dest
        self[idx_param] = param_value
        self.map[param_name] = param_value
        return self

def _set_attribute(obj, dest: str, value):
    target = obj
    if "." in dest:
        dest_list = dest.split(".")
        nw_dest = dest_list.pop(-1)
        for x in dest_list:
            target = getattr(target, x)
        dest = nw_dest
    setattr(target, dest, value)

def init_parameters(node: Node, params: ROS2ParamList) -> List[Parameter]:
    if not isinstance(params, list):
        params = [params]
    check = list(map(lambda x: isinstance(x, ROS2Param), params))
    assert all(check), "All params must a RO2Param instance"
    nw_params = [node.declare_parameter(x.name, value=x.value, descriptor=x.description) for x in params]
    p: ROS2Param
    for p in params:
        value = node.get_parameter(p.name).value
        p.value = value
        _set_attribute(node, p.dest, value)
    if hasattr(node, "_"):
        delattr(node, "_")
    return nw_params

def parameter_list_to_dict(params: List[Parameter]) -> Dict[str, Any]:
    params_dict = {p.name: p.value for p in params}
    return params_dict

def default_reconfigure(node: Node, params: List[Parameter], ros2_param_list: ROS2ParamList):
    for p in params:
        dest = ros2_param_list.get(p.name).dest
        _set_attribute(node, dest, p.value)
        node.get_logger().info("Reconfiguring {} to {}".format(p.name, p.value))
    if hasattr(node, "_"):
        delattr(node, "_")
    return SetParametersResult(successful=True)

def default_reconfigure_with_map(node: Node, params: List[Parameter], ros2_param_list: ROS2ParamList, mapper: dict):
    ros2_param_list_copy = deepcopy(ros2_param_list)
    for param_name, nw_dest in mapper.items():
        ros2_param_list_copy = ros2_param_list_copy.change_dest(param_name, nw_dest)
    return default_reconfigure(node, params, ros2_param_list_copy)


LATCH_PROFILE = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


def wait_for_message(msg_type, node: Node, topic: str, *, qos_profile=1, time_to_wait=-1):
    """
    Wait for the next message on a topic without spinning the node.

    Same behaviour as rclpy.wait_for_message, which is missing from the rclpy
    of the r35 image. Returns (True, msg), or (False, None) on timeout or SIGINT.
    """
    context = node.context
    wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
    wait_set.clear_entities()
    sub = node.create_subscription(msg_type, topic, lambda _: None, qos_profile=qos_profile)
    try:
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)
        wait_set.wait(timeout_sec_to_nsec(time_to_wait))

        if sigint_gc.handle.pointer in wait_set.get_ready_entities('guard_condition'):
            return False, None
        if sub.handle.pointer in wait_set.get_ready_entities('subscription'):
            msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
            if msg_info is not None:
                return True, msg_info[0]
    finally:
        node.destroy_subscription(sub)
    return False, None
