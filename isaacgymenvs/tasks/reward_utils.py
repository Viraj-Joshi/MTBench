
import torch
import math

@torch.jit.script
def _sigmoids(x, value_at_1, sigmoid):
    # type: (Tensor, float, str) -> Tensor
    """Returns 1 when `x` == 0, between 0 and 1 otherwise.

    Args:
        x: A scalar or numpy array.
        value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
        sigmoid: String, choice of sigmoid type.

    Returns:
        A numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
        `quadratic` sigmoids which allow `value_at_1` == 0.
        ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be nonnegative and smaller than 1, "
                "got {}.".format(value_at_1)
            )
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(
                "`value_at_1` must be strictly between 0 and 1, "
                "got {}.".format(value_at_1)
            )
    
    if sigmoid == "gaussian":
        scale = math.sqrt(-2 * math.log(value_at_1))
        return torch.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == "hyperbolic":
        scale = math.acosh(1 / value_at_1)
        return 1 / torch.cosh(x * scale)

    elif sigmoid == "long_tail":
        scale = math.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == "reciprocal":
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == "cosine":
        scale = math.acos(2 * value_at_1 - 1) / math.pi
        scaled_x = x * scale
        return torch.where(abs(scaled_x) < 1, (1 + math.cos(math.pi * scaled_x)) / 2, 0.0)

    elif sigmoid == "linear":
        scale = 1 - value_at_1
        scaled_x = x * scale
        return torch.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == "quadratic":
        scale = math.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return torch.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == "tanh_squared":
        scale = math.atanh(math.sqrt(1 - value_at_1))
        return 1 - torch.tanh(x * scale) ** 2
    else:
        raise ValueError(f"Unknown sigmoid type {sigmoid}.")

_DEFAULT_VALUE_AT_MARGIN = .1
@torch.jit.script
def tolerance(
    x,
    margin,
    bounds=(0.0, 0.0),
    sigmoid="gaussian",
    value_at_margin=_DEFAULT_VALUE_AT_MARGIN,
):
    # type: (Tensor, Tensor, Tuple[float,float], str, float) -> Tensor
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
        x: A torch tensor.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Tensor. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds for each environment.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
        'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    # margin = margin.clamp(min=0)
    if (margin < 0).any():
        b = margin < 0
        raise ValueError(f"`margin` must be non-negative. Neg at indices {b.nonzero()} with values {margin[b.nonzero()]}")

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    value = torch.where((margin==0) & (in_bounds), 1.0, 0.0)

    d = torch.where(x < lower, lower - x, x - upper) / margin
    value = torch.where((margin > 0) & (in_bounds), value, _sigmoids(d, value_at_margin, sigmoid))
    if torch.logical_or(value < 0, value > 1).any():
        raise ValueError(f"tolerance must range between 0 and 1. Outside bounds at indices {value.nonzero()} with values {value[value.nonzero()]}")
    return value

@torch.jit.script
def hamacher_product(a, b):
    """The hamacher (t-norm) product of a and b.

    computes (a * b) / ((a + b) - (a * b))

    Args:
        a (tensor): 1st term of hamacher product.
        b (tensor): 2nd term of hamacher product.

    Raises:
        ValueError: a and b must range between 0 and 1

    Returns:
        tensor: The hammacher product of a and b
    """
    # type: (Tensor, Tensor) -> Tensor
    if torch.logical_or(a < 0, a > 1).any():
        cond = (a < 0.0) | (a > 1.0)
        raise ValueError(f"a must range between 0 and 1 at indices {cond.nonzero()} with values {a[cond.nonzero()]}")
    if torch.logical_or(b < 0, b > 1).any():
        cond = (b < 0.0) | (b > 1.0)
        raise ValueError(f"b must range between 0 and 1 at indices {cond.nonzero()} with values {b[cond.nonzero()]}")

    denominator = (a + b) - (torch.mul(a, b))
    h_prod = torch.where(denominator>0,((torch.mul(a, b)) / denominator),0)

    if torch.logical_or(h_prod < 0, h_prod > 1).any():
        raise ValueError(f"hamacher product must range between 0 and 1. Outside bounds at indices {h_prod.nonzero()} with values {h_prod[h_prod.nonzero()]}")
    return h_prod

@torch.jit.script
def _gripper_caging_reward(
    obj_pos,
    franka_lfinger_pos,
    franka_rfinger_pos,
    tcp_center,
    init_tcp,
    actions,
    obj_init_pos,
    obj_radius,
    pad_success_thresh,
    object_reach_radius,
    xz_thresh,
    desired_gripper_effort=1.0,
    high_density=False,
    medium_density=False,
):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float, float, bool, bool ) -> torch.Tensor
    """Reward for agent grasping obj.

    Args:
        obj_pos,
        franka_lfinger_pos
        franka_rfinger_pos
        tcp_center
        init_tcp
        actions
        obj_init_pos
        obj_radius(float):radius of object's bounding sphere
        pad_success_thresh(float): successful distance of gripper_pad
            to object
        object_reach_radius(float): successful distance of gripper center
            to the object.
        xz_thresh(float): successful distance of gripper in x_z axis to the
            object. Y axis not included since the caging function handles
                successful grasping in the Y axis.
        desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
        high_density(bool): flag for high-density. Cannot be used with medium-density.
        medium_density(bool): flag for medium-density. Cannot be used with high-density.
    """
    if high_density and medium_density:
        raise ValueError("Can only be either high_density or medium_density")
    # MARK: Left-right gripper information for caging reward----------------
    left_pad = franka_lfinger_pos
    right_pad = franka_rfinger_pos

    # get current positions of left and right pads (Y axis)
    pad_y_lr = torch.hstack((left_pad[:,0].unsqueeze(-1), right_pad[:,0].unsqueeze(-1)))
    # compare *current* pad positions with *current* obj position (Y axis)
    pad_to_obj_lr = torch.abs(pad_y_lr - obj_pos[:,0].unsqueeze(-1))
    # compare *current* pad positions with *initial* obj position (Y axis)
    pad_to_objinit_lr = torch.abs(pad_y_lr - obj_init_pos[:,0].unsqueeze(-1))

    # Compute the left/right caging rewards. This is crucial for success,
    # yet counterintuitive mathematically because we invented it
    # accidentally.
    #
    # Before touching the object, `pad_to_obj_lr` ("x") is always separated
    # from `caging_lr_margin` ("the margin") by some small number,
    # `pad_success_thresh`.
    #
    # When far away from the object:
    #       x = margin + pad_success_thresh
    #       --> Thus x is outside the margin, yielding very small reward.
    #           Here, any variation in the reward is due to the fact that
    #           the margin itself is shifting.
    # When near the object (within pad_success_thresh):
    #       x = pad_success_thresh - margin
    #       --> Thus x is well within the margin. As long as x > obj_radius,
    #           it will also be within the bounds, yielding maximum reward.
    #           Here, any variation in the reward is due to the gripper
    #           moving *too close* to the object (i.e, blowing past the
    #           obj_radius bound).
    #
    # Therefore, before touching the object, this is very nearly a binary
    # reward -- if the gripper is between obj_radius and pad_success_thresh,
    # it gets maximum reward. Otherwise, the reward very quickly falls off.
    #
    # After grasping the object and moving it away from initial position,
    # x remains (mostly) constant while the margin grows considerably. This
    # penalizes the agent if it moves *back* toward `obj_init_pos`, but
    # offers no encouragement for leaving that position in the first place.
    # That part is left to the reward functions of individual environments.
    caging_lr_margin = torch.abs(pad_to_objinit_lr - pad_success_thresh)
    caging_lr = [
        tolerance(
            pad_to_obj_lr[:,i],  # "x" in the description above
            bounds=(obj_radius, pad_success_thresh),
            margin=caging_lr_margin[:,i],  # "margin" in the description above
            sigmoid="long_tail",
        )
        for i in range(2)
    ]
    caging_y = hamacher_product(caging_lr[0],caging_lr[1])

    # MARK: X-Z gripper information for caging reward-----------------------
    tcp = tcp_center
    xz = [1, 2]

    # Compared to the caging_y reward, caging_xz is simple. The margin is
    # constant (something in the 0.3 to 0.5 range) and x shrinks as the
    # gripper moves towards the object. After picking up the object, the
    # reward is maximized and changes very little
    caging_xz_margin = torch.norm(obj_init_pos[:,xz] - init_tcp[:,xz],dim=-1)
    
    caging_xz_margin -= xz_thresh
    caging_xz = tolerance(
        torch.norm(tcp[:,xz] - obj_pos[:,xz],dim=-1),  # "x" in the description above
        bounds=(0.0, xz_thresh),
        margin=caging_xz_margin,  # "margin" in the description above
        sigmoid="long_tail",
    )

    # MARK: Closed-extent gripper information for caging reward-------------
    gripper_closed = (
        torch.minimum(torch.maximum(torch.zeros_like(actions[:,-1]), actions[:,-1]),torch.ones_like(actions[:,-1]) * desired_gripper_effort)/desired_gripper_effort
    )
    
    # MARK: Combine components----------------------------------------------
    caging = hamacher_product(caging_y, caging_xz)
    gripping = torch.where(caging > 0.97, gripper_closed, torch.zeros_like(gripper_closed))
    caging_and_gripping = hamacher_product(caging, gripping)

    if high_density:
        caging_and_gripping = (caging_and_gripping + caging) / 2
    if medium_density:
        tcp = tcp_center
        tcp_to_obj = torch.norm(obj_pos - tcp,dim=-1)
        tcp_to_obj_init = torch.norm(obj_init_pos - init_tcp,dim=-1)
        # Compute reach reward
        # - We subtract `object_reach_radius` from the margin so that the
        #   reward always starts with a value of 0.1
        reach_margin = torch.abs(tcp_to_obj_init - object_reach_radius)
        reach = tolerance(
            tcp_to_obj,
            bounds=(0.0, object_reach_radius),
            margin=reach_margin,
            sigmoid="long_tail",
        )
        caging_and_gripping = (caging_and_gripping + reach) / 2

    return caging_and_gripping

@torch.jit.script
def in_range(a:torch.Tensor, b:torch.Tensor, c:torch.Tensor) -> torch.Tensor:
    return torch.where(c>=b,(b <= a) & (a <= c), (c <= a) & (a <= b))

@torch.jit.script
def rect_prism_tolerance(curr:torch.Tensor, zero:torch.Tensor, one:torch.Tensor) -> torch.Tensor:
    """Computes a reward if curr is inside a rectangular prism region.

    The 3d points curr and zero specify 2 diagonal corners of a rectangular
    prism that represents the decreasing region.

    one represents the corner of the prism that has a reward of 1.
    zero represents the diagonal opposite corner of the prism that has a reward
        of 0.
    Curr is the point that the prism reward region is being applied for.

    Args:
        curr(torch.Tensor): The point whose reward is being assessed.
            shape is (num_envs,3).
        zero(torch.Tensor): One corner of the rectangular prism, with reward 0.
            shape is (num_envs,3)
        one(torch.Tensor): The diagonal opposite corner of one, with reward 1.
            shape is (num_envs,3)
    """

    in_prism = (
        in_range(curr[:,0], zero[:,0], one[:,0])
        & in_range(curr[:,1], zero[:,1], one[:,1])
        & in_range(curr[:,2], zero[:,2], one[:,2])
    )

    diff = one - zero   
    x_scale = (curr[:,0] - zero[:,0]) / diff[:,0]
    y_scale = (curr[:,1] - zero[:,1]) / diff[:,1]
    z_scale = (curr[:,2] - zero[:,2]) / diff[:,2]

    return torch.where((in_prism & 1 == 1), x_scale * y_scale * z_scale, 1)