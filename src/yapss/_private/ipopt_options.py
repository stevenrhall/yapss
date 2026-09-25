"""

Provides the class ``IpoptOptions``, which is a container for Ipopt options.

Instances of ``IpoptOptions`` function much like SimpleNamespace instances, but have the
advantage that ``IpoptOptions`` has type annotations, which allows an IDE such as
PyCharm to provide type hints and autocompletions.

"""

from __future__ import annotations

import difflib
import math
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

from .exceptions import YapssWarning
from .ipopt_option_specs import IPOPT_OPTION_SPECS

__all__ = ["IpoptOptionSettingWarning", "IpoptOptions"]


class IpoptOptionSettingWarning(YapssWarning):
    """Ipopt refused an option.

    Issued at the start of a solve, when Ipopt refuses an option the problem set: the option is
    not applied, and the solve continues with Ipopt's default. Which options exist, and which
    values they take, depends on the Ipopt build, so only Ipopt can judge them. YAPSS refuses at
    the assignment only what is wrong on every build: an option YAPSS sets itself, or a value
    of the wrong kind.
    """


DEFAULT_IPOPT_OPTIONS = {
    "mu_strategy": "adaptive",
    # Ipopt otherwise passes a NaN or Inf Jacobian or Hessian to its linear solver, which
    # can crash the process (MUMPS on some sparsity patterns). With the check on, Ipopt
    # stops with status -13 (Invalid_Number_Detected) instead. The scan is one pass over
    # the nonzeros per evaluation, negligible next to a factorization.
    "check_derivatives_for_naninf": "yes",
}
"""Default Ipopt options."""

RESERVED_IPOPT_OPTIONS = {
    "nlp_scaling_method": (
        "YAPSS always solves with nlp_scaling_method='user-scaling'; "
        "set 'problem.sense' and 'problem.scale.objective' instead."
    ),
    "obj_scaling_factor": (
        "YAPSS manages objective scaling internally; set 'problem.sense' (sign) and "
        "'problem.scale.objective' (magnitude) instead."
    ),
    "hessian_approximation": (
        "YAPSS chooses this based on 'problem.derivatives.order'; set that instead."
    ),
    "warm_start_init_point": (
        "YAPSS does not pass warm-start dual/bound information to Ipopt, so this "
        "option has no effect and is not supported."
    ),
}
"""Ipopt options YAPSS configures itself; setting them directly is disallowed."""


_KIND_NAMES = {"int": "Integer", "float": "Number", "str": "String"}

_CONTAINER_METHODS = frozenset({"reset", "get_options"})
"""Methods of `IpoptOptions`; assigning them would shadow the method."""


def _coerce_option(name: str, value: Any) -> str | int | float:
    """Check `value` against the kind of Ipopt option `name`, and return it as a Python type.

    Ipopt keeps three option registries -- Integer, Number, and String -- and refuses a value
    sent to the wrong one, so the value must reach the backend as the Python type that maps
    to the option's registry. `IPOPT_OPTION_SPECS` records the kind of every documented
    option; the annotations on ``IpoptOptions`` repeat it for an IDE's completions. An
    Integer option takes any integer, a Number option any integer or real, a String option
    a string; NumPy scalars are converted to the Python type. An option the table does not
    have is checked only for being one of the three kinds. ``bool``
    is refused everywhere: Python makes it an integer, but no Ipopt option is boolean (the
    yes/no options are strings).
    """
    kind = IPOPT_OPTION_SPECS.get(name, {}).get("kind")
    label = (
        f"{_KIND_NAMES[kind]} option" if isinstance(kind, str) and kind in _KIND_NAMES else "option"
    )
    got = f"got {value!r} of type {type(value).__name__}"
    if isinstance(value, bool):
        msg = f"Ipopt {label} '{name}' does not take a bool ({got}); yes/no options take a str."
        raise TypeError(msg)
    if kind == "int":
        if isinstance(value, Integral):
            return int(value)
        msg = f"Ipopt Integer option '{name}' takes an int, {got}."
        raise TypeError(msg)
    if kind == "float":
        if isinstance(value, Real):
            number = float(value)
            if not math.isfinite(number):
                msg = (
                    f"Ipopt Number option '{name}' must be a finite number, got {number}. "
                    f"Ipopt compares option values against a range, and no comparison with "
                    f"NaN is true."
                )
                raise ValueError(msg)
            return number
        msg = f"Ipopt Number option '{name}' takes a float or int, {got}."
        raise TypeError(msg)
    if kind == "str":
        if isinstance(value, str):
            return value
        msg = f"Ipopt String option '{name}' takes a str, {got}."
        raise TypeError(msg)
    # a name the table does not have: the kind is unknown, so accept any of the three and
    # let Ipopt judge. `__setattr__` has already warned that YAPSS does not recognize it.
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        return value
    msg = f"Ipopt option '{name}' takes an int, float, or str, {got}."
    raise TypeError(msg)


class IpoptOptions:
    """Container for Ipopt options.

    For a ``Problem`` instance `problem`, the user can set an Ipopt option as follows:

    >>> problem.ipopt_options.max_iter = 500  # doctest: +SKIP

    An option can be deleted by setting it to ``None``, in which case Ipopt will use the default
    value for that option:

    >>> problem.ipopt_options.max_iter = None  # doctest: +SKIP

    There are a large number of Ipopt options (over 300!). See the YAPSS documentation for
    commonly used options. A complete list of options is available in the `Ipopt options
    reference <https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_REF>`_. Some options
    may conflict with options set by YAPSS, so some care is required when setting
    ipopt_options.
    """

    def __init__(self) -> None:
        """Initialize an IpoptOptions instance."""
        self.reset()

    # Hidden from type checkers, so that the annotations below are what they check: a
    # visible __setattr__ makes them accept every attribute name, and a misspelled option
    # would pass.
    if not TYPE_CHECKING:

        def __setattr__(self, name: str, value: str | float | None) -> None:
            """Set an option value, or delete the option if value is None.

            Raises
            ------
            ValueError
                If the option is one YAPSS manages itself.
            TypeError
                If the value is not of the option's kind (Integer, Number, or String).
            """
            if name in RESERVED_IPOPT_OPTIONS:
                msg = (
                    f"'{name}' is managed by YAPSS and cannot be set directly. "
                    f"{RESERVED_IPOPT_OPTIONS[name]}"
                )
                raise ValueError(msg)
            if name in _CONTAINER_METHODS:
                # options are stored as instance attributes, so this would shadow the method
                # and `reset()` would then fail with an int not being callable
                msg = (
                    f"'{name}' is a method of ipopt_options, not an Ipopt option, and cannot be "
                    f"assigned. Ipopt has no option of that name either."
                )
                raise AttributeError(msg)
            if value is None:
                if hasattr(self, name):
                    delattr(self, name)
            else:
                super().__setattr__(name, _coerce_option(name, value))

    def reset(self) -> None:
        """Reset all options to their default values."""
        for key in list(self.__dict__.keys()):
            if not key.startswith("_") and key not in ("reset", "get_options"):
                delattr(self, key)
        for k, v in DEFAULT_IPOPT_OPTIONS.items():
            setattr(self, k, v)

    def get_options(self) -> dict[str, str | int | float]:
        """Return a dictionary of all defined option values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    # retrieved from https://coin-or.github.io/Ipopt/OPTIONS.html 2024-10-27
    # Ipopt's documentation says output_file, file_print_level and file_append work only
    # when read from an ipopt.opt file. That is not true of the C interface for the first
    # two, which is what YAPSS uses: tests/modules/test_ipopt_defaults.py sets them and
    # reads back the log Ipopt wrote, so they are listed here like any other option.
    # file_append is left out for an unrelated reason -- Ipopt 3.14.11, which the pinned
    # casadi wheel bundles, has no such option and refuses it ("It is not a valid option").
    # It arrived later, so a conda build may well have it; setting it is allowed and warns.
    # file_append: str
    accept_after_max_steps: int | None
    accept_every_trial_step: str | None
    acceptable_compl_inf_tol: float | None
    acceptable_constr_viol_tol: float | None
    acceptable_dual_inf_tol: float | None
    acceptable_iter: int | None
    acceptable_obj_change_tol: float | None
    acceptable_tol: float | None
    adaptive_mu_globalization: str | None
    adaptive_mu_kkt_norm_type: str | None
    adaptive_mu_kkterror_red_fact: float | None
    adaptive_mu_kkterror_red_iters: int | None
    adaptive_mu_monotone_init_factor: float | None
    adaptive_mu_restore_previous_iterate: str | None
    alpha_for_y: str | None
    alpha_for_y_tol: float | None
    alpha_min_frac: float | None
    alpha_red_factor: float | None
    barrier_tol_factor: float | None
    bound_frac: float | None
    bound_mult_init_method: str | None
    bound_mult_init_val: float | None
    bound_mult_reset_threshold: float | None
    bound_push: float | None
    bound_relax_factor: float | None
    check_derivatives_for_naninf: str | None
    compl_inf_tol: float | None
    constr_mult_init_max: float | None
    constr_mult_reset_threshold: float | None
    constr_viol_tol: float | None
    constraint_violation_norm_type: str | None
    corrector_compl_avrg_red_fact: float | None
    corrector_type: str | None
    delta: float | None
    dependency_detection_with_rhs: str | None
    dependency_detector: str | None
    derivative_test: str | None
    derivative_test_first_index: int | None
    derivative_test_perturbation: float | None
    derivative_test_print_all: str | None
    derivative_test_tol: float | None
    diverging_iterates_tol: float | None
    dual_inf_tol: float | None
    eta_phi: float | None
    evaluate_orig_obj_at_resto_trial: str | None
    expect_infeasible_problem: str | None
    expect_infeasible_problem_ctol: float | None
    expect_infeasible_problem_ytol: float | None
    fast_step_computation: str | None
    file_print_level: int | None
    filter_margin_fact: float | None
    filter_max_margin: float | None
    filter_reset_trigger: int | None
    findiff_perturbation: float | None
    first_hessian_perturbation: float | None
    fixed_mu_oracle: str | None
    fixed_variable_treatment: str | None
    gamma_phi: float | None
    gamma_theta: float | None
    grad_f_constant: str | None
    gradient_approximation: str | None
    hessian_approximation: str | None
    hessian_approximation_space: str | None
    hessian_constant: str | None
    honor_original_bounds: str | None
    hsllib: str | None
    inf_pr_output: str | None
    jac_c_constant: str | None
    jac_d_constant: str | None
    jacobian_approximation: str | None
    jacobian_regularization_exponent: float | None
    jacobian_regularization_value: float | None
    kappa_d: float | None
    kappa_sigma: float | None
    kappa_soc: float | None
    least_square_init_duals: str | None
    least_square_init_primal: str | None
    limited_memory_aug_solver: str | None
    limited_memory_init_val: float | None
    limited_memory_init_val_max: float | None
    limited_memory_init_val_min: float | None
    limited_memory_initialization: str | None
    limited_memory_max_history: int | None
    limited_memory_max_skipping: int | None
    limited_memory_special_for_resto: str | None
    limited_memory_update_type: str | None
    line_search_method: str | None
    linear_scaling_on_demand: str | None
    linear_solver: str | None
    linear_system_scaling: str | None
    ma27_ignore_singularity: str | None
    ma27_la_init_factor: float | None
    ma27_liw_init_factor: float | None
    ma27_meminc_factor: float | None
    ma27_pivtol: float | None
    ma27_pivtolmax: float | None
    ma27_print_level: int | None
    ma27_skip_inertia_check: str | None
    ma28_pivtol: float | None
    ma57_automatic_scaling: str | None
    ma57_block_size: int | None
    ma57_node_amalgamation: int | None
    ma57_pivot_order: int | None
    ma57_pivtol: float | None
    ma57_pivtolmax: float | None
    ma57_pre_alloc: float | None
    ma57_print_level: int | None
    ma57_small_pivot_flag: int | None
    ma77_buffer_lpage: int | None
    ma77_buffer_npage: int | None
    ma77_file_size: int | None
    ma77_maxstore: int | None
    ma77_nemin: int | None
    ma77_order: str | None
    ma77_print_level: int | None
    ma77_small: float | None
    ma77_static: float | None
    ma77_u: float | None
    ma77_umax: float | None
    ma86_nemin: int | None
    ma86_order: str | None
    ma86_print_level: int | None
    ma86_scaling: str | None
    ma86_small: float | None
    ma86_static: float | None
    ma86_u: float | None
    ma86_umax: float | None
    ma97_nemin: int | None
    ma97_order: str | None
    ma97_print_level: int | None
    ma97_scaling1: str | None
    ma97_scaling2: str | None
    ma97_scaling3: str | None
    ma97_scaling: str | None
    ma97_small: float | None
    ma97_solve_blas3: str | None
    ma97_switch1: str | None
    ma97_switch2: str | None
    ma97_switch3: str | None
    ma97_u: float | None
    ma97_umax: float | None
    max_cpu_time: float | None
    max_filter_resets: int | None
    max_hessian_perturbation: float | None
    max_iter: int | None
    max_refinement_steps: int | None
    max_resto_iter: int | None
    max_soc: int | None
    max_soft_resto_iters: int | None
    max_wall_time: float | None
    mehrotra_algorithm: str | None
    min_hessian_perturbation: float | None
    min_refinement_steps: int | None
    mu_allow_fast_monotone_decrease: str | None
    mu_init: float | None
    mu_linear_decrease_factor: float | None
    mu_max: float | None
    mu_max_fact: float | None
    mu_min: float | None
    mu_oracle: str | None
    mu_strategy: str | None
    mu_superlinear_decrease_power: float | None
    mu_target: float | None
    mumps_dep_tol: float | None
    mumps_mem_percent: int | None
    mumps_mpi_communicator: int | None
    mumps_permuting_scaling: int | None
    mumps_pivot_order: int | None
    mumps_pivtol: float | None
    mumps_pivtolmax: float | None
    mumps_print_level: int | None
    mumps_scaling: int | None
    neg_curv_test_reg: str | None
    neg_curv_test_tol: float | None
    nlp_lower_bound_inf: float | None
    nlp_scaling_constr_target_gradient: float | None
    nlp_scaling_max_gradient: float | None
    nlp_scaling_method: str | None
    nlp_scaling_min_value: float | None
    nlp_scaling_obj_target_gradient: float | None
    nlp_upper_bound_inf: float | None
    nu_inc: float | None
    nu_init: float | None
    num_linear_variables: int | None
    obj_max_inc: float | None
    obj_scaling_factor: float | None
    option_file_name: str | None
    output_file: str | None
    pardiso_iter_coarse_size: int | None
    pardiso_iter_dropping_factor: float | None
    pardiso_iter_dropping_schur: float | None
    pardiso_iter_inverse_norm_factor: float | None
    pardiso_iter_max_levels: int | None
    pardiso_iter_max_row_fill: int | None
    pardiso_iter_relative_tol: float | None
    pardiso_iterative: str | None
    pardiso_matching_strategy: str | None
    pardiso_max_droptol_corrections: int | None
    pardiso_max_iter: int | None
    pardiso_max_iterative_refinement_steps: int | None
    pardiso_msglvl: int | None
    pardiso_order: str | None
    pardiso_redo_symbolic_fact_only_if_inertia_wrong: str | None
    pardiso_repeated_perturbation_means_singular: str | None
    pardiso_skip_inertia_check: str | None
    pardisolib: str | None
    pardisomkl_matching_strategy: str | None
    pardisomkl_max_iterative_refinement_steps: int | None
    pardisomkl_msglvl: int | None
    pardisomkl_order: str | None
    pardisomkl_redo_symbolic_fact_only_if_inertia_wrong: str | None
    pardisomkl_repeated_perturbation_means_singular: str | None
    pardisomkl_skip_inertia_check: str | None
    perturb_always_cd: str | None
    perturb_dec_fact: float | None
    perturb_inc_fact: float | None
    perturb_inc_fact_first: float | None
    point_perturbation_radius: float | None
    print_advanced_options: str | None
    print_frequency_iter: int | None
    print_frequency_time: float | None
    print_info_string: str | None
    print_level: int | None
    print_options_documentation: str | None
    print_options_mode: str | None
    print_timing_statistics: str | None
    print_user_options: str | None
    quality_function_balancing_term: str | None
    quality_function_centrality: str | None
    quality_function_max_section_steps: int | None
    quality_function_norm_type: str | None
    quality_function_section_qf_tol: float | None
    quality_function_section_sigma_tol: float | None
    recalc_y: str | None
    recalc_y_feas_tol: float | None
    replace_bounds: str | None
    required_infeasibility_reduction: float | None
    residual_improvement_factor: float | None
    residual_ratio_max: float | None
    residual_ratio_singular: float | None
    resto_failure_feasibility_threshold: float | None
    resto_penalty_parameter: float | None
    resto_proximity_weight: float | None
    rho: float | None
    s_max: float | None
    s_phi: float | None
    s_theta: float | None
    sb: str | None
    sigma_max: float | None
    sigma_min: float | None
    skip_corr_if_neg_curv: str | None
    skip_corr_in_monotone_mode: str | None
    skip_finalize_solution_call: str | None
    slack_bound_frac: float | None
    slack_bound_push: float | None
    slack_move: float | None
    soc_method: int | None
    soft_resto_pderror_reduction_factor: float | None
    spral_cpu_block_size: int | None
    spral_gpu_perf_coeff: float | None
    spral_ignore_numa: str | None
    spral_max_load_inbalance: float | None
    spral_min_gpu_work: float | None
    spral_nemin: int | None
    spral_order: str | None
    spral_pivot_method: str | None
    spral_print_level: int | None
    spral_scaling: str | None
    spral_scaling_1: str | None
    spral_scaling_2: str | None
    spral_scaling_3: str | None
    spral_small: float | None
    spral_small_subtree_threshold: float | None
    spral_switch_1: str | None
    spral_switch_2: str | None
    spral_switch_3: str | None
    spral_u: float | None
    spral_umax: float | None
    spral_use_gpu: str | None
    start_with_resto: str | None
    tau_min: float | None
    theta_max_fact: float | None
    theta_min_fact: float | None
    timing_statistics: str | None
    tiny_step_tol: float | None
    tiny_step_y_tol: float | None
    tol: float | None
    warm_start_bound_frac: float | None
    warm_start_bound_push: float | None
    warm_start_entire_iterate: str | None
    warm_start_init_point: str | None
    warm_start_mult_bound_push: float | None
    warm_start_mult_init_max: float | None
    warm_start_same_structure: str | None
    warm_start_slack_bound_frac: float | None
    warm_start_slack_bound_push: float | None
    warm_start_target_mu: float | None
    watchdog_shortened_iter_trigger: int | None
    watchdog_trial_iter_max: int | None
    wsmp_inexact_droptol: float | None
    wsmp_inexact_fillin_limit: float | None
    wsmp_max_iter: int | None
    wsmp_no_pivoting: str | None
    wsmp_num_threads: int | None
    wsmp_ordering_option2: int | None
    wsmp_ordering_option: int | None
    wsmp_pivtol: float | None
    wsmp_pivtolmax: float | None
    wsmp_scaling: int | None
    wsmp_singularity_threshold: float | None
    wsmp_skip_inertia_check: str | None
    wsmp_write_matrix_iteration: int | None


def refusal_message(name: str, value: str | int | float) -> str:
    """Return the warning for an option Ipopt refused.

    Ipopt says only that it refused the option; its console output says why. The table of
    documented options adds one hint where it can: a name close to a documented one may be a
    misspelling, and a documented option may not be provided by this build or may not take the
    value. The table describes one Ipopt release, so the hint never decides anything.
    """
    if name in IPOPT_OPTION_SPECS:
        hint = "This Ipopt build might not provide it, or might not accept that value. "
    else:
        near = difflib.get_close_matches(name, IPOPT_OPTION_SPECS, n=1, cutoff=0.8)
        hint = f"Did you mean '{near[0]}'? " if near else ""
    return (
        f"Ipopt refused option '{name}' with value {value!r}. {hint}The option was not "
        f"applied, and the solve continues with Ipopt's default. Check Ipopt's console output "
        f"above for the exact cause."
    )
