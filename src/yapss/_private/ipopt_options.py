"""

Provides the class ``IpoptOptions``, which is a container for Ipopt options.

Instances of ``IpoptOptions`` function much like SimpleNamespace instances, but have the
advantage that ``IpoptOptions`` has type annotations, which allows an IDE such as
PyCharm to provide type hints and autocompletions.

"""

from __future__ import annotations

__all__ = ["IpoptOptions"]

DEFAULT_IPOPT_OPTIONS = {
    "mu_strategy": "adaptive",
}
"""Default Ipopt options."""


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

    def __setattr__(self, name: str, value: str | float | None) -> None:
        """Set an option value, or delete the option if value is None."""
        if value is None:
            if hasattr(self, name):
                delattr(self, name)
        else:
            super().__setattr__(name, value)

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
    accept_after_max_steps: int
    accept_every_trial_step: str
    acceptable_compl_inf_tol: float
    acceptable_constr_viol_tol: float
    acceptable_dual_inf_tol: float
    acceptable_iter: int
    acceptable_obj_change_tol: float
    acceptable_tol: float
    adaptive_mu_globalization: str
    adaptive_mu_kkt_norm_type: str
    adaptive_mu_kkterror_red_fact: float
    adaptive_mu_kkterror_red_iters: int
    adaptive_mu_monotone_init_factor: float
    adaptive_mu_restore_previous_iterate: str
    alpha_for_y: str
    alpha_for_y_tol: float
    alpha_min_frac: float
    alpha_red_factor: float
    barrier_tol_factor: float
    bound_frac: float
    bound_mult_init_method: str
    bound_mult_init_val: float
    bound_mult_reset_threshold: float
    bound_push: float
    bound_relax_factor: float
    check_derivatives_for_naninf: str
    compl_inf_tol: float
    constr_mult_init_max: float
    constr_mult_reset_threshold: float
    constr_viol_tol: float
    constraint_violation_norm_type: str
    corrector_compl_avrg_red_fact: float
    corrector_type: str
    delta: float
    dependency_detection_with_rhs: str
    dependency_detector: str
    derivative_test: str
    derivative_test_first_index: int
    derivative_test_perturbation: float
    derivative_test_print_all: str
    derivative_test_tol: float
    diverging_iterates_tol: float
    dual_inf_tol: float
    eta_phi: float
    evaluate_orig_obj_at_resto_trial: str
    expect_infeasible_problem: str
    expect_infeasible_problem_ctol: float
    expect_infeasible_problem_ytol: float
    fast_step_computation: str
    # only works when read from the ipopt.opt options file
    # file_append: str
    # file_print_level: int
    filter_margin_fact: float
    filter_max_margin: float
    filter_reset_trigger: int
    findiff_perturbation: float
    first_hessian_perturbation: float
    fixed_mu_oracle: str
    fixed_variable_treatment: str
    gamma_phi: float
    gamma_theta: float
    grad_f_constant: str
    gradient_approximation: str
    hessian_approximation: str
    hessian_approximation_space: str
    hessian_constant: str
    honor_original_bounds: str
    hsllib: str
    inf_pr_output: str
    jac_c_constant: str
    jac_d_constant: str
    jacobian_approximation: str
    jacobian_regularization_exponent: float
    jacobian_regularization_value: float
    kappa_d: float
    kappa_sigma: float
    kappa_soc: float
    least_square_init_duals: str
    least_square_init_primal: str
    limited_memory_aug_solver: str
    limited_memory_init_val: float
    limited_memory_init_val_max: float
    limited_memory_init_val_min: float
    limited_memory_initialization: str
    limited_memory_max_history: int
    limited_memory_max_skipping: int
    limited_memory_special_for_resto: str
    limited_memory_update_type: str
    line_search_method: str
    linear_scaling_on_demand: str
    linear_solver: str
    linear_system_scaling: str
    ma27_ignore_singularity: str
    ma27_la_init_factor: float
    ma27_liw_init_factor: float
    ma27_meminc_factor: float
    ma27_pivtol: float
    ma27_pivtolmax: float
    ma27_print_level: int
    ma27_skip_inertia_check: str
    ma28_pivtol: float
    ma57_automatic_scaling: str
    ma57_block_size: int
    ma57_node_amalgamation: int
    ma57_pivot_order: int
    ma57_pivtol: float
    ma57_pivtolmax: float
    ma57_pre_alloc: float
    ma57_print_level: int
    ma57_small_pivot_flag: int
    ma77_buffer_lpage: int
    ma77_buffer_npage: int
    ma77_file_size: int
    ma77_maxstore: int
    ma77_nemin: int
    ma77_order: str
    ma77_print_level: int
    ma77_small: float
    ma77_static: float
    ma77_u: float
    ma77_umax: float
    ma86_nemin: int
    ma86_order: str
    ma86_print_level: int
    ma86_scaling: str
    ma86_small: float
    ma86_static: float
    ma86_u: float
    ma86_umax: float
    ma97_nemin: int
    ma97_order: str
    ma97_print_level: int
    ma97_scaling1: str
    ma97_scaling2: str
    ma97_scaling3: str
    ma97_scaling: str
    ma97_small: float
    ma97_solve_blas3: str
    ma97_switch1: str
    ma97_switch2: str
    ma97_switch3: str
    ma97_u: float
    ma97_umax: float
    max_cpu_time: float
    max_filter_resets: int
    max_hessian_perturbation: float
    max_iter: int
    max_refinement_steps: int
    max_resto_iter: int
    max_soc: int
    max_soft_resto_iters: int
    max_wall_time: float
    mehrotra_algorithm: str
    min_hessian_perturbation: float
    min_refinement_steps: int
    mu_allow_fast_monotone_decrease: str
    mu_init: float
    mu_linear_decrease_factor: float
    mu_max: float
    mu_max_fact: float
    mu_min: float
    mu_oracle: str
    mu_strategy: str
    mu_superlinear_decrease_power: float
    mu_target: float
    mumps_dep_tol: float
    mumps_mem_percent: int
    mumps_permuting_scaling: int
    mumps_pivot_order: int
    mumps_pivtol: float
    mumps_pivtolmax: float
    mumps_print_level: int
    mumps_scaling: int
    neg_curv_test_reg: str
    neg_curv_test_tol: float
    nlp_lower_bound_inf: float
    nlp_scaling_constr_target_gradient: float
    nlp_scaling_max_gradient: float
    nlp_scaling_method: str
    nlp_scaling_min_value: float
    nlp_scaling_obj_target_gradient: float
    nlp_upper_bound_inf: float
    nu_inc: float
    nu_init: float
    num_linear_variables: int
    obj_max_inc: float
    obj_scaling_factor: float
    option_file_name: str
    # only works when read from the ipopt.opt options file
    # output_file: str
    pardiso_iter_coarse_size: int
    pardiso_iter_dropping_factor: float
    pardiso_iter_dropping_schur: float
    pardiso_iter_inverse_norm_factor: float
    pardiso_iter_max_levels: int
    pardiso_iter_max_row_fill: int
    pardiso_iter_relative_tol: float
    pardiso_iterative: str
    pardiso_matching_strategy: str
    pardiso_max_droptol_corrections: int
    pardiso_max_iter: int
    pardiso_max_iterative_refinement_steps: int
    pardiso_msglvl: int
    pardiso_order: str
    pardiso_redo_symbolic_fact_only_if_inertia_wrong: str
    pardiso_repeated_perturbation_means_singular: str
    pardiso_skip_inertia_check: str
    pardisolib: str
    pardisomkl_matching_strategy: str
    pardisomkl_max_iterative_refinement_steps: int
    pardisomkl_msglvl: int
    pardisomkl_order: str
    pardisomkl_redo_symbolic_fact_only_if_inertia_wrong: str
    pardisomkl_repeated_perturbation_means_singular: str
    pardisomkl_skip_inertia_check: str
    perturb_always_cd: str
    perturb_dec_fact: float
    perturb_inc_fact: float
    perturb_inc_fact_first: float
    point_perturbation_radius: float
    print_advanced_options: str
    print_frequency_iter: int
    print_frequency_time: float
    print_info_string: str
    print_level: int
    print_options_documentation: str
    print_options_mode: str
    print_timing_statistics: str
    print_user_options: str
    quality_function_balancing_term: str
    quality_function_centrality: str
    quality_function_max_section_steps: int
    quality_function_norm_type: str
    quality_function_section_qf_tol: float
    quality_function_section_sigma_tol: float
    recalc_y: str
    recalc_y_feas_tol: float
    replace_bounds: str
    required_infeasibility_reduction: float
    residual_improvement_factor: float
    residual_ratio_max: float
    residual_ratio_singular: float
    resto_failure_feasibility_threshold: float
    resto_penalty_parameter: float
    resto_proximity_weight: float
    rho: float
    s_max: float
    s_phi: float
    s_theta: float
    sb: str
    sigma_max: float
    sigma_min: float
    skip_corr_if_neg_curv: str
    skip_corr_in_monotone_mode: str
    skip_finalize_solution_call: str
    slack_bound_frac: float
    slack_bound_push: float
    slack_move: float
    soc_method: int
    soft_resto_pderror_reduction_factor: float
    spral_cpu_block_size: int
    spral_gpu_perf_coeff: float
    spral_ignore_numa: str
    spral_max_load_inbalance: float
    spral_min_gpu_work: float
    spral_nemin: int
    spral_order: str
    spral_pivot_method: str
    spral_print_level: int
    spral_scaling: str
    spral_scaling_1: str
    spral_scaling_2: str
    spral_scaling_3: str
    spral_small: float
    spral_small_subtree_threshold: float
    spral_switch_1: str
    spral_switch_2: str
    spral_switch_3: str
    spral_u: float
    spral_umax: float
    spral_use_gpu: str
    start_with_resto: str
    tau_min: float
    theta_max_fact: float
    theta_min_fact: float
    timing_statistics: str
    tiny_step_tol: float
    tiny_step_y_tol: float
    tol: float
    warm_start_bound_frac: float
    warm_start_bound_push: float
    warm_start_entire_iterate: str
    warm_start_init_point: str
    warm_start_mult_bound_push: float
    warm_start_mult_init_max: float
    warm_start_same_structure: str
    warm_start_slack_bound_frac: float
    warm_start_slack_bound_push: float
    warm_start_target_mu: float
    watchdog_shortened_iter_trigger: int
    watchdog_trial_iter_max: int
    wsmp_inexact_droptol: float
    wsmp_inexact_fillin_limit: float
    wsmp_max_iter: int
    wsmp_no_pivoting: str
    wsmp_num_threads: int
    wsmp_ordering_option2: int
    wsmp_ordering_option: int
    wsmp_pivtol: float
    wsmp_pivtolmax: float
    wsmp_scaling: int
    wsmp_singularity_threshold: float
    wsmp_skip_inertia_check: str
    wsmp_write_matrix_iteration: int
