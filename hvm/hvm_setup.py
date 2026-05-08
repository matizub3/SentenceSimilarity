"""
Reusable function definitions for the STS17 HVM / normalizing-flow upgrade.

This implements a continuous-weight version of Ranganath, Tran, and Blei's
Hierarchical Variational Model idea:

    lambda0 ~ Normal(0, I)
    lambda  = planar_flow_K(lambda0; theta)
    z_i     ~ q(z_i | lambda_i)

Here z is the flattened vector of all Bayesian MLP weights and biases. Each
weight/bias z_i has two variational parameters inside lambda:

    lambda_i = (mean_i, rho_i)
    q(z_i | lambda_i) = Normal(mean_i, softplus(rho_i)^2)

Note: this implementation evaluates the auxiliary distribution on transformed
lambda, so the actual auxiliary term is r(lambda | z), not r(lambda0 | z).

The default training objective below uses a data-emphasized hierarchical ELBO
(DE-HVM ELBO) with stable per-datapoint minibatch scaling.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats

from baseline_setup import (
    parse_bool,
    parse_hidden_sizes,
    load_feature_npz,
    prepare_train_valid_test_arrays,
    make_nn_params_as_list_of_dicts,
    predict_y_given_x_sts,
    calc_logpdf_prior_sts,
    softplus_inverse,
    zeros_like_pytree,
    adam_update_pytree,
)


# ---------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------

def pytree_to_numpy(pytree):
    """Copy a JAX pytree to CPU-backed NumPy arrays so it is safe to pickle."""
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), pytree)


def pytree_to_jax(pytree):
    """Convert a NumPy-backed checkpoint pytree back to JAX arrays."""
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), pytree)


# ---------------------------------------------------------------------
# Flatten/unflatten helpers for BNN parameter pytrees
# ---------------------------------------------------------------------

def get_template_nn_params(n_dims_input, hidden_sizes):
    return make_nn_params_as_list_of_dicts(
        n_dims_input=n_dims_input,
        n_dims_output=1,
        n_dims_per_hidden_list=hidden_sizes,
    )


def flatten_nn_params(nn_params):
    leaves, treedef = jax.tree_util.tree_flatten(nn_params)
    flat = jnp.concatenate([jnp.ravel(x) for x in leaves], axis=0)
    shapes = [x.shape for x in leaves]
    sizes = [int(np.prod(s)) for s in shapes]
    return flat, treedef, shapes, sizes


def make_unflatten_fn(treedef, shapes, sizes):
    split_ids = np.cumsum(sizes)[:-1].tolist()

    def unflatten(flat_vec):
        pieces = jnp.split(flat_vec, split_ids)
        leaves = [jnp.reshape(piece, shape) for piece, shape in zip(pieces, shapes)]
        return jax.tree_util.tree_unflatten(treedef, leaves)

    return unflatten


def make_bnn_structure(n_dims_input, hidden_sizes):
    template = get_template_nn_params(n_dims_input, hidden_sizes)
    flat_template, treedef, shapes, sizes = flatten_nn_params(template)
    unflatten_fn = make_unflatten_fn(treedef, shapes, sizes)
    n_bnn_params = int(flat_template.shape[0])
    return template, unflatten_fn, n_bnn_params


# ---------------------------------------------------------------------
# Planar normalizing flow prior q(lambda; theta)
# ---------------------------------------------------------------------

def init_planar_flow_params(lambda_dim, flow_length=2, seed=101, init_scale=1e-3):
    rng = np.random.RandomState(seed)
    params = []

    for _ in range(flow_length):
        params.append({
            "u": jnp.asarray(rng.normal(0.0, init_scale, size=(lambda_dim,)).astype(np.float32)),
            "w": jnp.asarray(rng.normal(0.0, init_scale, size=(lambda_dim,)).astype(np.float32)),
            "b": jnp.asarray(rng.normal(0.0, init_scale, size=()).astype(np.float32)),
        })

    return params


def _planar_u_hat(u, w):
    """
    Rezende-Mohamed invertibility correction for planar flows.
    """
    wu = jnp.dot(w, u)
    m = -1.0 + jax.nn.softplus(wu)
    w_norm_sq = jnp.sum(w ** 2) + 1e-8
    return u + ((m - wu) * w / w_norm_sq)


def apply_planar_flow(lambda0_D, flow_params):
    """
    Forward planar flow.

    Returns:
        lambda_D: transformed vector
        log_abs_det_sum: sum_k log |det df_k/dlambda_{k-1}|
    """
    lam = lambda0_D
    log_abs_det_sum = 0.0

    for fp in flow_params:
        u = fp["u"]
        w = fp["w"]
        b = fp["b"]
        u_hat = _planar_u_hat(u, w)

        linear = jnp.dot(w, lam) + b
        h = jnp.tanh(linear)
        h_prime = 1.0 - h ** 2
        psi = h_prime * w

        det_term = 1.0 + jnp.dot(u_hat, psi)
        log_abs_det_sum = log_abs_det_sum + jnp.log(jnp.abs(det_term) + 1e-8)

        lam = lam + u_hat * h

    return lam, log_abs_det_sum


def sample_lambda_from_flow(flow_params, key, lambda_dim):
    lambda0_D = jax.random.normal(key, shape=(lambda_dim,))
    lambda_D, log_abs_det_sum = apply_planar_flow(lambda0_D, flow_params)

    log_q_lambda0 = jnp.sum(jstats.norm.logpdf(lambda0_D, loc=0.0, scale=1.0))
    log_q_lambda = log_q_lambda0 - log_abs_det_sum

    return lambda0_D, lambda_D, log_q_lambda


# ---------------------------------------------------------------------
# q(z | lambda): BNN weights/biases given random variational parameters
# ---------------------------------------------------------------------

def split_lambda_into_mean_rho(lambda_D):
    half = lambda_D.shape[0] // 2
    mean_P = lambda_D[:half]
    rho_P = lambda_D[half:]
    return mean_P, rho_P


def sample_z_from_lambda(lambda_D, key, min_stddev=1e-5):
    mean_P, rho_P = split_lambda_into_mean_rho(lambda_D)
    stddev_P = jax.nn.softplus(rho_P) + min_stddev
    eps_P = jax.random.normal(key, shape=mean_P.shape)
    z_P = mean_P + stddev_P * eps_P

    log_q_z_given_lambda = jnp.sum(
        jstats.norm.logpdf(z_P, loc=mean_P, scale=stddev_P)
    )

    return z_P, log_q_z_given_lambda


# ---------------------------------------------------------------------
# Auxiliary r(lambda | z; phi): product of Gaussians
# ---------------------------------------------------------------------

def init_aux_r_params(lambda_dim, seed=101, init_stddev=1.0):
    """
    r(lambda_j | z) = Normal(a_j * cond_j(z) + b_j, softplus(realstd_j)^2)

    cond(z) = [z, z], so lambda_dim must be 2 * n_bnn_params.
    """
    rng = np.random.RandomState(seed)
    init_realstd = softplus_inverse(init_stddev).astype(np.float32)

    return {
        "a": jnp.asarray(rng.normal(0.0, 1e-3, size=(lambda_dim,)).astype(np.float32)),
        "b": jnp.asarray(np.zeros((lambda_dim,), dtype=np.float32)),
        "realstd": jnp.asarray(np.full((lambda_dim,), init_realstd, dtype=np.float32)),
    }


def calc_log_r_lambda_given_z(lambda_D, z_P, aux_params):
    cond_D = jnp.concatenate([z_P, z_P], axis=0)
    mean_D = aux_params["a"] * cond_D + aux_params["b"]
    std_D = jax.nn.softplus(aux_params["realstd"]) + 1e-5

    return jnp.sum(
        jstats.norm.logpdf(lambda_D, loc=mean_D, scale=std_D)
    )


# ---------------------------------------------------------------------
# Learnable model hyperparameters eta
# ---------------------------------------------------------------------

def init_hyper_params(prior_stddev=3.0, likelihood_stddev=0.10):
    """
    Positive model hyperparameters are represented on the unconstrained
    real line and mapped through softplus during optimization.
    """
    return {
        "prior_realstd": jnp.asarray(softplus_inverse(prior_stddev).astype(np.float32)),
        "likelihood_realstd": jnp.asarray(softplus_inverse(likelihood_stddev).astype(np.float32)),
    }


def unpack_hyper_params(hyper_params):
    prior_stddev = jax.nn.softplus(hyper_params["prior_realstd"]) + 1e-5
    likelihood_stddev = jax.nn.softplus(hyper_params["likelihood_realstd"]) + 1e-5
    return prior_stddev, likelihood_stddev


def stop_fixed_hyper_grads(grad_hyper, learn_prior_stddev=True, learn_likelihood_stddev=False):
    """Zero gradients for hyperparameters the user wants fixed."""
    return {
        "prior_realstd": grad_hyper["prior_realstd"] if learn_prior_stddev else jnp.zeros_like(grad_hyper["prior_realstd"]),
        "likelihood_realstd": grad_hyper["likelihood_realstd"] if learn_likelihood_stddev else jnp.zeros_like(grad_hyper["likelihood_realstd"]),
    }


def hyper_params_to_float_dict(hyper_params):
    prior_stddev, likelihood_stddev = unpack_hyper_params(hyper_params)
    return {
        "prior_stddev": float(prior_stddev),
        "likelihood_stddev": float(likelihood_stddev),
    }


# ---------------------------------------------------------------------
# HVM hierarchical ELBO
# ---------------------------------------------------------------------

def calc_hvm_elbo_one_sample(
        flow_params,
        aux_params,
        hyper_params,
        key,
        x_ND,
        y_N,
        unflatten_fn,
        n_bnn_params,
        n_train_total,
        data_weight,
        use_sigmoid_output=False):
    key_lambda, key_z = jax.random.split(key)

    lambda_dim = 2 * n_bnn_params

    lambda0_D, lambda_D, log_q_lambda = sample_lambda_from_flow(
        flow_params=flow_params,
        key=key_lambda,
        lambda_dim=lambda_dim,
    )

    z_P, log_q_z_given_lambda = sample_z_from_lambda(
        lambda_D=lambda_D,
        key=key_z,
    )

    nn_params = unflatten_fn(z_P)

    pred_N = predict_y_given_x_sts(
        nn_params,
        x_ND,
        use_sigmoid_output=use_sigmoid_output,
    )

    prior_stddev, likelihood_stddev = unpack_hyper_params(hyper_params)

    log_lik = jnp.sum(
        jstats.norm.logpdf(y_N, loc=pred_N, scale=likelihood_stddev)
    )

    log_prior_z = calc_logpdf_prior_sts(
        nn_params,
        prior_stddev=prior_stddev,
    )

    log_r = calc_log_r_lambda_given_z(
        lambda_D=lambda_D,
        z_P=z_P,
        aux_params=aux_params,
    )

    B = x_ND.shape[0]

    global_terms = (
        log_prior_z
        + log_r
        - log_q_z_given_lambda
        - log_q_lambda
    )

    de_hvm_elbo_per_datapoint = (
        data_weight * (log_lik / B)
        + global_terms / n_train_total
    )

    return de_hvm_elbo_per_datapoint


def calc_hvm_elbo(
        flow_params,
        aux_params,
        hyper_params,
        key,
        x_ND,
        y_N,
        unflatten_fn,
        n_bnn_params,
        n_train_total,
        data_weight,
        n_mc_samples=5,
        use_sigmoid_output=False):
    keys = jax.random.split(key, n_mc_samples)
    total = 0.0

    for sample_id in range(n_mc_samples):
        total = total + calc_hvm_elbo_one_sample(
            flow_params=flow_params,
            aux_params=aux_params,
            hyper_params=hyper_params,
            key=keys[sample_id],
            x_ND=x_ND,
            y_N=y_N,
            unflatten_fn=unflatten_fn,
            n_bnn_params=n_bnn_params,
            n_train_total=n_train_total,
            data_weight=data_weight,
            use_sigmoid_output=use_sigmoid_output,
        )

    return total / n_mc_samples


value_and_grad_hvm_elbo = jax.value_and_grad(calc_hvm_elbo, argnums=(0, 1, 2))

fast_value_and_grad_hvm_elbo = jax.jit(
    value_and_grad_hvm_elbo,
    static_argnames=[
        "unflatten_fn",
        "n_bnn_params",
        "n_train_total",
        "n_mc_samples",
        "use_sigmoid_output",
    ],
)


# ---------------------------------------------------------------------
# Prediction/evaluation for HVM
# ---------------------------------------------------------------------

def sample_nn_params_from_hvm(flow_params, key, unflatten_fn, n_bnn_params):
    key_lambda, key_z = jax.random.split(key)
    lambda0_D, lambda_D, log_q_lambda = sample_lambda_from_flow(
        flow_params=flow_params,
        key=key_lambda,
        lambda_dim=2 * n_bnn_params,
    )
    z_P, _ = sample_z_from_lambda(lambda_D, key_z)
    return unflatten_fn(z_P)


def predict_with_hvm_samples(
        flow_params,
        x_ND,
        unflatten_fn,
        n_bnn_params,
        n_samples=50,
        use_sigmoid_output=False,
        seed=202):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_samples)
    preds = []

    for sample_id in range(n_samples):
        nn_params = sample_nn_params_from_hvm(
            flow_params=flow_params,
            key=keys[sample_id],
            unflatten_fn=unflatten_fn,
            n_bnn_params=n_bnn_params,
        )
        pred_N = predict_y_given_x_sts(
            nn_params,
            x_ND,
            use_sigmoid_output=use_sigmoid_output,
        )
        preds.append(pred_N)

    preds_SN = jnp.stack(preds, axis=0)
    pred_mean_N = jnp.mean(preds_SN, axis=0)
    pred_std_N = jnp.std(preds_SN, axis=0)
    return pred_mean_N, pred_std_N, preds_SN


def evaluate_hvm_rmse(
        flow_params,
        x_ND,
        y_N,
        unflatten_fn,
        n_bnn_params,
        n_samples=50,
        use_sigmoid_output=False,
        seed=202):
    pred_mean_N, pred_std_N, preds_SN = predict_with_hvm_samples(
        flow_params=flow_params,
        x_ND=x_ND,
        unflatten_fn=unflatten_fn,
        n_bnn_params=n_bnn_params,
        n_samples=n_samples,
        use_sigmoid_output=use_sigmoid_output,
        seed=seed,
    )
    rmse = jnp.sqrt(jnp.mean((pred_mean_N - y_N) ** 2))
    return rmse, pred_mean_N, pred_std_N, preds_SN


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

def train_hvm_bnn_upgrade(
        x_train_ND,
        y_train_N,
        x_valid_ND,
        y_valid_N,
        x_test_ND,
        y_test_N,
        hidden_sizes,
        n_iters=2000,
        batch_size=64,
        n_mc_samples=3,
        step_size=1e-4,
        prior_stddev=3.0,
        likelihood_stddev=0.10,
        learn_prior_stddev=True,
        learn_likelihood_stddev=False,
        hyper_step_size=None,
        flow_length=2,
        flow_init_scale=1e-3,
        aux_init_stddev=1.0,
        data_weight=None,
        use_sigmoid_output=False,
        n_valid_samples=20,
        seed=101,
        print_every=100):
    key = jax.random.PRNGKey(seed)
    rng = np.random.RandomState(seed)

    n_dims_input = x_train_ND.shape[1]
    template, unflatten_fn, n_bnn_params = make_bnn_structure(
        n_dims_input=n_dims_input,
        hidden_sizes=hidden_sizes,
    )
    lambda_dim = 2 * n_bnn_params

    N = x_train_ND.shape[0]

    if data_weight is None:
        data_weight = n_bnn_params / float(N)

    print("HVM BNN parameter count:", n_bnn_params)
    print("HVM lambda dimension:", lambda_dim)
    print("HVM flow length:", flow_length)
    print("HVM train size N:", N)
    print("DE-HVM data weight kappa:", data_weight)
    print("Initial prior_stddev:", prior_stddev, "| learn:", learn_prior_stddev)
    print("Initial likelihood_stddev:", likelihood_stddev, "| learn:", learn_likelihood_stddev)
    print("Hyperparameter step size:", hyper_step_size if hyper_step_size is not None else step_size)

    flow_params = init_planar_flow_params(
        lambda_dim=lambda_dim,
        flow_length=flow_length,
        seed=seed,
        init_scale=flow_init_scale,
    )

    aux_params = init_aux_r_params(
        lambda_dim=lambda_dim,
        seed=seed + 1,
        init_stddev=aux_init_stddev,
    )

    hyper_params = init_hyper_params(
        prior_stddev=prior_stddev,
        likelihood_stddev=likelihood_stddev,
    )

    if hyper_step_size is None:
        hyper_step_size = step_size

    m_flow = zeros_like_pytree(flow_params)
    v_flow = zeros_like_pytree(flow_params)
    m_aux = zeros_like_pytree(aux_params)
    v_aux = zeros_like_pytree(aux_params)
    m_hyper = zeros_like_pytree(hyper_params)
    v_hyper = zeros_like_pytree(hyper_params)

    history = {
        "iter": [],
        "train_de_hvm_elbo": [],
        "valid_rmse": [],
        "test_rmse": [],
        "prior_stddev": [],
        "likelihood_stddev": [],
        "is_best": [],
    }

    best_valid_rmse = np.inf
    best_checkpoint = None

    start_time = time.time()

    for iter_id in range(1, n_iters + 1):
        batch_ids = rng.choice(N, size=batch_size, replace=False)
        xb_BD = x_train_ND[batch_ids]
        yb_B = y_train_N[batch_ids]

        key, subkey = jax.random.split(key)

        elbo, (grad_flow, grad_aux, grad_hyper) = fast_value_and_grad_hvm_elbo(
            flow_params,
            aux_params,
            hyper_params,
            subkey,
            xb_BD,
            yb_B,
            unflatten_fn=unflatten_fn,
            n_bnn_params=n_bnn_params,
            n_train_total=N,
            data_weight=data_weight,
            n_mc_samples=n_mc_samples,
            use_sigmoid_output=use_sigmoid_output,
        )

        grad_hyper = stop_fixed_hyper_grads(
            grad_hyper,
            learn_prior_stddev=learn_prior_stddev,
            learn_likelihood_stddev=learn_likelihood_stddev,
        )

        flow_params, m_flow, v_flow = adam_update_pytree(
            flow_params, grad_flow, m_flow, v_flow, iter_id, step_size
        )
        aux_params, m_aux, v_aux = adam_update_pytree(
            aux_params, grad_aux, m_aux, v_aux, iter_id, step_size
        )
        hyper_params, m_hyper, v_hyper = adam_update_pytree(
            hyper_params, grad_hyper, m_hyper, v_hyper, iter_id, hyper_step_size
        )

        if iter_id == 1 or iter_id % print_every == 0 or iter_id == n_iters:
            valid_rmse, _, _ = evaluate_hvm_rmse(
                flow_params=flow_params,
                x_ND=x_valid_ND,
                y_N=y_valid_N,
                unflatten_fn=unflatten_fn,
                n_bnn_params=n_bnn_params,
                n_samples=n_valid_samples,
                use_sigmoid_output=use_sigmoid_output,
                seed=seed + 5000 + iter_id,
            )

            test_rmse, _, _ = evaluate_hvm_rmse(
                flow_params=flow_params,
                x_ND=x_test_ND,
                y_N=y_test_N,
                unflatten_fn=unflatten_fn,
                n_bnn_params=n_bnn_params,
                n_samples=n_valid_samples,
                use_sigmoid_output=use_sigmoid_output,
                seed=seed + 6000 + iter_id,
            )

            current_hyper = hyper_params_to_float_dict(hyper_params)

            history["iter"].append(iter_id)
            history["train_de_hvm_elbo"].append(float(elbo))
            history["valid_rmse"].append(float(valid_rmse))
            history["test_rmse"].append(float(test_rmse))
            history["prior_stddev"].append(current_hyper["prior_stddev"])
            history["likelihood_stddev"].append(current_hyper["likelihood_stddev"])

            is_best = float(valid_rmse) < best_valid_rmse
            history["is_best"].append(bool(is_best))

            if is_best:
                best_valid_rmse = float(valid_rmse)
                best_checkpoint = {
                    "iter": int(iter_id),
                    "valid_rmse": float(valid_rmse),
                    "test_rmse": float(test_rmse),
                    "train_de_hvm_elbo": float(elbo),
                    "flow_params": pytree_to_numpy(flow_params),
                    "aux_params": pytree_to_numpy(aux_params),
                    "hyper_params": pytree_to_numpy(hyper_params),
                    "prior_stddev": float(current_hyper["prior_stddev"]),
                    "likelihood_stddev": float(current_hyper["likelihood_stddev"]),
                }

            best_marker = " *BEST*" if is_best else ""
            print(
                "iter %6d/%d | time %7.1f sec | DE-HVM ELBO/datapoint %.6f | valid RMSE %.6f | test RMSE %.6f | prior_stddev %.6g | likelihood_stddev %.6g%s"
                % (
                    iter_id,
                    n_iters,
                    time.time() - start_time,
                    float(elbo),
                    float(valid_rmse),
                    float(test_rmse),
                    current_hyper["prior_stddev"],
                    current_hyper["likelihood_stddev"],
                    best_marker,
                )
            )

    info = {
        "unflatten_fn": unflatten_fn,
        "n_bnn_params": n_bnn_params,
        "lambda_dim": lambda_dim,
        "data_weight": float(data_weight),
        "n_train_total": int(N),
        "initial_prior_stddev": float(prior_stddev),
        "initial_likelihood_stddev": float(likelihood_stddev),
        "final_prior_stddev": hyper_params_to_float_dict(hyper_params)["prior_stddev"],
        "final_likelihood_stddev": hyper_params_to_float_dict(hyper_params)["likelihood_stddev"],
        "learn_prior_stddev": bool(learn_prior_stddev),
        "learn_likelihood_stddev": bool(learn_likelihood_stddev),
        "hyper_step_size": float(hyper_step_size),
        "best_iter": None if best_checkpoint is None else int(best_checkpoint["iter"]),
        "best_valid_rmse": np.nan if best_checkpoint is None else float(best_checkpoint["valid_rmse"]),
        "best_test_rmse": np.nan if best_checkpoint is None else float(best_checkpoint["test_rmse"]),
        "best_train_de_hvm_elbo": np.nan if best_checkpoint is None else float(best_checkpoint["train_de_hvm_elbo"]),
        "best_prior_stddev": np.nan if best_checkpoint is None else float(best_checkpoint["prior_stddev"]),
        "best_likelihood_stddev": np.nan if best_checkpoint is None else float(best_checkpoint["likelihood_stddev"]),
        "best_checkpoint": best_checkpoint,
    }

    return flow_params, aux_params, hyper_params, history, info
