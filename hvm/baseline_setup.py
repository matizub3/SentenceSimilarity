"""
Reusable function definitions for the STS17 mean-field Bayesian MLP baseline.

This file contains:
- MLP parameter creation and forward pass
- Data splitting and standardization helpers
- Mean-field variational posterior helpers
- Reparameterized ELBO calculation
- Custom Adam optimizer
- Training and evaluation functions
- Small argument-parsing helper functions used by run_baseline.py
"""

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats


# ---------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------

def parse_hidden_sizes(hidden_sizes_str):
    """
    Convert command-line architecture strings into a list of hidden layer sizes.

    Examples
    --------
    "100"        -> [100]
    "100-30"     -> [100, 30]
    "100-30-15"  -> [100, 30, 15]
    "[]"         -> []
    "linear"     -> []
    """
    hidden_sizes_str = str(hidden_sizes_str).strip()

    if hidden_sizes_str.lower() in ["[]", "linear", "none", ""]:
        return []

    return [int(x) for x in hidden_sizes_str.split("-")]


def parse_bool(s):
    """
    Convert command-line strings to bool.
    """
    s = str(s).lower()

    if s in ["true", "1", "yes", "y"]:
        return True

    if s in ["false", "0", "no", "n"]:
        return False

    raise argparse.ArgumentTypeError(f"Cannot interpret {s} as bool.")


# ---------------------------------------------------------------------
# Neural network helpers
# ---------------------------------------------------------------------

def make_nn_params_as_list_of_dicts(
        n_dims_input,
        n_dims_output,
        n_dims_per_hidden_list,
        weight_fill_func=None,
        bias_fill_func=None):
    """
    Create an MLP parameter pytree.

    Each layer is a dictionary:
        {
            "w": weight matrix of shape (input_dim, output_dim),
            "b": bias vector of shape (output_dim,)
        }

    The full network is a list of these dictionaries.
    """
    if weight_fill_func is None:
        weight_fill_func = lambda shape: np.zeros(shape, dtype=np.float32)

    if bias_fill_func is None:
        bias_fill_func = lambda shape: np.zeros(shape, dtype=np.float32)

    dims = [n_dims_input] + list(n_dims_per_hidden_list) + [n_dims_output]

    nn_params = []

    for layer_id in range(len(dims) - 1):
        n_in = dims[layer_id]
        n_out = dims[layer_id + 1]

        layer = {
            "w": jnp.asarray(weight_fill_func((n_in, n_out)), dtype=jnp.float32),
            "b": jnp.asarray(bias_fill_func((n_out,)), dtype=jnp.float32),
        }

        nn_params.append(layer)

    return nn_params


def predict_f_given_x(nn_params, x_ND):
    """
    Forward pass through an MLP.

    Uses tanh activation for hidden layers.
    Uses identity activation for output layer.
    """
    h = x_ND

    for layer_id, layer in enumerate(nn_params):
        w = layer["w"]
        b = layer["b"]

        h = jnp.dot(h, w) + b

        is_hidden_layer = layer_id < len(nn_params) - 1

        if is_hidden_layer:
            h = jnp.tanh(h)

    return h


def pretty_print_nn_param_list(nn_params, prefix=""):
    """
    Small debugging helper.
    """
    for layer_id, layer in enumerate(nn_params):
        print(f"{prefix}layer {layer_id}")
        print("  w shape:", layer["w"].shape)
        print("  b shape:", layer["b"].shape)
        print("  w mean/std:", float(jnp.mean(layer["w"])), float(jnp.std(layer["w"])))
        print("  b mean/std:", float(jnp.mean(layer["b"])), float(jnp.std(layer["b"])))


# ---------------------------------------------------------------------
# Generic pytree helpers
# ---------------------------------------------------------------------

def softplus_inverse(x):
    """
    Inverse of softplus for positive x.
    """
    x = np.asarray(x)
    return np.log(np.exp(x) - 1.0)


def zeros_like_pytree(pytree):
    return jax.tree.map(lambda x: jnp.zeros_like(x), pytree)


def softplus_of_pytree(pytree):
    return jax.tree.map(lambda x: jax.nn.softplus(x), pytree)


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------

def load_feature_npz(feature_file):
    """
    Load the precomputed STS17 feature .npz file.

    Expected arrays:
        x_ND
        y_raw_N
        pair_N, optional

    y_raw_N is divided by 5.0 to match the report's y in [0, 1].
    """
    data = np.load(feature_file, allow_pickle=True)

    raw_x_ND = data["x_ND"].astype(np.float32)
    raw_y_N = data["y_raw_N"].astype(np.float32)

    if "pair_N" in data:
        pair_N = data["pair_N"]
    else:
        pair_N = np.array(["unknown"] * raw_x_ND.shape[0])

    raw_y_N = raw_y_N / 5.0

    return raw_x_ND, raw_y_N, pair_N


def stratified_train_valid_test_split(pair_N, train_frac=0.80, valid_frac=0.10, seed=101):
    """
    80/10/10 split stratified by language pair.
    """
    rng = np.random.RandomState(seed)

    train_ids = []
    valid_ids = []
    test_ids = []

    for pair in np.unique(pair_N):
        ids = np.flatnonzero(pair_N == pair)
        rng.shuffle(ids)

        n = len(ids)
        n_train = int(np.floor(train_frac * n))
        n_valid = int(np.floor(valid_frac * n))

        train_ids.extend(ids[:n_train])
        valid_ids.extend(ids[n_train:n_train + n_valid])
        test_ids.extend(ids[n_train + n_valid:])

    train_ids = np.array(train_ids)
    valid_ids = np.array(valid_ids)
    test_ids = np.array(test_ids)

    rng.shuffle(train_ids)
    rng.shuffle(valid_ids)
    rng.shuffle(test_ids)

    return train_ids, valid_ids, test_ids


def standardize_using_train_stats(x_train, x_valid, x_test, eps=1e-8):
    """
    Standardize using training-set mean and std only.
    """
    x_mean_1D = np.mean(x_train, axis=0, keepdims=True)
    x_std_1D = np.std(x_train, axis=0, keepdims=True)

    x_std_1D = np.maximum(x_std_1D, eps)

    x_train_std = (x_train - x_mean_1D) / x_std_1D
    x_valid_std = (x_valid - x_mean_1D) / x_std_1D
    x_test_std = (x_test - x_mean_1D) / x_std_1D

    return x_train_std, x_valid_std, x_test_std, x_mean_1D, x_std_1D


def prepare_train_valid_test_arrays(raw_x_ND, raw_y_N, pair_N, seed=101):
    """
    Split, standardize, and convert arrays to JAX arrays.
    """
    train_ids, valid_ids, test_ids = stratified_train_valid_test_split(pair_N, seed=seed)

    x_train_ND_raw = raw_x_ND[train_ids]
    y_train_N_raw = raw_y_N[train_ids]

    x_valid_ND_raw = raw_x_ND[valid_ids]
    y_valid_N_raw = raw_y_N[valid_ids]

    x_test_ND_raw = raw_x_ND[test_ids]
    y_test_N_raw = raw_y_N[test_ids]

    x_train_ND, x_valid_ND, x_test_ND, x_mean_1D, x_std_1D = standardize_using_train_stats(
        x_train_ND_raw,
        x_valid_ND_raw,
        x_test_ND_raw,
    )

    arrays = {
        "x_train_ND": jnp.asarray(x_train_ND, dtype=jnp.float32),
        "y_train_N": jnp.asarray(y_train_N_raw, dtype=jnp.float32),
        "x_valid_ND": jnp.asarray(x_valid_ND, dtype=jnp.float32),
        "y_valid_N": jnp.asarray(y_valid_N_raw, dtype=jnp.float32),
        "x_test_ND": jnp.asarray(x_test_ND, dtype=jnp.float32),
        "y_test_N": jnp.asarray(y_test_N_raw, dtype=jnp.float32),
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "test_ids": test_ids,
        "x_mean_1D": x_mean_1D,
        "x_std_1D": x_std_1D,
    }

    return arrays


# ---------------------------------------------------------------------
# Mean-field variational posterior helpers
# ---------------------------------------------------------------------

def fill_q_params_with_draws_from_normal(
        n_dims_input,
        n_dims_output,
        n_dims_per_hidden_list,
        mean=0.0,
        stddev=1.0,
        random_state=101):
    """
    Create MLP-shaped pytree filled with Normal draws.
    """
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    def fill_func(shape):
        return random_state.normal(mean, stddev, size=shape).astype(np.float32)

    return make_nn_params_as_list_of_dicts(
        n_dims_input=n_dims_input,
        n_dims_output=n_dims_output,
        n_dims_per_hidden_list=n_dims_per_hidden_list,
        weight_fill_func=fill_func,
        bias_fill_func=fill_func,
    )


def init_mean_field_q(
        n_dims_input,
        hidden_sizes,
        init_mean_stddev=0.01,
        init_q_stddev=0.05,
        seed=101):
    """
    q(theta) = product Normal(mean, softplus(realstddev)^2)
    """
    rng = np.random.RandomState(seed)

    q_mean_params = fill_q_params_with_draws_from_normal(
        n_dims_input=n_dims_input,
        n_dims_output=1,
        n_dims_per_hidden_list=hidden_sizes,
        mean=0.0,
        stddev=init_mean_stddev,
        random_state=rng,
    )

    init_realstddev = softplus_inverse(init_q_stddev).astype(np.float32)

    def realstddev_fill_func(shape):
        return np.full(shape, init_realstddev, dtype=np.float32)

    q_realstddev_params = make_nn_params_as_list_of_dicts(
        n_dims_input=n_dims_input,
        n_dims_output=1,
        n_dims_per_hidden_list=hidden_sizes,
        weight_fill_func=realstddev_fill_func,
        bias_fill_func=realstddev_fill_func,
    )

    return q_mean_params, q_realstddev_params


def sample_nn_params_from_q_reparam(q_mean_params, q_realstddev_params, key):
    """
    Reparameterization trick:
        theta = mean + stddev * epsilon
        epsilon ~ Normal(0, 1)
        stddev = softplus(realstddev)
    """
    mean_leaves, treedef = jax.tree_util.tree_flatten(q_mean_params)
    realstddev_leaves, _ = jax.tree_util.tree_flatten(q_realstddev_params)

    keys = jax.random.split(key, len(mean_leaves))

    sampled_leaves = []

    for mean_leaf, realstddev_leaf, leaf_key in zip(mean_leaves, realstddev_leaves, keys):
        stddev_leaf = jax.nn.softplus(realstddev_leaf)
        eps_leaf = jax.random.normal(leaf_key, shape=mean_leaf.shape)
        sampled_leaf = mean_leaf + stddev_leaf * eps_leaf
        sampled_leaves.append(sampled_leaf)

    sampled_params = jax.tree_util.tree_unflatten(treedef, sampled_leaves)

    return sampled_params


# ---------------------------------------------------------------------
# Log densities and ELBO
# ---------------------------------------------------------------------

def calc_logpdf_prior_sts(nn_params, prior_stddev=3.0):
    """
    Mean-field Gaussian prior over all weights and biases.
    """
    total = 0.0

    for layer in nn_params:
        total += jnp.sum(jstats.norm.logpdf(layer["w"], loc=0.0, scale=prior_stddev))
        total += jnp.sum(jstats.norm.logpdf(layer["b"], loc=0.0, scale=prior_stddev))

    return total


def calc_logpdf_q_sts(nn_params, q_mean_params, q_realstddev_params):
    """
    Mean-field Gaussian variational density.
    """
    q_stddev_params = softplus_of_pytree(q_realstddev_params)

    total = 0.0

    for layer_id in range(len(nn_params)):
        w = nn_params[layer_id]["w"]
        b = nn_params[layer_id]["b"]

        w_mean = q_mean_params[layer_id]["w"]
        b_mean = q_mean_params[layer_id]["b"]

        w_stddev = q_stddev_params[layer_id]["w"]
        b_stddev = q_stddev_params[layer_id]["b"]

        total += jnp.sum(jstats.norm.logpdf(w, loc=w_mean, scale=w_stddev))
        total += jnp.sum(jstats.norm.logpdf(b, loc=b_mean, scale=b_stddev))

    return total


def predict_y_given_x_sts(nn_params, x_ND, use_sigmoid_output=False):
    """
    Predict scalar similarity score.
    """
    pred_N1 = predict_f_given_x(nn_params, x_ND)
    pred_N = jnp.squeeze(pred_N1, axis=-1)

    if use_sigmoid_output:
        pred_N = jax.nn.sigmoid(pred_N)

    return pred_N


def calc_logpdf_likelihood_sts(
        nn_params,
        x_ND,
        y_N,
        likelihood_stddev=0.10,
        use_sigmoid_output=False):
    """
    Gaussian likelihood:
        y_n | theta ~ Normal(f_theta(x_n), likelihood_stddev^2)
    """
    pred_N = predict_y_given_x_sts(
        nn_params,
        x_ND,
        use_sigmoid_output=use_sigmoid_output,
    )

    logpdf_N = jstats.norm.logpdf(
        y_N,
        loc=pred_N,
        scale=likelihood_stddev,
    )

    return jnp.sum(logpdf_N)


def calc_elbo_one_sample_reparam(
        q_mean_params,
        q_realstddev_params,
        key,
        x_ND,
        y_N,
        n_total_data=None,
        prior_stddev=3.0,
        likelihood_stddev=0.10,
        use_sigmoid_output=False):
    """
    One Monte Carlo estimate of the ELBO.
    """
    nn_params = sample_nn_params_from_q_reparam(
        q_mean_params,
        q_realstddev_params,
        key,
    )

    log_lik = calc_logpdf_likelihood_sts(
        nn_params,
        x_ND,
        y_N,
        likelihood_stddev=likelihood_stddev,
        use_sigmoid_output=use_sigmoid_output,
    )

    log_prior = calc_logpdf_prior_sts(
        nn_params,
        prior_stddev=prior_stddev,
    )

    log_q = calc_logpdf_q_sts(
        nn_params,
        q_mean_params,
        q_realstddev_params,
    )

    batch_size = x_ND.shape[0]
    n_total = batch_size if n_total_data is None else n_total_data
    elbo = (n_total / batch_size) * log_lik + log_prior - log_q

    return elbo / n_total


def calc_elbo_reparam(
        q_mean_params,
        q_realstddev_params,
        key,
        x_ND,
        y_N,
        n_mc_samples=5,
        n_total_data=None,
        prior_stddev=3.0,
        likelihood_stddev=0.10,
        use_sigmoid_output=False):
    """
    Average multiple Monte Carlo estimates of the ELBO.
    """
    keys = jax.random.split(key, n_mc_samples)

    total = 0.0

    for sample_id in range(n_mc_samples):
        total += calc_elbo_one_sample_reparam(
            q_mean_params,
            q_realstddev_params,
            keys[sample_id],
            x_ND,
            y_N,
            n_total_data=n_total_data,
            prior_stddev=prior_stddev,
            likelihood_stddev=likelihood_stddev,
            use_sigmoid_output=use_sigmoid_output,
        )

    return total / n_mc_samples


value_and_grad_elbo_reparam = jax.value_and_grad(
    calc_elbo_reparam,
    argnums=(0, 1),
)

fast_value_and_grad_elbo_reparam = jax.jit(
    value_and_grad_elbo_reparam,
    static_argnames=[
        "n_mc_samples",
        "prior_stddev",
        "likelihood_stddev",
        "use_sigmoid_output",
    ],
)


# ---------------------------------------------------------------------
# Optimizer, training, and evaluation
# ---------------------------------------------------------------------

def adam_update_pytree(
        params,
        grads,
        m,
        v,
        t,
        step_size,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8):
    """
    Adam update for pytrees.

    This performs gradient ascent because we maximize the ELBO.
    """
    m = jax.tree.map(
        lambda m_old, g: beta1 * m_old + (1.0 - beta1) * g,
        m,
        grads,
    )

    v = jax.tree.map(
        lambda v_old, g: beta2 * v_old + (1.0 - beta2) * (g ** 2),
        v,
        grads,
    )

    m_hat = jax.tree.map(
        lambda m_val: m_val / (1.0 - beta1 ** t),
        m,
    )

    v_hat = jax.tree.map(
        lambda v_val: v_val / (1.0 - beta2 ** t),
        v,
    )

    params = jax.tree.map(
        lambda p, mh, vh: p + step_size * mh / (jnp.sqrt(vh) + eps),
        params,
        m_hat,
        v_hat,
    )

    return params, m, v


def evaluate_rmse_with_posterior_mean(
        q_mean_params,
        x_ND,
        y_N,
        use_sigmoid_output=False):
    """
    Deterministic evaluation using posterior mean weights.
    """
    pred_N = predict_y_given_x_sts(
        q_mean_params,
        x_ND,
        use_sigmoid_output=use_sigmoid_output,
    )

    rmse = jnp.sqrt(jnp.mean((pred_N - y_N) ** 2))

    return rmse


def predict_with_posterior_samples(
        q_mean_params,
        q_realstddev_params,
        x_ND,
        n_samples=100,
        use_sigmoid_output=False,
        seed=202):
    """
    Bayesian predictive distribution from posterior samples.
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_samples)

    preds = []

    for sample_id in range(n_samples):
        nn_params = sample_nn_params_from_q_reparam(
            q_mean_params,
            q_realstddev_params,
            keys[sample_id],
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


def evaluate_rmse_with_posterior_predictive_mean(
        q_mean_params,
        q_realstddev_params,
        x_ND,
        y_N,
        n_samples=100,
        use_sigmoid_output=False,
        seed=202):
    """
    RMSE using posterior predictive mean.

    Returns:
        rmse, pred_mean_N, pred_std_N, preds_SN
        where preds_SN has shape [n_samples, n_examples].
    """
    pred_mean_N, pred_std_N, preds_SN = predict_with_posterior_samples(
        q_mean_params,
        q_realstddev_params,
        x_ND,
        n_samples=n_samples,
        use_sigmoid_output=use_sigmoid_output,
        seed=seed,
    )

    rmse = jnp.sqrt(jnp.mean((pred_mean_N - y_N) ** 2))

    return rmse, pred_mean_N, pred_std_N, preds_SN


def train_mean_field_bnn_baseline(
        x_train_ND,
        y_train_N,
        x_valid_ND,
        y_valid_N,
        hidden_sizes,
        n_iters=2000,
        batch_size=64,
        n_mc_samples=5,
        step_size=1e-3,
        prior_stddev=3.0,
        likelihood_stddev=0.10,
        init_q_stddev=0.05,
        use_sigmoid_output=False,
        seed=101,
        print_every=100):
    """
    Train the mean-field Bayesian MLP baseline using reparameterized BBVI.
    """
    key = jax.random.PRNGKey(seed)
    rng = np.random.RandomState(seed)

    n_dims_input = x_train_ND.shape[1]

    q_mean_params, q_realstddev_params = init_mean_field_q(
        n_dims_input=n_dims_input,
        hidden_sizes=hidden_sizes,
        init_q_stddev=init_q_stddev,
        seed=seed,
    )

    adam_m_mean = zeros_like_pytree(q_mean_params)
    adam_v_mean = zeros_like_pytree(q_mean_params)

    adam_m_realstddev = zeros_like_pytree(q_realstddev_params)
    adam_v_realstddev = zeros_like_pytree(q_realstddev_params)

    history = {
        "iter": [],
        "train_elbo": [],
        "valid_rmse": [],
    }

    N = x_train_ND.shape[0]

    start_time = time.time()

    for iter_id in range(1, n_iters + 1):
        batch_ids = rng.choice(N, size=batch_size, replace=False)

        xb_BD = x_train_ND[batch_ids]
        yb_B = y_train_N[batch_ids]

        key, subkey = jax.random.split(key)

        elbo, (grad_mean, grad_realstddev) = fast_value_and_grad_elbo_reparam(
            q_mean_params,
            q_realstddev_params,
            subkey,
            xb_BD,
            yb_B,
            n_mc_samples=n_mc_samples,
            n_total_data=N,
            prior_stddev=prior_stddev,
            likelihood_stddev=likelihood_stddev,
            use_sigmoid_output=use_sigmoid_output,
        )

        q_mean_params, adam_m_mean, adam_v_mean = adam_update_pytree(
            q_mean_params,
            grad_mean,
            adam_m_mean,
            adam_v_mean,
            iter_id,
            step_size,
        )

        q_realstddev_params, adam_m_realstddev, adam_v_realstddev = adam_update_pytree(
            q_realstddev_params,
            grad_realstddev,
            adam_m_realstddev,
            adam_v_realstddev,
            iter_id,
            step_size,
        )

        if iter_id == 1 or iter_id % print_every == 0 or iter_id == n_iters:
            valid_rmse = evaluate_rmse_with_posterior_mean(
                q_mean_params,
                x_valid_ND,
                y_valid_N,
                use_sigmoid_output=use_sigmoid_output,
            )

            history["iter"].append(iter_id)
            history["train_elbo"].append(float(elbo))
            history["valid_rmse"].append(float(valid_rmse))

            print(
                "iter %6d/%d | time %7.1f sec | normalized ELBO %.6f | valid RMSE %.6f"
                % (
                    iter_id,
                    n_iters,
                    time.time() - start_time,
                    float(elbo),
                    float(valid_rmse),
                )
            )

    return q_mean_params, q_realstddev_params, history
