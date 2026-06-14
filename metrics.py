"""
metrics.py
==========
Reservoir computing performance metrics used in:

    Hadaeghi, Fakhar, Khajehnejad & Hilgetag (2026)
    "A Computational Perspective on the No-Strong-Loops Principle
     in Brain Networks".

Provides four task-agnostic functions that accept any square weight
matrix W (numpy ndarray, shape N×N) and return performance scores:

    compute_memory_capacity(W, ...)   
    compute_kernel_rank(W, ...)      
    compute_narma(W, n, ...)         
    evaluate_mante(W, ...)       

Dependencies
------------
    numpy, scipy, sklearn  — standard scientific stack
    echoes                 — https://github.com/fabridamicelli/echoes
    neurogym               — required only for evaluate_mante()
                             https://github.com/neurogym/neurogym

import warnings
warnings.filterwarnings("ignore")


import numpy as np
from scipy import linalg, stats
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score

# ── Echoes (required for MC, KR, NARMA, Mante) ───────────────────────────────
try:
    from echoes import ESNRegressor
    from echoes.reservoir._leaky_numba import harvest_states
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "The 'echoes' package is required.  Install it with:\n"
        "    pip install echoes\n"
        "or visit https://github.com/fabridamicelli/echoes"
    ) from _e


def _build_esn(
    W: np.ndarray,
    W_in: np.ndarray,
    spectral_radius: float,
    n_transient: int,
    ridge_alpha: float,
    noise: float,
    random_state: int,
    fit_only_states: bool = True,
    store_states_train: bool = False,
    store_states_pred: bool = False,
    bias: float = 0.0,
    input_scaling: float = None,
) -> ESNRegressor:
    """
    Construct an ESNRegressor with the given weight matrix and parameters.
    W is passed directly (no further scaling inside Echoes).
    """
    kwargs = dict(
        n_reservoir       = W.shape[0],
        spectral_radius   = float(spectral_radius),
        W                 = W.astype(float),
        W_in              = W_in,
        bias              = float(bias),
        noise             = float(noise),
        n_transient       = int(n_transient),
        regression_method = 'ridge',
        ridge_alpha       = float(ridge_alpha),
        random_state      = int(random_state),
        fit_only_states   = fit_only_states,
        store_states_train = store_states_train,
        store_states_pred  = store_states_pred,
        leak_rate         = 1.0,
    )
    if input_scaling is not None:
        kwargs['input_scaling'] = float(input_scaling)
    return ESNRegressor(**kwargs)



#  Memory Capacity

def compute_memory_capacity(
    W: np.ndarray,
    spectral_radius: float = 0.95,
    sequence_length: int   = 3000,
    n_transient: int       = 100,
    w_in_scale: float      = 0.05,
    ridge_alpha: float     = 1e-10,
    noise: float           = 1e-5,
    train_fraction: float  = 0.7,
    random_state: int      = 283,
) -> float:
    """
    Compute Memory Capacity (MC) for a reservoir defined by weight matrix W.

    MC measures the ability of the reservoir to linearly reconstruct past
    inputs across increasing time lags τ = 1, 2, …, floor(1.4 * N).
    It is computed as the sum of squared Pearson correlations (R²) between
    the actual and predicted delayed inputs over all lags (Jaeger 2001;
    Farkaš et al. 2016).


    Parameters
    ----------
    W : ndarray, shape (N, N)
        Reservoir weight matrix (binary or weighted, directed).
    sequence_length : int, default 3000
        Total length of the i.i.d. Gaussian input signal u(t) ~ N(0,1).
    n_transient : int, default 100
        Number of initial time steps discarded from training to remove
        dependency on the zero initial state.
    w_in_scale : float, default 0.05
        Input weights W_in are drawn from Uniform(-w_in_scale, w_in_scale).
    ridge_alpha : float, default 1e-10
        Tikhonov regularisation coefficient for the linear readout.
    noise : float, default 1e-5
        Additive i.i.d. Gaussian noise injected into reservoir states
        during training to improve numerical stability.
    train_fraction : float, default 0.7
        Fraction of the sequence used for training; remainder for testing.
    random_state : int, default 283
        Seed for numpy RNG (input signal and W_in generation).

    Returns
    -------
    mc : float
        Memory capacity score (non-negative).  Theoretical maximum = N.

    References
    ----------
    Jaeger H (2001). Short term memory in echo state networks.
    Farkaš I, Bosák R, Gergel' P (2016). Neural Networks, 83, 109–120.
    """
    rng = np.random.default_rng(random_state)
    N   = W.shape[0]


    W_in = rng.uniform(-w_in_scale, w_in_scale, size=(N, 1)).astype(float)

    u = rng.standard_normal(sequence_length)

    #   num_delays = floor(1.4 * N) following Farkaš et al. (2016)
    num_delays = int(np.floor(1.4 * N))
    Y = np.zeros((sequence_length, num_delays))
    for lag in range(1, num_delays + 1):
        Y[lag:, lag - 1] = u[:-lag]

    X = u.reshape(-1, 1)

    split = int(np.floor(train_fraction * sequence_length))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    esn = _build_esn(
        W            = W.float(),
        W_in         = W_in,
        spectral_radius = spectral_radius,
        n_transient  = n_transient,
        ridge_alpha  = ridge_alpha,
        noise        = noise,
        random_state = int(random_state),
        fit_only_states = True,
    )
    esn.fit(X_train, Y_train)
    Y_pred = esn.predict(X_test)

    # Discard initial test steps that fall within the transient window
    offset  = esn.n_transient + 1
    Y_test_aligned  = Y_test[offset:]
    Y_pred_aligned  = Y_pred[offset:]

    mc = float(np.maximum(
        np.array([r2_score(Y_test_aligned[:, i], Y_pred_aligned[:, i])
                  for i in range(num_delays)]),
        0
    ).sum())

    return round(mc, 4)


# 2. Kernel Rank


def compute_kernel_rank(
    W: np.ndarray,
    spectral_radius: float = 0.95,
    num_inputs: int        = None,
    signal_length: int     = 500,
    n_transient: int       = 50,
    w_in_scale: float      = 0.05,
    svd_threshold: float   = 0.01,
    noise: float           = 1e-5,
    random_state: int      = 283,
) -> int:
    """
    Compute Kernel Rank (KR) for a reservoir defined by weight matrix W.

    KR quantifies the separation property of the reservoir: the number of
    linearly independent response dimensions when the network is driven by
    `num_inputs` distinct i.i.d. Gaussian input streams (Legenstein & Maass
    2007; Dale et al. 2021).

    Concretely, KR = number of singular values of the state matrix M
    (shape N × num_inputs) that exceed `svd_threshold * σ₁`, where σ₁
    is the largest singular value.

    The network is normalised internally so its spectral radius equals
    `spectral_radius`; the caller's matrix W is not modified.

    Parameters
    ----------
    W : ndarray, shape (N, N)
        Reservoir weight matrix (binary or weighted, directed).
    spectral_radius : float, default 0.99
        Target spectral radius after normalisation.
    num_inputs : int or None, default None
        Number of distinct input signals used to probe the reservoir.
        If None, set to N (square state matrix), following the paper.
    signal_length : int, default 200
        Length (timesteps) of each individual input signal.  Must exceed
        n_transient.  200 gives stable estimates for N up to ~1024.
    n_transient : int, default 50
        Initial steps discarded per signal to remove transient dynamics.
    w_in_scale : float, default 0.05
        Input weights W_in drawn from Uniform(−w_in_scale, w_in_scale),
        scaled by 0.5 * spectral_radius(W_raw) as in the paper.
    svd_threshold : float, default 0.01
        Relative SVD threshold: singular values below
        `svd_threshold * σ₁` are treated as zero.
    noise : float, default 1e-5
        Additive noise injected per time step (aids numerical stability).
    random_state : int, default 283
        Seed for numpy RNG.

    Returns
    -------
    kr : int
        Kernel rank.  Theoretical maximum = min(N, num_inputs).

    References
    ----------
    Legenstein R, Maass W (2007). Neural Networks, 20(3), 323–334.
    Dale M et al. (2021). Natural Computing, 20, 205–216.
    """
    rng = np.random.default_rng(random_state)
    N   = W.shape[0]

    if num_inputs is None:
        num_inputs = N   # square state matrix as in the paper
    W_in    = 
               rng.uniform(-w_in_scale, w_in_scale, size=(N, 1)).astype(float)

    
    #   harvest_states() requires the internally scaled W from ESNRegressor.
    #   We fit on a dummy signal (length > n_transient) to trigger it.
    esn = _build_esn(
        W               = W.float(),
        W_in            = W_in,
        spectral_radius = spectral_radius,
        n_transient     = n_transient,
        ridge_alpha     = 1e-10,
        noise           = noise,
        random_state    = int(random_state),
        fit_only_states = True,
        store_states_train = True,
    )
    dummy = np.zeros((n_transient + 2, 1))
    esn.fit(dummy, dummy)

    
    res           = esn.reservoir_
    initial_state = np.zeros(N)
    y_dummy       = np.zeros((signal_length, 1))

    state_matrix = np.zeros((N, num_inputs))
    for i in range(num_inputs):
        signal = rng.standard_normal(signal_length).reshape(-1, 1)
        states = harvest_states(
            signal,
            y_dummy,
            initial_state,
            res.W_in,
            res.W,
            res.W_fb,
            res.bias,
            res.activation,
            res.noise,
            res.leak_rate,
        )
        # Take the final state after discarding the transient
        state_matrix[:, i] = states[n_transient:, :][-1, :]

   
    singular_values = linalg.svd(state_matrix, compute_uv=False)
    threshold       = singular_values[0] * svd_threshold
    kr              = int(np.count_nonzero(singular_values > threshold))

    return kr



# 3. NARMA Task (n = 5 or 10)

def _generate_narma_sequence(
    sequence_length: int,
    n: int,
    seed: int = None,
) -> tuple:
    """
    Generate a NARMA-n target sequence (Jaeger 2001; Rodan & Tiňo 2010).

    y[t+1] = 0.3·y[t] + 0.05·y[t]·Σy[t-n+1..t] + 1.5·u[t]·u[t-n+1] + 0.1

    A warm-up period (200 steps) is prepended and discarded so the
    returned sequence is independent of zero initial conditions.
    Sequences that diverge to non-finite values are regenerated (up to
    50 retries) with different seeds.

    Returns
    -------
    u_out : ndarray (sequence_length, 1)   driving input u(t)
    y_out : ndarray (sequence_length, 1)   NARMA-n target y(t)
    y_var : float                          full-sequence variance for NRMSE
    """
    warmup   = 200
    total    = sequence_length + warmup
    attempts = 0

    while attempts < 50:
        rng = np.random.default_rng(
            seed if seed is not None else (attempts * 9973 + 1)
        )
        u = 0.5 * rng.random(total)
        y = np.zeros(total)

        for t in range(n, total - 1):
            past_sum = float(np.sum(y[t - n + 1: t + 1]))
            y[t + 1] = (
                0.3  * y[t]
                + 0.05 * y[t] * past_sum
                + 1.5  * u[t] * u[t - n + 1]
                + 0.1
            )

        u_out = u[warmup:].reshape(-1, 1)
        y_out = y[warmup:].reshape(-1, 1)

        if np.isfinite(y_out).all():
            return u_out, y_out, float(np.var(y_out))

        attempts += 1
        seed = (seed + 1) if seed is not None else attempts

    raise RuntimeError(
        f"NARMA-{n}: failed to generate a finite sequence after 50 retries."
    )


def compute_narma(
    W: np.ndarray,
    n: int             = 10,
    spectral_radius: float = 0.95,
    sequence_length: int   = 1000,
    train_fraction: float  = 0.7,
    washout_test: int      = 100,
    n_transient: int       = 300,
    w_in_scale: float      = 1.0,
    input_scaling: float   = 0.3,
    bias: float            = 0.1,
    ridge_alpha: float     = 1e-6,
    noise: float           = 1e-4,
    random_state: int      = 283,
    data_seed: int         = 283,
) -> float:
    """
    Evaluate reservoir performance on the NARMA-n task (n = 5 or 10).

    NARMA (Nonlinear Auto-Regressive Moving Average) tasks require the
    reservoir to reproduce a target sequence whose current value depends
    nonlinearly on past inputs and outputs over a horizon of n timesteps
    (Jaeger 2001; Rodan & Tiňo 2010).  They serve as canonical benchmarks
    for nonlinear memory in recurrent systems.

    Performance is measured as the Pearson correlation coefficient between
    the predicted and actual NARMA-n output on the held-out test segment,
    after discarding `washout_test` initial test steps.  Values close to
    1.0 indicate accurate reproduction; values near 0 indicate failure.


    Parameters
    ----------
    W : ndarray, shape (N, N)
        Reservoir weight matrix (binary or weighted, directed).
    n : int, default 10
        NARMA memory length.  Use 5 for NARMA-5 or 10 for NARMA-10.
    spectral_radius : float, default 0.95
        Target spectral radius after normalisation.
    sequence_length : int, default 1000
        Total length of the NARMA sequence (excluding warm-up).
    train_fraction : float, default 0.7
        Fraction of the sequence used for training.
    washout_test : int, default 100
        Initial test steps discarded before computing performance.
    n_transient : int, default 300
        Initial training steps discarded to remove initial reservoir dynamics.
    w_in_scale : float, default 1.0
        Input weights W_in drawn from Uniform(−w_in_scale, w_in_scale).
    input_scaling : float, default 0.3
        Scalar applied to W_in (absorbed into input weights).
    bias : float, default 0.1
        Constant bias added to the reservoir input at each step.
    ridge_alpha : float, default 1e-6
        Tikhonov regularisation coefficient for the linear readout.
    noise : float, default 1e-4
        Additive noise per time step (with small Gaussian perturbation
        σ=1e-4 added to training states to improve generalisation).
    random_state : int, default 283
        Seed for W_in generation.
    data_seed : int, default 283
        Seed for NARMA sequence generation.

    Returns
    -------
    pearson_r : float
        Pearson correlation coefficient on the test segment (after washout).
        Returns NaN if the reservoir produces non-finite states.

    References
    ----------
    Jaeger H (2001). The "echo state" approach to analysing and training
        recurrent neural networks. GMD-Report 148.
    Rodan A, Tiňo P (2010). IEEE Trans. Neural Networks, 22(1), 131–144.
    """
    rng = np.random.default_rng(random_state)
    N   = W.shape[0]

    u, y, _ = _generate_narma_sequence(sequence_length, n, seed=data_seed)

    u_in  = u[:-1].astype(np.float64)   # (T-1, 1)
    y_tgt = y[1:].astype(np.float64)    # (T-1, 1)

    split       = int(np.floor(train_fraction * len(u_in)))
    u_train     = u_in[:split]
    y_train     = y_tgt[:split]
    u_test      = u_in[split:]
    y_test      = y_tgt[split:]


    W_in = (rng.uniform(-w_in_scale, w_in_scale, size=(N, 1))
            * input_scaling).astype(float)

    esn = _build_esn(
        W               = W.float(),
        W_in            = W_in,
        spectral_radius = spectral_radius,
        n_transient     = n_transient,
        ridge_alpha     = ridge_alpha,
        noise           = noise,
        random_state    = int(random_state),
        fit_only_states = False,
        bias            = bias,
    )
    esn.fit(u_train, y_train)
    y_pred = esn.predict(u_test)

    if not np.isfinite(y_pred).all():
        return float('nan')

    y_test_aligned = y_test[washout_test:].ravel()
    y_pred_aligned = y_pred[washout_test:].ravel()
    n_aligned      = min(len(y_test_aligned), len(y_pred_aligned))
    r, _           = stats.pearsonr(
        y_test_aligned[:n_aligned],
        y_pred_aligned[:n_aligned],
    )
    return float(r)



# 4. Context-Dependent Decision-Making Task (Mante et al. 2013)

# Task constants (fixed, matching the paper)
_MANTE_TASK_NAME   = 'ContextDecisionMaking-v0'
_MANTE_TASK_DT     = 100        # ms per timestep
_MANTE_N_INPUTS    = 7          # observation channels
_MANTE_N_TRANSIENT = 15         # ~1 trial worth of steps
_MANTE_SEQ_LEN     = 100        # timesteps per trial in Dataset call


def _generate_mante_data(
    n_trials: int,
    seq_len: int,
    seed: int = None,
) -> tuple:
    """
    Generate inputs and labels for the ContextDecisionMaking-v0 task.

    Returns
    -------
    X : ndarray (n_trials * seq_len, 7)
    y : ndarray (n_trials * seq_len,)   labels {0=fixate, 1=left, 2=right}
    """
    try:
        import neurogym as ngym
    except ImportError as e:
        raise ImportError(
            "neurogym is required for evaluate_mante().  Install with:\n"
            "    pip install neurogym\n"
            "or visit https://github.com/neurogym/neurogym"
        ) from e

    if seed is not None:
        np.random.seed(seed)

    dataset = ngym.Dataset(
        _MANTE_TASK_NAME,
        env_kwargs={'dt': _MANTE_TASK_DT},
        batch_size=1,
        seq_len=seq_len,
    )

    X_list, y_list = [], []
    for _ in range(n_trials):
        x, y = dataset()
        X_list.append(x[:, 0, :])
        y_list.append(y[:, 0])

    X = np.concatenate(X_list, axis=0).astype(np.float64)
    y = np.concatenate(y_list, axis=0).astype(int)
    return X, y


def _collect_reservoir_states(
    W: np.ndarray,
    W_in: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    spectral_radius: float,
    ridge_alpha: float,
    noise: float,
    random_state: int,
    bias: float,
) -> tuple:
    """
    Run Echoes on train and test inputs and return stored reservoir states.
    The linear readout fitted by Echoes is discarded; only states are used.
    """
    esn = _build_esn(
        W               = W,
        W_in            = W_in,
        spectral_radius = spectral_radius,
        n_transient     = _MANTE_N_TRANSIENT,
        ridge_alpha     = ridge_alpha,
        noise           = noise,
        random_state    = random_state,
        fit_only_states = False,   # inputs concatenated with states (Jaeger)
        store_states_train = True,
        store_states_pred  = True,
        bias            = bias,
        input_scaling   = None,    # absorbed into W_in
    )
    dummy_train = np.zeros((X_train.shape[0], 1))
    esn.fit(X_train, dummy_train)
    _  = esn.predict(X_test)
    return esn.states_train_, esn.states_pred_


def evaluate_mante(
    W: np.ndarray,
    spectral_radius: float = 1.2,
    n_trials: int          = 1000,
    train_fraction: float  = 0.7,
    w_in_scale: float      = 1.0,
    input_scaling: float   = 0.1,
    bias: float            = 0.0,
    ridge_alpha: float     = 1e-6,
    clf_ridge_alpha: float = 5.0,
    noise: float           = 1e-4,
    random_state: int      = 42,
    data_seed: int         = 0,
) -> tuple:
    """
    Evaluate reservoir performance on the context-dependent decision-making
    task (Mante et al. 2013), implemented via NeuroGym.

    Task structure (dt = 100 ms)
    ----------------------------
    Input: 7 channels — fixation signal, context/rule cue, and two pairs
    of stimulus channels representing competing sensory evidence.
    Each trial proceeds through: fixation (300 ms) → stimulus (750 ms) →
    variable delay → decision (100 ms).
    Output: 3 classes — {0: fixate, 1: choose left, 2: choose right}.

    Pipeline
    --------
    1. Generate `n_trials` trials via NeuroGym Dataset.
    2. Concatenate into one temporal sequence; split 70/30 train/test.
    3. Collect reservoir states via Echoes (fit_only_states=False so
       inputs are concatenated with states at each time step, following
       the Jaeger convention and Suárez et al. 2024).
    4. Train a RidgeClassifier on decision-period timesteps only
       (labels 1 and 2; fixation timesteps, label 0, are excluded).
    5. Evaluate on the test set: balanced accuracy and macro F1.

    The spectral radius default (1.2) matches Suárez et al. (2024), who
    show that context tasks benefit from dynamics near or slightly above
    the edge of stability.  The caller's matrix W is not modified.

    Parameters
    ----------
    W : ndarray, shape (N, N)
        Reservoir weight matrix (binary or weighted, directed).
    spectral_radius : float, default 1.2
        Target spectral radius after normalisation.
    n_trials : int, default 1000
        Number of task trials (matching Suárez et al. 2024).
    train_fraction : float, default 0.7
        Fraction of trials used for training (temporal split).
    w_in_scale : float, default 1.0
        W_in drawn from Uniform(−w_in_scale, w_in_scale), then scaled
        by `input_scaling`.
    input_scaling : float, default 0.1
        Scalar applied to W_in after sampling.
    bias : float, default 0.0
        Constant bias added to reservoir input.
    ridge_alpha : float, default 1e-6
        Regularisation for the Echoes linear readout (discarded; only
        states are used, but Echoes requires a target to fit).
    clf_ridge_alpha : float, default 5.0
        Regularisation for the RidgeClassifier readout.
    noise : float, default 1e-4
        Additive noise per timestep.
    random_state : int, default 42
        Seed for W_in generation and Echoes internal RNG.
    data_seed : int, default 0
        Seed for NeuroGym trial generation.

    Returns
    -------
    balanced_accuracy : float
        Balanced accuracy on the test set, decision period only.
        Chance level = 0.5 (binary classification: left vs right).
    macro_f1 : float
        Macro-averaged F1 score on the test set, decision period only.
        Returns (NaN, NaN) if reservoir produces non-finite states.

    References
    ----------
    Mante V, Sussillo D, Shenoy KV, Newsome WT (2013). Nature, 503, 78–84.
    Suárez LE et al. (2024). Nature Communications, 15, 656.
    """
    rng = np.random.default_rng(random_state)
    N   = W.shape[0]

    X, y = _generate_mante_data(n_trials, _MANTE_SEQ_LEN, seed=data_seed)

    split = int(np.floor(train_fraction * len(X)))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    W_in = (rng.uniform(-w_in_scale, w_in_scale, size=(N, _MANTE_N_INPUTS))
            * input_scaling).astype(float)

    states_train, states_test = _collect_reservoir_states(
        W               = W,
        W_in            = W_in,
        X_train         = X_train,
        X_test          = X_test,
        spectral_radius = spectral_radius,
        ridge_alpha     = ridge_alpha,
        noise           = noise,
        random_state    = int(random_state),
        bias            = bias,
    )

    if (not np.isfinite(states_train).all() or
            not np.isfinite(states_test).all()):
        return float('nan'), float('nan')

    features_train = np.concatenate([states_train, X_train], axis=1)
    features_test  = np.concatenate([states_test,  X_test],  axis=1)

    y_train_aligned = y_train[_MANTE_N_TRANSIENT:]

    #   Fixation timesteps (label 0) are excluded from both train and test.
    train_mask = (y_train_aligned == 1) | (y_train_aligned == 2)
    test_mask  = (y_test          == 1) | (y_test          == 2)

    features_train_dec = features_train[train_mask]
    y_train_dec        = y_train_aligned[train_mask]
    features_test_dec  = features_test[test_mask]
    y_test_dec         = y_test[test_mask]

    clf = RidgeClassifier(alpha=clf_ridge_alpha)
    clf.fit(features_train_dec, y_train_dec)
    y_pred = clf.predict(features_test_dec)

    bal_acc = float(balanced_accuracy_score(y_test_dec, y_pred))
    f1      = float(f1_score(y_test_dec, y_pred, average='macro'))

    return bal_acc, f1


# ═════════════════════════════════════════════════════════════════════════════
# Quick-start example (run as: python metrics.py)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("metrics.py — quick-start example")
    print("=" * 50)
    print("Building a random 64-node reservoir …")

    rng = np.random.default_rng(0)
    N   = 64
    W   = rng.standard_normal((N, N))


    # ── Memory Capacity
    print("Computing Memory Capacity …")
    mc = compute_memory_capacity(W, random_state=0)
    print(f"  MC = {mc:.4f}   (theoretical max ≈ {N})\n")

    # ── Kernel Rank 
    print("Computing Kernel Rank …")
    kr = compute_kernel_rank(W, random_state=0)
    print(f"  KR = {kr}   (theoretical max = {N})\n")

    # ── NARMA-5 
    print("Running NARMA-5 …")
    r5 = compute_narma(W, n=5, random_state=0, data_seed=0)
    print(f"  Pearson r = {r5:.4f}\n")

    # ── NARMA-10 
    print("Running NARMA-10 …")
    r10 = compute_narma(W, n=10, random_state=0, data_seed=0)
    print(f"  Pearson r = {r10:.4f}\n")

    # ── Mante task (skipped if neurogym not installed) 
    try:
        import neurogym  # noqa: F401
        print("Running Mante context-dependent decision-making task …")
        bal_acc, f1 = evaluate_mante(W, random_state=42, data_seed=0)
        print(f"  Balanced accuracy = {bal_acc:.4f}")
        print(f"  Macro F1          = {f1:.4f}")
        print(f"  Chance level      = 0.5000\n")
    except ImportError:
        print("  [Mante task skipped — neurogym not installed]\n")

    print("Done.")
