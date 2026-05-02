import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# SIMULATION PARAMETERS (from PDF)
PARAMS = {
    'q':   4,
    'theta': 1,
    'O':   0.5,          # Mb
    'r_d': 1.2,          # Mb/s
    'r_u': 1.3,          # Mb/s
    'K':   100,
    'B':   0.5 / 1024,   # 0.5 kb → Mb
    'psi': 0.001,
    'alpha': 1/3,
    'beta':  1/3,
    'gamma': 1/3,
}

SETTINGS = {
    1: {
        'v': 5, 'M': 25, 't': 50, 'X': 500,
        'xi': [250.74, 187.36, 138.91, 245.20, 271.70, 276.80, 119.32, 194.45,
               213.02, 138.59, 207.21, 215.94, 236.32, 203.08, 145.41, 264.83,
               286.12, 123.30, 242.89, 121.47, 198.76, 254.19, 167.43, 289.05, 132.88],
    },
    2: {
        'v': 5, 'M': 40, 't': 50, 'X': 500,
        'xi': [121.12, 259.89, 198.40, 234.00, 272.30, 134.32, 157.55, 288.65,
               210.58, 255.98, 249.03, 201.28, 298.93, 157.16, 125.26, 167.22,
               261.86, 211.60, 184.22, 153.01, 116.73, 241.46, 201.56, 290.00,
               218.31, 232.87, 104.61, 268.79, 185.63, 115.13, 236.16, 102.55,
               259.15, 226.24, 100.15, 142.94, 199.64, 221.28, 150.30, 104.61],
    },
}


# OBJECTIVE FUNCTION

def compute_utility(m_set, n, setting_id, p=PARAMS):
    s = SETTINGS[setting_id]
    v, M, t, X = s['v'], s['M'], s['t'], s['X']
    xi = np.array(s['xi'])

    m = len(m_set)
    xi_selected = xi[list(m_set)]

    # Raw values
    C = xi_selected.sum() / n
    eta = p['theta'] * (m ** p['q'])
    L = (n * p['B'] / p['r_d']) + (p['K'] / xi[list(m_set)].min()) + \
        p['psi'] * (n * p['B']) ** m + (p['O'] / p['r_u'])

    # Bounds
    C_max = xi.sum() / t
    C_min = np.sort(xi)[:v].sum() / X

    eta_max = p['theta'] * (M ** p['q'])
    eta_min = p['theta'] * (v ** p['q'])

    xi_sorted_desc = np.sort(xi)[::-1]
    vth_highest = xi_sorted_desc[v - 1]

    L_max = (X * p['B'] / p['r_d']) + (p['K'] / xi.min()) + \
            p['psi'] * (X * p['B']) ** M + (p['O'] / p['r_u'])
    L_min = (t * p['B'] / p['r_d']) + (p['K'] / vth_highest) + \
            p['psi'] * (t * p['B']) ** v + (p['O'] / p['r_u'])

    def log_norm(val, lo, hi):
        lv, ll, lh = np.log(max(val, 1e-12)), np.log(max(lo, 1e-12)), np.log(max(hi, 1e-12))
        return (lv - ll) / (lh - ll + 1e-12)

    C_norm   = log_norm(C,   C_min,   C_max)
    eta_norm = 1 - log_norm(eta, eta_min, eta_max)
    L_norm   = log_norm(L,   L_min,   L_max)

    U = p['alpha'] * L_norm + p['beta'] * eta_norm + p['gamma'] * C_norm
    return U


def decode_solution(x_binary, n_continuous, setting_id):
    """Convert algorithm output → (m_set, n)."""
    s = SETTINGS[setting_id]
    v, M, t, X = s['v'], s['M'], s['t'], s['X']
    M_total = s['M']

    bits = np.round(x_binary).astype(int)
    selected = set(np.where(bits[:M_total] == 1)[0])

    # Enforce bounds  v ≤ m ≤ M
    if len(selected) < v:
        pool = list(set(range(M_total)) - selected)
        np.random.shuffle(pool)
        selected.update(pool[:v - len(selected)])
    while len(selected) > M:
        selected.pop()

    n = int(np.clip(round(n_continuous), t, X))
    return selected, n


def evaluate(x_binary, n_cont, setting_id):
    m_set, n = decode_solution(x_binary, n_cont, setting_id)
    return compute_utility(m_set, n, setting_id), m_set, n



# ALGORITHM PARAMETERS

POP_SIZE  = 30
MAX_ITER  = 100
N_RUNS    = 20


# ─────────────────────────────────────────────────────────────
# 1. GENETIC ALGORITHM – BINARY-CODED
# ─────────────────────────────────────────────────────────────
def ga_binary(setting_id, seed=0):
    np.random.seed(seed)
    s = SETTINGS[setting_id]
    M_total = s['M']
    dim = M_total + 1  # M bits + 1 for n (mapped 0-1 → t-X)

    pop = np.random.randint(0, 2, (POP_SIZE, dim)).astype(float)
    history = []

    def fitness(ind):
        bits = ind[:M_total]
        n_cont = s['t'] + ind[M_total] * (s['X'] - s['t'])
        u, _, _ = evaluate(bits, n_cont, setting_id)
        return u

    fits = np.array([fitness(p) for p in pop])
    best_fit = fits.min()
    best_ind = pop[fits.argmin()].copy()

    for it in range(MAX_ITER):
        # Tournament selection
        new_pop = []
        for _ in range(POP_SIZE):
            idx = np.random.choice(POP_SIZE, 4, replace=False)
            a, b = idx[:2][np.argmin(fits[idx[:2]])], idx[2:][np.argmin(fits[idx[2:]])]
            p1, p2 = pop[a], pop[b]
            # One-point crossover
            if np.random.rand() < 0.8:
                pt = np.random.randint(1, dim)
                child = np.concatenate([p1[:pt], p2[pt:]])
            else:
                child = p1.copy()
            # Mutation
            mask = np.random.rand(dim) < 0.02
            child[mask] = 1 - child[mask]
            child = np.clip(child, 0, 1)
            new_pop.append(child)

        pop = np.array(new_pop)
        fits = np.array([fitness(p) for p in pop])
        if fits.min() < best_fit:
            best_fit = fits.min()
            best_ind = pop[fits.argmin()].copy()
        history.append(best_fit)

    bits = best_ind[:M_total]
    n_cont = s['t'] + best_ind[M_total] * (s['X'] - s['t'])
    m_set, n = decode_solution(bits, n_cont, setting_id)
    return best_fit, m_set, n, history


# ─────────────────────────────────────────────────────────────
# 2. GENETIC ALGORITHM – REAL-CODED
# ─────────────────────────────────────────────────────────────
def ga_real(setting_id, seed=0):
    np.random.seed(seed)
    s = SETTINGS[setting_id]
    M_total = s['M']
    dim = M_total + 1

    pop = np.random.rand(POP_SIZE, dim)
    history = []

    def fitness(ind):
        bits = (ind[:M_total] > 0.5).astype(float)
        n_cont = s['t'] + ind[M_total] * (s['X'] - s['t'])
        u, _, _ = evaluate(bits, n_cont, setting_id)
        return u

    fits = np.array([fitness(p) for p in pop])
    best_fit = fits.min()
    best_ind = pop[fits.argmin()].copy()

    for it in range(MAX_ITER):
        new_pop = []
        for _ in range(POP_SIZE):
            idx = np.random.choice(POP_SIZE, 4, replace=False)
            a, b = idx[:2][np.argmin(fits[idx[:2]])], idx[2:][np.argmin(fits[idx[2:]])]
            p1, p2 = pop[a], pop[b]
            # SBX crossover
            if np.random.rand() < 0.9:
                eta_c = 2
                u = np.random.rand(dim)
                beta = np.where(u <= 0.5,
                                (2 * u) ** (1/(eta_c+1)),
                                (1/(2*(1-u))) ** (1/(eta_c+1)))
                child = 0.5 * ((1+beta)*p1 + (1-beta)*p2)
            else:
                child = p1.copy()
            # Polynomial mutation
            eta_m = 20
            mask = np.random.rand(dim) < 0.05
            u2 = np.random.rand(dim)
            delta = np.where(u2 < 0.5,
                             (2*u2)**(1/(eta_m+1)) - 1,
                             1 - (2*(1-u2))**(1/(eta_m+1)))
            child = np.where(mask, child + delta * 0.1, child)
            child = np.clip(child, 0, 1)
            new_pop.append(child)

        pop = np.array(new_pop)
        fits = np.array([fitness(p) for p in pop])
        if fits.min() < best_fit:
            best_fit = fits.min()
            best_ind = pop[fits.argmin()].copy()
        history.append(best_fit)

    bits = (best_ind[:M_total] > 0.5).astype(float)
    n_cont = s['t'] + best_ind[M_total] * (s['X'] - s['t'])
    m_set, n = decode_solution(bits, n_cont, setting_id)
    return best_fit, m_set, n, history


# ─────────────────────────────────────────────────────────────
# 3. PARTICLE SWARM OPTIMIZATION
# ─────────────────────────────────────────────────────────────
def pso(setting_id, seed=0):
    np.random.seed(seed)
    s = SETTINGS[setting_id]
    M_total = s['M']
    dim = M_total + 1
    w, c1, c2 = 0.7, 1.5, 1.5

    pos = np.random.rand(POP_SIZE, dim)
    vel = np.random.uniform(-0.5, 0.5, (POP_SIZE, dim))
    history = []

    def fitness(ind):
        bits = (ind[:M_total] > 0.5).astype(float)
        n_cont = s['t'] + ind[M_total] * (s['X'] - s['t'])
        u, _, _ = evaluate(bits, n_cont, setting_id)
        return u

    pbest = pos.copy()
    pbest_fit = np.array([fitness(p) for p in pbest])
    gbest_idx = pbest_fit.argmin()
    gbest = pbest[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    for it in range(MAX_ITER):
        r1, r2 = np.random.rand(POP_SIZE, dim), np.random.rand(POP_SIZE, dim)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, 0, 1)
        fits = np.array([fitness(p) for p in pos])
        improve = fits < pbest_fit
        pbest[improve] = pos[improve].copy()
        pbest_fit[improve] = fits[improve]
        if pbest_fit.min() < gbest_fit:
            gbest_fit = pbest_fit.min()
            gbest = pbest[pbest_fit.argmin()].copy()
        history.append(gbest_fit)

    bits = (gbest[:M_total] > 0.5).astype(float)
    n_cont = s['t'] + gbest[M_total] * (s['X'] - s['t'])
    m_set, n = decode_solution(bits, n_cont, setting_id)
    return gbest_fit, m_set, n, history


# ─────────────────────────────────────────────────────────────
# 4. TEACHING-LEARNING-BASED OPTIMIZATION
# ─────────────────────────────────────────────────────────────
def tlbo(setting_id, seed=0):
    np.random.seed(seed)
    s = SETTINGS[setting_id]
    M_total = s['M']
    dim = M_total + 1
    history = []

    pop = np.random.rand(POP_SIZE, dim)

    def fitness(ind):
        bits = (ind[:M_total] > 0.5).astype(float)
        n_cont = s['t'] + ind[M_total] * (s['X'] - s['t'])
        u, _, _ = evaluate(bits, n_cont, setting_id)
        return u

    fits = np.array([fitness(p) for p in pop])
    best_fit = fits.min()
    best_ind = pop[fits.argmin()].copy()

    for it in range(MAX_ITER):
        # Teaching phase
        teacher = pop[fits.argmin()]
        mean_pop = pop.mean(axis=0)
        Tf = np.random.randint(1, 3, dim)
        new_pop = pop + np.random.rand(POP_SIZE, dim) * (teacher - Tf * mean_pop)
        new_pop = np.clip(new_pop, 0, 1)
        new_fits = np.array([fitness(p) for p in new_pop])
        improve = new_fits < fits
        pop[improve] = new_pop[improve]
        fits[improve] = new_fits[improve]

        # Learner phase
        for i in range(POP_SIZE):
            j = np.random.choice([k for k in range(POP_SIZE) if k != i])
            if fits[i] < fits[j]:
                new_i = pop[i] + np.random.rand(dim) * (pop[i] - pop[j])
            else:
                new_i = pop[i] + np.random.rand(dim) * (pop[j] - pop[i])
            new_i = np.clip(new_i, 0, 1)
            f_new = fitness(new_i)
            if f_new < fits[i]:
                pop[i] = new_i
                fits[i] = f_new

        if fits.min() < best_fit:
            best_fit = fits.min()
            best_ind = pop[fits.argmin()].copy()
        history.append(best_fit)

    bits = (best_ind[:M_total] > 0.5).astype(float)
    n_cont = s['t'] + best_ind[M_total] * (s['X'] - s['t'])
    m_set, n = decode_solution(bits, n_cont, setting_id)
    return best_fit, m_set, n, history


# ─────────────────────────────────────────────────────────────
# 5. DIFFERENTIAL EVOLUTION
# ─────────────────────────────────────────────────────────────
def de(setting_id, seed=0):
    np.random.seed(seed)
    s = SETTINGS[setting_id]
    M_total = s['M']
    dim = M_total + 1
    F, CR = 0.8, 0.9
    history = []

    pop = np.random.rand(POP_SIZE, dim)

    def fitness(ind):
        bits = (ind[:M_total] > 0.5).astype(float)
        n_cont = s['t'] + ind[M_total] * (s['X'] - s['t'])
        u, _, _ = evaluate(bits, n_cont, setting_id)
        return u

    fits = np.array([fitness(p) for p in pop])
    best_fit = fits.min()
    best_ind = pop[fits.argmin()].copy()

    for it in range(MAX_ITER):
        for i in range(POP_SIZE):
            idxs = [k for k in range(POP_SIZE) if k != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), 0, 1)
            cross = np.random.rand(dim) < CR
            cross[np.random.randint(dim)] = True
            trial = np.where(cross, mutant, pop[i])
            f_trial = fitness(trial)
            if f_trial < fits[i]:
                pop[i] = trial
                fits[i] = f_trial
        if fits.min() < best_fit:
            best_fit = fits.min()
            best_ind = pop[fits.argmin()].copy()
        history.append(best_fit)

    bits = (best_ind[:M_total] > 0.5).astype(float)
    n_cont = s['t'] + best_ind[M_total] * (s['X'] - s['t'])
    m_set, n = decode_solution(bits, n_cont, setting_id)
    return best_fit, m_set, n, history


# ─────────────────────────────────────────────────────────────
# 6. ARTIFICIAL BEE COLONY
# ─────────────────────────────────────────────────────────────
def abc(setting_id, seed=0):
    np.random.seed(seed)
    s = SETTINGS[setting_id]
    M_total = s['M']
    dim = M_total + 1
    limit = POP_SIZE * dim // 2
    history = []

    pop = np.random.rand(POP_SIZE, dim)

    def fitness(ind):
        bits = (ind[:M_total] > 0.5).astype(float)
        n_cont = s['t'] + ind[M_total] * (s['X'] - s['t'])
        u, _, _ = evaluate(bits, n_cont, setting_id)
        return u

    fits = np.array([fitness(p) for p in pop])
    trial_cnt = np.zeros(POP_SIZE)
    best_fit = fits.min()
    best_ind = pop[fits.argmin()].copy()

    for it in range(MAX_ITER):
        # Employed bees
        for i in range(POP_SIZE):
            k = np.random.choice([j for j in range(POP_SIZE) if j != i])
            phi = np.random.uniform(-1, 1, dim)
            j = np.random.randint(dim)
            cand = pop[i].copy()
            cand[j] = pop[i][j] + phi[j] * (pop[i][j] - pop[k][j])
            cand = np.clip(cand, 0, 1)
            f_cand = fitness(cand)
            if f_cand < fits[i]:
                pop[i] = cand; fits[i] = f_cand; trial_cnt[i] = 0
            else:
                trial_cnt[i] += 1

        # Onlooker bees – roulette
        inv = 1 / (fits + 1e-12)
        probs = inv / inv.sum()
        for _ in range(POP_SIZE):
            i = np.random.choice(POP_SIZE, p=probs)
            k = np.random.choice([j for j in range(POP_SIZE) if j != i])
            phi = np.random.uniform(-1, 1, dim)
            j = np.random.randint(dim)
            cand = pop[i].copy()
            cand[j] = pop[i][j] + phi[j] * (pop[i][j] - pop[k][j])
            cand = np.clip(cand, 0, 1)
            f_cand = fitness(cand)
            if f_cand < fits[i]:
                pop[i] = cand; fits[i] = f_cand; trial_cnt[i] = 0
            else:
                trial_cnt[i] += 1

        # Scout bees
        exhausted = trial_cnt > limit
        pop[exhausted] = np.random.rand(exhausted.sum(), dim)
        fits[exhausted] = np.array([fitness(p) for p in pop[exhausted]])
        trial_cnt[exhausted] = 0

        if fits.min() < best_fit:
            best_fit = fits.min()
            best_ind = pop[fits.argmin()].copy()
        history.append(best_fit)

    bits = (best_ind[:M_total] > 0.5).astype(float)
    n_cont = s['t'] + best_ind[M_total] * (s['X'] - s['t'])
    m_set, n = decode_solution(bits, n_cont, setting_id)
    return best_fit, m_set, n, history


# ─────────────────────────────────────────────────────────────
# 7. ANT COLONY OPTIMIZATION  (continuous ACO variant)
# ─────────────────────────────────────────────────────────────
def aco(setting_id, seed=0):
    np.random.seed(seed)
    s = SETTINGS[setting_id]
    M_total = s['M']
    dim = M_total + 1
    n_ants = POP_SIZE
    archive_size = 20
    q_acor = 0.5
    xi_acor = 0.85
    history = []

    # Initialize archive with random solutions
    archive = np.random.rand(archive_size, dim)

    def fitness(ind):
        bits = (ind[:M_total] > 0.5).astype(float)
        n_cont = s['t'] + ind[M_total] * (s['X'] - s['t'])
        u, _, _ = evaluate(bits, n_cont, setting_id)
        return u

    archive_fits = np.array([fitness(p) for p in archive])
    sort_idx = np.argsort(archive_fits)
    archive = archive[sort_idx]
    archive_fits = archive_fits[sort_idx]
    best_fit = archive_fits[0]
    best_ind = archive[0].copy()

    for it in range(MAX_ITER):
        # Weights using Gaussian kernel
        ranks = np.arange(1, archive_size + 1)
        weights = (1 / (q_acor * archive_size * np.sqrt(2*np.pi))) * \
                  np.exp(-((ranks-1)**2) / (2 * q_acor**2 * archive_size**2))
        weights /= weights.sum()

        new_solutions = []
        for _ in range(n_ants):
            sol = np.zeros(dim)
            for d in range(dim):
                chosen = np.random.choice(archive_size, p=weights)
                sigma = xi_acor * np.sum(np.abs(archive[:, d] - archive[chosen, d])) / (archive_size - 1 + 1e-12)
                sol[d] = archive[chosen, d] + np.random.normal(0, max(sigma, 0.01))
            sol = np.clip(sol, 0, 1)
            new_solutions.append(sol)

        new_arr = np.vstack([archive, new_solutions])
        new_fits = np.concatenate([archive_fits, [fitness(p) for p in new_solutions]])
        sort_idx = np.argsort(new_fits)
        archive = new_arr[sort_idx[:archive_size]]
        archive_fits = new_fits[sort_idx[:archive_size]]
        if archive_fits[0] < best_fit:
            best_fit = archive_fits[0]
            best_ind = archive[0].copy()
        history.append(best_fit)

    bits = (best_ind[:M_total] > 0.5).astype(float)
    n_cont = s['t'] + best_ind[M_total] * (s['X'] - s['t'])
    m_set, n = decode_solution(bits, n_cont, setting_id)
    return best_fit, m_set, n, history


# ─────────────────────────────────────────────────────────────
# RUN ALL ALGORITHMS × SETTINGS × N_RUNS
# ─────────────────────────────────────────────────────────────
ALGORITHMS = {
    'GA-Binary': ga_binary,
    'GA-Real':   ga_real,
    'PSO':       pso,
    'TLBO':      tlbo,
    'DE':        de,
    'ABC':       abc,
    'ACO':       aco,
}

COLORS = {
    'GA-Binary': '#4A90D9',
    'GA-Real':   '#E74C3C',
    'PSO':       '#2ECC71',
    'TLBO':      '#F39C12',
    'DE':        '#9B59B6',
    'ABC':       '#1ABC9C',
    'ACO':       '#E67E22',
}

print("=" * 70)
print("  Blockchain Configuration Optimization — Metaheuristic Benchmark")
print("=" * 70)

all_results = {}   # [setting][algo] = {best_fits, avg_fits, conv_curves, runtimes, m_vals, n_vals}

for sid in [1, 2]:
    s = SETTINGS[sid]
    print(f"\n{'─'*70}")
    print(f"  SETTING {sid}  |  v={s['v']}  M={s['M']}  t={s['t']}  X={s['X']}")
    print(f"{'─'*70}")
    all_results[sid] = {}

    for aname, afunc in ALGORITHMS.items():
        best_fits, avg_fits_per_run = [], []
        conv_matrix = []
        runtimes, m_vals, n_vals = [], [], []

        best_m, best_n = None, None

        print(f"\n  [{aname:10s}] running {N_RUNS} seeds: ", end='', flush=True)
        for run in range(N_RUNS):
            t0 = time.time()
            bf, m_set, n, hist = afunc(sid, seed=run)
            elapsed = time.time() - t0
            best_fits.append(bf)
            conv_matrix.append(hist)
            runtimes.append(elapsed)
            m_vals.append(len(m_set))
            n_vals.append(n)
            if best_m is None or bf <= min(best_fits[:-1] or [float('inf')]):
                best_m = len(m_set)
                best_n = n
            print('.', end='', flush=True)

        # Average convergence across runs
        conv_arr = np.array(conv_matrix)       # (N_RUNS, MAX_ITER)
        avg_conv = conv_arr.mean(axis=0)

        # Best and average fitness
        best_fit_val  = np.min(best_fits)
        avg_fit_val   = np.mean(best_fits)
        avg_runtime   = np.mean(runtimes)
        std_fit       = np.std(best_fits)

        all_results[sid][aname] = {
            'best_fits':   best_fits,
            'best':        best_fit_val,
            'avg':         avg_fit_val,
            'std':         std_fit,
            'avg_conv':    avg_conv,
            'all_conv':    conv_arr,
            'runtimes':    runtimes,
            'avg_runtime': avg_runtime,
            'best_m':      best_m,
            'best_n':      best_n,
            'm_vals':      m_vals,
            'n_vals':      n_vals,
        }

        print(f"  Best={best_fit_val:.6f}  Avg={avg_fit_val:.6f}  Std={std_fit:.6f}"
              f"  RT={avg_runtime:.2f}s  Best_m={best_m}  Best_n={best_n}")


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────
def make_convergence_plot(sid):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f'Convergence Curves — Setting {sid}  (v={SETTINGS[sid]["v"]}, M={SETTINGS[sid]["M"]}, '
                 f't={SETTINGS[sid]["t"]}, X={SETTINGS[sid]["X"]})',
                 fontsize=14, fontweight='bold', y=1.01)

    algo_names = list(ALGORITHMS.keys())
    iters = np.arange(1, MAX_ITER + 1)

    for idx, aname in enumerate(algo_names):
        ax = axes[idx // 3][idx % 3]
        res = all_results[sid][aname]
        conv_all = res['all_conv']        # (N_RUNS, MAX_ITER)
        avg_conv = res['avg_conv']

        # Shaded min/max band
        ax.fill_between(iters, conv_all.min(0), conv_all.max(0),
                        alpha=0.15, color=COLORS[aname])
        # Individual runs (faint)
        for run_curve in conv_all:
            ax.plot(iters, run_curve, color=COLORS[aname], alpha=0.12, linewidth=0.6)
        # Mean curve
        ax.plot(iters, avg_conv, color=COLORS[aname], linewidth=2.0,
                label=f'Mean | Best={res["best"]:.5f}')

        ax.set_title(aname, fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=9)
        ax.set_ylabel('Utility (U)', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlim([1, MAX_ITER])
        ax.tick_params(labelsize=8)

    # 8th panel: all algorithms overlaid
    ax = axes[2][1]
    for aname in algo_names:
        res = all_results[sid][aname]
        ax.plot(iters, res['avg_conv'], color=COLORS[aname], linewidth=2, label=aname)
    ax.set_title('All Algorithms — Mean Convergence', fontsize=11, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=9)
    ax.set_ylabel('Utility (U)', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, MAX_ITER])
    ax.tick_params(labelsize=8)

    # 9th panel: box plot of final fitness
    ax = axes[2][2]
    data = [all_results[sid][a]['best_fits'] for a in algo_names]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch, aname in zip(bp['boxes'], algo_names):
        patch.set_facecolor(COLORS[aname])
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(algo_names)+1))
    ax.set_xticklabels([a.replace('-', '\n') for a in algo_names], fontsize=7)
    ax.set_title('Best Fitness Distribution (20 runs)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Best Utility (U)', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = f'/mnt/user-data/outputs/convergence_setting{sid}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")
    return path


def make_comparison_plot(sid):
    algo_names = list(ALGORITHMS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Algorithm Comparison — Setting {sid}  (v={SETTINGS[sid]["v"]}, M={SETTINGS[sid]["M"]})',
                 fontsize=13, fontweight='bold')

    colors = [COLORS[a] for a in algo_names]
    x = np.arange(len(algo_names))
    w = 0.35

    # 1. Best vs Average fitness
    ax = axes[0][0]
    bests = [all_results[sid][a]['best']  for a in algo_names]
    avgs  = [all_results[sid][a]['avg']   for a in algo_names]
    stds  = [all_results[sid][a]['std']   for a in algo_names]
    b1 = ax.bar(x - w/2, bests, w, color=colors, alpha=0.85, label='Best Fitness', edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + w/2, avgs,  w, color=colors, alpha=0.45, label='Avg Fitness',  edgecolor='white', linewidth=0.5, hatch='///')
    ax.errorbar(x + w/2, avgs, yerr=stds, fmt='none', color='black', capsize=3, linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(algo_names, fontsize=8, rotation=15)
    ax.set_ylabel('Utility (U)', fontsize=9); ax.set_title('Best vs Average Fitness', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y'); ax.tick_params(labelsize=8)
    for rect, val in zip(b1, bests):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=6.5, rotation=70)

    # 2. Runtime comparison
    ax = axes[0][1]
    rts = [all_results[sid][a]['avg_runtime'] for a in algo_names]
    bars = ax.bar(x, rts, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(algo_names, fontsize=8, rotation=15)
    ax.set_ylabel('Time (s)', fontsize=9); ax.set_title('Average Runtime per Run', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y'); ax.tick_params(labelsize=8)
    for rect, rt in zip(bars, rts):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01,
                f'{rt:.1f}s', ha='center', va='bottom', fontsize=8)

    # 3. Best m (validators selected from best run)
    ax = axes[1][0]
    ms = [all_results[sid][a]['best_m'] for a in algo_names]
    bars = ax.bar(x, ms, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    v_val, M_val = SETTINGS[sid]['v'], SETTINGS[sid]['M']
    ax.axhline(v_val, color='red',   linestyle='--', linewidth=1.2, label=f'v_min={v_val}')
    ax.axhline(M_val, color='green', linestyle='--', linewidth=1.2, label=f'M_max={M_val}')
    ax.set_xticks(x); ax.set_xticklabels(algo_names, fontsize=8, rotation=15)
    ax.set_ylabel('# Validators (m)', fontsize=9)
    ax.set_title('Best-Run Selected Validators (m)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y'); ax.tick_params(labelsize=8)
    for rect, m_v in zip(bars, ms):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.1,
                f'{m_v}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 4. Best n (transactions per block from best run)
    ax = axes[1][1]
    ns = [all_results[sid][a]['best_n'] for a in algo_names]
    bars = ax.bar(x, ns, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    t_val, X_val = SETTINGS[sid]['t'], SETTINGS[sid]['X']
    ax.axhline(t_val, color='red',   linestyle='--', linewidth=1.2, label=f't_min={t_val}')
    ax.axhline(X_val, color='green', linestyle='--', linewidth=1.2, label=f'X_max={X_val}')
    ax.set_xticks(x); ax.set_xticklabels(algo_names, fontsize=8, rotation=15)
    ax.set_ylabel('Transactions (n)', fontsize=9)
    ax.set_title('Best-Run Transactions per Block (n)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y'); ax.tick_params(labelsize=8)
    for rect, n_v in zip(bars, ns):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1,
                f'{n_v}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = f'/mnt/user-data/outputs/comparison_setting{sid}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def make_summary_table():
    """Print a formatted summary table for both settings."""
    print("\n" + "=" * 90)
    print(f"  {'SUMMARY TABLE':^86}")
    print("=" * 90)

    for sid in [1, 2]:
        s = SETTINGS[sid]
        print(f"\n  Setting {sid}: v={s['v']}, M={s['M']}, t={s['t']}, X={s['X']}")
        print(f"  {'Algorithm':<14} {'Best Fitness':>14} {'Avg Fitness':>14} {'Std':>10} "
              f"{'Best m':>8} {'Best n':>8} {'Avg RT(s)':>10}")
        print("  " + "-" * 84)
        for aname in ALGORITHMS.keys():
            r = all_results[sid][aname]
            print(f"  {aname:<14} {r['best']:>14.6f} {r['avg']:>14.6f} {r['std']:>10.6f} "
                  f"{r['best_m']:>8} {r['best_n']:>8} {r['avg_runtime']:>10.2f}")

    print("\n" + "=" * 90)


# ─────────────────────────────────────────────────────────────
# GENERATE ALL PLOTS
# ─────────────────────────────────────────────────────────────
saved_paths = []
for sid in [1, 2]:
    print(f"\n  Generating plots for Setting {sid}...")
    saved_paths.append(make_convergence_plot(sid))
    saved_paths.append(make_comparison_plot(sid))

make_summary_table()

print("\n  All done! Files saved to /mnt/user-data/outputs/")
print("  " + "\n  ".join(saved_paths))
