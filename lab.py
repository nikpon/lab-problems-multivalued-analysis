import typing as tp
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

def func(x: tp.Sequence[float]) -> float:
    return np.maximum(np.abs(2 * x[0] + x[1] - 3), np.abs(x[0] + 2) + np.abs(x[1] - 1))

gm = (0, 1) # Wolfram Alpha

SIZE = 10
DPI = 201

x1 = np.linspace(-SIZE, SIZE, DPI)
x2 = np.linspace(-SIZE, SIZE, DPI)
X1, X2 = np.meshgrid(x1, x2)
f = func((X1, X2))

def nonsmooth(DPI, SIZE):
    # x1 = -2
    l1x1 = np.array(DPI * [-2])
    l1x2 = np.linspace(-SIZE, SIZE, DPI)
    l1z = func((l1x1, l1x2))

    # x2 = 1
    l2x1 = np.linspace(-SIZE, SIZE, DPI)
    l2x2 = np.array(DPI * [1])
    l2z = func((l2x1, l2x2))

    # 2x1 + x2 = 3
    l3x2 = np.linspace(-SIZE, SIZE, DPI)
    l3x1 = (3 - l3x2) / 2
    l3z = func((l3x1, l3x2))

    # x1 = 4
    l4x1 = np.array(DPI * [4])
    l4x2 = np.linspace(-SIZE, SIZE, DPI)
    l4z = func((l4x1, l4x2))

    # x1 + 2x2 = 6
    l5x1 = np.linspace(-SIZE, SIZE, DPI)
    l5x2 = (6 - l5x1) / 2
    l5z = func((l5x1, l5x2))

    # x1 = 0
    l6x1 = np.array(DPI * [0])
    l6x2 = np.linspace(-SIZE, SIZE, DPI)
    l6z = func((l6x1, l6x2))

    # 3/2 x1 + x2 = 1
    l7x2 = np.linspace(-SIZE, SIZE, DPI)
    l7x1 = 2 * (1 - l7x2) / 3
    l7z = func((l7x1, l7x2))

    lsx1 = [l1x1, l2x1, l3x1, l4x1, l5x1, l6x1, l7x1]
    lsx2 = [l1x2, l2x2, l3x2, l4x2, l5x2, l6x2, l7x2]
    lsz = [l1z, l2z, l3z, l4z, l5z, l6z, l7z]

    return lsx1, lsx2, lsz

lsx1, lsx2, lsz = nonsmooth(DPI, SIZE)
fig = go.Figure(data=[
    go.Surface(z=f, x=X1, y=X2),
    *[
        go.Scatter3d(
            x=lsx1[i], y=lsx2[i], z=lsz[i],
            marker=dict(color='white', size=1),
            line=dict(color='white', width=5),
            name=''
        ) for i in range(7)
    ],
    go.Scatter3d(
        x=[gm[0]], y=[gm[1]], z=[func(gm)],
        marker=dict(color='white', size=5),
        name=''
    )
])
fig.update_layout(width=800, height=600)
fig.show()

def sub_grad(x: tp.Sequence[float]) -> tp.Sequence[float]:
    g1 = np.array([
        np.zeros(x[0].shape),
        np.zeros(x[1].shape),
    ])

    if x[0] + 2 < 0:
        g1[0] -= 1
    elif x[0] + 2 > 0:
        g1[0] += 1
    else:
        g1[0] += np.random.uniform(-1, 1, g1[0].shape)

    if x[1] - 1 < 0:
        g1[1] -= 1
    elif x[1] - 1 > 0:
        g1[1] += 1
    else:
        g1[1] += np.random.uniform(-1, 1, g1[1].shape)

    g2 = np.array([
        np.zeros(x[0].shape),
        np.zeros(x[1].shape),
    ])

    if 2 * x[0] + x[1] - 3 < 0:
        g2[0] -= 2
        g2[1] -= 1
    elif 2 * x[0] + x[1] - 3 > 0:
        g2[0] += 2
        g2[1] += 1
    else:
        g2[0] += np.random.uniform(-1, 1, g2[0].shape)
        g2[1] += np.random.uniform(-2, 2, g2[1].shape)

    if np.abs(2 * x[0] + x[1] - 3) < np.abs(x[0] + 2) + np.abs(x[1] - 1):
        return g1
    elif np.abs(2 * x[0] + x[1] - 3) > np.abs(x[0] + 2) + np.abs(x[1] - 1):
        return g2
    else:
        t = np.random.uniform(0, 1)
        return t * g1 + (1 - t) * g2

SIZE = 5
DPI = 21

lsx1, lsx2, lsz = nonsmooth(DPI, SIZE)

x1 = np.linspace(-SIZE, SIZE, DPI)
x2 = np.linspace(-SIZE, SIZE, DPI)
X1, X2 = np.meshgrid(x1, x2)

for _ in range(4):
    for i in range(7):
        X1 = np.vstack([X1, lsx1[i]])
        X2 = np.vstack([X2, lsx2[i]])

f = func((X1, X2))

g = np.array([sub_grad((x1i, x2i)) for (x1i, x2i) in zip(X1.ravel(), X2.ravel())]).reshape((*X1.shape, 2))
u = np.array([np.array([dx for (dx, dy) in gi]) for gi in g])
v = np.array([np.array([dy for (dx, dy) in gi]) for gi in g])

w = 0.5 * np.sqrt(u**2 + v**2)
u /= w
v /= w

fig = ff.create_quiver(x=X1, y=X2, u=-u, v=-v, name='gradient')
for i in range(7):
    fig.add_trace(
        go.Scatter(x=lsx1[i], y=lsx2[i], mode='lines',
                line=dict(color='black', width=1),
                connectgaps=True))
fig.add_trace(
    go.Scatter(x=[gm[0]], y=[gm[1]],
               name='optimum'))
fig.update_layout(width=800, height=800)
fig.show()

def optimize(
    func: tp.Callable[[tp.Sequence[float]], float],
    sub_grad: tp.Callable[[tp.Sequence[float]], tp.Sequence[float]],
    init: tp.Sequence[float],
    step_rule: str,
    step_params: tp.Mapping[str, float] = {},
    max_iter: int = 10_000,
    tol: float = 1e-6,
) -> tp.Tuple[tp.Sequence[tp.Sequence[float]], tp.Sequence[float]]:
    """Unconditionally optimizes func with subgradient method.

    Parameters:
    func:         R^n -> R function to be optimized.
    sub_grad:     subgradient of func.
    init:         initial value to start the optimization from.
    step_rule:    rule to select the step size.
                  Should be one of:
                    'constant-size'
                    'constant-length'
                    'square-summable'
                    'diminishing'
    step_params:  parameters of the step size rule.
    max_iter:     maximum number of iterations to perform.
    tol:          tolerance up to which the optimization is performed.

    Returns:
    points: a list of visited points.
    values: a list of function values in these points.
    """
    points = [init]
    values = [func(init)]

    for k in range(max_iter):
        g = sub_grad(points[-1])

        step_size: float = 0
        if step_rule == 'constant-size':
            step_size = step_params['h']
        elif step_rule == 'constant-length':
            step_size = step_params['h'] / np.linalg.norm(g)
        elif step_rule == 'square-summable':
            step_size = step_params['a'] / (step_params['b'] + k + 1)
        elif step_rule == 'diminishing':
            step_size = step_params['a'] / np.sqrt(k + 1)
        else:
            raise ValueError("please refer to the docstring for supported values of step_rule")

        points.append(points[-1] - step_size * g)
        values.append(func(points[-1]))

        if np.linalg.norm(points[-1] - points[-2]) < tol and \
           np.linalg.norm(values[-1] - values[-2]) < tol:
            break

    return np.vstack(points), np.vstack(values)

COLORS = ['red', 'blue', 'green']
STEPS = [0.1, 0.01, 1e-3]
fig = go.Figure()
for (color, step) in zip(COLORS, STEPS):
    points, values = optimize(func, sub_grad, np.array([0, 0]), 'constant-size', {'h': step}, max_iter=100)
    values = values.ravel()
    fig.add_trace(
        go.Scatter(x=np.arange(len(values)), y=values, mode='lines',
                   name=f'constant-size, {step}',
                   line=dict(color=color, width=1),
                   connectgaps=True)
    )
fig.show()

COLORS = ['red', 'blue', 'green']
STEPS = [0.1, 0.01, 0.001]
fig = go.Figure()
for (color, step) in zip(COLORS, STEPS):
    points, values = optimize(func, sub_grad, np.array([0, 0]), 'constant-length', {'h': step}, max_iter=100)
    values = values.ravel()
    fig.add_trace(
        go.Scatter(x=np.arange(len(values)), y=values, mode='lines',
                   name=f'constant-length, {step}',
                   line=dict(color=color, width=1),
                   connectgaps=True)
    )
fig.show()

COLORS = ['red', 'blue', 'green']
PARAMS = [{'h': 0.1}, {'a': 1, 'b': 1}, {'a': 0.5}]
METHODS = ['constant-size', 'square-summable', 'diminishing']
fig = go.Figure()
for (color, params, method) in zip(COLORS, PARAMS, METHODS):
    points, values = optimize(func, sub_grad, np.array([0, 0]), method, params, max_iter=100)
    values = values.ravel()
    fig.add_trace(
        go.Scatter(x=np.arange(len(values)), y=values, mode='lines',
                   name=f'{method}, {params}',
                   line=dict(color=color, width=1),
                   connectgaps=True)
    )
fig.show()

points, values = optimize(func, sub_grad, np.array([0, 0]), 'square-summable', {'a': 1, 'b': 0}, max_iter=10000)

best = values.argmin()
points[best], values[best]
