"""Fixed data + evaluation. Do not modify."""
import random

def load_data(n=1000, seed=0):
    rng = random.Random(seed)
    xs = [rng.uniform(-1, 1) for _ in range(n)]
    ys = [3.0 * x + 0.5 + rng.gauss(0, 0.1) for x in xs]
    return xs, ys

def evaluate(predict, xs, ys):
    """Mean squared error of predict(x) against y."""
    return sum((predict(x) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)
