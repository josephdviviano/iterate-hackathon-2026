"""Fit y = w*x + b with gradient descent and report the validation MSE."""
from prepare import load_data, evaluate

LR = 0.1
STEPS = 200

xs, ys = load_data()
w, b = 0.0, 0.0
for step in range(STEPS):
    gw = gb = 0.0
    for x, y in zip(xs, ys):
        err = (w * x + b) - y
        gw += err * x
        gb += err
    # BUG: gradients are summed but never averaged over the dataset
    w -= LR * gw
    b -= LR * gb

val = evaluate(lambda x: w * x + b, *load_data(seed=1))
print("---")
print(f"val_mse: {val:.6f}")
print(f"w: {w:.4f}")
print(f"b: {b:.4f}")
