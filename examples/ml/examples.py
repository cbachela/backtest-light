import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor

# --- 1. Generate data ---
np.random.seed(0)
n = 30
t = np.linspace(0, 5, n)
g = 9.81
x_true = 0.5 * g * t**2
noise = np.random.normal(0, 3.1, size=n)
x = x_true + noise * np.sqrt(t).flatten()

t = t.reshape(-1, 1)


# split
t_train, t_val, x_train, x_val = train_test_split(t, x, test_size=0.4, random_state=42)


# helper plot
# helper plot
def plot_model(model, title):
    t_plot = np.linspace(0, 5, 200).reshape(-1, 1)
    x_pred = model.predict(t_plot)

    plt.scatter(t_train, x_train, label="train")
    plt.scatter(t_val, x_val, label="val")
    plt.plot(t_plot, x_pred, color="black")

    # Vertical distances for validation points
    x_pred_val = model.predict(t_val)
    plt.vlines(t_val, x_val, x_pred_val, colors="red", alpha=0.5, linestyles="dashed")

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_model_train(model, title):
    t_plot = np.linspace(0, 5, 200).reshape(-1, 1)
    x_pred = model.predict(t_plot)
    x_pred_train = model.predict(t_train)

    plt.scatter(t_train, x_train, label="train")
    plt.plot(t_plot, x_pred, color="black")
    # Vertical distances
    plt.vlines(t_train, x_train, x_pred_train, colors="red", alpha=0.5, linestyles="dashed")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_model_val(model, title):
    t_plot = np.linspace(0, 5, 200).reshape(-1, 1)
    x_pred = model.predict(t_plot)
    x_pred_val = model.predict(t_val)

    plt.scatter(t_val, x_val, label="val", color="orange")
    plt.plot(t_plot, x_pred, color="black")
    # Vertical distances
    plt.vlines(t_val, x_val, x_pred_val, colors="red", alpha=0.5, linestyles="dashed")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title(title)
    plt.legend()
    plt.show()


# --- 2. Linear model (underfits) ---
lin = LinearRegression()
lin.fit(t_train, x_train)
plot_model(lin, "1. Linear model (underfits)")
# plot_model_train(lin, "1. Linear model (underfits)")
# plot_model_val(lin, "1. Linear model (underfits)")


# --- 3. Feature engineering (correct model) ---
quad = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
quad.fit(t_train, x_train)
plot_model(quad, "2. Add t^2 feature (works)")

# --- 4. Neural net (overfits) ---
nn_overfit = MLPRegressor(
    hidden_layer_sizes=(100, 100),
    activation="relu",
    alpha=0.0,
    solver="lbfgs",
    max_iter=5000,
    random_state=0,
)

nn_overfit.fit(t_train, x_train)
plot_model(nn_overfit, "3. Neural Net (overfits small data)")


# --- 5. Neural net + regularization (CV) ---
param_grid_nn = {
    "hidden_layer_sizes": [(10,), (50,), (100,), (50, 50)],
    "alpha": np.logspace(-5, 1, 10),
}

grid_nn = GridSearchCV(
    MLPRegressor(solver="lbfgs", max_iter=5000, random_state=0),
    param_grid_nn,
    scoring="neg_mean_squared_error",
    cv=5,
    return_train_score=True,
)

grid_nn.fit(t_train, x_train)

best_nn = grid_nn.best_estimator_
plot_model(best_nn, "4. Neural Net (regularized via CV)")

# --- 6. Bias-variance curve (alpha) ---
alphas = grid_nn.cv_results_["param_alpha"].data
train_err = -grid_nn.cv_results_["mean_train_score"]
val_err = -grid_nn.cv_results_["mean_test_score"]

plt.scatter(alphas, train_err, label="train error")
plt.scatter(alphas, val_err, label="validation error")
plt.xscale("log")
plt.xlabel("alpha (regularization)")
plt.ylabel("MSE")
plt.title("Bias-Variance Tradeoff (Neural Net)")
plt.legend()
plt.show()

print("Best NN params:", grid_nn.best_params_)
