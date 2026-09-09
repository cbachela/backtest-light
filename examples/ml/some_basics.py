import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def y_yhat_plot(y, y_pred, title: str = ""):
    plt.figure(figsize=(5, 4))
    plt.title(title)
    plt.scatter(y_pred, y, s=15, alpha=0.8)
    plt.xlabel(r"$\hat{y}$", fontsize=10)
    plt.ylabel(r"$y$", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout() 
    plt.show()

def y_yhat_plot_overlay(y_train, y_pred_train, y_test, y_pred_test, title=""):

    plt.figure(figsize=(5, 4))
    plt.title(title)
    plt.scatter(y_pred_train, y_train, 
                s=15, alpha=0.7, label="Train", color="tab:blue")
    plt.scatter(y_pred_test, y_test, 
                s=15, alpha=0.7, label="Test", color="tab:orange")
    plt.xlabel(r"$\hat{y}$", fontsize=10)
    plt.ylabel(r"$y$", fontsize=10)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

#-----------------------------------------
# What we want to learn
#-----------------------------------------

# Download 5 years of Apple data
data = yf.download("AAPL", period="5y", 
                   interval="1d")

# Compute returns
price = data['Close'].resample('W').last()
ret = price.pct_change().dropna()

# Label
y = ret.shift(-1).dropna()

# Past returns dont help
y_yhat_plot(y, ret.reindex(y.index),
            title="past returns as predictors")


#-----------------------------------------
# Features :)
#-----------------------------------------

# Let's get some features
np.random.seed(42)

X = pd.DataFrame(index=y.index)
X['A'] = np.random.randn(len(X))
X['B'] = np.random.randn(len(X))
X['C'] = np.random.randn(len(X))

X.head()

X = X.reindex(y.index)


#-----------------------------------------
# ML - Linearity won't cut it
#-----------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

linear_model = LinearRegression()
linear_model.fit(X, y)

# Predict on training data
y_pred_linear = linear_model.predict(X)

# Evaluation
mse_linear = mean_squared_error(y, y_pred_linear)
print("Linear Model Training MSE:", mse_linear)

y_yhat_plot(y, y_pred_linear, title="Linear Model")

#-----------------------------------------
# ML - the overfit
#-----------------------------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X, y)

# Predict on training data
y_pred = model.predict(X)

# Evaluation
mse = mean_squared_error(y, y_pred)
print("Training MSE:", mse)
y_yhat_plot(y, y_pred, title="ML predictions")


#-----------------------------------------
# Train / Test split (time-aware)
#-----------------------------------------
split = int(0.6 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

#-----------------------------------------
# Fit model
#-----------------------------------------
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


y_yhat_plot_overlay(
    y_train, y_pred_train,
    y_test, y_pred_test,
    title="Train vs Test Predictions"
)
# Errors
print("Train MSE:", mean_squared_error(y_train, y_pred_train))
print("Test  MSE:", mean_squared_error(y_test, y_pred_test))


#-----------------------------------------
# Lets add an actual feature
# with marginal value
#-----------------------------------------
X['momentum'] = ret.shift(1).reindex(X.index)
X = X.dropna()
y = y.reindex(X.index)


#-----------------------------------------
# Hyperparameter Tuning
#-----------------------------------------
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

split = int(0.6 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [2, 3, 5, None],
    'min_samples_leaf': [1, 5, 10]
}

model = RandomForestRegressor(random_state=42)

grid = GridSearchCV(
    model,
    param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid.fit(X_train, y_train)

# Best tuned model
best_model = grid.best_estimator_

y_pred_train_tuned = best_model.predict(X_train)
y_pred_test_tuned = best_model.predict(X_test)


y_yhat_plot_overlay(
    y_train, y_pred_train_tuned,
    y_test, y_pred_test_tuned,
    title="Tuned Model"
)

print("Tuned Train MSE:", mean_squared_error(y_train, y_pred_train_tuned))
print("Tuned Test  MSE:", mean_squared_error(y_test, y_pred_test_tuned))



#-----------------------------------------
# Information advantage or just leakage
# will be the name of the game!
#-----------------------------------------
X = X.reindex(y.index)

# Leakage feature 
np.random.seed(42)
X['leak'] = y.squeeze().values + 0.2 * np.random.randn(len(y))

# splitting
split = int(0.6 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [2, 3, 5, None],
    'min_samples_leaf': [1, 5, 10]
}

model = RandomForestRegressor(random_state=42)

grid = GridSearchCV(
    model,
    param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid.fit(X_train, y_train)

# Best tuned model
best_model = grid.best_estimator_

y_pred_train_tuned = best_model.predict(X_train)
y_pred_test_tuned = best_model.predict(X_test)


y_yhat_plot_overlay(
    y_train, y_pred_train_tuned,
    y_test, y_pred_test_tuned,
    title="Tuned Model (with information advantage)"
)

print("Tuned Train MSE:", mean_squared_error(y_train, y_pred_train_tuned))
print("Tuned Test  MSE:", mean_squared_error(y_test, y_pred_test_tuned))



import shap

# Use TreeExplainer for RandomForest
explainer = shap.TreeExplainer(best_model)

# Compute SHAP values on test set
shap_values = explainer.shap_values(X_test)

# Summary plot (global feature importance)
# will correlate by chance in small samples
shap.summary_plot(shap_values, X_test)