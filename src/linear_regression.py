import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Generate synthetic data with more scatter on the sides
np.random.seed(42)
X_left = np.linspace(0, 2, 20)
X_middle = np.linspace(2, 8, 30)
X_right = np.linspace(8, 10, 20)

y_true_left = 2 * X_left + 1 + np.random.normal(scale=10, size=len(X_left))
y_true_middle = 2 * X_middle + 1 + np.random.normal(scale=2, size=len(X_middle))
y_true_right = 2 * X_right + 1 + np.random.normal(scale=10, size=len(X_right))

X = np.concatenate((X_left, X_middle, X_right))
y_true = np.concatenate((y_true_left, y_true_middle, y_true_right))

# Fit linear regression model
X_with_intercept = sm.add_constant(X)
model = sm.OLS(y_true, X_with_intercept)
results = model.fit()

# Get coefficients and confidence intervals
coef = results.params
conf_int = results.conf_int()

# Create predictions and prediction intervals
X_pred = sm.add_constant(np.linspace(0, 10, 100))
y_pred = results.get_prediction(X_pred).summary_frame()

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y_true, label='True data')
plt.plot(X_pred[:, 1], y_pred['mean'], color='red', label='Predicted mean')
plt.plot(X_pred[:, 1], y_pred['obs_ci_lower'], linestyle='dashed', color='orange', label='95% Prediction Interval')
plt.plot(X_pred[:, 1], y_pred['obs_ci_upper'], linestyle='dashed', color='orange')
plt.fill_between(X_pred[:, 1], y_pred['obs_ci_lower'], y_pred['obs_ci_upper'], color='orange', alpha=0.2)
plt.title('Linear Regression with Uncertainty')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Display coefficients and confidence intervals
print("Coefficients:")
print(coef)
print("\nConfidence Intervals:")
print(conf_int)
