from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score

# Load datasets
diabetes = datasets.load_diabetes()
x_data = diabetes.data
y_data = diabetes.target

# Split, 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=30)

# Make the models
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor(random_state=30)
forest_model = RandomForestRegressor(random_state=30)

# Train the models
linear_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

# Predictions
linear_predict = linear_model.predict(X_test)
tree_predict = tree_model.predict(X_test)
forest_predict = forest_model.predict(X_test)

# Evaluate mean squared error for each model
evaluate_linear = mean_squared_error(y_test, linear_predict)
evaluate_tree = mean_squared_error(y_test, tree_predict)
evaluate_forest = mean_squared_error(y_test, forest_predict)

print(f'Linear MSE: {evaluate_linear: .2f}')
print(f'Tree MSE: {evaluate_tree: .2f}')
print(f'Forest MSE: {evaluate_forest: .2f}')

# Evaluate explained variance score
var_linear = explained_variance_score(y_test, linear_predict)
var_tree = explained_variance_score(y_test, tree_predict)
var_forest = explained_variance_score(y_test, forest_predict)

print('\n')
print(f'Var Linear Score: {var_linear: .2f}')
print(f'Var Tree Score: {var_tree: .2f}')
print(f'Var Forest Score: {var_forest: .2f}')

# Evaluate r2 score
r2_linear = r2_score(y_test, linear_predict)
r2_tree = r2_score(y_test, tree_predict)
r2_forest = r2_score(y_test, forest_predict)

print('\n')
print(f'r2 Linear Score: {r2_linear: .2f}')
print(f'r2 Tree Score: {r2_tree: .2f}')
print(f'r2 Forest Score: {r2_forest: .2f}')
