

# End-to-End Hitters Machine Learning Pipeline

from helpers import *

df = pd.read_csv("hitters.csv")
df.info()
df.describe()
cat_cols, num_cols, cat_but_car = classify_columns_types(df)
categorical_column_summary(df, cat_cols, "Salary", True)
numeric_column_summary(df, num_cols, "Salary", True)


# Create new variable
# Rate of AtBat for whole career
df["RAtBat"] = df["AtBat"] / df["CAtBat"]

# Rate of Hits for whole career
df["RHits"] = df["Hits"] / df["CHits"]

# Rate of HmRun for whole career
df["RHmRun"] = df["HmRun"] / df["CHmRun"]

# Rate of Runs for whole career
df["RRuns"] = df["Runs"] / df["CRuns"]

# Rate of RBI for whole career
df["RRBI"] = df["RBI"] / df["CRBI"]

# Rate of Walks for whole career
df["RWalks"] = df["Walks"] / df["CWalks"]

# Succession rate of Hits for AtBat
df["SHits"] = df["Hits"] / df["AtBat"]

# Succession rate of CHits for CAtBat
df["SCHits"] = df["CHits"] / df["CAtBat"]

# Succession rate of HmRun for Runs
df["SRuns"] = df["HmRun"] / df["Runs"]

# Succession rate of CHmRun for CRuns
df["SCHmRun"] = df["CHmRun"] / df["CRuns"]

# Determine outlier and replace with thresholds
cat_cols, num_cols, cat_but_car = classify_columns_types(df)
plot_quartiles_outlier(df, num_cols)
for col in num_cols:
    replace_outliers_with_thresholds(df, col, 0.05, 0.95)

# Standardization
num_cols = [col for col in num_cols if 'Salary' not in col]
standardization(df, num_cols, RobustScaler())

# One-hot encoding
df = one_hot_encoder(df, cat_cols, True)

# Fill NaN values of 'Salary' with KNN method
df = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(df), columns=df.columns)

# Define X and y variables
y = df["Salary"]
X = df.drop("Salary", axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

# Modeling
# Linear Regression
linreg = LinearRegression().fit(X_train, y_train)
evaluate_model(linreg, X_train, y_train, X_test, y_test, "Linear Regression")

# Lasso Regression
lasreg = Lasso().fit(X_train,y_train)
evaluate_model(lasreg, X_train, y_train, X_test, y_test, "Lasso Regression")

# LightGBM - Light Gradient Boosting Model
lgb_model = LGBMRegressor().fit(X_train, y_train)
evaluate_model(lgb_model, X_train, y_train, X_test, y_test, "Light Gradient Boosting Model")
plot_importance(lgb_model, X_test)

# Random Forest
rf_model = RandomForestRegressor().fit(X_train, y_train)
evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")
plot_importance(rf_model, X_test)