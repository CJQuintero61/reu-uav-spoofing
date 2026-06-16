# Model 1 Documentation
This file has the documentation for model 1's design decisions and explanations

## Design Decisions
This section contains the design decisions for the code

### Constants
```
SEED = 0
TEST_SIZE = 0.25
```
The seed is defined early on so that everything that accepts a `random_state` parameter gets
set to the same value for reproducability. The `test_size` is set to 0.25 and this was chosen
as the specific value since `train_test_split()` defaults to 0.25.

### Removing Columns
```
columns_to_drop = []
for col in data.columns:
    if data[col].nunique() == 1:
        columns_to_drop.append(col)
```
This loop identifies which columns have only 1 unique value and adds them to
the list of columns to drop from the dataframe. These values were dropped because they
do not help in classifying the data as benign or malicious since they are constant
in both benign and malicious entries.

```
timing = ['timestamp', 'time_utc_usec', 'timestamp_time_relative']
columns_to_drop += timing
```
These timing columns are also added to the list of columns to remove since
timing features should have no effect on classifying data as benign or malicious.

### Converting Labels to 0 and 1
```
# convert the labels to 0 and 1
data['label'] = data['label'].replace('benign', 0)
data['label'] = data['label'].replace('malicious', 1)
```
The labels in the data are converted to 0 and 1 instead
of strings for easier handling.

### Using a Pipeline
```
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='linear', class_weight='balanced', random_state=SEED))
])
```
The pipeline defines a sequence of transforms to apply
to the data. This pipeline is created because in each
fold when applying cross validation, we want the
scale to be applied based on the current fold and not
have any data leakage from other folds. The same applies
to the SVC where the SVC should be fit to the current fold
and should not have any knowledge of other folds.

### Using StandardScaler
Standard Scaler was chosen as an arbitrary choice due to
its name.

### SVC Parameters
```
kernel='linear'
```
A linear kernel was chosen since the results from using a
linear kernel were slightly better than the rbf kernel.

The results from the linear kernel's cross validation scores were:
- Average test_accuracy: 0.9996 +/- 0.0007
- Average test_precision: 1.0 +/- 0.0
- Average test_recall: 0.9973 +/- 0.0053
- Average test_f1: 0.9987 +/- 0.0027

While the results from the rbf kernel's cross validation scores were:
- Average test_accuracy: 0.9989 +/- 0.0009
- Average test_precision: 1.0 +/- 0.0
- Average test_recall: 0.992 +/- 0.0066
- Average test_f1: 0.996 +/- 0.0033

```
# extract the trained svm from the pipeline
svc_model = pipeline.named_steps['svc']

signed_coefficients = pd.Series(
    svc_model.coef_[0],
    index=X_train.columns
).sort_values(ascending=False)

# the top features that contribute to the model predicting a sample as malicious
print('Positive coefficients:')
print(signed_coefficients.head(5))
print()

# the top features that contribute to the model predicting a sample as benign
print('Negative coefficients:')
print(signed_coefficients.tail(5))
```
The linear kernel also allows feature coefficients
to be extracted to see which features contributed the most
to the model's decision.

Lastly, the linear kernel is easier to understand
as this means the data was linearly seperable.

```
class_weight='balanced'
```
Balanced class weights were used since the data was very
unbalanced with there being far more benign samples
compared to malicious samples.

### Model Scoring Categories
```
scoring = ['accuracy', 'precision', 'recall', 'f1']
```
These 4 basic categories were chosen as simple metrics
to view and compare to other models and evaluate the model's
performance.

### Cross Valiation
```
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)
```
Stratified K Folds was chosen due to the imbalanced
dataset. `n_splits=5` was chosen, since 5 is the default
value used by the `StratifiedKFold()` constructor.