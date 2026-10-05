# Mental Health Score Predictor

A regression project that predicts a mental health score from student social media usage, stress and related habits. I compared Linear Regression against Support Vector Machines (SVM), then I finetuned the SVM to see how much tuning helps.

## Dataset
- **Source:** https://www.kaggle.com/datasets/saurabhstudyadda/student-social-media-and-mental-health-impact-csv
- **Size:** 5000 rows, 12 columns
- **Target variable:** Mental_Health_Score
- **Input features:** Age, Gender, Country, Academic_Level, Most_Used_Platform, Purpose_Of_Use, Avg_Daily_Usage_Hours,
                      Daily_Unlocks, Study_Hours, Physical_Activity_Hours, Sleep_Hours_Per_Night, Stress_Level


### 1. Exploratory data analysis
- Looked through the dataset and checked each column's distribution.
- Found outliers in Stress_Level column and all the numeric features columns only.
- Checked how each input feature correlates with the target.
- Cleaned the dataset, i.e dropped the duplicate rows, clipped rows etc.

### 2. Feature engineering
- **Country column:** it has 111 unique values, but only a few appear often. I kept 10 most occurring countries and grouped all other countries to "Other" to reduce the number of features and ease the load on the models.
- **Skewed numeric column:** Study_Hours column was slightly right-skewed, so I applied a log transformation. The other numeric columns were fine as they were.
- **Stress level:** it carries an inherent order, so I used **ordinal encoding** to keep that order.
- **Other categorical columns:** one-hot encoded.

### 3. Preprocessing pipelines
I built separate preprocessing pipelines for each column type and combined them with scikit-learn's `ColumnTransformer`. This keeps preprocessing and modeling in a single object, so the same transformations are applied to the training and test data.

### 4. Models
Each model (Linear Regression and SVM) sits in a full pipeline (preprocessing followed by the regressor):

```python
lr_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regression_model', LinearRegression())
])

lr_pipeline.fit(X_train, Y_train)
predictions = lr_pipeline.predict(X_test)
```
The SVM follows the same structure.

## Results

| Model | [R2 score] 
|---|---|
| Linear Regression | [0.74] | 
| SVM | [0.89] |
| Fine-tuned SVM | [0.87] | 

## Author
**Morsalina**: [GitHub](https://github.com/Morsalina)
