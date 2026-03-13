



<!-- ![header image](/assets/header_image.png)-->

<!-- ![](/assets/imdb.png) -->



<h1 align="center"> Opinion Polarity Detection</h2>


<!--![imdb header](/assets/imdb_header.png)-->




<p>What I learned</p>

- How natural language processing (NLP) classifies text into sentiment categories, making it easier to analyze opinions in large datasets (e.g., reviews or social media) and support insights like customer feedback or social media monitoring.


<!--
- How natural language processing (NLP) to classifies text into sentiment categories (e.g., positive or negative).

- Understand overall opinions in large collections of text, such as reviews or social media content.

- Simplifies sentiment into measurable categories for analysis.

- Supports insights for applications like customer feedback analysis or social media monitoring.

-->


<p> What I did</p>




<!--
- Built a modular Python sentiment analysis pipeline covering data loading, preprocessing, model training, evaluation, and prediction.

- Processed and engineered features from the IMDb dataset (50,000 reviews), handling missing values and encoding sentiments.

- Trained and compared multiple ML models (Logistic Regression, Decision Tree, Naive Bayes) with hyperparameter tuning, achieving ~89–90% accuracy.

- Created visualizations (sentiment distribution, confusion matrices, model comparisons) and documented the workflow in a clear README.

-->


- Developed a sentiment analysis pipeline on the IMDb dataset, including data loading, preprocessing, feature engineering, and model training with multiple algorithms (Logistic Regression, Decision Tree, Naive Bayes) and hyperparameter tuning (~81–93% accuracy).

- Generated visualizations (sentiment distribution, confusion matrices, model comparisons) and documented the entire workflow for clarity and reproducibility.












<h2 align="center">Workflow</h2>

- <h4><a href="/src/data_loader.py">Data Loading</a></h4>
    <div>
<!-- <style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>-->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>One of the other reviewers has mentioned that ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A wonderful little production. &lt;br /&gt;&lt;br /&gt;The...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I thought this was a wonderful way to spend ti...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Basically there's a family where a little boy ...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Petter Mattei's "Love in the Time of Money" is...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49995</th>
      <td>I thought this movie did a down right good job...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>49996</th>
      <td>Bad plot, bad dialogue, bad acting, idiotic di...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>49997</th>
      <td>I am a Catholic taught in parochial elementary...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>49998</th>
      <td>I'm going to have to disagree with the previou...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>49999</th>
      <td>No one expects the Star Trek movies to be high...</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>
<p>50000 rows × 2 columns</p>
</div>

<a href="/data/imdb.csv">Check out dataset</a>


- <h4>Data Information</h4>

    | # | Column    | Non-Null Count | Dtype  |
    |---|-----------|----------------|--------|
    | 0 | review    | 50000 non-null | object |
    | 1 | sentiment | 50000 non-null | object |


<!-- - Pre-Training Target Distribution

    ![Target Distribution](/assets/distribution.png)  -->

- <h4><a href="/src/data_preprocessing.py">Data Preprocessing</a></h4>

    - Check for missing values
    - Check for duplicated rows
    - Drop duplicated rows

- <h4><a href="/assets/sentinment_distribution.png">Pre-Training Visualization</a></h4>

    ![sentinment distribution](/assets/sentinment_distribution.png)

- <h4><a href="/src/data_preprocessing.py">Data Encoding</a></h4>

    - Encode the categorical target variable ("sentinment") into numerical format (0 and 1)

    - ``positive -> 1``    ``Negative -> 0``

- <h4><a href="/src/feature_engineering.py">Feature Engineering</a></h4>

    - X = ``df["review"]``

    - y = ``df["sentiment_encoded"]``

- <h4><a href="/src/data_preprocessing.py">Data Splitting</a></h4>

    - 80% training, 20% testing

- <h4><a href="/src/model_training.py">Model Pipelines Setup and Comparison</a></h4>








- <h4><a href="/src/model_evaluation.py">Model Training and Evaluation</a></h4>


    ![logistic regression confusion matrix](/assets/confusion_matrix_logistic.png)
    ![decison tree confusion matrix](/assets/confusion_matrix_decision_tree.png)
    ![naive bayes confusion matrix](/assets/confusion_matix_naive_bayes.png)
<!--
<h4> Logistic Regression Results</h4>

<b>Accuracy:</b> 0.8925  

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| negative  | 0.91      | 0.87   | 0.89     | 4939    |
| positive  | 0.88      | 0.91   | 0.89     | 4978    |

| Metric        | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| accuracy      | -         | -      | 0.89     | 9917    |
| macro avg     | 0.89      | 0.89   | 0.89     | 9917    |
| weighted avg  | 0.89      | 0.89   | 0.89     | 9917    |

-->

<!--![logistic regression confusion matrix](/assets/confusion_matrix_logistic.png)-->



<!--

<h4>Decision Tree Results</h4>

<b>Accuracy:</b> 0.7134  

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| negative  | 0.71      | 0.71   | 0.71     | 4939    |
| positive  | 0.71      | 0.71   | 0.71     | 4978    |

| Metric        | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| accuracy      | -         | -      | 0.71     | 9917    |
| macro avg     | 0.71      | 0.71   | 0.71     | 9917    |
| weighted avg  | 0.71      | 0.71   | 0.71     | 9917    |

-->

<!--![decison tree confusion matrix](/assets/confusion_matrix_decision_tree.png)-->

<!--
<h4>Multinomial Naive Bayes Results</h4>

<b>Accuracy:</b> 0.8638  

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| negative  | 0.86      | 0.87   | 0.86     | 4939    |
| positive  | 0.87      | 0.86   | 0.86     | 4978    |

| Metric        | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| accuracy      | -         | -      | 0.86     | 9917    |
| macro avg     | 0.86      | 0.86   | 0.86     | 9917    |
| weighted avg  | 0.86      | 0.86   | 0.86     | 9917    |
-->


<!-- ![naive bayes confusion matrix](/assets/confusion_matix_naive_bayes.png)-->




- <h4><a href="/src/hyperparameter_tuning.py">Hyperparameter Tuning</a></h4>

    ![tuned logistic regression confusion matrix](/assets/confusion_matrix_tuned_logistic.png)

<!--
<h4> Tuned Logistic Regression Results</h4>

<b>Test Accuracy:</b> 0.8999  

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| negative  | 0.91      | 0.88   | 0.90     | 4939    |
| positive  | 0.89      | 0.92   | 0.90     | 4978    |

| Metric        | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| accuracy      | -         | -      | 0.90     | 9917    |
| macro avg     | 0.90      | 0.90   | 0.90     | 9917    |
| weighted avg  | 0.90      | 0.90   | 0.90     | 9917    |

-->




<!--![tuned logistic regression confusion matrix](/assets/confusion_matrix_tuned_logistic.png)-->


- <h4><a href="/assets/model_accuracy_2.png">Model Accuracy Before Tuning</a></h4>

    ![model accuarcy before tuning](/assets/model_accuracy_2.png)



- <h4><a href="/assets/model_accuracy.png">Model Accuracy After Tuning</a></h4>

    ![model accuracy after tuning](/assets/model_accuracy.png)