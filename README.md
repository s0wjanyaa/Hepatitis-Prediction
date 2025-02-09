# Hepatitis Life Expectancy Prediction

This project uses various machine learning regression models to predict the life expectancy of hepatitis patients based on various health indicators.The code for this project was developed and run in Google Colab.It explores and compares the performance of three different regression models:

*   Linear Regression
*   Support Vector Machines (SVM)
*   Logistic Regression  

## Dataset

The dataset used in this project is sourced from Kaggle:[ [Link to the Kaggle dataset]](https://www.kaggle.com/code/mragpavank/hepatitis-using-svm/input).  It contains various health indicators for hepatitis patients. The dataset has the following features:


*   **Target Variable:**
    *   `target`:  `DIE (1)`: Patient died. `LIVE (2)`: Patient survived.

*   **Features:**
    *   `age`: Age of the patient (in years).
    *   `gender`: Gender of the patient. `male (1)`, `female (2)`.
    *   `steroid`: Use of steroid medication. `no`, `yes`.
    *   `antivirals`: Use of antiviral medication. `no`, `yes`.
    *   `fatique`: Presence of fatigue. `no`, `yes`.
    *   `malaise`: Presence of malaise. `no`, `yes`.
    *   `anorexia`: Presence of anorexia. `no`, `yes`.
    *   `liverBig`: Enlarged liver. `no`, `yes`.
    *   `liverFirm`: Firm liver. `no`, `yes`.
    *   `spleen`: Enlarged spleen. `no`, `yes`.
    *   `spiders`: Presence of spider angiomas. `no`, `yes`.
    *   `ascites`: Presence of ascites. `no`, `yes`.
    *   `varices`: Presence of varices. `no`, `yes`.
    *   `histology`: Liver biopsy results. `no`, `yes`.
    *   `bilirubin`: Bilirubin level 
    *   `alk`: Alkaline phosphatase level 
    *   `sgot`: SGOT (AST) level 
    *   `albu`: Albumin level 
    *   `protime`: Prothrombin time
    *   `ID`: Patient ID 


## Code
The code for this project is implemented in a Colab notebook (`HepatitisPred.ipynb`) and was developed and run in Google Colab. The notebook includes data preprocessing, model training, evaluation, and comparison of Linear Regression, SVM, and Logistic Regression models.

## Running the Code

1.  Clone the repository: `git clone https://github.com/s0wjanyaa/Hepatitis-Prediction.git` 
 Open the Colab notebook (`HepatitisPred.ipynb`) in Google Colab.
3.  **Place the dataset:**  Place the `hepatitis.csv` file in your Google Drive.  You can put it anywhere you like.  For example, you could create a folder called `data` in your My Drive and put it there.
4.  **Mount your Drive (if needed):**  At the beginning of the Colab notebook, add the following code to mount your Google Drive:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

5.  **Set the correct path in the notebook:** In the code where you load the data, *modify* the path to point to the location where you placed the `hepatitis.csv` file in your Drive.  For example, if you put it in a folder called `data` in your My Drive, the code should look like this:

    ```python
    df = pd.read_csv('/content/drive/MyDrive/data/hepatitis.csv', na_values='?')
    ```

    If you put it directly in your My Drive, it would be:

    ```python
    df = pd.read_csv('/content/drive/MyDrive/hepatitis.csv', na_values='?')
    ```

6.  Run the cells in the notebook sequentially.

This notebook can also be run in a local Jupyter environment (ensure dependencies are installed).

## Models and Evaluation

This project explores the use of Linear Regression, Support Vector Machines (SVM), and Logistic Regression for predicting life expectancy in hepatitis patients.

*   **Linear Regression:** Used to predict the target variable directly. Evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

    *   Training MAE: 0.191
    *   Test MAE: 0.2996
    *   Training MSE: 0.0669
    *   Test MSE: 0.2002
    *   Training R-squared: 0.5842
    *   Test R-squared: 0.1451

The Linear Regression model shows a reasonable fit on the training data but performs poorly on the test set, indicating potential overfitting. The low R-squared value on the test set suggests that the model does not explain much of the variance in the target variable.

*   **SVM:** Used for classification (predicting `DIE`(1) or `LIVE`(2)). Evaluated using accuracy, F1-score, and confusion matrix. A linear kernel was used.The SVM model achieved a test accuracy of 77.4%. 

*   **Logistic Regression:** Used for classification. Evaluated using accuracy, precision, recall, F1-score, and confusion matrix.The Logistic Regression model achieved a test accuracy of 74.2%. The model achieved a precision of  0.93, recall of 0.98, and an F1-score of 0.96 on the test set. 

The SVM model achieved the highest accuracy on the test set (77.4%), followed closely by Logistic Regression (74.2%).  However, it's important to consider other metrics such as precision, recall, and F1-score, especially for imbalanced datasets.  Linear Regression, while showing a reasonable fit on the training data, performed poorly on the test data, suggesting overfitting or that it's not a suitable model for this classification task.  Further investigation into feature engineering or trying different models might be beneficial.
 
**Feature Importance (Logistic Regression):**

The feature importance analysis for the Logistic Regression model identified the following as the most influential features in predicting life expectancy (in descending order of importance):

    1.  `anorexia`
    2.  `gender`
    3.  `malaise`
    4.  `albu` (Albumin level)
    5.  `spiders`
    6.  `bili` (Bilirubin level)
    7.  `ascites`
    8.  `fatigue`
    9.  `varices`
    10. `histology`
    11. `steroid`
    12. `antivirals`
    13. `spleen`
    14. `liverFirm`
    15. `liverBig`
    16. `age`
    17. `protime` (Prothrombin time)
    18. `alk` (Alkaline phosphatase level)
    19. `sgot` (SGOT/AST level)

These results suggest that the presence of anorexia, gender, and malaise are strong indicators of life expectancy in hepatitis patients, according to the logistic regression model.  Albumin and bilirubin levels, along with the presence of spiders and ascites, also play a significant role.  It's important to note that feature importance is specific to the model used and should be interpreted in the context of the model's performance.

## Dependencies

Python libraries used in the project:

*   pandas
*   numpy
*   scikit-learn
*   matplotlib (used for plotting)
*   seaborn (used for plotting)
