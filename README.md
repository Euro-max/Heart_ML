# Heart Disease Prediction Project

This project is a comprehensive machine learning pipeline designed to predict the presence and type of heart disease using the UCI Heart Disease dataset. It covers a full end-to-end workflow, from data preprocessing and exploratory analysis to model deployment in a web application.

## 🚀 Features

* **Data Preprocessing:** Imputation of missing values, one-hot encoding for categorical features, np.log() for numerical data to reduce skewness, and data scaling.
* **Dimensionality Reduction:** Application of Principal Component Analysis (PCA) to reduce the number of features while retaining critical information.
* **Supervised Learning:** Implementation and hyperparameter tuning of traditional models : **Logistic Regression**, **Random Forest Classifier**, and **Support Vector Machines**
* **Unsupervised Learning:** Exploration of hidden patterns in the data using **K-Means Clustering**.
* **Deep Learning:** Development of a simple feedforward neural network using **TensorFlow** and **Keras** for prediction.
* **Web Application:** A user-friendly, interactive prediction app built with **Streamlit**.
* **Deployment & Sharing:** The app is served locally and shared with a public URL using **ngrok**.

## 📊 Dataset

The project utilizes the **Heart Disease UCI** dataset, a well-known benchmark for heart disease prediction. The dataset contains 14 features and a multi-class target variable (`num`) indicating the presence and type of heart disease (from 0 to 4).

---
## 📁 Project Structure

├── data/                    # Contains the raw dataset and the preprocessed data
│   └── heart_disease_uci.csv
│   └── cleaned_heart_disease_uci.csv
├── models/                  # Stores trained model and preprocessors
│   ├── final_model.pkl
│   ├── pca.pkl
│   └── scaler.pkl
├── ui/                    # Streamlit web application
│   └── app.py                 
├── notebooks      # Jupyter/Colab notebooks
│   └── Data_PreProcessing.ipynb
│   └── Feature_Selection.ipynb
│   └── HyperParameter_Tuning.ipynb
│   └── PCA_Analysis.ipynb
│   └── Simple_feedforward_NN.ipynb
│   └── Supervised_Learning.ipynb
│   └── UnSupervised_Learning.ipynb

├── README.md                # Project documentation
└── requirements.txt         # Project dependencies

---
## 🧠 Methodology

### 1. Data Preprocessing

The initial raw data undergoes several steps to prepare it for modeling:
* Missing values are identified and handled.
* Categorical features are converted into numerical format using one-hot encoding.
* Numpy log for numerical data to handle its skewedness in order to improve the model's performance especially Logistic Regression,PCA,and Neural Networks by making the distribution more symmetrical.
* Numerical features are scaled to a standard range using `StandardScaler` to ensure all features contribute equally to the model.

### 2. Model Training & Evaluation

The project explores both classical and modern machine learning approaches:
* **Supervised Learning:**
    * **Random Forest Classifier:** A powerful ensemble model known for its high accuracy. It was fine-tuned using `GridSearchCV` and a `class_weight='balanced'` parameter to handle the multi-class imbalance.
    * **Logistic Regression:** Used as a baseline and fine-tuned for comparison.
* **Unsupervised Learning:**
    * **K-Means Clustering:** Applied to find natural groupings or subtypes of patients within the dataset.

### 3. Deep Learning

* A simple feedforward neural network was constructed with TensorFlow and Keras. This model learns complex, non-linear relationships in the data to improve prediction performance.
  
### 4. Results
 
 <img width="1500" height="1300" alt="model_accuracies" src="https://github.com/user-attachments/assets/998162e9-deca-405d-aaf1-6eac5f595aa7" />

### 5. Deployment

The final trained model is deployed as a web application:
* **Streamlit:** The `app.py` file contains the logic for the web interface, which allows users to input patient data and receive a real-time heart disease prediction.
* **ngrok:** `ngrok` is used to create a secure tunnel from the local machine to the internet, providing a public URL to share the application with others.

## 🚀 Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Euro-max/Heart_ML.git](https://github.com/Euro-max/Heart_ML.git)
    cd Heart_ML
    ```

2.  **Install dependencies:**
    Make sure you have all the required libraries by installing them from the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    Start the Streamlit server, which will run the app on your localhost (typically `http://localhost:8501`).

    ```bash
    streamlit run app.py
    ```

4.  **Make it public with ngrok:**
    In a separate terminal, start `ngrok` to expose your local app to the public internet.

    ```bash
    ngrok http 8501
    ```

    `ngrok` will provide a public URL (e.g., `https://random-string.ngrok.io`) that you can share.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

* UCI Machine Learning Repository for providing the dataset.
* The developers of `scikit-learn`, `pandas`, `streamlit`, `tensorflow`, and `ngrok` for their excellent tools.
