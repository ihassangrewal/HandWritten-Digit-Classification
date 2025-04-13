🧠 SVM-KNN Hybrid Classifier with PCA

This project demonstrates a hybrid machine learning approach combining Support Vector Machines (SVM) and K-Nearest Neighbors (KNN) for classification tasks. It also utilizes Principal Component Analysis (PCA) to reduce feature dimensionality before training, improving efficiency and performance.

📊 Project Highlights
	•	Hybrid Model: SVM + KNN
	•	Dimensionality Reduction: PCA with 60 components
	•	Dataset: train.csv (with features and target labels)
	•	Accuracy Achieved: 96.95%
	•	Execution Time: ~7.6 seconds

🚀 Workflow
	1.	Load and explore the dataset
	2.	Apply PCA to reduce dimensionality
	3.	Split data into training and testing sets
	4.	Train a KNN classifier (k=2) and evaluate
	5.	Measure model accuracy and execution time

🛠️ Tech Stack
	•	Python
	•	NumPy
	•	Pandas
	•	scikit-learn (PCA, KNN, SVM)

🖼️ Output

Model accuracy and execution time shown in output.jpeg

📁 Files in This Repo

/AI-Project
│
├── model.py         - Main training + evaluation script
├── train.csv        - Input dataset (not included)
├── output.jpeg      - Screenshot of final results
└── README.md        - Project documentation (this file)

📦 How to Run

Make sure you have the required packages installed (numpy, pandas, scikit-learn), and that train.csv is in the same folder.

Run the script with:

python model.py

✅ Future Ideas
	•	Add more models for comparison (Random Forest, XGBoost)
	•	Visualize PCA components
	•	Include confusion matrix and classification report
	•	Deploy model with Flask or Streamlit

Thanks for checking out this project! 🚀
