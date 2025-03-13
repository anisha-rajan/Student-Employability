import gradio as gr
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_or_train_models():
    """Load models from disk or train and save them if not found."""
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("logistic_regression.pkl", "rb") as f:
            logistic_regression = pickle.load(f)
        with open("perceptron.pkl", "rb") as f:
            perceptron = pickle.load(f)
    except FileNotFoundError:
        print("Training models...")
        df = pd.read_excel("Student-Employability-Datasets.xlsx", sheet_name="Data")
        X = df.iloc[:, 1:-2].values
        y = (df["CLASS"] == "Employable").astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logistic_regression = LogisticRegression(random_state=42)
        logistic_regression.fit(X_train_scaled, y_train)
        
        perceptron = Perceptron(random_state=42)
        perceptron.fit(X_train_scaled, y_train)
        
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open("logistic_regression.pkl", "wb") as f:
            pickle.dump(logistic_regression, f)
        with open("perceptron.pkl", "wb") as f:
            pickle.dump(perceptron, f)
    
    return scaler, logistic_regression, perceptron
scaler, logistic_regression, perceptron = load_or_train_models()

def predict_employability(name, ga, mos, pc, ma, sc, api, cs, model_choice):
    """Predict employability based on input features and selected model."""
    name = name if name else "The candidate"
    input_data = np.array([[ga, mos, pc, ma, sc, api, cs]])
    input_scaled = scaler.transform(input_data)
    
    model = logistic_regression if model_choice == "Logistic Regression" else perceptron
    prediction = model.predict(input_scaled)[0]
    
    return f"{name} is {'Employable 😊' if prediction == 1 else 'Less Employable - Work Hard! 💪'}"
with gr.Blocks() as app:
    gr.Markdown("# Employability Evaluation ")
    
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Name")
            sliders = [
                gr.Slider(1, 5, step=1, label=label) for label in [
                    "General Appearance", "Manner of Speaking", "Physical Condition",
                    "Mental Alertness", "Self Confidence", "Ability to Present Ideas",
                    "Communication Skills"
                ]
            ]
            model_choice = gr.Radio(["Logistic Regression", "Perceptron"], label="Select Model")
            predict_btn = gr.Button("Get Evaluation")
        
        with gr.Column():
            result_output = gr.Textbox(label="Employability Prediction")
    
    predict_btn.click(
        fn=predict_employability,
        inputs=[name] + sliders + [model_choice],
        outputs=[result_output]
    )
app.launch(share=True)
