# Explainable Mental Health Analysis (Hinglish + MuRIL)

### 🌟 Project Overview
This project addresses the critical gap in mental health screening for code-mixed Indian languages. While standard NLP models excel in pure English, they often fail to capture the emotional nuances of **Hinglish** (Hindi mixed with English, written in Roman script). 

Our research develops a fine-tuned **MuRIL (Multilingual Representations for Indian Languages)** framework to detect emotional distress (Anxiety, Depression, Stress, Normal, and Suicidal Ideation) with high precision and transparency.

### 🚀 Key Highlights
- **Specialized for Hinglish:** Leverages MuRIL's deep understanding of Indian linguistic context.
- **Safety-First Calibration:** Implemented a custom **0.75 probability threshold** for high-risk categories, achieving **87% Precision in Suicidal Ideation detection**.
- **Explainable AI (XAI):** Integrated **SHAP (Shapley Additive exPlanations)** to provide token-level transparency, allowing clinicians to see *which words* triggered a specific risk assessment.
- **Academic Validation:** This research has been officially **published and presented** at [ICIDSSD 2026,Jamia Hamdard].

### 🔗 Live Links
- **Interactive Dashboard:** [https://jyotisangam-hinglish-mental-health-dashboard.hf.space]
- **Published Research Paper:** [https://drive.google.com/file/d/1eELKmIzR7_PJFu3-YnabiCpdOSRYAFId/view?usp=drivesdk]

### 🛠️ Tech Stack
- **Model:** Fine-tuned MuRIL (Bidirectional Transformer)
- **Frameworks:** PyTorch, Transformers (Hugging Face)
- **XAI Tool:** SHAP
- **Deployment:** Streamlit

### 📁 Project Structure
- `app.py`: Main Streamlit dashboard code.
- `requirements.txt`: List of necessary Python libraries.
- `notebooks/`: Jupyter notebooks detailing the training and fine-tuning process.
- `documents/`: Copy of the published research paper and presentation.

### 🔧 Installation & Local Setup
To run this project locally, clone the repository and install the dependencies:
```bash
git clone [https://github.com/jyotisangam/muril-hinglish-mental-health-detection-analysis.git](https://github.com/jyotisangam/muril-hinglish-mental-health-detection-analysis.git)
cd [muril-hinglish-mental-health-detection-analysis]
pip install -r requirements.txt
streamlit run app.py
