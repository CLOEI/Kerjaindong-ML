# Kerjaindong-ML

Kerjaindong-ML is a university project that also serves as a practical tool for Kerjaindong.id. It leverages AI-generated datasets and machine learning models to provide instant insights into project requirements.

## Objective

The goal of this project is to:
- Help **users** instantly estimate the approximate budget and timeline for their projects based on a supplied project description.
- Assist **service providers** in understanding the purpose of the project, whether it requires hosting, web design, and other key details.

## Features

- **Instant Predictions**:
  - **Estimated Budget**: Predicts the approximate cost of the project.
  - **Estimated Deadline**: Provides an estimated timeline for project completion.
  - **Project Objective**: Identifies the purpose of the project (e.g., E-Commerce, Company Profile).
  - **Additional Requirements**: Determines if the project needs hosting or UI/UX design.
  - **Platform and Industry**: Predicts the platform (e.g., Web, Mobile) and industry (e.g., Retail, Healthcare).

- **AI-Generated Dataset**:
  - The dataset is generated using the latest Claude model, ensuring realistic and diverse project descriptions.

## Machine Learning Models

The project uses two machine learning models:
1. **Regression**:
   - Model: `RandomForestRegressor`
   - Purpose: Predicts the estimated budget and deadline.
2. **Classification**:
   - Model: `LogisticRegression`
   - Purpose: Predicts categorical outputs such as project objective, platform, industry, and additional requirements.

### Why Random Forest and Logistic Regression?
- After extensive testing, `RandomForestRegressor` provided better results for regression tasks compared to alternatives like XGBoost.
- `LogisticRegression` was chosen for classification due to its simplicity and effectiveness for this dataset.

## How It Works

1. **Input**: The user provides a project description.
2. **Processing**:
   - The description is vectorized using a `TfidfVectorizer`.
   - Predictions are made using the regression and classification models.
3. **Output**: The system returns:
   - Estimated budget and deadline.
   - Project objective, platform, industry, and additional requirements.

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Required Python packages (listed in `requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/cloei/Kerjaindong-ML.git
   cd Kerjaindong-ML