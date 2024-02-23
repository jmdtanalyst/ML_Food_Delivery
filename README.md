# Machine Learning Food Delivery Performance Indicator

**Unlocking insights into the factors that influences Food Delivery Time.**

You can test this running project here:[(http://mldelivery.jmcloudpro.com:82/](http://mldelivery.jmcloudpro.com:82/).

## Overview

#### Life cycle of Machine learning Project

- Understanding the Problem Statement
- Data Collection
- Data Checks to perform
- Exploratory data analysis
- Data Pre-Processing
- Model Training
- Choose best model

### 1) Problem statement

- This project aim to predict the delivery time based some caracteristcs.

### 2) Data Collection

- Dataset Source -https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset

### 3) About Dataset

Food delivery is a courier service in which a restaurant, store, or independent food-delivery company delivers food to a customer. An order is typically made either through a restaurant or grocer's website or mobile app, or through a food ordering company. The delivered items can include entrees, sides, drinks, desserts, or grocery items and are typically delivered in boxes or bags. The delivery person will normally drive a car, but in bigger cities where homes and restaurants are closer together, they may use bikes or motorized scooters.

- **Exploring the Impact of Factors on Food Delivery Time:** This project delves into the realm of machine learning to uncover how various factors, such as Vehicle type, wheather condictions, Time, Type of Order, impact in Delivery time.

- **Key Features:**
  - Exploratory Data Analysis (EDA) to uncover patterns and relationships within the dataset.
  - Model Training and Evaluation using a diverse range of machine learning algorithms.
  - Hyperparameter Tuning to optimize model performance.
  - Docker Image for seamless reproducibility and deployment.

## Project Structure

- **data/:** Houses the raw dataset and any preprocessed data.
- **models/:** Stores trained machine learning models.
- **notebooks/:** Includes Jupyter notebooks for data exploration, model training, and analysis.
- **src/:** Contains Python scripts for data preprocessing, model training, and evaluation.
- **Dockerfile:** Provides instructions for building the Docker image.

## Getting Started

1. **Clone the Repository:**

```bash
  git clone https://github.com/jmdtanalyst/ML_Food_Delivery.git
```

2. **Build the Docker Image**

```bash
docker build -t ml_food_delivery .

```

3. **Run the Docker Container**

```bash
docker run -p 80:80 ml_food_delivery

```

### Usage

    Explore Notebooks: Dive into detailed analysis and model development within the notebooks.
    Utilize Scripts: Use Python scripts in src/ for data preprocessing and model training.
    Experiment with Hyperparameters: Fine-tune model performance by adjusting hyperparameters.
    Deploy Docker Image: Utilize the Docker image for production use.

### Contributions

Contributions are welcome!

### License

This project is licensed under the MIT License. See the LICENSE file for details.

Let's collaborate to predict academic success and empower student achievement!
