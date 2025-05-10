import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from shap import Explanation
from torch.optim import Adam
import warnings
import math

from sklearn.preprocessing import StandardScaler
import ipywidgets as widgets
from IPython.display import display

from tabulate import tabulate

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
import shap


CATEGORICAL_FEATURES = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality', 'extracurricular_participation', 'exercise_frequency']
CONTINUOUS_FEATURES = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
                       'attendance_percentage', 'sleep_hours', 'mental_health_rating']

warnings.filterwarnings("ignore", message="The '.*_dataloader' does not have many workers.*")
warnings.filterwarnings("ignore", message="The number of training batches.*log_every_n_steps.*")
warnings.filterwarnings("ignore", message="Using default `ModelCheckpoint`. Consider installing `litmodels` package to enable `LitModelCheckpoint` for automatic upload to the Lightning model registry.")


class MultiInputSingleOutputModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.training_losses = []
        self.validation_losses = []
        self.linear1 = nn.Linear(14, 32)
        self.dropout1 = nn.Dropout(0.15)

        self.linear2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.15)

        self.linear3 = nn.Linear(16, 1)
        self.loss = nn.MSELoss()

    def forward(self, input):
        x = F.relu(self.linear1(input))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)
        x = self.linear3(x)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.005) # Slightly higher learning rate
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        self.training_losses.append(loss.item())
        # self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.validation_losses.append(loss.item())
        self.log('val_loss', loss, prog_bar=True)  # this must match monitor='val_loss'
        return loss


def tabulate_column_uniques(column: pd.Series):
    unique_vals, encoded_vals = pd.factorize(column)
    table_data = [[i, val] for i, val in enumerate(encoded_vals)]
    friendly_names = {
        "gender": "Gender",
        "part_time_job": "Part-time Job",
        "diet_quality": "Diet Quality",
        "parental_education_level": "Parental Education Level",
        "internet_quality": "Internet Quality",
        "extracurricular_participation": "Extracurricular Participation"
    }
    print(tabulate(table_data, headers=['Value', friendly_names[column.name]], tablefmt="psql"), end='\n')

def print_all_tabulated_columns(df: pd.DataFrame):
    for column in ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality', 'extracurricular_participation']:
        tabulate_column_uniques(input_values[column])

def factorize_columns(df: pd.DataFrame):
    df_copy = df.copy()
    for column in ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality', 'extracurricular_participation']:
        df_copy[column] = df_copy[column].factorize()[0].astype('float32')
    return df_copy

def initialize_sliders():
    sliders = {
        'age': widgets.IntSlider(description='Age', min=17, max=24, value=20),
        'gender': widgets.Dropdown(description='Gender', options=['Male', 'Female', 'Other'], value='Male'),
        'study_hours_per_day': widgets.FloatSlider(description='Study Hours', min=0, max=8.3, step=0.1, value=2.5),
        'social_media_hours': widgets.FloatSlider(description='Social Media Hours', min=0, max=6.2, step=0.1, value=0.0),
        'netflix_hours': widgets.FloatSlider(description='Netflix Hours', min=0, max=5.5, step=0.1, value=0.0),
        'part_time_job': widgets.Dropdown(description='Part-time Job', options=['Yes', 'No'], value='No'),
        'attendance_percentage': widgets.FloatSlider(description='Attendance Percentage', min=56, max=100, step=0.1, value=84.0),
        'sleep_hours': widgets.FloatSlider(description='Sleep Hours', min=3.5, max=10, step=0.5, value=7.0),
        'diet_quality': widgets.Dropdown(description='Diet Quality', options=['Poor', 'Fair', 'Good'], value='Fair'),
        'exercise_frequency': widgets.IntSlider(description='Exercise Frequency', min=0, max=7, step=1, value=0),
        'parental_education_level': widgets.Dropdown(description='Parental Education Level', options=['None', 'High School', 'Bachelor', 'Master'], value='None'),
        'internet_quality': widgets.Dropdown(description='Internet Quality', options=['Poor', 'Average', 'Good'], value='Average'),
        'mental_health_rating': widgets.IntSlider(description='Mental Health Rating', min=1, max=10, step=1, value=5),
        'extracurricular_participation': widgets.Dropdown(description='Extracurricular Participation', options=['Yes', 'No'], value='No')
    }

    predict_button = widgets.Button(description='Predict Exam Score')
    output = widgets.Output()

    def on_button_click(b):
        with output:
            output.clear_output()
            # Gather all values into a dictionary
            values = {name: widget.value for name, widget in sliders.items()}

            # Convert to DataFrame
            input_df = pd.DataFrame([values])

            # Apply the same preprocessing as training data
            X = factorize_columns(input_df)
            X[CONTINUOUS_FEATURES] = scaler.transform(X[CONTINUOUS_FEATURES])

            # Convert to tensor and get prediction
            input_tensor = torch.tensor(X.values, dtype=torch.float32)

            model.eval()
            with torch.no_grad():
                prediction = model(input_tensor)
                clamped_prediction = prediction.clamp(min=0.0, max=100.0)
                print(f"Predicted Exam Score: {clamped_prediction.item():.2f}")

    predict_button.on_click(on_button_click)

    for widget in sliders.values():
        display(widget)
    display(predict_button)
    display(output)

    return sliders


df = pd.read_csv('./data/student_habits_performance.csv', na_values=[''], keep_default_na=False)
L.seed_everything(6)




input_values = df[['age', 'gender', 'study_hours_per_day', 'social_media_hours', 'netflix_hours', 'part_time_job', 'attendance_percentage',
                   'sleep_hours', 'diet_quality','exercise_frequency', 'parental_education_level', 'internet_quality',
                   'mental_health_rating', 'extracurricular_participation']]

label_values = df['exam_score'] #observed values


# ----- Changing categorical columns to numerical values -----
X = factorize_columns(input_values)
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[CONTINUOUS_FEATURES] = scaler.fit_transform(X[CONTINUOUS_FEATURES])
scaled_input = pd.DataFrame(X_scaled, columns=input_values.columns)


input_train, input_test, label_train, label_test = train_test_split(scaled_input, label_values, test_size=0.25) #splitting the data into train and test sets
input_train, input_val, label_train, label_val = train_test_split(input_train, label_train, test_size=0.1)

input_train_tensors = torch.tensor(input_train.values, dtype=torch.float32) #Converting input train data into tensors. We use .values since torch.tensor() doesn't like DataFrames
input_test_tensors = torch.tensor(input_test.values, dtype=torch.float32) #converting input test data into tensors
input_val_tensors = torch.tensor(input_val.values, dtype=torch.float32) #converting input validation data into tensors

label_train_tensors = torch.tensor(label_train.values, dtype=torch.float32).view(-1, 1)
label_test_tensors = torch.tensor(label_test.values, dtype=torch.float32).view(-1, 1)
label_val_tensors = torch.tensor(label_val.values, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(input_train_tensors, label_train_tensors) #converting input train data and labels into a dataset
train_dataloader = DataLoader(train_dataset, batch_size=16) #we convert to a DataLoader to make it easier to iterate over batches

test_dataset = TensorDataset(input_test_tensors, label_test_tensors) #same as above, but for test data
test_dataloader = DataLoader(test_dataset, batch_size=16)

validation_dataset = TensorDataset(input_val_tensors, label_val_tensors)
validation_dataloader = DataLoader(validation_dataset, batch_size=8)


model = MultiInputSingleOutputModel()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
    min_delta=1e-4
)

trainer = L.Trainer(max_epochs=100, callbacks=[early_stop], gradient_clip_val=0.5, enable_progress_bar=False)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

test_results = trainer.test(model, dataloaders=test_dataloader)
test_loss = math.sqrt(test_results[0]['test_loss'])

print(f"Test Loss: {test_loss:.2f}")


explainer = shap.DeepExplainer(model, input_train_tensors)

explanation = explainer.shap_values(input_train_tensors, check_additivity=False).squeeze()


shap_explanation = Explanation(explanation,
                             feature_names=input_values.columns,
                             data=input_train_tensors.numpy())  # Add the background data
shap.plots.beeswarm(shap_explanation,
                    max_display=14,  # Show all features
                    plot_size=(12,8),
                    show=False)
plt.subplots_adjust(left=0.25)  # Increase left margin for labels
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(model.training_losses, label='Training Loss')
plt.plot(model.validation_losses, label='Validation Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    predictions = model(input_test_tensors).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(label_test_tensors.numpy(), predictions, alpha=0.5)
plt.plot([label_test_tensors.min(), label_test_tensors.max()],
         [label_test_tensors.min(), label_test_tensors.max()],
         'r--', lw=2)
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Actual vs Predicted Exam Scores')
plt.show()

sliders = initialize_sliders()