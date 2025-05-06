import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from shap import Explanation
from sympy import false
from torch.optim import Adam
import numpy as np

from sklearn.preprocessing import StandardScaler

from tabulate import tabulate

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.random import seed
import shap



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



df = pd.read_csv('./data/student_habits_performance.csv', na_values=[''], keep_default_na=False)
L.seed_everything(6) #44 is good 6 is better

categorical_columns = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality', 'extracurricular_participation', 'exercise_frequency']
continuous_features = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
                       'attendance_percentage', 'sleep_hours', 'mental_health_rating']

input_values = df[['age', 'gender', 'study_hours_per_day', 'social_media_hours', 'netflix_hours', 'part_time_job', 'attendance_percentage',
                   'sleep_hours', 'diet_quality','exercise_frequency', 'parental_education_level', 'internet_quality',
                   'mental_health_rating', 'extracurricular_participation']]

label_values = df['exam_score'] #observed values


# print_all_tabulated_columns(input_values) TODO reactivate somehow later

# ----- Changing categorical columns to numerical values -----
input_values = factorize_columns(input_values)
X = input_values
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[continuous_features] = scaler.fit_transform(X[continuous_features])
scaled_input = pd.DataFrame(X_scaled, columns=input_values.columns)

# print(tabulate(input_values.head(), headers='keys', tablefmt='psql'))
# print(tabulate(scaled_input.head(), headers='keys', tablefmt='psql'))

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

trainer = L.Trainer(max_epochs=100, callbacks=[early_stop], gradient_clip_val=0.5, enable_progress_bar=False) #TODO change back to 100 AND REMOVE DEBUGGER
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)

trainer.test(model, dataloaders=test_dataloader)


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
