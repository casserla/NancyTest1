#FIXED IN VISUAL STUDIO CODE, JUST TAKING A HOT MINUTE TO SAVE, #addition of the shapefile path gui.
import os
import sqlite3
import numpy as np
import tensorflow as tf
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import time
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tkinter as tk
from tkinter import filedialog, messagebox
#shapefile_path = '/content/sample_data/GWS_Merge_With_Physics.shp'
#gdf = gpd.read_file(shapefile_path, encoding='ISO-8859-1', low_memory=False)


def get_shapefile_path():
    def select_reference_file():
        filepath = filedialog.askopenfilename(
            title="Select your reference shapefile",
            filetypes=[("Shapefiles", "*.shp")]
        )
        if filepath:
            root.quit()
            root.destroy()
            main_callback(filepath)

    def select_gws_standard():
        root.quit()
        root.destroy()
        default_path = os.path.join(current_directory, 'GWS_Merge_With_Physics.shp')
        main_callback(default_path)

    def main_callback(path):
        global shapefile_path
        shapefile_path = path
        print(f"Selected shapefile: {shapefile_path}")

    root = tk.Tk()
    root.title("Select Shapefile Source")
    root.geometry("400x200")

    label = tk.Label(root, text="Would you like to use a reference file, or GWSStandard?", font=("Helvetica", 12))
    label.pack(pady=20)

    ref_button = tk.Button(root, text="Reference File", command=select_reference_file, width=20)
    ref_button.pack(pady=5)

    gws_button = tk.Button(root, text="GWSStandard", command=select_gws_standard, width=20)
    gws_button.pack(pady=5)

    root.mainloop()

# Set the working directory
current_directory = os.getcwd()
print("Current Directory location:", current_directory)

# Prompt for shapefile selection
get_shapefile_path()

# Proceed with loading the shapefile
gdf = gpd.read_file(shapefile_path, encoding='ISO-8859-1', low_memory=False)
print(f"Shapefile loaded from: {shapefile_path}")


# # Get the current working directory (directory of the script)
# current_directory = os.getcwd()
# print ("Current Directory location:",current_directory)

# # Define the relative path to the shapefile in your project
# shapefile_filename = 'GWS_Merge_With_Physics.shp'

# # Combine the current directory with the relative path
# shapefile_path = os.path.join(current_directory, shapefile_filename)

# # Read the shapefile using geopandas
# gdf = gpd.read_file(shapefile_path, encoding='ISO-8859-1', low_memory=False)

# print(f"Shapefile loaded from: {shapefile_path}")

#DATA PRE-PROCESSING
columns_to_normalize = ['Gamma_CPS', 'CntTimeS', 'BKGD', 'r']
def normalize_columns(dataframe, columns):
    if not all(col in dataframe.columns for col in columns):
        raise ValueError("Some columns do not exist in the DataFrame.")
    if dataframe[columns].isnull().any().any():
        raise ValueError("DataFrame contains NaN values in columns to be normalized.")
    if not all(np.issubdtype(dataframe[col].dtype, np.number) for col in columns):
        raise ValueError("Some columns are not numeric.")
    scaler = MinMaxScaler()
    dataframe[columns] = scaler.fit_transform(dataframe[columns])
    print("Normalization complete for columns:", columns)
    return dataframe
gdf = normalize_columns(gdf, columns_to_normalize)


acquisition_time = 1
background_count_rate_bq = 79.28
background_count_rate_counts = background_count_rate_bq * 60
bkgd_std = np.std(background_count_rate_counts)
alarm_level_factor = 3
alarm_level = alarm_level_factor * bkgd_std
Pdesired = 0.95
activity = 5.2
branch = gdf['branch'].mean() #0.8519
mu_a_soil = 0.087
soil_density = 1.38
mu_a = gdf['mu_a'].mean() #0.02
r = gdf['r'].mean()  #.3048
air_density = 0.0294
v = 1
v = tf.cast(v, dtype=tf.float32)
N_theta = 0.09

#Interpolating the efficiency to specific angles. So, if we're considering a detector at an angle that’s not listed, we could use interpolation to estimate the relative efficiency at that angle
#x: The array of new angles where we want to interpolate the efficiency
#xp: The known angles in the angles array
#fp: The corresponding relative efficiencies from rel_eff_data
angles = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
rel_eff_data = np.array([0.09, 0.24, 0.39, 0.53, 0.64, 0.74, 0.83, 0.91, 0.97, 1.0, 0.97, 0.90, 0.81, 0.73, 0.62, 0.51, 0.38, 0.25, 0.11])

#defines specific angles for interpolation
specific_angles = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])

#interpolates the relative efficiency values at the specific angles
interpolated_efficiency = np.interp(specific_angles, angles, rel_eff_data)

#prints the interpolated relative efficiency values u00B0 is the degree symbol
for angle, efficiency in zip(specific_angles, interpolated_efficiency):
    print(f"Angle: {angle}u00B0 -> Interpolated Efficiency: {efficiency:.2f}")

angles_rad = np.deg2rad(angles)
N_90 = rel_eff_data[angles.tolist().index(90)]
if N_90 == 0:
    raise ValueError("N_90 cannot be zero.")
delta_theta = rel_eff_data / N_90

angles_train, angles_validation, rel_eff_train, rel_eff_validation = train_test_split(
    angles_rad, rel_eff_data, test_size=0.2, random_state=42
)

input_train = tf.convert_to_tensor(angles_train[:, np.newaxis], dtype=tf.float32)
rel_eff_train = tf.convert_to_tensor(rel_eff_train[:, np.newaxis], dtype=tf.float32)

input_validation = tf.convert_to_tensor(angles_validation[:, np.newaxis], dtype=tf.float32)
rel_eff_validation = tf.convert_to_tensor(rel_eff_validation[:, np.newaxis], dtype=tf.float32)

#Hyperparameter Tuning
initial_learning_rate = 1e-3  #increase for faster convergence
batch_size = 64
physics_loss_weight = 0.7 #increase to emphasize physics constraints
early_stopping_patience = 80  #allows for more training time
weight_decay = 1e-6  #reduces to prevent excessive regularization
dropout = 0.2  #reduce to prevent underfitting

#Define PINN Model
class EnhancedPINN(torch.nn.Module):
    def __init__(self):
        super(EnhancedPINN, self).__init__()
        self.fc1 = torch.nn.Linear(4, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

N_90_value = rel_eff_data[angles == 90][0]
delta_theta = rel_eff_data / N_90_value

params = (Pdesired, activity, branch, mu_a, air_density, r, v, acquisition_time, background_count_rate_counts, alarm_level, N_90_value)
print(f"Params before calling physics_loss: {params}")
print(f"Length of params before unpacking: {len(params)}")

def physics_loss(model, angles, rel_eff_data, params):
    Pdesired, activity, branch, mu_a_soil, soil_density, r, v, acquisition_time, background_count_rate, alarm_level, N_90_value = params
    print(f"Params received in physics_loss: {params}")
    r_t = tf.sqrt(r**2 + (v * acquisition_time)**2)
    angles_float = tf.convert_to_tensor(angles[:, np.newaxis], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(angles_float)
        rel_eff = tf.convert_to_tensor(model(angles_float), dtype=tf.float32)
        fluence_rate = (activity * branch * rel_eff * tf.exp(-mu_a_soil * r_t)) / (4 * np.pi * r_t**2)
        detection_probability = 1 - tf.math.exp(-fluence_rate * r_t * soil_density)
    B = background_count_rate * acquisition_time
    B_tensor = tf.convert_to_tensor(B, dtype=tf.float32)
    false_positive_prob = 1 - tf.math.exp(-B_tensor)
    loss_detection = tf.reduce_mean(tf.square(detection_probability - Pdesired))
    loss_false_positive = tf.reduce_mean(tf.square(false_positive_prob - alarm_level))
    loss = loss_detection + loss_false_positive
    return loss

def calculate_linear_attenuation_coefficient(mu_a, air_density):
    return mu_a * air_density

def speed_adjustment_factor(speed, max_speed=10):
    return max(0.1, 1 / (1 + (speed / max_speed)))

def acquisition_time_adjustment_factor(acquisition_time, max_time=5):
    return min(1.0, acquisition_time / (max_time + acquisition_time))

def adjust_background_count_rate(background_count_rate_bq, alarm_threshold_factor=1.5):
    min_rate = 0.0
    max_rate = 8000.0

    #convert Bq to CPM
    background_count_rate_cpm = background_count_rate_bq * 60

    #apply alarm threshold factor
    adjusted_rate = min(max(background_count_rate_cpm, min_rate), max_rate) * alarm_threshold_factor

    return min(adjusted_rate, max_rate)

background_count_rate_bq = 79.28
adjusted_background_rate = adjust_background_count_rate(background_count_rate_bq, alarm_threshold_factor=1.5)
print("Adjusted Background Count Rate:", adjusted_background_rate)

#setting up the framework for real-time device capture with relation to PINN model
#creates device object
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#instantiate the model and move it to the device graphic/central processing unit (GPU or CPU)
#CPU - processing time from a computer standpoint (binary, 0 or 1)
#GPU - processing time that is understandable to the human eye (sprectra)
model = EnhancedPINN().to(device)

#function to measure inference time
def measure_inference_time(model, input_data, runs=100):
    model.eval()
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)  #moves input data to the device

    #warms-up inference to prevent timing the first execution
    with torch.no_grad():
        _ = model(input_tensor)

    #measures the inference time over multiple runs
    start_time = time.time()
    for _ in range(runs):
        with torch.no_grad():  #disables gradient calculation for inference
            _ = model(input_tensor)
    end_time = time.time()

    #calculates average inference time
    avg_time_per_run = (end_time - start_time) / runs
    print(f"Average Inference Time on {device}: {avg_time_per_run:.6f} seconds")

    return avg_time_per_run

def calculate_immobile_MDD(model, angles, params, tolerance=1e-3):
    Pdesired, activity, branch, mu_a_soil, soil_density, r, v, acquisition_time, background_count_rate, alarm_level, N_90_value = params
    adjusted_background_count_rate = adjust_background_count_rate(background_count_rate)

    R_min = 0
    R_max = 10000

    #initializes angles_rad here
    angles_rad = np.deg2rad(angles)

    while (R_max - R_min) > tolerance:
        R_fix = (R_min + R_max) / 2
        current_params = (
            Pdesired, activity, branch, soil_density, v, acquisition_time, adjusted_background_count_rate,
            alarm_level, R_fix)

        #converts angles to PyTorch tensor
        angles_rad_tensor = torch.tensor(np.deg2rad(angles), dtype=torch.float32).to(device)
        angles_rad_tensor = angles_rad_tensor.clone().detach().to(device)

        #reshapes to match the expected input shape
        angles_rad_tensor = angles_rad_tensor.view(-1, 1)
        angles_rad_tensor = angles_rad_tensor.repeat(1, 4)

        #passes the PyTorch tensor to the model
        model_output = model(angles_rad_tensor)

        #assigns model_output to rel_eff
        rel_eff = tf.convert_to_tensor(model_output.cpu().detach().numpy(), dtype=tf.float32)  #convert model_output to TensorFlow tensor
        #reshape rel_eff
        rel_eff = tf.reshape(rel_eff, [-1, 1])

        fluence_rate = activity * branch * rel_eff
        detection_probability = 1 - tf.math.exp(-fluence_rate * R_fix * N_theta * mu_a_soil * soil_density / adjusted_background_count_rate)
        mean_detection_probability = tf.reduce_mean(detection_probability).numpy()

        print(
            f"R_min: {R_min}, R_max: {R_max}, R_fix: {R_fix}, Mean Detection Probability: {mean_detection_probability}")

        if mean_detection_probability >= Pdesired:
            R_max = R_fix
        else:
            R_min = R_fix

    return R_fix

def calculate_mobile_MDD(model, angles, params, tolerance=1e-3):
    Pdesired, activity, branch, mu_a_soil, soil_density, r, v, acquisition_time, background_count_rate_counts, alarm_level, N_90_value = params
    adjusted_background_count_rate = adjust_background_count_rate(background_count_rate_counts)

    R_min = 0
    R_fix = calculate_immobile_MDD(model, angles, params, tolerance) #initial doubling of R_fix
    R_max = 2 * R_fix  #initial doubling of R_fix

    while (R_max - R_min) > tolerance:
        R_test = (R_max + R_min) / 2
        total_detection_prob = 0.0
        distance_traveled = 0.0
        time_elapsed = 0.0
        current_distance_per_step = v * acquisition_time
        speed_factor = 1 / (1 + (v / 15)**2)  #adjusting constant scaling factor as needed

        step_count = 0
        while distance_traveled < R_test:
            #converts angles_rad to a PyTorch tensor inside the loop
            angles_rad_tensor = torch.tensor(angles_rad[:, np.newaxis], dtype=torch.float32).to(device)

            #reshapes to match the expected input shape
            angles_rad_tensor = angles_rad_tensor.view(-1, 1)
            angles_rad_tensor = angles_rad_tensor.repeat(1, 4)

            #passes the PyTorch tensor to the model
            model_output = model(angles_rad_tensor)

            #gets the PyTorch tensor output
            rel_eff = model_output

            #convert PyTorch tensor to TensorFlow tensor
            rel_eff = tf.convert_to_tensor(rel_eff.cpu().detach().numpy(), dtype=tf.float32)
            #reshape rel_eff
            rel_eff = tf.reshape(rel_eff, [-1, 1])

            fluence_rate = activity * branch * rel_eff
            detection_prob = 1 - tf.math.exp((-fluence_rate * R_fix * N_theta * mu_a_soil * soil_density / adjusted_background_count_rate * v)) * speed_factor
            total_detection_prob += detection_prob
            distance_traveled += current_distance_per_step

            step_count += 1
            if step_count > 10000:  #adjusting constant scaling factor as needed
                print(f"Breaking loop at step {step_count} to prevent infinite loop.")
                break

        mean_total_detection_prob = tf.reduce_mean(total_detection_prob).numpy()
        print(f"R_test: {R_test}, Mean Total Detection Probability: {mean_total_detection_prob}, Speed: {v}, Steps: {step_count}")

        if mean_total_detection_prob >= Pdesired:
            R_max = R_test
        else:
            R_min = R_test

    return R_test

def calculate_mdd(model, angles, params, is_mobile, tolerance):
    if is_mobile:
        return calculate_mobile_MDD(model, angles, params, tolerance)
    else:
        return calculate_immobile_MDD(model, angles, params, tolerance)


#compile the 'Enhanced' PINN model
model = EnhancedPINN()

#learning rate schedule
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
early_stopping = EarlyStopping(patience=25, verbose=True)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=2000, #adjust for slower decay
    decay_rate=0.95
)

#comparisions between the losses with time (creating some sort of correction factor between PINN and device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

angles = np.linspace(0, 90, 19)

#converts angles to PyTorch tensor
angles_rad = torch.from_numpy(np.deg2rad(angles)[:, np.newaxis]).to(torch.float32).to(device)

#expands angles_rad to match the shape
angles_rad = angles_rad.expand(-1, 4)

#makes sure it's moved to the device with no gradient tracking
angles_rad = angles_rad.clone().detach()

model.to(device)

rel_eff_data = torch.tensor(rel_eff_data, dtype=torch.float32).to(device)

#modify the physics loss function for weighting
def physics_loss(model, angles_rad, rel_eff_data, params):
    #calculate the predicted efficiencies
    predictions = model(angles_rad)

    #make sure params are in tensor format
    params = torch.tensor(params, dtype=torch.float32).to(device)

    #extract necessary parameters
    r = params[0]  #distance to the source (m)
    mu_a = params[1]  #attenuation coefficient (1/m)

    #apply the physical attenuation law (exponential decay based on distance)
    expected_efficiency = torch.exp(-mu_a * r) * (1 / (1 + (r/10)**2))  #distance-based scaling

    #calculate loss based on how well the model's predictions match the expected physical behavior
    attenuation_loss = torch.mean(torch.abs(predictions - expected_efficiency))

    #inverse square law penalty
    inverse_square_penalty = torch.mean(torch.abs(predictions - (1 / (r ** 2))))

    #angular dependence
    #reshape angles_rad to match predictions shape
    angles_rad = angles_rad[:, 0].repeat(1, predictions.shape[1]).view(predictions.shape)
    angular_penalty = torch.mean(torch.abs(predictions - (torch.cos(angles_rad) + 0.5*torch.sin(angles_rad))))

    #adjust total physics loss with weighted terms
    attenuation_weight = 0.7  #increased weight for attenuation loss
    inverse_square_weight = 0.3  #weight for the inverse square penalty

    total_physics_loss = attenuation_weight * attenuation_loss + inverse_square_weight * inverse_square_penalty

    return total_physics_loss

#modified total loss function with weighted physics loss
def total_loss(pred, true_val, angles, lower_bound, upper_bound):
    #calculate data loss (mean squared error)
    data_loss = torch.mean((pred - true_val) ** 2)
    normalized_data_loss = data_loss / (data_loss + 1e-8)  #normalize data loss to prevent large values

    #calculate the physics-informed loss
    physics_loss_val = physics_loss(pred, true_val, angles, params)
    normalized_physics_loss = physics_loss_val / (physics_loss_val + 1e-8)  #normalize physics loss to avoid extreme values

    #boundary loss (ensure predictions stay within limits)
    bound_loss = boundary_loss(pred, lower_bound, upper_bound)

    #combine all losses, adjust the weight of physics loss
    total_loss = normalized_data_loss + normalized_physics_loss + bound_loss
    return total_loss

#TRAINING STEPS
#define normalization factors outside the train_step function
data_loss_normalization_factor = 1.0  #no normalization for data loss
physics_loss_normalization_factor = 1.0  #no normalization for physics loss

@tf.function
#modified train_step function
#def train_step(input_batch, rel_eff_batch, data_loss_normalization_factor, physics_loss_normalization_factor):
    #with tf.GradientTape() as tape:
        #rel_eff_predicted = model(input_batch)
        #data_loss = torch.mean((rel_eff_predicted - rel_eff_batch)**2)
        #pass N_90_value to physics_loss
        #physics_loss_value = physics_loss(model, input_batch, rel_eff_batch, params)

        #physics_loss_normalized = physics_loss_value / physics_loss_normalization_factor
        #data_loss_normalized = data_loss / data_loss_normalization_factor

        #total_loss = (0.5 * physics_loss_norm) + data_loss_norm

    #gradients = tape.gradient(total_loss, model.trainable_variables)
    #clip gradients before applying to prevent exploding gradients
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    #apply gradients
    #optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #return total_loss

#physical boundary condition, like detection max range
def boundary_loss(pred, max_boundary_value=10000):
    #apply sigmoid to predictions before applying boundary loss
    pred_sigmoid = torch.sigmoid(pred)
    #penalty for exceeding the maximum boundary
    upper_bound_loss = torch.mean(torch.relu(pred_sigmoid - max_boundary_value))
    return upper_bound_loss

#final combined loss function (data + physics + boundary)
def combined_loss(model, input_batch, rel_eff_batch, params, max_boundary_value=10000, physics_loss_weight=1.0):
    data_loss = torch.mean((model(input_batch) - rel_eff_batch)**2)
    physics_loss_value = physics_loss(model, input_batch, rel_eff_batch, params)
    boundary_penalty = boundary_loss(model(input_batch), max_boundary_value)

    total_loss = data_loss + physics_loss_weight * physics_loss_value + boundary_penalty
    return total_loss

#in training loop, replace the previous loss calculation with `combined_loss`
def train_step(input_batch, rel_eff_batch, params, max_boundary_value=10000, physics_loss_weight=1.0):
    with tf.GradientTape() as tape:
        rel_eff_predicted = model(input_batch)

        #use combined loss function
        total_loss = combined_loss(model, input_batch, rel_eff_batch, params, max_boundary_value, physics_loss_weight)

    #convert total_loss to a TensorFlow tensor
    total_loss = tf.convert_to_tensor(total_loss, dtype=tf.float32)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss

#Validation Arrays
train_mae_list = []
train_rmse_list = []
val_mae_list = []
val_rmse_list = []
train_r2_list = []
val_r2_list = []

num_epochs = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#move tensors to CPU before converting to NumPy arrays
input_train_cpu = input_train.cpu().numpy()
rel_eff_train_cpu = rel_eff_train.cpu().numpy()
input_validation_cpu = input_validation.cpu().numpy()
rel_eff_validation_cpu = rel_eff_validation.cpu().numpy()

#convert input_train to a PyTorch tensor
input_train = torch.tensor(input_train_cpu, dtype=torch.float32).to(device)

#in case input_train is not divisible by batch size, make sure it has the correct shape
num_elements = input_train.shape[0]
remainder = num_elements % batch_size
if remainder != 0:
    padding_size = batch_size - remainder
    #create the padding tensor on the same device as input_train
    padding_tensor = torch.zeros(padding_size, input_train.shape[1]).to(device)
    #concatenate the tensors
    padded_input_train = torch.cat([input_train, padding_tensor], dim=0)

input_train = padded_input_train
input_train = input_train.view(-1, 4)  #reshape input_train

#move input_validation to CPU before reshaping
input_validation_cpu = input_validation.cpu().numpy()
#reshape the input_validation_cpu NumPy array
input_validation_reshaped = input_validation_cpu.reshape(-1, 4)
#convert back to a PyTorch tensor if necessary
input_validation = torch.from_numpy(input_validation_reshaped).to(device)

#move rel_eff_train to CPU before converting to NumPy
rel_eff_train_cpu = rel_eff_train.cpu().numpy()

#create the PyTorch tensor from the NumPy array
rel_eff_train = torch.tensor(rel_eff_train_cpu, dtype=torch.float32).to(device)

#check if the input_train and rel_eff_train are on the correct device
print(f"input_train type: {type(input_train)}, shape: {input_train.shape}, device: {input_train.device}")
print(f"rel_eff_train type: {type(rel_eff_train)}, shape: {rel_eff_train.shape}, device: {rel_eff_train.device}")

#initialize data_loss and physics_loss_val to avoid NameError
data_loss = torch.tensor(1.0, device=device)
physics_loss_val = torch.tensor(1.0, device=device)

#define normalization factors outside the train_step function
data_loss_normalization_factor = data_loss.detach()  #normalize by current data loss
physics_loss_normalization_factor = physics_loss_val.detach()  #normalize by current physics loss

#train step function with normalization factors
def train_step(input_batch, rel_eff_batch, data_loss_normalization_factor, physics_loss_normalization_factor):
    #forward pass to get model predictions
    rel_eff_predicted = model(input_batch)

    #compute data loss (Mean Squared Error)
    data_loss = torch.mean((rel_eff_predicted - rel_eff_batch)**2)

    #compute physics loss
    physics_loss_value = physics_loss(model, input_batch, rel_eff_batch, params)

    #normalize the data and physics loss
    physics_loss_normalized = physics_loss_value / physics_loss_normalization_factor
    data_loss_normalized = data_loss / data_loss_normalization_factor

    #compute the total loss (adjust the weight of each component)
    total_loss = (physics_loss_weight * physics_loss_norm) + (1.0 * data_loss_norm) #increased data loss during training

    #backpropagation and optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss

#PINN Training Loop
accumulation_steps = 2
prev_val_loss = float('inf')
patience_count = 0

def normalize_loss(loss, normalization_factor):
    return loss if normalization_factor == 0 else loss / normalization_factor

for epoch in range(num_epochs):
    start_time = time.time()
    num_features = input_train.shape[1]

    #prepare training data
    input_train_batch = input_train[:len(rel_eff_train)].view(len(rel_eff_train), num_features)
    rel_eff_train_batch = rel_eff_train.view(-1, 1)

    #validation data prep
    num_val_samples = len(rel_eff_validation_cpu)
    if input_validation_cpu.ndim == 1:
        input_validation_cpu = input_validation_cpu.reshape(num_val_samples, 1)
    else:
        input_validation_cpu = input_validation_cpu.reshape(num_val_samples, -1)

    rel_eff_validation_batch = torch.tensor(rel_eff_validation_cpu, dtype=torch.float32).view(num_val_samples, 1).to(device)
    val_input_batch = torch.tensor(input_validation_cpu, dtype=torch.float32).to(device)

    #training
    model.train()
    model.zero_grad()

    train_predictions_list = []
    train_targets_list = []
    running_train_loss = 0.0
    running_train_mae = 0.0

    for batch_idx in range(0, len(input_train_batch), accumulation_steps):
        input_batch = input_train_batch[batch_idx: batch_idx + accumulation_steps]
        target_batch = rel_eff_train_batch[batch_idx: batch_idx + accumulation_steps]

        input_batch = input_batch.view(len(target_batch), num_features)
        predictions = model(input_batch)

        train_predictions_list.append(predictions.detach())
        train_targets_list.append(target_batch.detach())

        data_loss = nn.MSELoss()(predictions, target_batch)
        physics_loss_val = physics_loss(model, input_batch, target_batch, params)

        data_loss_norm = normalize_loss(data_loss, data_loss_normalization_factor)
        physics_loss_norm = normalize_loss(physics_loss_val, physics_loss_normalization_factor)

        total_loss = (physics_loss_weight * physics_loss_norm) + data_loss_norm
        total_loss = total_loss / accumulation_steps  # Normalize total loss for accumulation
        total_loss.backward()

        running_train_loss += total_loss.item()
        running_train_mae += mean_absolute_error(target_batch.cpu().numpy(), predictions.detach().cpu().numpy())

        if ((batch_idx // accumulation_steps + 1) % accumulation_steps) == 0:
            optimizer.step()
            model.zero_grad()

            avg_train_loss = running_train_loss / accumulation_steps
            avg_train_mae = running_train_mae / accumulation_steps
            print(f"Epoch {epoch}, Step {batch_idx // accumulation_steps}: Train Loss: {avg_train_loss:.6f}, Train MAE: {avg_train_mae:.6f}")
            running_train_loss = 0.0
            running_train_mae = 0.0

    #overall training metrics
    train_preds_np = torch.cat(train_predictions_list).cpu().numpy()
    train_targets_np = torch.cat(train_targets_list).cpu().numpy()
    train_mae = mean_absolute_error(train_targets_np, train_preds_np)
    train_rmse = np.sqrt(mean_squared_error(train_targets_np, train_preds_np))
    train_r2 = r2_score(train_targets_np, train_preds_np) if len(train_targets_np) > 1 else 0.0

    #validating the model
    model.eval()
    with torch.no_grad():
        if val_input_batch.shape == (4, 1):
            val_input_batch = val_input_batch.repeat(1, 4).view(4, 4)

        val_predictions = model(val_input_batch)

        if val_predictions.shape[0] != rel_eff_validation_batch.shape[0]:
            rel_eff_validation_batch = rel_eff_validation_batch[:val_predictions.shape[0]]

        #compute validation metrics
        val_preds_np = val_predictions.cpu().numpy()
        val_targets_np = rel_eff_validation_batch.cpu().numpy()

        val_mae = mean_absolute_error(val_targets_np, val_preds_np)
        val_rmse = np.sqrt(mean_squared_error(val_targets_np, val_preds_np))
        val_r2 = r2_score(val_targets_np, val_preds_np) if val_targets_np.shape[0] > 1 else 0.0  # Avoid NaN R²

        #compute the training metrics across all batches for the epoch
        all_train_preds_np = torch.cat(train_predictions_list).cpu().numpy()
        all_train_targets_np = torch.cat(train_targets_list).cpu().numpy()

        train_mae = mean_absolute_error(all_train_targets_np, all_train_preds_np)
        train_rmse = np.sqrt(mean_squared_error(all_train_targets_np, all_train_preds_np))
        train_r2 = r2_score(all_train_targets_np, all_train_preds_np) if all_train_targets_np.shape[0] > 1 else 0.0

        #total validation loss
        total_loss_validation = nn.MSELoss()(val_predictions, rel_eff_validation_batch)
        
        #R\u00B2 is R^2
        print(f"Epoch {epoch}: Validation Loss: {total_loss_validation.item():.6f}")
        print(f"Epoch {epoch}: Training MAE: {train_mae:.6f}, Training RMSE: {train_rmse:.6f}, Training R\u00B2: {train_r2:.6f}")
        print(f"Epoch {epoch}: Validation MAE: {val_mae:.6f}, Validation RMSE: {val_rmse:.6f}, Validation R\u00B2: {val_r2:.6f}")

    #early stopping
    if val_rmse > prev_val_loss:
        patience_count += 1
        if patience_count >= early_stopping_patience:
            print("Early stopping triggered.")
            break
    else:
        patience_count = 0
        prev_val_loss = val_rmse

#validation set evaluation
#convert angles_validation and rel_eff_validation to PyTorch tensors
angles_validation_pt = torch.tensor(angles_validation, dtype=torch.float32).to(device)

#move rel_eff_validation to CPU before converting to NumPy
rel_eff_validation_pt = torch.tensor(rel_eff_validation.cpu().numpy(), dtype=torch.float32).to(device)

#ensure angles_validation_pt has the correct shape for the model
angles_validation_pt = angles_validation_pt.view(-1, 1)  #reshape to (num_samples, 1)
angles_validation_pt = angles_validation_pt.repeat(1, 4)  #repeat to match the model's input shape (num_samples, 4)

#calculate physics loss using PyTorch tensors
physics_loss_value_validation = physics_loss(model, angles_validation_pt, rel_eff_validation_pt, params)

#calculate data loss using PyTorch tensors and operations
#pass angles_validation_pt (PyTorch tensor) to the model
rel_eff_validation_predicted = model(angles_validation_pt)
data_loss_value_validation = torch.mean((rel_eff_validation_predicted - rel_eff_validation_pt)**2)

#normalize losses (using PyTorch operations)
physics_loss_value_validation_normalized = physics_loss_value_validation / physics_loss_normalization_factor
data_loss_value_validation_normalized = data_loss_value_validation / data_loss_normalization_factor

#calculate total loss (using PyTorch operations)
total_loss_validation = (physics_loss_weight * physics_loss_value_validation_normalized) + data_loss_value_validation_normalized

#print the total loss (using .item() to get the scalar value from the PyTorch tensor)
print(f"Final Validation Loss: {total_loss_validation.item()}")

#ADDED to measure inference time before computing MDD
#operating outside of the model
class EnhancedPINN(torch.nn.Module):
    def __init__(self):
        super(EnhancedPINN, self).__init__()
        self.fc1 = torch.nn.Linear(4, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#create device object
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#instantiate the model and move it to the device (GPU or CPU)
model = EnhancedPINN().to(device)

#function to measure inference time
def measure_inference_time(model, input_data, runs=100):
    model.eval()
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)  #move input data to the device

    #warms-up inference to prevent timing the first execution
    with torch.no_grad():
        _ = model(input_tensor)

    #measure the inference time over multiple runs
    start_time = time.time()
    for _ in range(runs):
        with torch.no_grad():  #disable gradient calculation for inference
            _ = model(input_tensor)
    end_time = time.time()

    #calculate the average inference time
    avg_time_per_run = (end_time - start_time) / runs
    print(f"Average Inference Time on {device}: {avg_time_per_run:.6f} seconds")

    return avg_time_per_run

is_mobile = True
if is_mobile:
    mdd_mobile = calculate_mdd(model, angles, params, is_mobile=True, tolerance=0.95)
    mdd_mobile_meters = mdd_mobile / 39.37  #converting MDD to meters
    print(f"Maximum Detectable Distance (Mobile): {mdd_mobile} inches ({mdd_mobile_meters:.2f} meters)")
else:
    mdd_immobile = calculate_immobile_MDD(model, angles, params, tolerance=0.95)
    mdd_immobile_meters = mdd_immobile / 39.37
    print(f"Maximum Detectable Distance (Immobile): {mdd_immobile} inches ({mdd_immobile_meters:.2f} meters)")



