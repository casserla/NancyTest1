import numpy as np
import tensorflow as tf
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Origin of files when using google colab
shapefile_path = '/content/sample_data/GWS_Merge_With_Physics.shp' 
gdf = gpd.read_file(shapefile_path, encoding='ISO-8859-1', low_memory=False)

#Data preprocessing

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

angles = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
rel_eff_data = np.array([0.09, 0.24, 0.39, 0.53, 0.64, 0.74, 0.83, 0.91, 0.97, 1.0, 0.97, 0.90, 0.81, 0.73, 0.62, 0.51, 0.38, 0.25, 0.11])

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


#angles defintion and training model
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


#model for PINN
class EnhancedPINN(tf.keras.Model):
    def __init__(self):
        super(EnhancedPINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(100, activation='relu')
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.2)

        self.dense2 = tf.keras.layers.Dense(100, activation='relu')
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.2)

        self.dense3 = tf.keras.layers.Dense(50, activation='relu')
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(0.2)

        self.dense4 = tf.keras.layers.Dense(50, activation='relu')
        self.batch_norm4 = tf.keras.layers.BatchNormalization()
        self.dropout4 = tf.keras.layers.Dropout(0.2)

        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        #reshaped inputs to 2D
        inputs = tf.reshape(inputs, [-1, 1])

        hidden1 = self.dense1(inputs)
        hidden1 = self.batch_norm1(hidden1)
        hidden1 = self.dropout1(hidden1)

        hidden2 = self.dense2(hidden1)
        hidden2 = self.batch_norm2(hidden2)
        hidden2 = self.dropout2(hidden2)

        hidden3 = self.dense3(hidden2)
        hidden3 = self.batch_norm3(hidden3)
        hidden3 = self.dropout3(hidden3)

        hidden4 = self.dense4(hidden3)
        hidden4 = self.batch_norm4(hidden4)
        hidden4 = self.dropout4(hidden4)

        output = self.output_layer(hidden4)
        return output

    #DEFITIONS
    #physics loss
def physics_loss(model, angles, rel_eff_data, params):
    Pdesired, activity, branch, mu_a_soil, soil_density, r, v, acquisition_time, background_count_rate_counts_value, alarm_level, N_90_value = params  # Changed here
    print(f"Params received in physics_loss: {params}")
    r_t = tf.sqrt(r**2 + (v * acquisition_time)**2)
    angles_float = tf.convert_to_tensor(angles[:, np.newaxis], dtype=tf.float32)
    #delta_theta = rel_eff_data / N_90

    with tf.GradientTape() as tape:
        tape.watch(angles_float)
        #rel_eff = model(angles_float)
        #rel_eff = tf.convert_to_tensor(rel_eff, dtype=tf.float32)
        rel_eff = tf.convert_to_tensor(model(angles_float), dtype=tf.float32) #added
        fluence_rate = (activity * branch * rel_eff * tf.exp(-mu_a_soil * r_t)) / (4 * np.pi * r_t**2)
        detection_probability = 1 - tf.math.exp(-fluence_rate * r_t * soil_density)
    B = background_count_rate_counts_value * acquisition_time  # Changed here
    B_tensor = tf.convert_to_tensor(B, dtype=tf.float32)
    false_positive_prob = 1 - tf.math.exp(-B_tensor)
    loss_detection = tf.reduce_mean(tf.square(detection_probability - Pdesired))
    loss_false_positive = tf.reduce_mean(tf.square(false_positive_prob - alarm_level))
    loss = loss_detection + loss_false_positive
    return loss

#linear attenutation 
def calculate_linear_attenuation_coefficient(mu_a, air_density):
    return mu_a * air_density
#background count
def adjust_background_count_rate(background_count_rate_counts, alarm_threshold_factor=1.5):
    min_rate = 0.0
    max_rate = 8000.0
    #adjusting count rate with the alarm threshold factor
    return min(max(background_count_rate_counts, min_rate), max_rate) * alarm_threshold_factor
adjusted_background_rate = adjust_background_count_rate(background_count_rate_counts, alarm_threshold_factor=1.5)
print("Adjusted Background Count Rate:", adjusted_background_rate)

#immobile MDD function
def calculate_immobile_MDD(model, angles, params, tolerance=1e-3):
    Pdesired, activity, branching_ratio, air_density, v, acquisition_time, background_count_rate = params
    adjusted_background_count_rate = adjust_background_count_rate(background_count_rate)

    R_min = 0
    R_max = 10000

    while (R_max - R_min) > tolerance:
        R_fix = (R_min + R_max) / 2
        current_params = (
            Pdesired, activity, branching_ratio, air_density, v, acquisition_time, adjusted_background_count_rate,
            alarm_level, R_fix)

        angles_rad = np.deg2rad(angles)
        angles_rad = tf.convert_to_tensor(angles_rad[:, np.newaxis], dtype=tf.float32)

        model_output = model(angles_rad)

        #make sure model output is a tensor
        if isinstance(model_output, dict) and 'output' in model_output:
            rel_eff = model_output['output']
        elif isinstance(model_output, tf.Tensor):
            rel_eff = model_output
        else:
            raise ValueError("Unexpected format for model output.")

        if rel_eff.shape != (angles_rad.shape[0], 1):
            raise ValueError(f"Unexpected shape for rel_eff: {rel_eff.shape}, expected {(angles_rad.shape[0], 1)}")

        fluence_rate = activity * branching_ratio * rel_eff
        detection_probability = 1 - tf.math.exp(-fluence_rate * R_fix * air_density / adjusted_background_count_rate)
        mean_detection_probability = tf.reduce_mean(detection_probability).numpy()

        print(
            f"R_min: {R_min}, R_max: {R_max}, R_fix: {R_fix}, Mean Detection Probability: {mean_detection_probability}")

        if mean_detection_probability >= Pdesired:
            R_max = R_fix
        else:
            R_min = R_fix

    return R_fix

#mobile MDD function
def calculate_mobile_MDD(model, angles, params, tolerance=1e-3):
    # Adjust unpacking to match the number of elements in 'params'
    Pdesired, activity, branching_ratio, mu_a_soil, air_density, r, v, acquisition_time, background_count_rate_counts, alarm_level, N_90 = params
    adjusted_background_count_rate = adjust_background_count_rate(background_count_rate_counts) # Use background_count_rate_counts here

    R_min = 0
    # Use original params for calculate_immobile_MDD
    R_fix = calculate_immobile_MDD(model, angles, (Pdesired, activity, branch, air_density, v, acquisition_time, background_count_rate_counts), tolerance=tolerance)  #starting point for mobile MDD
    R_max = 2 * R_fix  #initial doubling of R_fix

    while (R_max - R_min) > tolerance:
        R_test = (R_max + R_min) / 2
        total_detection_prob = 0.0
        distance_traveled = 0.0
        current_distance_per_step = v * acquisition_time

        step_count = 0
        while distance_traveled < R_test:
            angle_rad = np.arcsin(distance_traveled / R_test)
            angles_rad = tf.convert_to_tensor([[angle_rad]], dtype=tf.float32)

            model_output = model(angles_rad)

            if isinstance(model_output, dict) and 'output' in model_output:
                rel_eff = model_output['output']
            elif isinstance(model_output, tf.Tensor):
                rel_eff = model_output
            else:
                raise ValueError("Unexpected format for model output.")

            if rel_eff.shape != (angles_rad.shape[0], 1):
                raise ValueError(f"Unexpected shape for rel_eff: {rel_eff.shape}, expected {(angles_rad.shape[0], 1)}")

            # Use activity and branch from params
            fluence_rate = activity * branch * rel_eff
            detection_prob = 1 - tf.math.exp(-fluence_rate * R_test * air_density / adjusted_background_count_rate)
            total_detection_prob += detection_prob
            distance_traveled += current_distance_per_step

            step_count += 1
            if step_count > 10000:  #break if too many steps to prevent infinite loop
                print(f"Breaking loop at step {step_count} to prevent infinite loop.")
                break

        mean_total_detection_prob = tf.reduce_mean(total_detection_prob).numpy()
        print(f"R_test: {R_test}, Mean Total Detection Probability: {mean_total_detection_prob}, Speed: {v}, Steps: {step_count}")

        if mean_total_detection_prob >= Pdesired:
            R_max = R_test
        else:
            R_min = R_test

    return R_test

#compile the 'Enhanced' PINN model
model = EnhancedPINN()

initial_learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96
)

#adjusted loss function and physics loss weight
physics_loss_weight = 0.1  #adjust based on the problem and losses scaling
params = (Pdesired, activity, branch, mu_a_soil, air_density, r, v, acquisition_time, background_count_rate_counts, alarm_level, N_90)

#calculate normalization factors
data_loss_normalization_factor = tf.reduce_mean(tf.square(rel_eff_train))
angles_rad = tf.convert_to_tensor(np.deg2rad(angles)[:, np.newaxis], dtype=tf.float32)
rel_eff_initial = model(angles_rad)
physics_loss_initial = physics_loss(model, angles, rel_eff_data, params)
physics_loss_normalization_factor = tf.reduce_mean(tf.square(physics_loss_initial))

#early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

num_epochs = 100
batch_size = 32

@tf.function
def train_step(input_batch, rel_eff_batch):
    with tf.GradientTape() as tape:
        rel_eff_predicted = model(input_batch)
        data_loss = tf.reduce_mean(tf.square(rel_eff_predicted - rel_eff_batch))
        physics_loss_value = physics_loss(model, input_batch, rel_eff_batch, params)

        data_loss_normalized = data_loss / data_loss_normalization_factor
        physics_loss_normalized = physics_loss_value / physics_loss_normalization_factor

        total_loss = (physics_loss_weight * physics_loss_normalized) + data_loss_normalized

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss

train_mae_list = []
train_rmse_list = []
val_mae_list = []
val_rmse_list = []
train_r2_list = []
val_r2_list = []

for epoch in range(num_epochs):
    #batch the training data
    indices = tf.range(start=0, limit=tf.shape(input_train)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    input_train_shuffled = tf.gather(input_train, shuffled_indices)
    rel_eff_train_shuffled = tf.gather(rel_eff_train, shuffled_indices)

    for i in range(0, len(input_train), batch_size):
        input_batch = input_train_shuffled[i:i+batch_size]
        rel_eff_batch = rel_eff_train_shuffled[i:i+batch_size]

        total_loss = train_step(input_batch, rel_eff_batch)

    #loss calculation validation
    rel_eff_validation_predicted = model(input_validation)
    data_loss_validation = tf.reduce_mean(tf.square(rel_eff_validation_predicted - rel_eff_validation))
    physics_loss_validation = physics_loss(model, input_validation, rel_eff_validation, params)

    data_loss_validation_normalized = data_loss_validation / data_loss_normalization_factor
    physics_loss_validation_normalized = physics_loss_validation / physics_loss_normalization_factor

    total_loss_validation = (physics_loss_weight * physics_loss_validation_normalized) + data_loss_validation_normalized

    train_predictions = model(input_train)
    val_predictions = model(input_validation)

    train_mae = mean_absolute_error(rel_eff_train.numpy(), train_predictions.numpy())
    train_rmse = np.sqrt(mean_squared_error(rel_eff_train.numpy(), train_predictions.numpy()))
    val_mae = mean_absolute_error(rel_eff_validation.numpy(), val_predictions.numpy())
    val_rmse = np.sqrt(mean_squared_error(rel_eff_validation.numpy(), val_predictions.numpy()))

    train_r2 = r2_score(rel_eff_train.numpy(), train_predictions.numpy())
    val_r2 = r2_score(rel_eff_validation.numpy(), val_predictions.numpy())

    #metrics to lists
    train_mae_list.append(train_mae)
    train_rmse_list.append(train_rmse)
    train_r2_list.append(train_r2)
    val_mae_list.append(val_mae)
    val_rmse_list.append(val_rmse)
    val_r2_list.append(val_r2)

    print(f"Epoch {epoch}: Validation Loss: {total_loss_validation.numpy()}")
    print(f"Epoch {epoch}: Training MAE: {train_mae}, Training RMSE: {train_rmse}, Training R²: {train_r2}")
    print(f"Epoch {epoch}: Validation MAE: {val_mae}, Validation RMSE: {val_rmse}, Validation R²: {val_r2}")

    #early stopping check
    if epoch > 0 and total_loss_validation > prev_val_loss:
        patience_count += 1
    else:
        patience_count = 0

    if patience_count > early_stopping.patience:
        print(f"Early stopping at epoch {epoch} with validation loss: {total_loss_validation.numpy()}")
        break

    prev_val_loss = total_loss_validation

#validation set evaluation
physics_loss_value_validation = physics_loss(model, angles_validation, rel_eff_validation, params)
data_loss_value_validation = tf.reduce_mean(tf.square(model(input_validation) - rel_eff_validation))

physics_loss_value_validation_normalized = physics_loss_value_validation / physics_loss_normalization_factor
data_loss_value_validation_normalized = data_loss_value_validation / data_loss_normalization_factor

total_loss_validation = (physics_loss_weight * physics_loss_value_validation_normalized) + data_loss_value_validation_normalized

print(f"Final Validation Loss: {total_loss_validation.numpy()}")

#training data metrics
rel_eff_train_predicted = model(input_train)
data_loss_train = tf.reduce_mean(tf.square(rel_eff_train_predicted - rel_eff_train))
physics_loss_train = physics_loss(model, input_train, rel_eff_train, params)

data_loss_train_normalized = data_loss_train / data_loss_normalization_factor
physics_loss_train_normalized = physics_loss_train / physics_loss_normalization_factor

total_loss_train = (physics_loss_weight * physics_loss_train_normalized) + data_loss_train_normalized

print(f"Final Training Loss: {total_loss_train.numpy()}")

is_mobile = True
if is_mobile:
    mdd_mobile = calculate_mdd(model, angles, params, is_mobile=True, tolerance=0.95)
    mdd_mobile_meters = mdd_mobile / 39.37  #converting MDD to meters
    print(f"Maximum Detectable Distance (Mobile): {mdd_mobile} inches ({mdd_mobile_meters:.2f} meters)")
else:
    mdd_immobile = calculate_immobile_MDD(model, angles, params, tolerance=0.95)
    mdd_immobile_meters = mdd_immobile / 39.37
    print(f"Maximum Detectable Distance (Immobile): {mdd_immobile} inches ({mdd_immobile_meters:.2f} meters)")