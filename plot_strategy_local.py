#Import the modules
import fastf1 as ff1 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gradio as gr
import seaborn as sns

# Make a list of the grand prix files
grand_prix_files = {
    "Austrian Grand Prix": "race_stints_2024/austrian_grand_prix_stints.csv",
    "British Grand Prix": "race_stints_2024/british_grand_prix_stints.csv",
    "Bahrain Grand Prix": "race_stints_2024/bahrain_grand_prix_stints.csv",
    "Saudi Arabian Grand Prix": "race_stints_2024/saudi_arabian_grand_prix_stints.csv",
    "Australian Grand Prix": "race_stints_2024/australian_grand_prix_stints.csv",
    "Emilia Romagna Grand Prix": "race_stints_2024/emilia_romagna_grand_prix_stints.csv",
    "Monaco Grand Prix": "race_stints_2024/monaco_grand_prix_stints.csv",
    "Spanish Grand Prix": "race_stints_2024/spanish_grand_prix_stints.csv",
    "Canadian Grand Prix": "race_stints_2024/canadian_grand_prix_stints.csv",
    "French Grand Prix": "race_stints_2024/french_grand_prix_stints.csv",
    "Hungarian Grand Prix": "race_stints_2024/hungarian_grand_prix_stints.csv",
    "Belgian Grand Prix": "race_stints_2024/belgian_grand_prix_stints.csv",
    "Dutch Grand Prix": "race_stints_2024/dutch_grand_prix_stints.csv",
    "Italian Grand Prix": "race_stints_2024/italian_grand_prix_stints.csv",
    "Singapore Grand Prix": "race_stints_2024/singapore_grand_prix_stints.csv",
    "Japanese Grand Prix": "race_stints_2024/japanese_grand_prix_stints.csv",
    "United States Grand Prix": "race_stints_2024/united_states_grand_prix_stints.csv",
    "Mexican Grand Prix": "race_stints_2024/mexican_grand_prix_stints.csv",
    "Brazilian Grand Prix": "race_stints_2024/brazilian_grand_prix_stints.csv",
    "Qatar Grand Prix": "race_stints_2024/qatar_grand_prix_stints.csv",
    "Abu Dhabi Grand Prix": "race_stints_2024/abu_dhabi_grand_prix_stints.csv",
    "Miami Grand Prix": "race_stints_2024/miami_grand_prix_stints.csv",
    "Las Vegas Grand Prix": "race_stints_2024/las_vegas_grand_prix_stints.csv"
}

grand_prix = "Austrian Grand Prix"
# Load the selected file
file_path = grand_prix_files[grand_prix]
data = pd.read_csv(file_path) 
print(data.head())

# Pre-Process and mapping the data

# Mapping the tyre compounds
compound_map = {
    "SOFT": 1,
    "MEDIUM": 2,
    "HARD": 3
}

data["Compound"] = data["Compound"].map(compound_map)

# Mapping the drivers
drivers_map = {
    "ALB": 1,  # Alexander Albon
    "ALO": 2,  # Fernando Alonso
    "BOT": 3,  # Valtteri Bottas
    "HAM": 4,  # Lewis Hamilton
    "HUL": 5,  # Nico Hulkenberg
    "LEC": 6,  # Charles Leclerc
    "PIA": 7,  # Oscar Piastri
    "NOR": 8,  # Lando Norris
    "PER": 9,  # Sergio Perez
    "RIC": 10, # Daniel Ricciardo
    "STR": 11, # Lance Stroll
    "TSU": 12, # Yuki Tsunoda
    "ZHO": 14, # Zhou Guanyu
    "GAS": 15, # Pierre Gasly
    "SAI": 16, # Carlos Sainz
    "VER": 17, # Max Verstappen
    "RUS": 18, # George Russell
    "OCO": 19, # Esteban Ocon
    "SAR": 20, # Logan Sargeant
    "MAG": 21, # Kevin Magnussen
    "LAW": 22, # Liam Lawson
    "COL": 23, # Franco Colapinto
    "DOO": 24, # Jack Doohan
    "BEA": 25  # Oliver Bearman
}

data["Driver"] = data["Driver"].map(drivers_map)

# Drop rows with NaN values in any column
data = data.dropna()

x = data[["Stint", "Compound", "Driver"]]
y = data["Laps"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(predictions)

# Accuracy of the data
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Plot the data
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=predictions, alpha=0.7, color='b', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Laps Actual')
plt.ylabel('Laps Predicted to Pit')
plt.title('Optimal Lap to Pit')
plt.show()