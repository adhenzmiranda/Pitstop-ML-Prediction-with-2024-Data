#Import the modules
import fastf1 as ff1 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gradio as gr

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

# def predict_optimal_strategy(grand_prix):

    # # Your prediction logic here
    # # For example:
    # X = data[['feature1', 'feature2']]  # Replace with actual feature columns
    # y = data['target']  # Replace with actual target column
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    
    # mse = mean_squared_error(y_test, predictions)
    # mae = mean_absolute_error(y_test, predictions)
    # r2 = r2_score(y_test, predictions)
    
    # return f"MSE: {mse}, MAE: {mae}, R2: {r2}"

# Create the Gradio interface
# iface = gradio.Interface(
#     fn=predict_optimal_strategy,
#     inputs= gradio.components.Dropdown(choices=list(grand_prix_files.keys()), label="Select Grand Prix"),
#     outputs="text"
# )
# iface.launch()