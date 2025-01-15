# Import modules
import fastf1 as ff1 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
import gradio as gr
import seaborn as sns
import numpy as np

# Dictionary of file locations
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
def optimal_pit_stop_lap(grand_prix):
    file_path = grand_prix_files[grand_prix]
    data = pd.read_csv(file_path)

    # Mapping the tyre compounds
    compound_map = {
        "SOFT": 1, "MEDIUM": 2, "HARD": 3,
        "INTERMEDIATE": 4, "WET": 5
    }
    data["Compound"] = data["Compound"].map(compound_map)

    # Mapping the drivers
    drivers_map = {
        "ALB": 1, "ALO": 2, "BOT": 3, "HAM": 4, "HUL": 5,
        "LEC": 6, "PIA": 7, "NOR": 8, "PER": 9, "RIC": 10,
        "STR": 11, "TSU": 12, "ZHO": 14, "GAS": 15, "SAI": 16,
        "VER": 17, "RUS": 18, "OCO": 19, "SAR": 20, "MAG": 21,
        "LAW": 22, "COL": 23, "DOO": 24, "BEA": 25
    }
    data["Driver"] = data["Driver"].map(drivers_map)

    # Prepare data for model
    x = data[["Stint", "Compound", "Driver"]]
    y = data["Laps"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    # Calculate metrics and residuals
    residuals = np.round(np.abs(y_test - predictions), decimals=2)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Create visualizations
    fig = plt.figure(figsize=(15, 10))
    
    # Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, predictions)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Laps')
    plt.ylabel('Predicted Laps')
    plt.title('Actual vs Predicted Laps')

    # Residuals Distribution
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=20)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')

    # Feature Importance
    plt.subplot(2, 2, 3)
    importance = pd.DataFrame({
        'Feature': ['Stint', 'Compound', 'Driver'],
        'Importance': model.feature_importances_
    })
    plt.bar(importance['Feature'], importance['Importance'])
    plt.title('Feature Importance')

    # Metrics Text
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, 
            f'R² Score: {r2*100:.2f}%\nMSE: {mse:.2f}\nMAE: {mae:.2f}',
            horizontalalignment='center', 
            verticalalignment='center', 
            fontsize=12)
    plt.axis('off')
    plt.title('Model Metrics')

    plt.tight_layout()

    # Prepare results
    results = x_test.copy()
    results["Actual Laps"] = y_test
    results["Predicted Laps"] = np.round(predictions, decimals=2)
    results["Residuals"] = residuals
    
    # Get top 3 predictions
    top_3_optimal_laps = results.sort_values(by="Residuals").head(3).reset_index(drop=True)

    return [top_3_optimal_laps, fig]

# Create Gradio interface
iface = gr.Interface(
    fn=optimal_pit_stop_lap,
    inputs=gr.Dropdown(
        choices=list(grand_prix_files.keys()),
        label="Select Grand Prix"
    ),
    outputs=[
        gr.DataFrame(label="Top 3 Optimal Pit Stops"),
        gr.Plot(label="Model Performance Metrics")
    ],
    title="F1 Pit Stop Optimizer",
    description="Displays optimal pit stop strategies and model performance metrics"
)

if __name__ == "__main__":
    iface.launch(share=True)