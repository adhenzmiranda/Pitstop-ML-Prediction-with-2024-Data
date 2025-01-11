import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import gradio as gr

# Race files dictionary
grand_prix_files = {
    "Austrian Grand Prix": "race_stints_2024/austrian_grand_prix_stints.csv",
    "Abu Dhabi Grand Prix": "race_stints_2024/abu_dhabi_grand_prix_stints.csv"
}

# Load selected race file

selected_race_dropdown = grand_prix_files["Austrian Grand Prix"]
file_path = grand_prix_files[selected_race_dropdown]

print(file_path)


# data = pd.read_csv(file_path)

# #Pre-processing the data and variable for the model

# # Train model


# # Gradio UI
# def gradio_interface(grand_prix, compound):
#     predicted_lap = predict_lap(grand_prix, compound)
#     return f"Predicted optimal lap to pit is: {predicted_lap}"

# compound_dropdown = ["Soft", "Medium", "Hard"]  # Example compounds

# inputs = [
#     gr.Dropdown(label="Grand Prix", choices=race_dropdown, value=race_dropdown[0]),
#     gr.Dropdown(label="Compound", choices=compound_dropdown, value=compound_dropdown[1])
# ]

# outputs = gr.Textbox(label="Prediction")

# gr.Interface(fn=gradio_interface, inputs=inputs, outputs=outputs).launch()
