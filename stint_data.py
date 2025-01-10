import fastf1 as ff1
import os

# Create directories if they don't exist
cache_dir = 'cache'
output_dir = 'race_stints_2024'

for directory in [cache_dir, output_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Enable caching
ff1.Cache.enable_cache(cache_dir)

# Get the 2024 F1 schedule
schedule = ff1.get_event_schedule(2024)

# Process each race
for _, event in schedule.iterrows():
    try:
        event_name = event['EventName']
        country = event['Country']
        
        print(f"\n{'='*50}")
        print(f"Processing {event_name} ({country})")
        print(f"{'='*50}")
        
        # Load race session
        race_session = ff1.get_session(2024, event_name, 'R')
        race_session.load()
        laps = race_session.laps
        
        # Get drivers and their abbreviations
        drivers = race_session.drivers
        print(f"\nDrivers in {event_name}:")
        driver_abbreviations = [race_session.get_driver(driver)["Abbreviation"] for driver in drivers]
        print(driver_abbreviations)
        
        # Analyze stints
        print(f"\nStint analysis for {event_name}:")
        stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
        stints = stints.groupby(["Driver", "Stint", "Compound"])
        stints = stints.count().reset_index()
        stints = stints.rename(columns={"LapNumber": "Laps"})
        print(stints)
        
        # Create filename - replace spaces with underscores and make lowercase
        filename = f"{event_name.lower().replace(' ', '_')}_stints.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save to CSV
        stints.to_csv(filepath, index=False)
        print(f"\nSaved stint data to: {filepath}")
        
    except Exception as e:
        print(f"Error processing {event_name}: {str(e)}")
        continue

print("\nProcessing complete! Check the 'race_stints_2024' directory for all CSV files.")