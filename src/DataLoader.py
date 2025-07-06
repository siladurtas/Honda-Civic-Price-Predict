import json
import os

class DataLoader:
    def load_civic_data():
        try:

            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(os.path.dirname(current_dir), 'data', 'civic.json')
            
            with open(data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            print(f"Error: civic.json file not found at {data_path}")
            return None
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in civic.json")
            return None
        except Exception as e:
            print(f"Error loading civic.json: {str(e)}")
            return None
