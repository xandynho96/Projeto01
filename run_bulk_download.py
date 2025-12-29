from data_manager import DataManager

def download_history():
    print("Starting Bulk Data Download (2020 - Present)...")
    print("This may take a few minutes depending on connection and rate limits.")
    
    dm = DataManager()
    dm.fetch_full_history(start_year=2020)
    
    print("Download finished!")

if __name__ == "__main__":
    download_history()
