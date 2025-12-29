import pandas as pd
import numpy as np
from ai_brain import AIBrain
import joblib
from tensorflow.keras.callbacks import ModelCheckpoint

def train_classifier():
    print("Loading Training Data (Winners & Losers)...")
    try:
        df = pd.read_csv("training_data_filtered.csv", index_col=0)
    except:
        print("CSV not found. Run extract_winning_signals.py first.")
        return

    # Check Columns
    # We need 'target' column and features
    if 'target' not in df.columns:
        print("Target column missing.")
        return

    # Balance Dataset
    winners = df[df['target'] == 1]
    losers = df[df['target'] == 0]
    
    n_winners = len(winners)
    print(f"Winners: {n_winners}")
    
    if n_winners < 10:
        print("Not enough winners to train.")
        return
        
    # Sample losers to match winners (1:1 ratio for balanced training)
    # This ensures the model isn't biased towards "doing nothing"
    losers_sampled = losers.sample(n=n_winners * 1, random_state=42)
    
    training_set = pd.concat([winners, losers_sampled]).sort_index()
    print(f"Training Dictionary Size: {len(training_set)} samples.")
    
    # Prepare X, y
    brain = AIBrain()
    
    # Manual Data Prep for Classification specific to this indexed DF
    # We need to reconstruct sequences. 
    # The DF has 'target' for row i. The sequence is i-60 to i.
    # The DF is a slice of history, but contiguous?
    # extract_winning_signals saved the WHOLE df with 'target' column.
    
    # We need to map the "training_set" indices back to the full sequence logic
    # But training_set is sparse rows. We need the CONTEXT (past 60 rows) for each training row.
    # So we must load the FULL df to get context, but only select indices from training_set for training.
    
    full_df = df # The loaded csv IS the full df with targets marked
    
    X = []
    y = []
    
    features = ['close', 'rsi', 'macd', 'bb_width', 'adx']
    
    # Pre-scale entire dataframe to enable fast slicing
    data_values = full_df[features].values
    scaled_values = brain.scaler.fit_transform(data_values) # Re-fit scaler to this data? Better utilize existing scaler if possible, but fit is safer for offline training 
    # Actually, we should use the SAVED scaler to match runtime.
    # But AIBrain loads it. Let's just use transform.
    # If error (UserWarning not fitted), then fit.
    try:
        scaled_values = brain.scaler.transform(data_values)
    except:
        scaled_values = brain.scaler.fit_transform(data_values)
        joblib.dump(brain.scaler, brain.scaler_path)
    
    target_indices = training_set.index # Timestamps?
    # pd.read_csv index might be int if we didn't save index properly or whatever
    # verify index from extract_winning_signals
    
    # Iterate through training_set indices (which are timestamps or int index labels from original)
    # The original df has integer index implied?
    # Let's check how it solves indices.
    
    # Simple approach: Re-index full_df 0..N
    full_df = full_df.reset_index(drop=True)
    # Re-identify targets
    indices_of_interest = full_df[full_df['target'].notnull()].index # All rows?
    # Wait, extract_winning_signals set target=1 for winners, 0 for others.
    # So we pick indices where target=1 and sample others.
    
    winner_indices = full_df[full_df['target'] == 1].index.tolist()
    loser_indices = full_df[full_df['target'] == 0].index.tolist()
    
    import random
    if len(loser_indices) > len(winner_indices) * 2:
        loser_indices = random.sample(loser_indices, len(winner_indices) * 2)
        
    training_indices = sorted(winner_indices + loser_indices)
    
    for i in training_indices:
        if i < 60: continue
        
        seq = scaled_values[i-60:i]
        label = full_df.iloc[i]['target']
        
        X.append(seq)
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    
    print(f"Final X shape: {X.shape}, y shape: {y.shape}")
    
    # Train
    brain.build_classifier((X.shape[1], X.shape[2]))
    
    print("Treinando Classificador IA...")
    checkpoint = ModelCheckpoint(brain.classifier_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    brain.classifier.fit(X, y, epochs=15, batch_size=32, validation_split=0.2, verbose=1, callbacks=[checkpoint])
    
    # Save final (incase validation wasn't best? No, keep best)
    # brain.classifier.save(brain.classifier_path) # Checkpoint handles it
    print(f"Classificador (Melhor Versão) salvo em {brain.classifier_path}")

if __name__ == "__main__":
    train_classifier()
