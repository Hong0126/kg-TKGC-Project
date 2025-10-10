# main.py (Final True-Metric Optimization Edition)
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import pandas as pd
import os
import numpy as np

from tkg_loader import load_all_tkg_data
from igat_predictor import IGATPredictor
from igat_batcher import InductiveTKGStitcher, TKGCollator
from metrics import calculate_giou

def analyze_entity_sets(df_train, df_valid, df_test):
    print("\n--- Inductive Capability Analysis ---")

    if df_train is None or df_train.empty:
        print("Train is empty")
        return

    # 1. extract all entities from train
    train_entities = set(df_train['s']) | set(df_train['o'])
    print(f"Number of unique entities in train: {len(train_entities):,}")

    # 2. Analysis val dataset
    if df_valid is not None and not df_valid.empty:
        valid_entities = set(df_valid['s']) | set(df_valid['o'])
        print(f"Unique entities in val: {len(valid_entities):,}")
        
        new_entities_in_valid = valid_entities - train_entities
        
        print(f"New entities in the val dataset: {len(new_entities_in_valid):,}")
        
        if len(valid_entities) > 0:
            percentage_new_in_valid = (len(new_entities_in_valid) / len(valid_entities)) * 100
            print(f"New entities propotion in the test dataset {percentage_new_in_valid:.2f}%")
    else:
        print("VAl is empty")

    # 3. Analysis test dataset
    if df_test is not None and not df_test.empty:
        test_entities = set(df_test['s']) | set(df_test['o'])
        print(f"Unique entities in test: {len(test_entities):,}")
        
        new_entities_in_test = test_entities - train_entities
        
        print(f"New entities in the test dataset: {len(new_entities_in_test):,}")
        
        if len(test_entities) > 0:
            percentage_new_in_test = (len(new_entities_in_test) / len(test_entities)) * 100
            print(f"New entities propotion in the test dataset: {percentage_new_in_test:.2f}%")
    else:
        print("Test is empty")
    print("----------------------------------------------------\n")
    
class TKGDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty: self.facts = np.array([])
        else: self.facts = df[['s_id', 'r_id', 'o_id', 'start_time', 'end_time']].to_numpy()
    def __len__(self): return len(self.facts)
    def __getitem__(self, idx):
        s, r, o, start_t, end_t = self.facts[idx]
        return {"s": int(s), "r": int(r), "o": int(o), "start_time": int(start_t), "end_time": int(end_t)}

def evaluate(model, data_loader, stitcher, device):
    model.eval()
    all_giou_scores = []
    pbar = tqdm(data_loader, desc="evaluating", leave=False)
    for batch_data in pbar:
        if not batch_data: continue
        with torch.no_grad(), autocast(device_type=device.type):
            dummy_intervals = batch_data["intervals"].to(device)
            dummy_intervals_norm = stitcher._normalize_time(dummy_intervals)
            _, predicted_interval_norm = model(
                batch_data["s_hist"].to(device), batch_data["o_hist"].to(device),
                batch_data["s_lens"].to(device), batch_data["o_lens"].to(device),
                batch_data["queries"].to(device), dummy_intervals_norm
            )
            t_span, t_min = stitcher.t_span, stitcher.t_min
            predictions_real = predicted_interval_norm.cpu() * t_span + t_min
            true_intervals_real = batch_data["intervals"].squeeze(1).cpu()
            giou = calculate_giou(predictions_real, true_intervals_real)
            all_giou_scores.append(giou)
    if not all_giou_scores: return {"gIoU": 0.0}
    return {"gIoU": torch.cat(all_giou_scores).mean().item() * 100}

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS, BATCH_SIZE = 100, 128
    LEARNING_RATE, WARMUP_STEPS, NUM_NEG_SAMPLES = 1e-4, 2000, 64
    
    REGRESSION_LOSS_WEIGHT = 1.0 
    
    FEATURE_DIM, HIDDEN_DIM, EMBEDDING_DIM = 128, 256, 128
    NUM_HEADS, NUM_LAYERS, DROPOUT = 8, 8, 0.1
    NUM_WORKERS = max(4, os.cpu_count() // 2) if os.cpu_count() else 4
    MODEL_SAVE_DIR = "saved_igat_models_final"
    BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_igat_final_model.pth")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    print("--- Final Hybrid (Ranking + gIoU Loss) Mode ---")
    
    df_train, df_valid, df_test, entity_map, relation_map, time_range = load_all_tkg_data()
    analyze_entity_sets(df_train, df_valid, df_test)
    full_df = pd.concat([df for df in [df_train, df_valid, df_test] if df is not None]).sort_values('start_time')
    
    model = IGATPredictor(len(entity_map), len(relation_map), FEATURE_DIM, HIDDEN_DIM, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT).to(DEVICE)
    print(f"Param size: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scaler = GradScaler()
    ranking_loss_fn = torch.nn.BCEWithLogitsLoss()

    stitcher = InductiveTKGStitcher(full_df, entity_map, relation_map, time_range)
    train_collator = TKGCollator(stitcher, num_neg_samples=NUM_NEG_SAMPLES)
    valid_collator = TKGCollator(stitcher, num_neg_samples=0)

    train_loader = DataLoader(TKGDataset(df_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=train_collator, pin_memory=True)
    valid_loader = DataLoader(TKGDataset(df_valid), batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS, collate_fn=valid_collator)

    best_valid_giou = -float('inf')
    global_step = 0
    print("\n--- Train IGAT (Final Hybrid Mode) ---")
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")
        for batch_data in pbar:
            if not batch_data: continue
            global_step += 1
            if global_step < WARMUP_STEPS:
                lr_scale = min(1.0, float(global_step) / float(WARMUP_STEPS))
                for pg in optimizer.param_groups: pg['lr'] = LEARNING_RATE * lr_scale
            
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type):
                intervals = batch_data["intervals"].to(DEVICE)
                intervals_norm = stitcher._normalize_time(intervals)
                
                scores, predicted_interval_norm = model(
                    batch_data["s_hist"].to(DEVICE), batch_data["o_hist"].to(DEVICE),
                    batch_data["s_lens"].to(DEVICE), batch_data["o_lens"].to(DEVICE),
                    batch_data["queries"].to(DEVICE), intervals_norm
                )
                
                # 1. core loss
                ranking_loss = ranking_loss_fn(scores, batch_data["labels"].to(DEVICE))
                
                # 2. gIoU loss
                true_interval_norm = intervals_norm[:, 0, :]
                giou_loss = (1 - calculate_giou(predicted_interval_norm, true_interval_norm)).mean()
                
                # 3. TOtal loss
                loss = ranking_loss + REGRESSION_LOSS_WEIGHT * giou_loss
            
            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Rank L": f"{ranking_loss.item():.4f}", "gIoU L": f"{giou_loss.item():.4f}"})

        if df_valid is not None and len(valid_loader) > 0:
            valid_results = evaluate(model, valid_loader, stitcher, DEVICE)
            print(f"\n--- Epoch {epoch+1} eva ---")
            print(pd.DataFrame([valid_results]).round(2).to_markdown(index=False, tablefmt="grid"))
            current_giou = valid_results["gIoU"]
            if current_giou > best_valid_giou:
                best_valid_giou = current_giou
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"** best gIoU: {best_valid_giou:.2f}. Model saved **\n")

    print(f"\n--- Training done --- \nbest gIoU: {best_valid_giou:.2f}")

if __name__ == "__main__":
    main()