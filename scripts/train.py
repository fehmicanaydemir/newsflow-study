# scripts/train.py
import argparse
from pathlib import Path
import optuna
import numpy as np
import torch
import torch.optim as optim
import pandas as pd

# Make "src" importable when running from scripts/
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import load_and_preprocess_data
from src.utils import recommender_evaluations, sample_indices
from src.models import MLP, VAE, NCF
from src.models import MLP_model, GMF_model  # for NCF


def MLP_objective(trial, kw_dict, train_data, static_test_data, items_array, checkpoints_path, data_name):
    lr = trial.suggest_float('learning_rate', 0.001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    beta = trial.suggest_float('beta', 0.0, 4.0)
    epochs = kw_dict.get('epochs', 10)

    model = MLP(hidden_dim, **kw_dict).to(kw_dict['device'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Ensure DataFrame for sample_indices (it uses .iloc)
    train_df_local = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
    num_training = train_df_local.shape[0]
    num_batches = int(np.ceil(num_training / batch_size))

    print(f"[trial {trial.number}] start: lr={lr:.4g}, batch={batch_size}, hidden={hidden_dim}, "
          f"beta={beta:.3f}, epochs={epochs}, users={num_training}", flush=True)

    best_hr = -1.0
    for epoch in range(epochs):
        print(f"[trial {trial.number}] epoch {epoch+1}/{epochs} - sampling positives/negatives ...", flush=True)

        # sample once per epoch; returns numpy [U, I+2] with last two float columns (pos/neg idx)
        train_matrix = sample_indices(train_df_local.copy(), **kw_dict)

        perm = np.random.permutation(num_training)
        running_loss = 0.0
        skipped_batches = 0

        # Optional LR decay (kept for parity with your baseline)
        if epoch != 0 and epoch % 10 == 0:
            lr = 0.1 * lr
            for g in optimizer.param_groups:
                g['lr'] = lr
            print(f"[trial {trial.number}] lr decayed to {lr:.4g}", flush=True)

        num_items = items_array.shape[0]

        for b in range(num_batches):
            # batch indices
            batch_idx = perm[b * batch_size:] if (b + 1) * batch_size >= num_training else perm[b * batch_size:(b + 1) * batch_size]

            # features
            Xb = torch.as_tensor(train_matrix[batch_idx, :-2], dtype=torch.float32, device=kw_dict['device'])

            # ---- robust index prep (THE FIX) ----
            # extract, sanitize NaNs, cast to int64, clamp to valid item IDs
            pos_raw = train_matrix[batch_idx, -2]
            neg_raw = train_matrix[batch_idx, -1]

            pos_idx = np.nan_to_num(pos_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)
            neg_idx = np.nan_to_num(neg_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)

            if num_items <= 0:
                raise ValueError("items_array has zero items")

            pos_idx = np.clip(pos_idx, 0, num_items - 1)
            neg_idx = np.clip(neg_idx, 0, num_items - 1)

            # (optional) debug first batch
            if b == 0:
                print(f"[trial {trial.number}] batch {b+1}: pos_idx dtype={pos_idx.dtype}, "
                      f"neg_idx dtype={neg_idx.dtype}, Xb={tuple(Xb.shape)}", flush=True)

            # gather item one-hots with robust integer indexing
            try:
                pos_items_np = np.take(items_array, pos_idx, axis=0)
                neg_items_np = np.take(items_array, neg_idx, axis=0)
            except Exception as e:
                skipped_batches += 1
                print(f"[trial {trial.number}] batch {b+1} skip: np.take error: {e}", flush=True)
                continue

            pos_items = torch.as_tensor(pos_items_np, dtype=torch.float32, device=kw_dict['device'])
            neg_items = torch.as_tensor(neg_items_np, dtype=torch.float32, device=kw_dict['device'])

            try:
                optimizer.zero_grad()
                pos_output = torch.diagonal(model(Xb, pos_items))
                neg_output = torch.diagonal(model(Xb, neg_items))

                # losses
                pos_loss = torch.mean((torch.ones_like(pos_output) - pos_output) ** 2)
                neg_loss = torch.mean((neg_output) ** 2)
                batch_loss = pos_loss + beta * neg_loss
                batch_loss.backward()
                optimizer.step()
                running_loss += float(batch_loss.item())
            except Exception as e:
                skipped_batches += 1
                print(f"[trial {trial.number}] batch {b+1} skip: forward/backward error: {e}", flush=True)
                continue

            if (b + 1) % 50 == 0 or (b + 1) == num_batches:
                done = b + 1
                denom = max(1, done - skipped_batches)
                print(f"[trial {trial.number}] epoch {epoch+1} batch {done}/{num_batches} "
                      f"loss={running_loss/denom:.4f} (skipped {skipped_batches})", flush=True)

        # Evaluate on capped static test set
        model.eval()
        try:
            hr10, _, _, _, _ = recommender_evaluations(model, **kw_dict)
        except Exception as e:
            print(f"[trial {trial.number}] evaluation error: {e}", flush=True)
            hr10 = 0.0
        print(f"[trial {trial.number}] epoch {epoch+1} HR@10={hr10:.4f}", flush=True)
        model.train()

        if hr10 > best_hr:
            best_hr = hr10
            best_path = checkpoints_path / f"MLP_{data_name}_BEST_trial{trial.number}.pt"
            torch.save(model.state_dict(), best_path)
            print(f"[trial {trial.number}] new best HR@10={best_hr:.4f} -> {best_path.name}", flush=True)
            if data_name == "NU":
                canon = checkpoints_path / "mlp_NU.pt"
                torch.save(model.state_dict(), canon)
                print(f"[trial {trial.number}] updated canonical NU checkpoint -> {canon.name}", flush=True)

    return best_hr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to use (ML1M, Yahoo, Pinterest, NU)')
    parser.add_argument('--recommender', type=str, required=True,
                        choices=['MLP', 'VAE', 'NCF'],
                        help='Recommender to train')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Number of Optuna trials')

    # speed knobs
    parser.add_argument('--epochs', type=int, default=3, help='Epochs per trial (default 3)')
    parser.add_argument('--limit_train_users', type=int, default=0, help='Use only first N train users (0=all)')
    parser.add_argument('--eval_users', type=int, default=2000, help='Evaluate on first N static-test users')
    args = parser.parse_args()

    data_name = args.dataset
    recommender_name = args.recommender
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files_path = Path("data/processed", data_name)
    checkpoints_path = Path("checkpoints")
    checkpoints_path.mkdir(parents=True, exist_ok=True)

    print(f"[info] dataset={data_name}, recommender={recommender_name}, device={device}", flush=True)

    if data_name == "NU":
        train_df = pd.read_csv(files_path / f"train_data_{data_name}.csv", index_col=0)
        test_df = pd.read_csv(files_path / f"test_data_{data_name}.csv", index_col=0)
        static_test_df = pd.read_csv(files_path / f"static_test_data_{data_name}.csv", index_col=0)

        if args.limit_train_users and args.limit_train_users > 0:
            train_df = train_df.iloc[:args.limit_train_users].copy()
            print(f"[info] limit_train_users -> using first {len(train_df)} train users", flush=True)
        if args.eval_users and args.eval_users > 0:
            static_test_df = static_test_df.iloc[:args.eval_users].copy()
            print(f"[info] eval_users -> evaluating on first {len(static_test_df)} static-test users", flush=True)

        train_data = train_df                   # keep DataFrame for sample_indices
        static_test_data = static_test_df

        num_items = train_df.shape[1]
        items_array = np.eye(num_items, dtype=np.float32)
        all_items_tensor = torch.eye(num_items, dtype=torch.float32, device=device)

        import pickle
        with open(files_path / f"pop_dict_{data_name}.pkl", "rb") as f:
            pop_dict = pickle.load(f)
        pop_array = np.zeros(num_items, dtype=np.float32)
        for i in range(num_items):
            pop_array[i] = float(pop_dict.get(i, 0.0))

        output_type_dict = {"VAE": "multiple", "MLP": "single", "NCF": "single"}
        kw_dict = {
            'device': device,
            'num_items': num_items,
            'num_features': num_items,
            'demographic': False,
            'pop_array': pop_array,
            'all_items_tensor': all_items_tensor,
            'static_test_data': static_test_data,
            'items_array': items_array,
            'output_type': output_type_dict[recommender_name],
            'recommender_name': recommender_name,
            'epochs': args.epochs,
        }
    else:
        (train_data_np, test_data, static_test_data, pop_dict,
         train_array, test_array_static, items_array, all_items_tensor,
         pop_array) = load_and_preprocess_data(data_name, files_path, device)

        train_data = pd.DataFrame(train_data_np)  # DataFrame for sample_indices
        num_items = test_data.shape[1]

        output_type_dict = {"VAE": "multiple", "MLP": "single", "NCF": "single"}
        kw_dict = {
            'device': device,
            'num_items': num_items,
            'num_features': num_items,
            'demographic': False,
            'pop_array': pop_array,
            'all_items_tensor': all_items_tensor,
            'static_test_data': static_test_data,
            'items_array': items_array,
            'output_type': output_type_dict[recommender_name],
            'recommender_name': recommender_name,
            'epochs': args.epochs,
        }

    if recommender_name == "MLP":
        print(f"[info] starting Optuna with {args.n_trials} trial(s)", flush=True)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: MLP_objective(
            trial, kw_dict, train_data, static_test_data, items_array, checkpoints_path, data_name
        ), n_trials=args.n_trials)
        print("[info] Best trial params:", study.best_trial.params, " Best HR@10:", study.best_value, flush=True)
    else:
        raise NotImplementedError("This script currently supports only MLP for NU; extend for VAE/NCF if needed.")


if __name__ == "__main__":
    main()
