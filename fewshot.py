import sys
import os
import json
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import argparse
import ast

from utils.utils import DataLoader
from utils.dataset import ProteinMoleculeDataset
from utils.metrics import evaluate_reg
from models.net import net
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from utils.trainer_utils import beta_loss_with_mixed_targets
from utils.protein_init import protein_init_with_keys

# --- Helper functions ---
def smiles_to_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        if not scaffold_smiles:
            raise ValueError("No scaffold found")
        return scaffold_smiles
    except Exception as e:
        return smiles

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def list_type(s):
    try:
        value = ast.literal_eval(s)
        if not isinstance(value, list):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid list value: {s}")
    return value

def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            reg_alpha, reg_beta, *_ = model(data)
            pred = reg_alpha / (reg_alpha + reg_beta)
            pred = pred * 12  # unscale if needed
            all_preds.append(pred.cpu())
            all_labels.append(data.reg_y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return evaluate_reg(all_labels, all_preds)

# Add compute_loss function for this script
def compute_loss(batch, reg_alpha, reg_beta, o_loss, cl_loss, regression_weight=1.0):
    # Use beta_loss_with_mixed_targets for regression loss
    reg_loss = beta_loss_with_mixed_targets(
        reg_alpha, reg_beta,
        batch.reg_y.squeeze(), batch.cls_y.squeeze(),
        min_pKi=0, max_pKi=12, threshold=4
    ) * regression_weight
    total_loss = reg_loss + o_loss + cl_loss
    return total_loss, reg_loss

# --- Main fine-tuning script ---
def main():
    parser = argparse.ArgumentParser(description="Finetune PSICHIC model with KL regularization.")
    parser.add_argument('--seed', type=int, default=168, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--config_path', type=str, default='config.json', help='Config file path')
    parser.add_argument('--datafolder', type=str, default='./PSICHIC_Hit_Opt/', help='Data folder')
    parser.add_argument('--result_path', type=str, default='./PSICHIC_Hit_Opt/frac20/', help='Result path')
    parser.add_argument('--trained_model_path', type=str, default='trained_weights/Application525K_QuadraticPrior', help='Pretrained model path')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--total_iters', type=int, default=5000, help='Total iterations')
    parser.add_argument('--lrate', type=float, default=1e-5, help='Learning rate for head modules')
    parser.add_argument('--backbone_lrate', type=float, default=1e-6, help='Learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--kl_weight', type=float, default=0.1, help='KL regularization weight')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--finetune_modules', type=list_type, default=['mol_out', 'prot_out', 'mu_out', 'sigma_out'], help='List of head modules to finetune (modular optimizer)')
    parser.add_argument('--partial_finetune_modules', type=list_type, default=None, help='List of modules to finetune (freeze backbone, only finetune these)')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value (default: 1.0)')
    parser.add_argument('--train_frac', type=float, default=0.2, help='Fraction of data to use for training (default: 0.8)')
    args = parser.parse_args()

    set_seed(args.seed)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # --- Load config and model ---
    with open(os.path.join(args.trained_model_path, 'config.json'), 'r') as f:
        config = json.load(f)
    degree_dict = torch.load(os.path.join(args.trained_model_path, 'degree.pt'))
    mol_deg, clique_deg, prot_deg = degree_dict['ligand_deg'], degree_dict['clique_deg'], degree_dict['protein_deg']

    model = net(
        mol_deg, prot_deg,
        mol_in_channels=config['params']['mol_in_channels'],
        mol_edge_channels=config['params']['mol_edge_channels'],
        clique_pe_walk_length=config['params']['clique_pe_walk_length'],
        prot_in_channels=config['params']['prot_in_channels'],
        prot_evo_channels=config['params']['prot_evo_channels'],
        hidden_channels=config['params']['hidden_channels'],
        pre_layers=config['params']['pre_layers'],
        post_layers=config['params']['post_layers'],
        aggregators=config['params']['aggregators'],
        scalers=config['params']['scalers'],
        total_layer=config['params']['total_layer'],
        K=config['params']['K'],
        heads=config['params']['heads'],
        dropout=config['params']['dropout'],
        dropout_attn_score=config['params']['dropout_attn_score'],
        device=args.device
    ).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.trained_model_path, 'model.pt'), map_location=args.device), strict=False)

    # --- Load pretrained model for KL ---
    import copy
    pre_trained_model = copy.deepcopy(model)
    pre_trained_model.eval()
    for p in pre_trained_model.parameters():
        p.requires_grad = False

    # --- Prepare data ---
    full_df = pd.read_csv('fep_test.csv')
    full_df['scaffold'] = full_df['Ligand'].apply(smiles_to_scaffold)
    all_proteins = full_df['Protein'].tolist()
    all_seqs = full_df['Protein_Sequence'].tolist()

    protein_seqs = dict(zip(all_proteins, all_seqs))

    all_proteins = []
    all_seqs = []
    for protein, seq in protein_seqs.items():
        all_proteins.append(protein)
        all_seqs.append(seq)

    protein_dict = protein_init_with_keys(all_proteins, all_seqs)

    for protein in all_proteins:
        protein_df = full_df[full_df['Protein'] == protein].reset_index(drop=True)
        train_df = protein_df.sample(frac=args.train_frac, random_state=args.seed)
        test_df = protein_df.drop(train_df.index).reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)

        ligand_path = os.path.join(args.datafolder, 'ligands')
        if not os.path.exists(ligand_path):
            os.makedirs(ligand_path)
        ligand_dict = {}

        train_dataset = ProteinMoleculeDataset(
            train_df, ligand_dict, 
            ligand_path, 
            protein_dict=protein_dict,
            protein_folder=None,
            result_path=args.result_path, 
            dataset_tag='train', device=args.device
        )
        test_dataset = ProteinMoleculeDataset(
            test_df, ligand_dict, 
            ligand_path, 
            protein_dict=protein_dict,
            protein_folder=None,
            result_path=args.result_path, 
            dataset_tag='test', 
            device=args.device
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                        follow_batch=['mol_x', 'clique_x', 'prot_node_aa'],
                        num_workers=8, pin_memory=True) # Added pin_memory

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                follow_batch=['mol_x', 'clique_x', 'prot_node_aa'],
                                num_workers=8, pin_memory=True) # Added pin_memory

        # --- Optimizer ---
        if args.partial_finetune_modules is not None:
            print(f"Using freezing strategy. Fine-tuning modules: {args.partial_finetune_modules}")
            optimizer = model.freeze_backbone_optimizers(
                partial_finetune_modules=args.partial_finetune_modules,
                weight_decay=args.weight_decay,
                learning_rate=args.lrate,
                betas=(0.9, 0.999),
                eps=1e-5,
                amsgrad=False
            )
        else:
            print(f"Using differential learning rates. Head LR: {args.lrate}, Backbone LR: {args.backbone_lrate}")
            optimizer = model.configure_modular_optimizers(
                weight_decay=args.weight_decay,
                learning_rate=args.lrate,
                backbone_lrate=args.backbone_lrate,
                betas=(0.9, 0.999),
                eps=1e-5,
                amsgrad=False,
                head_modules=set(args.finetune_modules)
            )

        # --- Training loop ---
        model.train()
        iter_count = 0
        pbar = tqdm(total=args.total_iters, desc=f'Finetuning {protein}')
        # Logging structures
        log_list = []
        best_rmse = float('inf')
        best_iter = 0
        best_metrics = None
        while iter_count < args.total_iters:
            for batch in train_loader:
                batch = batch.to(args.device)
                optimizer.zero_grad()
                reg_alpha, reg_beta, o_loss, cl_loss, _ = model(batch)
                # pred = reg_alpha / (reg_alpha + reg_beta)
                # pred_loss = torch.nn.functional.mse_loss(pred * 12, batch.reg_y)  # unscale if needed
                loss, reg_loss = compute_loss(batch, reg_alpha, reg_beta, o_loss, cl_loss)
                if args.kl_weight > 0:
                    # KL regularization
                    with torch.no_grad():
                        pre_reg_alpha, pre_reg_beta, *_ = pre_trained_model(batch)
                    kl_div = torch.distributions.kl_divergence(
                        torch.distributions.Beta(reg_alpha, reg_beta),
                        torch.distributions.Beta(pre_reg_alpha, pre_reg_beta)
                    ).mean()
                    loss = loss + args.kl_weight * kl_div
                else:
                    kl_div = torch.tensor(0.0)

                loss.backward()
                if args.clip is not None and args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

                iter_count += 1
                pbar.update(1)

                if iter_count % args.eval_interval == 0 or iter_count == args.total_iters:
                    eval_metrics = evaluate(model, test_loader, args.device)
                    rmse = eval_metrics.get('rmse', float('nan'))
                    log_entry = {
                        'iteration': iter_count,
                        'rmse': rmse,
                        'eval_metrics': eval_metrics,
                        'loss': float(loss.detach().cpu().item()),
                        'reg_loss': float(reg_loss.detach().cpu().item()),
                        'kl_div': float(kl_div.detach().cpu().item()),
                    }
                    log_list.append(log_entry)
                    print(f"[{protein}] Iter {iter_count}: Test RMSE: {rmse:.4f}")
                    torch.save(model.state_dict(), os.path.join(args.result_path, f"best_{protein}_finetuned_seed{args.seed}.pt"))
                    # Track best
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_iter = iter_count
                        best_metrics = eval_metrics.copy()
                    # Save log_entry immediately to a .jsonl file
                    log_jsonl_path = os.path.join(args.result_path, f"{protein}_seed{args.seed}_log.jsonl")
                    with open(log_jsonl_path, 'a') as f_jsonl:
                        f_jsonl.write(json.dumps(log_entry) + '\n')
                    print(f"Appended log entry to {log_jsonl_path}")

                if iter_count >= args.total_iters:
                    break
        pbar.close()
        print(f"Finished finetuning for protein: {protein}")

        # Save log as JSON
        log_path = os.path.join(args.result_path, f"{protein}_seed{args.seed}_log.json")
        with open(log_path, 'w') as f:
            json.dump(log_list, f, indent=2)

        # Save summary as CSV
        summary = {
            'protein': protein,
            'seed': args.seed,
            'best_rmse': best_rmse,
            'best_iter': best_iter,
            'final_rmse': log_list[-1]['rmse'] if log_list else float('nan'),
        }
        if best_metrics is not None:
            for k, v in best_metrics.items():
                summary[f'best_{k}'] = v
        summary_path = os.path.join(args.result_path, f"{protein}_seed{args.seed}_summary.csv")
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        print(f"Log saved to {log_path}")
        print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()