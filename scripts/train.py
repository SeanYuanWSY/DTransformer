import os
import json
import csv
import numpy as np
import shutil
from argparse import ArgumentParser
from datetime import datetime
from sklearn.model_selection import KFold

import torch
import tomlkit
from tqdm import tqdm

from DTransformer.data import KTData
from DTransformer.eval import Evaluator

DATA_DIR = "data"

# configure the main parser
parser = ArgumentParser()

# general options
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument("-bs", "--batch_size", help="batch size", default=8, type=int)
parser.add_argument(
    "-tbs", "--test_batch_size", help="test batch size", default=64, type=int
)

# data setup
datasets = tomlkit.load(open(os.path.join(DATA_DIR, "datasets.toml")))
parser.add_argument(
    "-d",
    "--dataset",
    help="choose from a dataset",
    choices=datasets.keys(),
    required=True,
)
parser.add_argument(
    "-p", "--with_pid", help="provide model with pid", action="store_true"
)

# model setup
parser.add_argument("-m", "--model", help="choose model")
parser.add_argument("--d_model", help="model hidden size", type=int, default=128)
parser.add_argument("--n_layers", help="number of layers", type=int, default=3)
parser.add_argument("--n_heads", help="number of heads", type=int, default=8)
parser.add_argument(
    "--n_know", help="dimension of knowledge parameter", type=int, default=32
)
parser.add_argument("--dropout", help="dropout rate", type=float, default=0.2)
parser.add_argument("--proj", help="projection layer before CL", action="store_true")
parser.add_argument(
    "--hard_neg", help="use hard negative samples in CL", action="store_true"
)

# training setup
parser.add_argument("-n", "--n_epochs", help="training epochs", type=int, default=100)
parser.add_argument(
    "-es",
    "--early_stop",
    help="early stop after N epochs of no improvements",
    type=int,
    default=10,
)
parser.add_argument(
    "-lr", "--learning_rate", help="learning rate", type=float, default=1e-3
)
parser.add_argument("-l2", help="L2 regularization", type=float, default=1e-5)
parser.add_argument(
    "-cl", "--cl_loss", help="use contrastive learning loss", action="store_true"
)
parser.add_argument(
    "--lambda", help="CL loss weight", type=float, default=0.1, dest="lambda_cl"
)
parser.add_argument("--window", help="prediction window", type=int, default=1)

# k-fold cross validation setup
parser.add_argument("--k_fold", help="number of folds for cross validation", type=int, default=1)
parser.add_argument("--random_state", help="random state for k-fold split", type=int, default=42)

# snapshot setup
parser.add_argument("-o", "--output_dir", help="directory to save model files and logs")
parser.add_argument(
    "-f", "--from_file", help="resume training from existing model file", default=None
)


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.start_time = datetime.now()

        # 创建日志目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化训练历史记录
        self.history = {
            'config': {},
            'training_start': self.start_time.isoformat(),
            'epochs': []
        }

        # 创建CSV文件记录详细训练数据
        self.csv_file = os.path.join(output_dir, 'training_log.csv')
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_pred_loss', 'train_cl_loss',
                'val_auc', 'val_acc', 'val_rmse', 'learning_rate', 'timestamp'
            ])

    def log_config(self, config):
        """记录训练配置"""
        self.history['config'] = {k: v for k, v in config.items()}
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.history['config'], f, indent=2, default=str)

    def log_epoch(self, epoch, train_metrics, val_metrics, lr=None):
        """记录每个epoch的训练和验证结果"""
        timestamp = datetime.now().isoformat()

        # 更新历史记录
        epoch_record = {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'timestamp': timestamp
        }
        if lr is not None:
            epoch_record['learning_rate'] = lr

        self.history['epochs'].append(epoch_record)

        # 写入CSV文件 - 只记录数值型指标
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics.get('loss', 0),
                train_metrics.get('pred_loss', 0),
                train_metrics.get('cl_loss', 0),
                val_metrics.get('auc', 0),
                val_metrics.get('acc', 0),
                val_metrics.get('rmse', 0),
                lr if lr is not None else '',
                timestamp
            ])

        # 实时保存历史记录
        self.save_history()

    def log_best_result(self, best_epoch, best_metrics):
        """记录最佳结果"""
        self.history['best_epoch'] = best_epoch
        self.history['best_metrics'] = best_metrics
        self.history['training_end'] = datetime.now().isoformat()
        self.history['training_duration'] = str(datetime.now() - self.start_time)

        # 保存最佳结果到单独文件
        best_result = {
            'best_epoch': best_epoch,
            'best_metrics': best_metrics,
            'training_duration': self.history['training_duration']
        }

        best_result_path = os.path.join(self.output_dir, 'best_result.json')
        with open(best_result_path, 'w') as f:
            json.dump(best_result, f, indent=2, default=str)

        self.save_history()

    def save_history(self):
        """保存完整训练历史"""
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def print_epoch_summary(self, epoch, train_metrics, val_metrics):
        """打印epoch摘要"""
        print(f"\nEpoch {epoch} Summary:")
        print("-" * 50)
        print("Training Metrics:")
        for metric, value in train_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.6f}")
            else:
                print(f"  {metric}: {value}")

        print("Validation Metrics:")
        for metric, value in val_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            elif isinstance(value, int):
                print(f"  {metric}: {value}")
            else:
                print(f"  {metric}: {value}")
        print("-" * 50)


def train_single_fold(args, train_data, valid_data, fold_idx, output_dir):
    """训练单个fold"""
    # prepare logger and output directory
    fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_output_dir, exist_ok=True)

    logger = TrainingLogger(fold_output_dir)
    logger.log_config(vars(args))
    print(f"Fold {fold_idx} logs will be saved to: {fold_output_dir}")

    # prepare model and optimizer
    dataset = datasets[args.dataset]
    if args.model == "DKT":
        from baselines.DKT import DKT
        model = DKT(dataset["n_questions"], args.d_model)
    elif args.model == "DKVMN":
        from baselines.DKVMN import DKVMN
        model = DKVMN(dataset["n_questions"], args.batch_size)
    elif args.model == "AKT":
        from baselines.AKT import AKT
        model = AKT(
            dataset["n_questions"],
            dataset["n_pid"],
            d_model=args.d_model,
            n_heads=args.n_heads,
            dropout=args.dropout,
        )
    else:
        from DTransformer.model import DTransformer
        model = DTransformer(
            dataset["n_questions"],
            dataset["n_pid"],
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_know=args.n_know,
            lambda_cl=args.lambda_cl,
            dropout=args.dropout,
            proj=args.proj,
            hard_neg=args.hard_neg,
            window=args.window,
        )

    if args.from_file:
        model.load_state_dict(torch.load(args.from_file, map_location=lambda s, _: s))
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.l2
    )
    model.to(args.device)

    # training
    best = {"auc": 0}
    best_epoch = 0
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None

    print(f"Starting training for fold {fold_idx}")
    print("=" * 60)

    for epoch in range(1, args.n_epochs + 1):
        epoch_start_time = datetime.now()
        print(f"\nFold {fold_idx}, Epoch {epoch}/{args.n_epochs} - {epoch_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # =========================================================================
        # Training phase (Corrected Logic)
        # =========================================================================
        model.train()
        it = tqdm(iter(train_data), desc=f"Training Fold {fold_idx} Epoch {epoch}")
        total_loss = 0.0
        total_pred_loss = 0.0
        total_cl_loss = 0.0
        total_samples = 0

        for batch in it:
            q = batch.get("q").to(args.device)
            s = batch.get("s").to(args.device)
            pid = batch.get("pid").to(args.device) if args.with_pid else None

            batch_size = q.size(0)

            # Process the whole batch at once
            optim.zero_grad()
            if args.cl_loss:
                loss, pred_loss, cl_loss = model.get_cl_loss(q, s, pid)
                total_pred_loss += pred_loss.item() * batch_size
                total_cl_loss += cl_loss.item() * batch_size
            else:
                loss = model.get_loss(q, s, pid)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Update progress bar
            postfix = {"loss": total_loss / total_samples}
            if args.cl_loss:
                postfix["pred_loss"] = total_pred_loss / total_samples
                postfix["cl_loss"] = total_cl_loss / total_samples
            it.set_postfix(postfix)

        # Calculate average training metrics
        train_metrics = {"loss": total_loss / total_samples if total_samples > 0 else 0}
        if args.cl_loss:
            train_metrics["pred_loss"] = total_pred_loss / total_samples if total_samples > 0 else 0
            train_metrics["cl_loss"] = total_cl_loss / total_samples if total_samples > 0 else 0

        # =========================================================================
        # Validation phase (Corrected Logic)
        # =========================================================================
        model.eval()
        evaluator = Evaluator()

        with torch.no_grad():
            it = tqdm(iter(valid_data), desc=f"Validation Fold {fold_idx} Epoch {epoch}")
            for batch in it:
                q = batch.get("q").to(args.device)
                s = batch.get("s").to(args.device)
                pid = batch.get("pid").to(args.device) if args.with_pid else None

                # Process the whole batch at once
                y, *_ = model.predict(q, s, pid)
                evaluator.evaluate(s, torch.sigmoid(y))

        val_metrics = evaluator.report()

        # 添加额外的验证信息
        val_samples = len(evaluator.y_true)
        val_metrics["samples"] = val_samples
        epoch_duration = datetime.now() - epoch_start_time
        val_metrics["epoch_duration_seconds"] = epoch_duration.total_seconds()

        # Print epoch summary
        logger.print_epoch_summary(epoch, train_metrics, val_metrics)

        # Log epoch results
        current_lr = optim.param_groups[0]['lr'] if optim.param_groups else None
        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)

        # Check for best model
        if val_metrics["auc"] > best["auc"]:
            best = val_metrics.copy()
            best_epoch = epoch
            print(f"Fold {fold_idx}: New best model! AUC: {best['auc']:.4f}")

            # Save best model
            model_path = os.path.join(fold_output_dir, f"best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Fold {fold_idx}: Best model saved to: {model_path}")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(fold_output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Fold {fold_idx}: Checkpoint saved to: {checkpoint_path}")

        # Early stopping check
        if args.early_stop > 0 and epoch - best_epoch > args.early_stop:
            print(f"Fold {fold_idx}: Early stopping - no improvement for {args.early_stop} epochs")
            break

    # Log final best results for this fold
    logger.log_best_result(best_epoch, best)
    print(f"\nFold {fold_idx} completed! Best results saved to: {fold_output_dir}")

    # Print fold summary
    print("\n" + "=" * 60)
    print(f"FOLD {fold_idx} SUMMARY")
    print("=" * 60)
    print(f"Best Epoch: {best_epoch}")
    print("Best Metrics:")
    for metric, value in best.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        elif isinstance(value, int):
            print(f"  {metric}: {value}")
        else:
            print(f"  {metric}: {value}")
    print("=" * 60)

    return best_epoch, best


def main(args):
    # prepare dataset
    dataset = datasets[args.dataset]
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None

    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        config_path = os.path.join(args.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
    else:
        print("Warning: No output directory specified, results will not be saved!")
        return 0, {}

    # Check if we're doing k-fold cross validation
    if args.k_fold <= 1:
        # Single train/valid split (original behavior)
        train_data = KTData(
            os.path.join(DATA_DIR, dataset["train"]),
            dataset["inputs"],
            seq_len=seq_len,
            batch_size=args.batch_size,
            shuffle=True,
        )
        valid_data = KTData(
            os.path.join(
                DATA_DIR, dataset["valid"] if "valid" in dataset else dataset["test"]
            ),
            dataset["inputs"],
            seq_len=seq_len,
            batch_size=args.test_batch_size,
        )

        best_epoch, best = train_single_fold(args, train_data, valid_data, 1, args.output_dir)
        return best_epoch, best

    else:
        # K-fold cross validation
        print(f"Starting {args.k_fold}-fold cross validation")
        print(f"Random state: {args.random_state}")
        print("=" * 60)

        print("Preparing data for k-fold cross validation...")

        all_data_records = []
        header = ""
        train_path = os.path.join(DATA_DIR, dataset["train"])
        with open(train_path, 'r') as f:
            header = f.readline()
            all_data_records.extend(f.readlines())

        if "valid" in dataset:
            valid_path = os.path.join(DATA_DIR, dataset["valid"])
            if os.path.exists(valid_path):
                with open(valid_path, 'r') as f:
                    f.readline()
                    all_data_records.extend(f.readlines())

        print(f"Loaded a total of {len(all_data_records)} records for K-fold splitting.")

        temp_data_dir = os.path.join(args.output_dir, "kfold_temp_data")
        os.makedirs(temp_data_dir, exist_ok=True)

        try:
            kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.random_state)

            fold_results = []
            all_best_metrics = []

            for fold_idx, (train_indices, val_indices) in enumerate(kf.split(all_data_records)):
                fold_num = fold_idx + 1
                print(f"\n{'=' * 60}")
                print(f"Training Fold {fold_num}/{args.k_fold}")
                print(f"{'=' * 60}")

                train_fold_path = os.path.join(temp_data_dir, f"train_fold_{fold_num}.csv")
                val_fold_path = os.path.join(temp_data_dir, f"val_fold_{fold_num}.csv")

                current_train_records = [all_data_records[i] for i in train_indices]
                current_val_records = [all_data_records[i] for i in val_indices]

                with open(train_fold_path, 'w') as f:
                    f.write(header)
                    f.writelines(current_train_records)

                with open(val_fold_path, 'w') as f:
                    f.write(header)
                    f.writelines(current_val_records)

                print(
                    f"Fold {fold_num}: Training with {len(current_train_records)} records, Validating with {len(current_val_records)} records.")

                train_data = KTData(
                    train_fold_path,
                    dataset["inputs"],
                    seq_len=seq_len,
                    batch_size=args.batch_size,
                    shuffle=True,
                )
                valid_data = KTData(
                    val_fold_path,
                    dataset["inputs"],
                    seq_len=seq_len,
                    batch_size=args.test_batch_size,
                )

                best_epoch, best_metrics = train_single_fold(
                    args, train_data, valid_data, fold_num, args.output_dir
                )

                fold_results.append({
                    'fold': fold_num,
                    'best_epoch': best_epoch,
                    'best_metrics': best_metrics
                })
                all_best_metrics.append(best_metrics)

                print(f"Fold {fold_num} completed. Best AUC: {best_metrics['auc']:.4f}")

        finally:
            print(f"\nCleaning up temporary data directory: {temp_data_dir}")
            shutil.rmtree(temp_data_dir)
            print("Cleanup complete.")

        print(f"\n{'=' * 80}")
        print(f"{args.k_fold}-FOLD CROSS VALIDATION RESULTS")
        print(f"{'=' * 80}")

        metrics_to_analyze = ['auc', 'acc', 'rmse']
        cv_results = {}

        for metric in metrics_to_analyze:
            if all_best_metrics and metric in all_best_metrics[0]:
                values = [result[metric] for result in all_best_metrics]
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv_results[metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'values': values
                }

                print(f"{metric.upper()}: {mean_val:.4f} \u00B1 {std_val:.4f}")
                print(f"  Fold values: {[f'{v:.4f}' for v in values]}")

        cv_summary = {
            'config': vars(args),
            'fold_results': fold_results,
            'cv_summary': cv_results,
            'timestamp': datetime.now().isoformat()
        }

        cv_results_path = os.path.join(args.output_dir, "cross_validation_results.json")
        with open(cv_results_path, 'w') as f:
            json.dump(cv_summary, f, indent=2, default=str)

        summary_path = os.path.join(args.output_dir, "cross_validation_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"{args.k_fold}-FOLD CROSS VALIDATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Number of folds: {args.k_fold}\n")
            f.write(f"Random state: {args.random_state}\n\n")

            for metric, stats in cv_results.items():
                f.write(f"{metric.upper()}: {stats['mean']:.4f} \u00B1 {stats['std']:.4f}\n")
                f.write(f"  Values: {[f'{v:.4f}' for v in stats['values']]}\n\n")

            f.write(f"\nDetailed results saved to: {cv_results_path}\n")

        print(f"\nCross-validation results saved to:")
        print(f"  {cv_results_path}")
        print(f"  {summary_path}")

        best_auc = cv_results['auc']['mean'] if 'auc' in cv_results else 0
        return 0, {'auc': best_auc, 'cv_summary': cv_results}


if __name__ == "__main__":
    args = parser.parse_args()

    print("Training Configuration:")
    print("-" * 40)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-" * 40)

    best_epoch, final_metrics = main(args)

    print(f"\nFinal Overall Results:")
    if args.k_fold > 1:
        print("Results are based on cross-validation averages.")
        if 'cv_summary' in final_metrics and final_metrics['cv_summary']:
            for k, v in final_metrics['cv_summary'].items():
                print(f"  Average {k.upper()}: {v['mean']:.4f} \u00B1 {v['std']:.4f}")
    else:
        print(f"Best Epoch: {best_epoch}")
        print("Best Metrics:")
        for k, v in final_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            elif isinstance(v, int):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {v}")