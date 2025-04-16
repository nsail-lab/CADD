import logging
from copy import deepcopy
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
import numpy as np
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        device: str = "cuda",
        epochs: int = 20,
        batch_size: int = 16,
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 1e-3},
        use_scheduler: bool = False,
        use_context: bool = False,
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.use_scheduler = use_scheduler
        self.use_context = use_context

    def train(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
    ):
        train_loader, test_loader = self._init_loaders(train_dataset, test_dataset)

        model = model.to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader) * 2, T_mult=1, eta_min=5e-6) if self.use_scheduler else None

        best_model, best_acc = None, 0
        self.metrics = {"train": [], "test": []}

        for epoch in range(self.epochs):
            LOGGER.info(f"Starting epoch {epoch + 1}")

            train_metrics = self._train_epoch(model, train_loader, criterion, optimizer, scheduler)
            test_metrics = self._evaluate(model, test_loader, criterion)

            LOGGER.info(f"Epoch [{epoch + 1}/{self.epochs}]: "
                        f"train_loss: {train_metrics['loss']:.4f}, train_acc: {train_metrics['accuracy']:.2f}%, "
                        f"test_loss: {test_metrics['loss']:.4f}, test_acc: {test_metrics['accuracy']:.2f}%")

            self.metrics["train"].append(train_metrics)
            self.metrics["test"].append(test_metrics)

            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
                best_model = deepcopy(model.state_dict())

        model.load_state_dict(best_model)
        return model, self.metrics
    
    def _init_loaders(self, train_dataset, test_dataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=6,
        )
        return train_loader, test_loader

    def _forward_and_loss(self, model, criterion, batch_x, batch_y):
        if self.use_context:
            wave, c_e = batch_x
            wave, c_e = wave.to(self.device), c_e.to(self.device)
            batch_out = model(wave, c_e=c_e.float())
        else:
            batch_x = batch_x.to(self.device)
            batch_out = model(batch_x)

        return batch_out, criterion(batch_out, batch_y)

    def _train_epoch(self, model, train_loader, criterion, optimizer, scheduler):
        model.train()
        running_loss, total = 0.0, 0
        all_preds, all_targets = [], []

        pbar = tqdm(train_loader, desc=f"[Training] Loss: n/a")
        for batch_x, batch_y in pbar:
            batch_y = batch_y.unsqueeze(1).float().to(self.device)
            batch_out, loss = self._forward_and_loss(model, criterion, batch_x, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.use_scheduler:
                scheduler.step()

            running_loss += loss.item() * batch_y.size(0)
            total += batch_y.size(0)

            all_preds.append(torch.sigmoid(batch_out).detach().cpu().numpy())
            all_targets.append(batch_y.detach().cpu().numpy())

            pbar.set_description(f"[Training] Loss: {running_loss / total:.4f}")

        return self._compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets), running_loss / total)

    def _evaluate(self, model, data_loader, criterion=torch.nn.BCEWithLogitsLoss(), return_preds=False):
        model.eval()
        running_loss, total = 0.0, 0
        all_preds, all_targets = [], []

        pbar = tqdm(data_loader, desc="[Evaluating]")
        with torch.no_grad():
            for batch_x, batch_y in pbar:
                batch_y = batch_y.unsqueeze(1).float().to(self.device)
                batch_out, loss = self._forward_and_loss(model, criterion, batch_x, batch_y)

                running_loss += loss.item() * batch_y.size(0)
                total += batch_y.size(0)

                all_preds.append(torch.sigmoid(batch_out).cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics = self._compute_metrics(all_preds, all_targets, running_loss / total)
        if return_preds:
            return metrics, all_preds, all_targets
        else:
            return metrics

    def _compute_metrics(self, y_pred, y_true, loss):
        y_pred_label = (y_pred >= 0.5).astype(int)

        accuracy = accuracy_score(y_true, y_pred_label)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_label, average=None, labels=[0, 1])
        precision_overall, recall_overall, f1_overall, _ = precision_recall_fscore_support(y_true, y_pred_label, average='weighted')
        auc = roc_auc_score(y_true, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()

        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        # Convert all numpy types to native types for JSON serialization
        return {
            "loss": float(loss),
            "accuracy": float(100 * accuracy),
            "auc": float(auc),
            "eer": float(eer),
            "precision_real": float(precision[1]),
            "recall_real": float(recall[1]),
            "f1_real": float(f1[1]),
            "precision_fake": float(precision[0]),
            "recall_fake": float(recall[0]),
            "f1_fake": float(f1[0]),
            "precision_overall": float(precision_overall),
            "recall_overall": float(recall_overall),
            "f1_overall": float(f1_overall),
            "true_real": int(tp),
            "false_fake": int(fn),
            "false_real": int(fp),
            "true_fake": int(tn),
        }