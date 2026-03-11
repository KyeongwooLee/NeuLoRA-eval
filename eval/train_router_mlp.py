from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

try:
    from .retrieval import EmbeddingBackend
except ImportError:  # pragma: no cover - script execution fallback
    from retrieval import EmbeddingBackend


STYLE_ORDER = ("direct", "socratic", "scaffolding", "feedback")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _validate_label_coverage(rows: list[dict]) -> None:
    seen = {str(row.get("label", "")).strip() for row in rows}
    missing = [label for label in STYLE_ORDER if label not in seen]
    if missing:
        raise RuntimeError(f"Training data is missing style labels required for router: {sorted(missing)}")


def _split_by_conversation(rows: list[dict], *, val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for idx, row in enumerate(rows):
        conv_id = str(row.get("conversation_id", "")).strip() or f"row_{idx}"
        by_conv[conv_id].append(row)

    conv_ids = list(by_conv.keys())
    rng = random.Random(seed)
    rng.shuffle(conv_ids)
    val_size = max(1, int(len(conv_ids) * val_ratio))
    if val_size >= len(conv_ids):
        val_size = max(1, len(conv_ids) - 1)

    val_set = set(conv_ids[:val_size])
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for conv_id, items in by_conv.items():
        if conv_id in val_set:
            val_rows.extend(items)
        else:
            train_rows.extend(items)
    return train_rows, val_rows


def _rows_to_xy(rows: list[dict], label_to_idx: dict[str, int]) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        label = str(row.get("label", "")).strip()
        if not text or label not in label_to_idx:
            continue
        texts.append(text)
        labels.append(label_to_idx[label])
    return texts, labels


def _chunked_embeddings(embedder: EmbeddingBackend, texts: list[str], *, batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), max(1, batch_size)):
        batch = texts[start : start + max(1, batch_size)]
        vectors.extend(embedder.encode(batch))
    return vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeuLoRA 2-layer MLP style router.")
    parser.add_argument(
        "--train-data-path",
        type=Path,
        default=Path(__file__).resolve().parent / "cache" / "router" / "router_train_multiturn.jsonl",
    )
    parser.add_argument(
        "--router-model-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "LangGraph" / "router_model.json",
    )
    parser.add_argument("--embedding-model-name", type=str, default="BAAI/bge-m3")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as error:
        raise RuntimeError("PyTorch is required to train the mlp router.") from error

    rows = _read_jsonl(args.train_data_path)
    if not rows:
        raise RuntimeError(f"No training rows found: {args.train_data_path}")
    _validate_label_coverage(rows)

    label_to_idx = {label: idx for idx, label in enumerate(STYLE_ORDER)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    train_rows, val_rows = _split_by_conversation(rows, val_ratio=args.val_ratio, seed=args.seed)
    if not train_rows:
        raise RuntimeError("Conversation split produced an empty train set.")
    if not val_rows:
        raise RuntimeError("Conversation split produced an empty validation set.")

    train_texts, train_labels = _rows_to_xy(train_rows, label_to_idx)
    val_texts, val_labels = _rows_to_xy(val_rows, label_to_idx)
    if not train_texts or not val_texts:
        raise RuntimeError("No usable samples after filtering empty text/labels.")

    embedder = EmbeddingBackend(model_name=args.embedding_model_name)
    train_vectors = _chunked_embeddings(embedder, train_texts, batch_size=max(1, args.embed_batch_size))
    val_vectors = _chunked_embeddings(embedder, val_texts, batch_size=max(1, args.embed_batch_size))

    input_dim = len(train_vectors[0])
    num_classes = len(STYLE_ORDER)
    hidden_dim = max(1, int(args.hidden_dim))

    train_x = torch.tensor(train_vectors, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    val_x = torch.tensor(val_vectors, dtype=torch.float32)
    val_y = torch.tensor(val_labels, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=max(1, args.batch_size),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=max(1, args.batch_size),
        shuffle=False,
    )

    class RouterMLP(nn.Module):
        def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hid_dim)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(hid_dim, out_dim)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    model = RouterMLP(input_dim, hidden_dim, num_classes)

    counts = defaultdict(int)
    for label_idx in train_labels:
        counts[int(label_idx)] += 1
    total = max(1, len(train_labels))
    class_weights = []
    for idx in range(num_classes):
        c = max(1, counts.get(idx, 1))
        class_weights.append(total / (num_classes * c))
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    best_val_loss = float("inf")
    best_state = None
    best_metrics = None
    bad_epochs = 0
    metrics_per_epoch: list[dict] = []

    for epoch in range(1, max(1, int(args.epochs)) + 1):
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            train_steps += 1

        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_steps = 0
            val_correct = 0
            val_total = 0
            for xb, yb in val_loader:
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_loss_sum += float(loss.item())
                val_steps += 1
                pred = torch.argmax(logits, dim=1)
                val_correct += int((pred == yb).sum().item())
                val_total += int(yb.shape[0])

            val_loss = val_loss_sum / max(1, val_steps)
            val_acc = val_correct / max(1, val_total)
            train_loss = train_loss_sum / max(1, train_steps)

        metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        }
        metrics_per_epoch.append(metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= max(1, int(args.patience)):
                break

    if best_state is None:
        raise RuntimeError("MLP training failed to produce a valid checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_logits = model(train_x)
        train_pred = torch.argmax(train_logits, dim=1)
        train_acc = float((train_pred == train_y).float().mean().item())
        val_logits = model(val_x)
        val_pred = torch.argmax(val_logits, dim=1)
        val_acc = float((val_pred == val_y).float().mean().item())

    router_payload = {}
    if args.router_model_path.exists():
        router_payload = json.loads(args.router_model_path.read_text(encoding="utf-8"))
    else:
        router_payload = {"styles": list(STYLE_ORDER), "centroids": {}}

    if "styles" not in router_payload:
        router_payload["styles"] = list(STYLE_ORDER)

    state_dict = model.state_dict()
    router_payload["classifier"] = {
        "type": "mlp2",
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "labels": [idx_to_label[idx] for idx in range(num_classes)],
        "state_dict": {
            "fc1.weight": state_dict["fc1.weight"].detach().cpu().tolist(),
            "fc1.bias": state_dict["fc1.bias"].detach().cpu().tolist(),
            "fc2.weight": state_dict["fc2.weight"].detach().cpu().tolist(),
            "fc2.bias": state_dict["fc2.bias"].detach().cpu().tolist(),
        },
        "metrics": {
            "train_samples": len(train_texts),
            "val_samples": len(val_texts),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "epochs_ran": len(metrics_per_epoch),
            "history": metrics_per_epoch,
            "best": best_metrics or {},
            "seed": args.seed,
            "embedding_model_name": args.embedding_model_name,
        },
    }

    args.router_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.router_model_path.write_text(
        json.dumps(router_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output = {
        "router_model_path": str(args.router_model_path),
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "label_distribution_train": {
            label: int(sum(1 for idx in train_labels if idx == label_to_idx[label]))
            for label in STYLE_ORDER
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
