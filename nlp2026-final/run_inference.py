"""최종 모델로 dev/test 예측 생성 (CUDA assert 우회: CPU 로드 후 GPU 이동)"""
import torch
import argparse
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from paraphrase_detection import (
    ParaphraseGPT, add_arguments, load_paraphrase_data,
    ParaphraseDetectionDataset, ParaphraseDetectionTestDataset
)
from torch.utils.data import DataLoader

YES_TOKEN_ID = 8505
NO_TOKEN_ID  = 3919


def infer(model, snapshots, dataloader, device, has_labels=True):
    all_preds, all_ids = [], []
    total_correct = total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="infer"):
            b_ids  = batch["token_ids"].to(device)
            b_mask = batch["attention_mask"].to(device)
            sent_ids = batch["sent_ids"]

            batch_probs = []
            for sd in snapshots:
                # CPU 상태의 state_dict을 GPU로 이동 후 로드
                sd_gpu = {k: v.to(device) for k, v in sd.items()}
                model.load_state_dict(sd_gpu, strict=False)
                if model.boosting:
                    for layer in model.adaptor_layers:
                        layer.is_activated = True
                model.eval()
                logits = model(b_ids, b_mask)
                batch_probs.append(torch.softmax(logits, dim=-1))

            avg_probs = torch.stack(batch_probs).mean(0)
            preds = torch.argmax(avg_probs, dim=1)          # 0=no, 1=yes
            token_map = torch.tensor([NO_TOKEN_ID, YES_TOKEN_ID], device=device)
            token_preds = token_map[preds].cpu().numpy()

            if has_labels and "labels" in batch:
                labels = batch["labels"].to(device)
                is_yes = (labels == YES_TOKEN_ID) if labels.dim() == 1 \
                         else torch.any(labels == YES_TOKEN_ID, dim=1)
                mapped = torch.where(is_yes,
                                     torch.tensor(1, device=device),
                                     torch.tensor(0, device=device))
                total_correct += (preds == mapped).sum().item()
                total_samples += labels.size(0)

            all_preds.extend(token_preds.tolist())
            all_ids.extend(sent_ids)

    acc = total_correct / total_samples if total_samples > 0 else 0.0
    return all_preds, all_ids, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--filepath", type=str, default="10-1e-05-paraphrase.pt")
    parser.add_argument("--para_dev",      type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test",     type=str, default="data/quora-test-input.csv")
    parser.add_argument("--para_dev_out",  type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda") if args.use_gpu and torch.cuda.is_available() \
             else torch.device("cpu")
    print(f"Device: {device}")

    # 앙상블 파일 로드 (CPU로)
    ensemble_path = args.filepath + ".ensemble"
    if os.path.exists(ensemble_path):
        ens = torch.load(ensemble_path, map_location="cpu")
        snapshots  = ens["snapshots"]
        saved_args = ens["args"]
        print(f"Ensemble snapshots loaded: {len(snapshots)}")
    else:
        saved = torch.load(args.filepath, map_location="cpu")
        snapshots  = [saved["model"]]
        saved_args = saved["args"]
        print("Single model loaded.")

    model = ParaphraseGPT(saved_args).to(device)

    # Dev
    dev_data   = ParaphraseDetectionDataset(load_paraphrase_data(args.para_dev), args)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False,
                            collate_fn=dev_data.collate_fn)
    dev_preds, dev_ids, dev_acc = infer(model, snapshots, dev_loader, device, has_labels=True)
    print(f"Dev acc: {dev_acc:.3f}")

    with open(args.para_dev_out, "w+", encoding="utf-8") as f:
        f.write("id \t Predicted_Is_Paraphrase \n")
        for s, p in zip(dev_ids, dev_preds):
            f.write(f"{s}, {p} \n")
        f.write(f"# Summary:: Dev acc={dev_acc:.3f}\n")

    # Test
    test_data   = ParaphraseDetectionTestDataset(load_paraphrase_data(args.para_test, split='test'), args)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                             collate_fn=test_data.collate_fn)
    test_preds, test_ids, _ = infer(model, snapshots, test_loader, device, has_labels=False)

    with open(args.para_test_out, "w+", encoding="utf-8") as f:
        f.write("id \t Predicted_Is_Paraphrase \n")
        for s, p in zip(test_ids, test_preds):
            f.write(f"{s}, {p} \n")

    print("예측 파일 저장 완료!")
    print(f"  dev  → {args.para_dev_out}")
    print(f"  test → {args.para_test_out}")


if __name__ == "__main__":
    main()
