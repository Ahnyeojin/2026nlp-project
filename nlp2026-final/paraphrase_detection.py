'''
Paraphrase detection을 위한 시작 코드.

고려 사항:
 - ParaphraseGPT: 여러분이 구현한 GPT-2 분류 모델 .
 - train: Quora paraphrase detection 데이터셋에서 ParaphraseGPT를 훈련시키는 절차.
 - test: Test 절차. 프로젝트 결과 제출에 필요한 파일들을 생성함.

실행:
  `python paraphrase_detection.py --use_gpu`
ParaphraseGPT model을 훈련 및 평가하고, 필요한 제출용 파일을 작성한다.
'''

import os
import argparse
import random
import torch
import copy
import math

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW

YES_TOKEN_ID = 8505
NO_TOKEN_ID = 3919

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.backends.cuda.matmul.allow_tf32 = True   # Tensor Core 활성화
  torch.backends.cudnn.allow_tf32 = True


# class for Boosting
class BoostingAdaptor(nn.Module):
  def __init__(self, in_dim, out_dim, rank=4, alpha=8):
    super().__init__()
    self.rank = rank
    self.scaling = alpha / rank

    # compressor: input dimension -> rank
    self.compressor = nn.Parameter(torch.zeros(in_dim, rank))
    # expander: rank -> output dimension
    self.expander = nn.Parameter(torch.zeros(rank, out_dim))

    self.compressor.requires_grad = False
    self.expander.requires_grad = False

    self.is_activated = False
    self.reset_params()

  def reset_params(self):
    nn.init.kaiming_uniform_(self.compressor, a=math.sqrt(5))
    nn.init.zeros_(self.expander)

  def activate_boosting(self):
    self.compressor.requires_grad_(True)
    self.expander.requires_grad_(True)
    self.is_activated = True

  def forward(self, x):
    if self.compressor.requires_grad or self.is_activated:
      compressed = torch.matmul(x, self.compressor)
      expanded = torch.matmul(compressed, self.expander)

      return expanded * self.scaling
    
    return 0.0


class ParaphraseGPT(nn.Module):
  """Paraphrase Detection을 위해 설계된 여러분의 GPT-2 Model."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection 의 출력은 두 가지: 1 (yes) or 0 (no).

    # 기본적으로, 전체 모델을 finetuning 한다.
    for param in self.gpt.parameters():
      param.requires_grad = True

    self.boosting = getattr(args, 'boosting', False)
    if self.boosting:
      self.adaptor_layers = nn.ModuleList([
        BoostingAdaptor(in_dim=args.d, out_dim=args.d, rank=4)
        for _ in range(args.l)
      ]) 

  def forward(self, input_ids, attention_mask):
    """
    TODO: paraphrase_detection_head Linear layer를 사용하여 토큰의 레이블을 예측하시오.

    입력은 다음과 같은 구조를 갖는다:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    따라서, 문장의 끝에서 다음 토큰에 대한 예측을 해야 할 것이다. 
    훈련이 잘 되었다면, 패러프레이즈인 경우에는 토큰 "yes"(BPE index 8505)가, 
    패러프레이즈가 아닌 경우에는 토큰 "no" (BPE index 3919)가 될 것이다.
    """
    ### 완성시켜야 할 빈 코드 블록
    # raise NotImplementedError

    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    last_token = outputs['last_token']
    
    if self.boosting:
      correction = 0.0
      for adaptor_layer in self.adaptor_layers:
        correction += adaptor_layer(last_token)
      last_token = last_token + correction

    logits = self.paraphrase_detection_head(last_token)
    return logits
    
  def activate_adaptor_layer(self):
    for param in self.gpt.parameters():
      param.requires_grad = False
    for param in self.paraphrase_detection_head.parameters():
      param.requires_grad = False

    for adaptor_layer in self.adaptor_layers:
      adaptor_layer.activate_boosting()
  
  def deactivate_and_reset_adaptor_layer(self):
        for param in self.gpt.parameters():
            param.requires_grad = True
        for param in self.paraphrase_detection_head.parameters():
            param.requires_grad = True

        if self.boosting:
            for adaptor_layer in self.adaptor_layers:
                adaptor_layer.compressor.requires_grad_(False)
                adaptor_layer.expander.requires_grad_(False)
                adaptor_layer.is_activated = False
                adaptor_layer.reset_params()
    

def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Quora 데이터셋에서 Paraphrase Detection을 위한 GPT-2 훈련."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # 데이터, 해당 데이터셋 및 데이터로드 생성하기.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_dataset = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_dataset = ParaphraseDetectionDataset(para_dev_data, args)

  current_dataloader = DataLoader(para_train_dataset, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_dataset.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_dataset, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_dataset.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
  best_dev_acc = 0
  snapshots = []    # List for snapshot ensemble

  is_ensemble = args.ensemble > 1
  cycle_length = args.epochs / args.ensemble if is_ensemble else float(args.epochs)

  #pre_boosting = args.epochs // 2
  is_boosting = args.boosting
  hard_samples = {}   # hard sample dictionary for boosing

  for epoch in range(args.epochs):
    epoch_in_cycle = epoch % cycle_length if is_ensemble else epoch
    current_cycle_idx = epoch // cycle_length if is_ensemble else 0
    
    is_boosting_epoch = is_boosting and (epoch_in_cycle == cycle_length - 1)
    is_cycle_restart_epoch = is_ensemble and (epoch_in_cycle == 0) and (current_cycle_idx > 0)

    # Case: Use boosting - implement boosting
    if is_boosting_epoch:
      hard_samples_pool = list(hard_samples.values())
      print(f"\n=== debug:: [Boosting] {epoch} epoch")

      if len(hard_samples_pool) > 0:
        model.activate_adaptor_layer()
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.lr, weight_decay=0.)
        boosting_dataset = copy.copy(para_train_dataset)
        boosting_dataset.data = hard_samples_pool
        current_dataloader = DataLoader(boosting_dataset, shuffle=True, batch_size=args.batch_size,
                                            collate_fn=boosting_dataset.collate_fn)
      else:
        print("\n=== debug:: No Hard Samples")
    elif is_cycle_restart_epoch:
      print(f"\n=== debug:: [Cycle Restart] {epoch} epoch")
      model.deactivate_and_reset_adaptor_layer()
      optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
      current_dataloader = DataLoader(para_train_dataset, shuffle=True, batch_size=args.batch_size, 
                                      collate_fn=para_train_dataset.collate_fn)

    model.train()
    train_loss = 0
    num_batches = 0
    
    num_samples = 0
    train_disagr_count = 0

    # Case: Use snapshot ensemble
    if is_ensemble:
      epoch_per_cycle = epoch % cycle_length
      current_lr = args.lr * (1.0 + np.cos(np.pi * epoch_per_cycle / cycle_length)) / 2.0
      for param_group in optimizer.param_groups:
        param_group['lr'] = max(current_lr, 1e-6)

    for batch in tqdm(current_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      if labels.dim() == 2:
        is_yes = torch.any(labels == YES_TOKEN_ID, dim=1)
      else:
        is_yes = (labels == YES_TOKEN_ID)
      # yes_t = torch.tensor(YES_TOKEN_ID, dtype=torch.long, device=labels.device)
      # no_t  = torch.tensor(NO_TOKEN_ID,  dtype=torch.long, device=labels.device)
      mapped_labels = torch.where(is_yes, torch.tensor(1, device=device), torch.tensor(0, device=device))

      # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트. 
      optimizer.zero_grad()

      # Case: Use reverse pair
      if args.reverse > 0.0:
        logits_forward = model(b_ids, b_mask)
        loss_cross_entropy = F.cross_entropy(logits_forward, mapped_labels, reduction='mean')

        b_ids_reverse = batch['reverse_token_ids'].to(device)
        b_mask_reverse = batch['reverse_attention_mask'].to(device)
        logits_reverse = model(b_ids_reverse, b_mask_reverse)

        p_forward = F.log_softmax(logits_forward, dim=-1)
        p_backward = F.softmax(logits_reverse, dim=-1)
        loss_kl_div = F.kl_div(p_forward, p_backward, reduction='batchmean')

        loss = loss_cross_entropy + (args.reverse * loss_kl_div)

        # check disagreement of forward-reverse pair
        preds_forward = torch.argmax(logits_forward, dim=1)
        preds_reverse = torch.argmax(logits_reverse, dim=1)
        train_disagr_count += (preds_forward != preds_reverse).sum().item()
        num_samples += labels.size(0)

        preds = preds_forward
        correct_forward = (preds_forward == mapped_labels)
        correct_reverse = (preds_reverse == mapped_labels)
        is_correct = correct_forward & correct_reverse
      else:
        logits = model(b_ids, b_mask)
        preds = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, mapped_labels, reduction='mean')
        is_correct = (preds == mapped_labels)

      loss.backward()
      optimizer.step()

      if is_boosting and (not is_boosting_epoch):
        correct_mask = is_correct.cpu().numpy()
        for i in range(len(b_ids)):
          sample_id = batch['sent_ids'][i]

          if not correct_mask[i]:
            sample_data = {k: v[i].cpu() if isinstance(v, torch.Tensor) else v[i]
                           for k, v in batch.items()}
            hard_samples[sample_id] = sample_data
          else:
            hard_samples.pop(sample_id, None)
          
      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    train_disagr_rate = (train_disagr_count / num_samples) if num_samples > 0 else 0.0    # forward-reverse disagreement rate

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train disagreement :: {train_disagr_rate :.3f} dev acc :: {dev_acc :.3f}")

    if is_ensemble:
      is_cycle_end = int((epoch + 1) % cycle_length) == 0 or (epoch == args.epochs - 1)
      if is_cycle_end and len(snapshots) < args.ensemble:
        print(f"Snapshot Captured - Epoch: {epoch}")
        snapshots.append(copy.deepcopy(model.state_dict()))
        torch.save({'snapshots': snapshots, 'args': args}, args.filepath + '.ensemble')


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  
  is_ensemble = args.ensemble > 1
  if is_ensemble and os.path.exists(args.filepath + ".ensemble"):
    saved = torch.load(args.filepath + ".ensemble", map_location=device)
    snapshots = saved['snapshots']
    saved_args = saved['args']
  else:
    saved = torch.load(args.filepath, map_location=device)
    snapshots = [saved['model']]
    saved_args = saved['args']

  model = ParaphraseGPT(saved_args)
  model = model.to(device)

  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)
  def ensemble_inference(dataloader, is_test=False):
    all_preds = []
    all_sent_ids = []

    # variables for calculating forward-reverse disagreement rate
    num_samples = 0
    test_disagr_count = 0
    disagr_rate = 0.0

    total_correct = 0
    total_samples = 0

    for batch in dataloader:
      b_ids = batch['token_ids'].to(device)
      b_mask = batch['attention_mask'].to(device)
      sent_ids = batch['sent_ids']

      has_reverse = 'reverse_token_ids' in batch
      if has_reverse:
        b_ids_reverse = batch['reverse_token_ids'].to(device)
        b_mask_reverse = batch['reverse_attention_mask'].to(device)

      batch_probs = []
      batch_reverse_probs = []    # for reverse pair test

      for state_dict in snapshots:
        model.load_state_dict(state_dict, strict=False)

        if model.boosting:
          for layer in model.adaptor_layers:
            layer.is_activated = True

        model.eval()

        logits = model(b_ids, b_mask)
        batch_probs.append(F.softmax(logits, dim=-1))

        if has_reverse:
          logits_reverse = model(b_ids_reverse, b_mask_reverse)
          yes_no_logits_reverse = logits_reverse[:, [NO_TOKEN_ID, YES_TOKEN_ID]]
          batch_reverse_probs.append(F.softmax(yes_no_logits_reverse, dim=-1))

      avg_probs = torch.stack(batch_probs, dim=0).mean(dim=0)   # [batch, 2]
      binary_preds = torch.argmax(avg_probs, dim=1)              # 0=no, 1=yes
      
      if 'labels' in batch:
        labels = batch['labels'].to(device)
        is_yes = torch.any(labels == YES_TOKEN_ID, dim=1) if labels.dim() == 2 else (labels == YES_TOKEN_ID)
        mapped_labels = torch.where(is_yes, torch.tensor(1, device=device), torch.tensor(0, device=device))
            
        # 예측값과 실제 정답 라벨 비교 후 일치하는 개수 누적
        total_correct += (binary_preds == mapped_labels).sum().item()
        total_samples += labels.size(0)
        
      if has_reverse:
        avg_reverse_probs = torch.stack(batch_reverse_probs, dim=0).mean(dim=0)
        binary_preds_reverse = torch.argmax(avg_reverse_probs, dim=1)
        test_disagr_count += (binary_preds != binary_preds_reverse).sum().item()
        num_samples += b_ids.size(0)


      token_map = torch.tensor([NO_TOKEN_ID, YES_TOKEN_ID], device=device)
      preds = token_map[binary_preds].cpu().numpy()

      all_preds.extend(preds.tolist())
      all_sent_ids.extend(sent_ids)

    disagr_rate = (test_disagr_count / num_samples) if num_samples > 0 else 0.0
    accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0

    return all_preds, all_sent_ids, disagr_rate, accuracy


  # dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  # test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)
  
  
  dev_preds, dev_ids, dev_disagr_rate, dev_para_acc = ensemble_inference(para_dev_dataloader, is_test=False)
  test_preds, test_ids, test_disagr_rate, _ = ensemble_inference(para_test_dataloader, is_test=True)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")


  with open(args.para_dev_out, "w+", encoding='utf-8') as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    # for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
    #   f.write(f"{p}, {s} \n")
    for p, s in zip(dev_ids, dev_preds):
      f.write(f"{p}, {s} \n")
    
    f.write(f"# Summary:: Lambda={args.reverse}, Disagr Rate={dev_disagr_rate:.3f}")

  with open(args.para_test_out, "w+", encoding='utf-8') as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    # for p, s in zip(test_para_sent_ids, test_para_y_pred):
    #   f.write(f"{p}, {s} \n")
    for p, s in zip(test_ids, test_preds):
      f.write(f"{p}, {s} \n")

    f.write(f"# Summary:: Lambda={args.reverse}, Disagr Rate={test_disagr_rate:.3f}")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  
  parser.add_argument("--reverse", type=float, default=0.0, help='coefficient value of KL divergence loss')
  parser.add_argument("--ensemble", type=int,  default=1, help='number of ensemble snapshots')
  parser.add_argument("--boosting", action='store_true', help='Activate LoRA based boosting')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """모델 크기에 따라 결정되는 인수들을 추가."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'  # 경로명 저장.
  seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
  train(args)
  test(args)
