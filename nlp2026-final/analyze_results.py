"""
실험 결과 분석 및 시각화 + 에러 케이스 분석
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

PRED_DIR = 'predictions'
DATA_DIR = 'data'
OUT_DIR  = 'analysis_output'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 헬퍼 ──────────────────────────────────────────────────────────────

def load_sentiment_pred(filepath):
    preds = {}
    with open(filepath, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # header
        for row in f:
            row = row.strip()
            if not row: continue
            parts = [x.strip() for x in row.split(',')]
            if len(parts) >= 2:
                preds[parts[0].strip()] = int(parts[1].strip())
    return preds

def load_sentiment_dev(filepath):
    labels = {}
    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sent_id = row['id'].strip()
            labels[sent_id] = int(row['sentiment'].strip())
    return labels

def load_para_pred(filepath):
    preds = {}
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()[1:]  # skip header
    for line in lines:
        line = line.strip()
        if not line: continue
        parts = [x.strip() for x in line.split(',')]
        if len(parts) >= 2:
            preds[parts[0].strip()] = int(parts[1].strip())
    return preds

def load_para_dev(filepath):
    labels = {}
    sents  = {}
    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                sid = row['id'].strip()
                labels[sid] = int(float(row['is_duplicate']))
                sents[sid]  = (row['sentence1'].strip(), row['sentence2'].strip())
            except: pass
    return labels, sents

# ── 1. Sentiment Analysis 결과 ────────────────────────────────────────

configs = [
    ('last-linear-layer', 'sst'),
    ('last-linear-layer', 'cfimdb'),
    ('full-model',        'sst'),
    ('full-model',        'cfimdb'),
]

baselines = {
    ('last-linear-layer','sst'):    0.462,
    ('full-model',       'sst'):    0.513,
    ('last-linear-layer','cfimdb'): 0.861,
    ('full-model',       'cfimdb'): 0.976,
}

results = {}
print("=" * 55)
print("  Sentiment Analysis 실험 결과")
print("=" * 55)

for mode, dataset in configs:
    pred_file = os.path.join(PRED_DIR, f'{mode}-{dataset}-dev-out.csv')
    dev_file  = os.path.join(DATA_DIR, f'ids-{dataset}-dev.csv')

    try:
        with open(pred_file, encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines()[1:] if l.strip()]
        preds, ids = [], []
        for line in lines:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 2:
                ids.append(parts[0])
                preds.append(int(parts[1]))

        true_map = load_sentiment_dev(dev_file)
        trues = [true_map[i] for i in ids if i in true_map]
        preds = [p for i, p in zip(ids, preds) if i in true_map]

        acc = accuracy_score(trues, preds)
        baseline = baselines[(mode, dataset)]
        diff = acc - baseline
        results[(mode, dataset)] = acc
        print(f"  {mode:<22} | {dataset.upper():<7} | acc={acc:.3f}  (기준:{baseline:.3f}, {diff:+.3f})")
    except Exception as e:
        print(f"  {mode} / {dataset}: 오류 — {e}")

print()

# ── 2. 시각화: 막대 그래프 ────────────────────────────────────────────

labels_x  = ['SST\nlast-linear', 'SST\nfull-model',
              'CFIMDB\nlast-linear', 'CFIMDB\nfull-model']
our_vals  = [
    results.get(('last-linear-layer','sst'),    0),
    results.get(('full-model',       'sst'),    0),
    results.get(('last-linear-layer','cfimdb'), 0),
    results.get(('full-model',       'cfimdb'), 0),
]
base_vals = [baselines[k] for k in configs]

x = np.arange(len(labels_x))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, base_vals, w, label='기준 모델(Baseline)', color='#9DB2CE', edgecolor='white')
b2 = ax.bar(x + w/2, our_vals,  w, label='본 모델(Ours)',       color='#4472C4', edgecolor='white')

ax.set_ylabel('Dev Accuracy', fontsize=12)
ax.set_title('Sentiment Analysis: 기준 모델 vs 본 모델', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels_x, fontsize=10)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=11)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

for bar in b1: ax.annotate(f'{bar.get_height():.3f}',
    xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
    xytext=(0,3), textcoords='offset points', ha='center', fontsize=9, color='#555')
for bar in b2: ax.annotate(f'{bar.get_height():.3f}',
    xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
    xytext=(0,3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_sentiment_accuracy.png'), dpi=150)
plt.close()
print("  [저장] fig1_sentiment_accuracy.png")

# ── 3. Paraphrase Detection 에러 케이스 분석 ─────────────────────────

print()
print("=" * 55)
print("  Paraphrase Detection 에러 케이스 분석")
print("=" * 55)

YES_TOKEN = 8505
NO_TOKEN  = 3919

try:
    true_labels, sents = load_para_dev(os.path.join(DATA_DIR, 'quora-dev.csv'))
    pred_map = load_para_pred(os.path.join(PRED_DIR, 'para-dev-output.csv'))

    common_ids = [sid for sid in pred_map if sid in true_labels]

    # 토큰 ID → 0/1 변환
    def tok_to_label(tok):
        if tok == YES_TOKEN: return 1
        if tok == NO_TOKEN:  return 0
        return 1 if tok == YES_TOKEN else 0  # default

    trues = [true_labels[sid] for sid in common_ids]
    preds = [tok_to_label(pred_map[sid]) for sid in common_ids]

    acc = accuracy_score(trues, preds)
    print(f"  Paraphrase Detection Dev Accuracy: {acc:.3f}")

    # Confusion matrix
    cm = confusion_matrix(trues, preds)
    print(f"  Confusion Matrix:\n  {cm}")

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Not Para (0)', 'Para (1)'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Paraphrase Detection\nConfusion Matrix (Dev Set)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig2_para_confusion.png'), dpi=150)
    plt.close()
    print("  [저장] fig2_para_confusion.png")

    # 에러 케이스 추출
    errors = [(sid, trues[i], preds[i]) for i, sid in enumerate(common_ids) if trues[i] != preds[i]]
    fp = [(sid, sents[sid]) for sid, t, p in errors if t == 0 and p == 1]  # 아닌데 맞다고
    fn = [(sid, sents[sid]) for sid, t, p in errors if t == 1 and p == 0]  # 맞는데 아니라고

    print(f"\n  전체 에러: {len(errors)}개 / {len(common_ids)}개")
    print(f"  False Positive (아닌데 paraphrase라고): {len(fp)}개")
    print(f"  False Negative (맞는데 아니라고):       {len(fn)}개")

    # 에러 케이스 파일 저장
    with open(os.path.join(OUT_DIR, 'error_cases.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("FALSE POSITIVE (실제 0, 예측 1) — 아닌데 paraphrase라고 판단\n")
        f.write("=" * 70 + "\n")
        for sid, (s1, s2) in fp[:10]:
            f.write(f"ID: {sid}\n")
            f.write(f"  Q1: {s1}\n")
            f.write(f"  Q2: {s2}\n\n")

        f.write("=" * 70 + "\n")
        f.write("FALSE NEGATIVE (실제 1, 예측 0) — 맞는데 paraphrase 아니라고 판단\n")
        f.write("=" * 70 + "\n")
        for sid, (s1, s2) in fn[:10]:
            f.write(f"ID: {sid}\n")
            f.write(f"  Q1: {s1}\n")
            f.write(f"  Q2: {s2}\n\n")

    print("  [저장] error_cases.txt")

    # 에러 비율 파이차트
    labels_pie = ['정답 (Correct)', 'False Positive', 'False Negative']
    sizes_pie  = [len(common_ids)-len(errors), len(fp), len(fn)]
    colors_pie = ['#4472C4', '#ED7D31', '#FF0000']
    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(sizes_pie, labels=labels_pie, colors=colors_pie,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 11})
    ax.set_title('Paraphrase Detection 예측 결과 분포\n(Dev Set)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig3_para_error_dist.png'), dpi=150)
    plt.close()
    print("  [저장] fig3_para_error_dist.png")

except Exception as e:
    import traceback
    print(f"  Paraphrase 분석 오류: {e}")
    traceback.print_exc()

# ── 4. 학습 곡선 (로그에서 추출한 데이터) ────────────────────────────

print()
print("=" * 55)
print("  학습 곡선 (train_log.txt 기반)")
print("=" * 55)

# train_log에서 추출한 CFIMDB full-model epoch별 결과
epochs_fm = [5, 6, 7, 8, 9]
dev_acc_fm = [0.984, 0.976, 0.984, 0.947, 0.984]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epochs_fm, dev_acc_fm, 'o-', color='#4472C4', linewidth=2, markersize=7, label='CFIMDB full-model')
ax.axhline(y=0.976, color='#ED7D31', linestyle='--', linewidth=1.5, label='기준 모델 (0.976)')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Dev Accuracy', fontsize=12)
ax.set_title('CFIMDB Full-Model 학습 곡선', fontsize=13, fontweight='bold')
ax.set_ylim(0.9, 1.01)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_learning_curve.png'), dpi=150)
plt.close()
print("  [저장] fig4_learning_curve.png")

print()
print(f"  분석 완료! 결과 폴더: {os.path.abspath(OUT_DIR)}")
