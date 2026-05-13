import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def logit_analysis(logits, correct, target):
    auroc = get_auroc(logits, correct)
    confidence = averaged_confidence(logits)
    entropy = averaged_entropy(logits)
    atc_score = get_atc_score(logits, target, threshold=0.95)
    margin = averaged_logit_margin(logits, target)
    energy = averaged_energy_score(logits)

    return {
        'auroc': auroc,
        'confidence': confidence,
        'entropy': entropy,
        'atc_score': atc_score,
        'logit_margin': margin,
        'energy_score': energy,
    }


def get_auroc(logits, correct):
    logits = torch.from_numpy(logits)
    probs = torch.softmax(logits, dim=1)
    scores, _ = probs.max(dim=1)
    scores = scores.numpy()
    labels = correct.astype(int)
    auroc = roc_auc_score(labels, scores)
    return auroc

def logits_to_probs(logits):
    return torch.softmax(logits, dim=1)

def confidence_score(logits):
    logits = torch.from_numpy(logits)
    probs = torch.softmax(logits, dim=1)
    conf, _ = probs.max(dim=1)
    return conf  # shape: (N,)

def averaged_confidence(logits):
    return confidence_score(logits).mean().item()

def entropy_score(logits):
    logits = torch.from_numpy(logits)
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs + 1e-12)).sum(dim=1)

def averaged_entropy(logits):
    return entropy_score(logits).mean().item()

def get_atc_score(logits, targets, threshold=0.95):
    logits = torch.from_numpy(logits)
    targets = torch.as_tensor(targets)
    probs = torch.softmax(logits, dim=1)
    max_probs, preds = probs.max(dim=1)

    selected = max_probs >= threshold
    if selected.sum() == 0:
        return 0.0  # no confident predictions

    acc = (preds[selected] == targets[selected]).float().mean().item()
    return acc

def logit_margin(logits, targets):
    logits = torch.from_numpy(logits)
    targets = torch.as_tensor(targets)
    correct_class_logits = logits[torch.arange(len(logits)), targets]
    top2 = torch.topk(logits, k=2, dim=1).values
    best_non_gt = torch.where(
        top2[:,0] == correct_class_logits, top2[:,1], top2[:,0]
    )
    return correct_class_logits - best_non_gt

def averaged_logit_margin(logits, targets):
    return logit_margin(logits, targets).mean().item()

def energy_score(logits):
    logits = torch.from_numpy(logits)
    return -torch.logsumexp(logits, dim=1)

def averaged_energy_score(logits):
    return energy_score(logits).mean().item()
