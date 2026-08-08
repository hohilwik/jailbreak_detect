# probably requires python3.12
# does not work with plain python3.8 for me

"""
Evaluate the trained jailbreak detector on edu data and known jailbreak data
for a range of thresholds
- False Positive Rate (FPR) and count of false positives
- False Negative Rate (FNR) and count of false negatives
"""

import os
import json
import pickle
import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

# config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# env vars / default
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
SVM_DIR = ARTIFACTS_DIR / "atk_svm_models"
JAILBREAK_DIR = ARTIFACTS_DIR / "jailbreak"
EMBEDDINGS_DATA_DIR = Path(os.getenv("EMBEDDINGS_DATA_DIR", "local_data_mdbr"))
INDEX_PATH = ARTIFACTS_DIR / "topic_index.json"

# all helped functions

def load_topic_index() -> dict:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Topic index not found at {INDEX_PATH}")
    with open(INDEX_PATH, "r") as f:
        return json.load(f)

def get_topic_embeddings_path(topic_id: str) -> Path:
    return EMBEDDINGS_DATA_DIR / topic_id / "questions_embeddings.npy"

def load_full_question_embeddings(topic_id: str) -> np.ndarray:
    emb_file = get_topic_embeddings_path(topic_id)
    if not emb_file.exists():
        raise FileNotFoundError(f"Embeddings not found at {emb_file}.")
    return np.load(emb_file)

# JailbreakDetector with max prob extraction

class JailbreakDetector:
    def __init__(self):
        self.attack_models = self._load_models("attack_class")
        self.bypass_models = self._load_models("bypass_technique")

    def _load_models(self, category: str) -> dict:
        models = {}
        pattern = f"jailbreak_{category}_*.pkl"
        for path in SVM_DIR.glob(pattern):
            stem = path.stem
            prefix = f"jailbreak_{category}_"
            if not stem.startswith(prefix):
                continue
            class_val = stem[len(prefix):]
            with open(path, "rb") as f:
                models[class_val] = pickle.load(f)
        log.info(f"Loaded {len(models)} {category} models.")
        return models

    def get_max_probability(self, embedding: np.ndarray) -> float:
        # return the max prob across all attack and bypass models
        # this val compared against threshold for decision
        max_prob = 0.0
        emb = embedding.reshape(1, -1)

        # attack models
        for model in self.attack_models.values():
            emb_scaled = model["scaler"].transform(emb)
            prob = model["svm"].predict_proba(emb_scaled)[0][1]
            if prob > max_prob:
                max_prob = prob

        # bypass models
        for model in self.bypass_models.values():
            emb_scaled = model["scaler"].transform(emb)
            prob = model["svm"].predict_proba(emb_scaled)[0][1]
            if prob > max_prob:
                max_prob = prob

        return float(max_prob)

# evaluate across metrics
# (precompute max probabilities for speed and to enable testing across multiple thresholds)

def compute_edu_max_probs(detector: JailbreakDetector, max_samples_per_topic: int = None) -> np.ndarray:
    # compute max prob for each edu sample, returns 1D array
    topic_index = load_topic_index()
    all_max_probs = []

    log.info("Computing max probs for educational data...")
    for topic_name, info in tqdm(topic_index.items(), desc="Topics"):
        topic_id = info["topic_id"]
        try:
            embeddings = load_full_question_embeddings(topic_id)
        except FileNotFoundError:
            log.warning(f"Embeddings missing for {topic_name} ({topic_id}), skipping")
            continue

        if max_samples_per_topic and len(embeddings) > max_samples_per_topic:
            idx = np.random.choice(len(embeddings), max_samples_per_topic, replace=False)
            embeddings = embeddings[idx]

        for emb in embeddings:
            max_prob = detector.get_max_probability(emb)
            all_max_probs.append(max_prob)

    if not all_max_probs:
        raise RuntimeError("No edu samples processed.")
    return np.array(all_max_probs)


def compute_jailbreak_max_probs(detector: JailbreakDetector, max_samples: int = None) -> np.ndarray:
    # compute max prob for each jailbreak sample, returns 1D array
    emb_file = JAILBREAK_DIR / "embeddings.npy"
    meta_file = JAILBREAK_DIR / "metadata.json"
    if not emb_file.exists() or not meta_file.exists():
        raise FileNotFoundError("Jailbreak embeddings or metadata not found. Run --prepare first")

    jailbreak_emb = np.load(emb_file)
    if max_samples and len(jailbreak_emb) > max_samples:
        idx = np.random.choice(len(jailbreak_emb), max_samples, replace=False)
        jailbreak_emb = jailbreak_emb[idx]

    all_max_probs = []
    log.info("Computing max probs for jailbreak data...")
    for emb in tqdm(jailbreak_emb, desc="Jailbreak samples"):
        max_prob = detector.get_max_probability(emb)
        all_max_probs.append(max_prob)

    return np.array(all_max_probs)


def compute_rates_and_counts(edu_probs: np.ndarray, jail_probs: np.ndarray, thresholds: list):
    """
    for each: threshold, calculate:
    - FPR = (edu_probs >= t).mean()
    - FNR = (jail_probs < t).mean()
    - false_positives = (edu_probs >= t).sum()
    - false_negatives = (jail_probs < t).sum()
    return dict
    """
    fpr_list = []
    fnr_list = []
    fp_count_list = []
    fn_count_list = []
    for t in thresholds:
        fp_mask = edu_probs >= t
        fn_mask = jail_probs < t
        fp_count = fp_mask.sum()
        fn_count = fn_mask.sum()
        # handle divide by zero
        fpr = fp_count / len(edu_probs) if len(edu_probs) > 0 else float('nan')
        fnr = fn_count / len(jail_probs) if len(jail_probs) > 0 else float('nan')
        fpr_list.append(fpr)
        fnr_list.append(fnr)
        fp_count_list.append(int(fp_count))
        fn_count_list.append(int(fn_count))
    return {
        "fpr": fpr_list,
        "fnr": fnr_list,
        "false_positives": fp_count_list,
        "false_negatives": fn_count_list,
    }

# arg options for script

def main():
    parser = argparse.ArgumentParser(description="Evaluate jailbreak detector over thresholds")
    parser.add_argument("--thresholds", type=float, nargs='+', default=None,
                        help="List of thresholds to evaluate (e.g., 0.1 0.2 0.3). "
                             "If not provided, uses 11 points from 0 to 1")
    parser.add_argument("--max-edu-per-topic", type=int, default=None,
                        help="Max edu samples per topic (for speed)")
    parser.add_argument("--max-jailbreak", type=int, default=None,
                        help="Max jailbreak samples to evaluate")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to a JSON file")
    args = parser.parse_args()

    # thresholds
    if args.thresholds is None:
        thresholds = np.linspace(0, 1, 11).tolist()  # 0.0, 0.1, ..., 1.0
    else:
        thresholds = args.thresholds

    # init detector
    log.info("Loading models...")
    detector = JailbreakDetector()

    # precompute max probs
    edu_probs = compute_edu_max_probs(detector, max_samples_per_topic=args.max_edu_per_topic)
    jail_probs = compute_jailbreak_max_probs(detector, max_samples=args.max_jailbreak)

    # calculate rates and counts
    results_dict = compute_rates_and_counts(edu_probs, jail_probs, thresholds)

    # print summary table
    print("\n" + "="*80)
    print("EVALUATION RESULTS FOR DIFFERENT THRESHOLDS")
    print("\n")
    print(f"{'Threshold':>10} | {'FPR':>10} | {'FNR':>10} | {'FP Count':>10} | {'FN Count':>10}")
    print("-"*80)
    for i, t in enumerate(thresholds):
        print(f"{t:10.3f} | {results_dict['fpr'][i]:10.4f} | {results_dict['fnr'][i]:10.4f} | "
              f"{results_dict['false_positives'][i]:10d} | {results_dict['false_negatives'][i]:10d}")
    print("="*80)

    # add metadata to results
    full_results = {
        "thresholds": thresholds,
        "num_edu_samples": len(edu_probs),
        "num_jailbreak_samples": len(jail_probs),
        **results_dict  
        # unpacks fpr, fnr, false_positives, false_negatives
    }

    # save to JSON if output filename provided
    if args.output:
        with open(args.output, "w") as f:
            json.dump(full_results, f, indent=2)
        log.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
