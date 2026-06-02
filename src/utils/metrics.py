from rdkit import Chem


def string_exact_match_accuracy(predictions, targets):
    if not targets:
        return 0.0
    return sum(pred == target for pred, target in zip(predictions, targets)) / len(targets)


def canonicalize_smiles_components(smiles):
    parts = [part for part in smiles.split(".") if part]
    if not parts:
        return None
    canonical_parts = []
    for part in parts:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return None
        canonical_parts.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return ".".join(sorted(canonical_parts))


def chemically_equivalent(prediction, target):
    prediction_canonical = canonicalize_smiles_components(prediction)
    target_canonical = canonicalize_smiles_components(target)
    return prediction_canonical is not None and prediction_canonical == target_canonical


def chemical_equivalence_accuracy(predictions, targets):
    if not targets:
        return 0.0
    return sum(chemically_equivalent(pred, target) for pred, target in zip(predictions, targets)) / len(targets)



def exact_match_accuracy(predictions, targets):
    """Primary correctness metric: exact chemical equivalence after canonicalization."""
    return chemical_equivalence_accuracy(predictions, targets)


def token_accuracy(pred_token_ids, target_token_ids, pad_token_id=0):
    correct = 0
    total = 0
    for pred, target in zip(pred_token_ids, target_token_ids):
        for pred_id, target_id in zip(pred, target):
            if target_id == pad_token_id:
                continue
            total += 1
            correct += int(pred_id == target_id)
    return correct / total if total else 0.0


def levenshtein_distance(left, right):
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + int(left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def average_levenshtein_distance(predictions, targets):
    if not targets:
        return 0.0
    return sum(levenshtein_distance(pred, target) for pred, target in zip(predictions, targets)) / len(targets)


def valid_smiles_rate(smiles_list):
    if not smiles_list:
        return 0.0
    valid = 0
    for smiles in smiles_list:
        parts = [part for part in smiles.split(".") if part]
        valid += int(bool(parts) and all(Chem.MolFromSmiles(part) is not None for part in parts))
    return valid / len(smiles_list)


def categorize_prediction(prediction, target):
    categories = []
    if len(target) > 120:
        categories.append("Long sequences")
    is_equivalent = chemically_equivalent(prediction, target)
    if any(ch.isdigit() for ch in prediction + target) and not is_equivalent:
        categories.append("Ring closure errors")
    if "@" in prediction + target and not is_equivalent:
        categories.append("Stereochemistry errors")
    if any(Chem.MolFromSmiles(part) is None for part in prediction.split(".") if part):
        categories.append("Invalid SMILES")
    if not is_equivalent and levenshtein_distance(prediction, target) <= max(5, len(target) * 0.10):
        categories.append("Near misses")
    return categories or ["Other"]


def error_analysis_rows(predictions, targets, limit_correct=50, limit_incorrect=50):
    rows = []
    correct_count = 0
    incorrect_count = 0
    for prediction, target in zip(predictions, targets):
        is_correct = chemically_equivalent(prediction, target)
        if is_correct and correct_count >= limit_correct:
            continue
        if not is_correct and incorrect_count >= limit_incorrect:
            continue
        rows.append(
            {
                "prediction": prediction,
                "target": target,
                "correct": is_correct,
                "string_exact_match": prediction == target,
                "levenshtein": levenshtein_distance(prediction, target),
                "prediction_canonical": canonicalize_smiles_components(prediction),
                "target_canonical": canonicalize_smiles_components(target),
                "categories": categorize_prediction(prediction, target),
            }
        )
        correct_count += int(is_correct)
        incorrect_count += int(not is_correct)
        if correct_count >= limit_correct and incorrect_count >= limit_incorrect:
            break
    return rows
