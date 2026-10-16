#!/usr/bin/env python3
"""Analyze old vs new pipeline results to find patterns in wins, losses, and failure modes."""
import json
import sys

def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return {item['req']['case']['caseNo']: item for item in data}

def molecule_match(expected_meds, predicted_meds):
    """Match at molecule level (ignoring brand name differences)."""
    expected_molecules = set()
    for m in expected_meds:
        mol = m.get('canonical_molecule', '')
        if mol:
            expected_molecules.add(mol.lower())
    
    predicted_molecules = set()
    for m in predicted_meds:
        mol = m.get('canonical_molecule', '')
        if mol:
            predicted_molecules.add(mol.lower())
    
    if not expected_molecules and not predicted_molecules:
        return 1.0, set(), set(), set()
    if not expected_molecules:
        return 0.0 if predicted_molecules else 1.0, set(), predicted_molecules, set()
    
    matched = expected_molecules & predicted_molecules
    missed = expected_molecules - predicted_molecules
    extra = predicted_molecules - expected_molecules
    
    # Jaccard-like score
    if len(expected_molecules) == 0:
        score = 1.0 if len(predicted_molecules) == 0 else 0.0
    else:
        score = len(matched) / len(expected_molecules)
    
    return score, matched, extra, missed

def dosage_group_match(expected_meds, predicted_meds):
    """Match at dosage_group_id level (more precise)."""
    expected_groups = set()
    for m in expected_meds:
        grp = m.get('dosage_group_id', '')
        if grp:
            expected_groups.add(grp)
    
    predicted_groups = set()
    for m in predicted_meds:
        grp = m.get('dosage_group_id', '')
        if grp:
            predicted_groups.add(grp)
    
    if not expected_groups and not predicted_groups:
        return 1.0
    if not expected_groups:
        return 0.0 if predicted_groups else 1.0
    
    matched = expected_groups & predicted_groups
    return len(matched) / len(expected_groups)

old_path = '/home/hardik/Projects/meldai_pro/tmp/prescribe_pipeline_20261012_104613.json'
new_path = '/home/hardik/Projects/meldai_pro/tmp/prescribe_pipeline_20261014_102216.json'

old_data = load_results(old_path)
new_data = load_results(new_path)

# All case numbers
all_cases = sorted(set(list(old_data.keys()) + list(new_data.keys())))

print(f"Old run: {len(old_data)} cases")
print(f"New run: {len(new_data)} cases")
print(f"Common cases: {len(set(old_data.keys()) & set(new_data.keys()))}")
print()

# Case-by-case analysis
results = []
for case_no in all_cases:
    if case_no not in old_data or case_no not in new_data:
        continue
    
    old_item = old_data[case_no]
    new_item = new_data[case_no]
    
    expected = old_item['expectedMeds']  # Same in both
    if 'medications' not in old_item['res'] or 'medications' not in new_item['res']:
        continue
    old_pred = old_item['res']['medications']
    new_pred = new_item['res']['medications']
    
    old_score, old_matched, old_extra, old_missed = molecule_match(expected, old_pred)
    new_score, new_matched, new_extra, new_missed = molecule_match(expected, new_pred)
    
    old_grp_score = dosage_group_match(expected, old_pred)
    new_grp_score = dosage_group_match(expected, new_pred)
    
    visit_type = old_item['req']['case']['visitType']
    diagnosis = old_item['req']['case']['diagnosis']
    past_cases = old_item['res'].get('pastCasesCount', 0)
    similar_cases = old_item['res'].get('similarCasesCount', 0)
    
    delta = new_score - old_score
    
    results.append({
        'case_no': case_no,
        'visit_type': visit_type,
        'diagnosis': diagnosis,
        'past_cases': past_cases,
        'similar_cases': similar_cases,
        'old_score': old_score,
        'new_score': new_score,
        'delta': delta,
        'old_matched': old_matched,
        'old_extra': old_extra,
        'old_missed': old_missed,
        'new_matched': new_matched,
        'new_extra': new_extra,
        'new_missed': new_missed,
        'expected_count': len(expected),
        'old_pred_count': len(old_pred),
        'new_pred_count': len(new_pred),
        'old_grp_score': old_grp_score,
        'new_grp_score': new_grp_score,
    })

# Summary
old_avg = sum(r['old_score'] for r in results) / len(results) * 100
new_avg = sum(r['new_score'] for r in results) / len(results) * 100
print(f"Average molecule recall: OLD={old_avg:.1f}%  NEW={new_avg:.1f}%  Δ={new_avg - old_avg:+.1f}%")

old_grp_avg = sum(r['old_grp_score'] for r in results) / len(results) * 100
new_grp_avg = sum(r['new_grp_score'] for r in results) / len(results) * 100
print(f"Average dosage-group recall: OLD={old_grp_avg:.1f}%  NEW={new_grp_avg:.1f}%  Δ={new_grp_avg - old_grp_avg:+.1f}%")

# Regimen size analysis
old_size_delta = sum(abs(r['old_pred_count'] - r['expected_count']) for r in results) / len(results)
new_size_delta = sum(abs(r['new_pred_count'] - r['expected_count']) for r in results) / len(results)
print(f"Average regimen size error: OLD={old_size_delta:.2f}  NEW={new_size_delta:.2f}")

# Extra drugs (false positives)
old_extra_avg = sum(len(r['old_extra']) for r in results) / len(results)
new_extra_avg = sum(len(r['new_extra']) for r in results) / len(results)
print(f"Average extra molecules: OLD={old_extra_avg:.2f}  NEW={new_extra_avg:.2f}")

print()

# Cases that improved
improved = [r for r in results if r['delta'] > 0]
regressed = [r for r in results if r['delta'] < 0]
unchanged = [r for r in results if r['delta'] == 0]

print(f"Improved: {len(improved)} cases")
print(f"Regressed: {len(regressed)} cases")
print(f"Unchanged: {len(unchanged)} cases")
print()

# === DETAILED: Cases that REGRESSED ===
print("=" * 80)
print("REGRESSED CASES (new prompt did worse)")
print("=" * 80)
for r in sorted(regressed, key=lambda x: x['delta']):
    print(f"\nCase {r['case_no']} ({r['visit_type']}) — {', '.join(r['diagnosis'])}")
    print(f"  Score: {r['old_score']:.0%} → {r['new_score']:.0%} (Δ={r['delta']:+.0%})")
    print(f"  Expected: {r['expected_count']} meds, Old pred: {r['old_pred_count']}, New pred: {r['new_pred_count']}")
    print(f"  OLD matched: {r['old_matched']}")
    print(f"  OLD missed:  {r['old_missed']}")
    print(f"  OLD extra:   {r['old_extra']}")
    print(f"  NEW matched: {r['new_matched']}")
    print(f"  NEW missed:  {r['new_missed']}")
    print(f"  NEW extra:   {r['new_extra']}")

# === DETAILED: Cases still at 0% ===
print()
print("=" * 80)
print("ZERO-SCORE CASES (new prompt scores 0%)")
print("=" * 80)
zeros = [r for r in results if r['new_score'] == 0 and r['expected_count'] > 0]
for r in sorted(zeros, key=lambda x: x['case_no']):
    print(f"\nCase {r['case_no']} ({r['visit_type']}) — {', '.join(r['diagnosis'])}")
    print(f"  Expected {r['expected_count']} meds, predicted {r['new_pred_count']}")
    print(f"  Missed: {r['new_missed']}")
    print(f"  Extra:  {r['new_extra']}")

# === Pattern analysis: by visit type ===
print()
print("=" * 80)
print("SCORE BY VISIT TYPE")
print("=" * 80)
for vt in ['new consultation', 'follow-up']:
    vt_results = [r for r in results if r['visit_type'] == vt]
    if vt_results:
        old_avg_vt = sum(r['old_score'] for r in vt_results) / len(vt_results) * 100
        new_avg_vt = sum(r['new_score'] for r in vt_results) / len(vt_results) * 100
        print(f"  {vt}: OLD={old_avg_vt:.1f}%  NEW={new_avg_vt:.1f}%  (n={len(vt_results)})")

# === Pattern analysis: by past cases presence ===
print()
print("=" * 80)
print("SCORE BY PATIENT HISTORY AVAILABILITY")
print("=" * 80)
for label, filt in [("Has past cases (>0)", lambda r: r['past_cases'] > 0), 
                     ("No past cases (=0)", lambda r: r['past_cases'] == 0)]:
    filtered = [r for r in results if filt(r)]
    if filtered:
        old_avg_f = sum(r['old_score'] for r in filtered) / len(filtered) * 100
        new_avg_f = sum(r['new_score'] for r in filtered) / len(filtered) * 100
        print(f"  {label}: OLD={old_avg_f:.1f}%  NEW={new_avg_f:.1f}%  (n={len(filtered)})")

# === Most commonly missed molecules ===
print()
print("=" * 80)
print("MOST COMMONLY MISSED MOLECULES (new run)")
print("=" * 80)
from collections import Counter
missed_counter = Counter()
for r in results:
    for mol in r['new_missed']:
        missed_counter[mol] += 1

for mol, count in missed_counter.most_common(15):
    print(f"  {mol}: missed in {count}/{len(results)} cases")

# === Most commonly predicted extra molecules ===
print()
print("=" * 80)
print("MOST COMMONLY EXTRA-PREDICTED MOLECULES (new run)")
print("=" * 80)
extra_counter = Counter()
for r in results:
    for mol in r['new_extra']:
        extra_counter[mol] += 1

for mol, count in extra_counter.most_common(15):
    print(f"  {mol}: extra in {count}/{len(results)} cases")

# === Cases with large regimen size mismatch ===
print()
print("=" * 80)
print("REGIMEN SIZE MISMATCH (|predicted - expected| >= 3)")
print("=" * 80)
for r in sorted(results, key=lambda x: abs(x['new_pred_count'] - x['expected_count']), reverse=True):
    diff = r['new_pred_count'] - r['expected_count']
    if abs(diff) >= 3:
        print(f"  Case {r['case_no']}: expected {r['expected_count']}, predicted {r['new_pred_count']} (diff={diff:+d}) — {', '.join(r['diagnosis'])}")

# === Cases that improved ===
print()
print("=" * 80)
print("IMPROVED CASES (selected)")
print("=" * 80)
for r in sorted(improved, key=lambda x: -x['delta'])[:10]:
    print(f"\nCase {r['case_no']} ({r['visit_type']}) — {', '.join(r['diagnosis'])}")
    print(f"  Score: {r['old_score']:.0%} → {r['new_score']:.0%} (Δ={r['delta']:+.0%})")
    print(f"  NEW matched: {r['new_matched']}")
    if r['new_missed']:
        print(f"  Still missed: {r['new_missed']}")

# Dump full results as JSON for reference
with open('/home/hardik/.gemini/antigravity-ide/brain/377ae0ba-02bc-43ca-92b1-4e61f26b95f3/scratch/analysis_data.json', 'w') as f:
    # Convert sets to lists for JSON serialization
    serializable = []
    for r in results:
        sr = dict(r)
        for k in ['old_matched', 'old_extra', 'old_missed', 'new_matched', 'new_extra', 'new_missed']:
            sr[k] = sorted(list(sr[k]))
        serializable.append(sr)
    json.dump(serializable, f, indent=2)
