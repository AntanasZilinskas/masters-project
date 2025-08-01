sk_tss, sk_auc, sk_brier = 0.9852, 0.9971, 0.0136
ev_tss, ev_auc, ev_brier = 0.990, 0.999, 0.0019
sk_ece, ev_ece = 0.0148, 0.0022
print("CORRECTED RESULTS:")
tss_imp = (ev_tss - sk_tss) / sk_tss * 100
brier_imp = (sk_brier - ev_brier) / sk_brier * 100
ece_imp = (sk_ece - ev_ece) / sk_ece * 100
print(f"TSS: {sk_tss:.4f} → {ev_tss:.4f} (+{tss_imp:.1f}%)")
print(f"Brier: {sk_brier:.4f} → {ev_brier:.4f} (+{brier_imp:.1f}%)")
print(f"ECE: {sk_ece:.4f} → {ev_ece:.4f} (+{ece_imp:.1f}%)")
