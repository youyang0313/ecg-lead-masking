"""ECG classification under missing leads: a 10-minute execution-explanation talk."""
import numpy as np  # @invisible
import torch  # @invisible
from lectrace import image, link, note, plot, table, text, url_reference  # @invisible
from _helpers import *  # @invisible   (data, archived model, BatchNorm hooks, AUROC, tables; not traced)


def main() -> None:
    text("# ECG classification when leads go missing")
    text("""
        A remote ECG reading service receives **12-lead** hospital ECGs, **2-lead** ambulatory
        recordings and **6-lead** recordings from handheld devices. A classifier trained only on
        complete 12-lead ECGs may quietly degrade on the others, and the service has no way to
        say how reliable each output is for that lead set.

        Some information is genuinely lost with fewer leads. Some is not: the six limb leads are
        linear combinations of any two of them, so a model that fails on I and II alone is
        failing for a reason other than missing information.
        """)
    text("""
        ## Question

        Does training one classifier on a mix of complete and randomly lead-masked ECGs
        improve discrimination under predefined missing-lead conditions, compared with matched
        complete-input training? And **why** does complete-input training fail?

        ## Approach

        | | Arm A | Arm B |
        | --- | --- | --- |
        | training input | always complete | 50% complete, 50% random whole-lead masks |
        | network | same residual 1-D CNN (359,749 parameters) | same |
        | data, budget, seeds, checkpoint rule | identical | identical |

        PTB-XL 1.0.3, five diagnostic superclasses (NORM, MI, STTC, CD, HYP), patient-separated
        test fold. Missing leads are zeroed after standardization. Test inputs: S12 (all leads),
        S6 (limb leads), S4 (I, II, III, V2), S3 (I, II, V2), S2 (I, II).
        """)
    link(url_reference("https://arxiv.org/abs/2207.01024", title="Reyna et al., Issues in the automated classification of multilead ECGs"))
    note("Keep this under 2.5 minutes. The table is the whole design: the only difference between A and B is the mask.")

    text("## State 1 — a zero-filled ECG breaks a physical law")
    x, y, rows = test_set()  # @invisible
    i = demo_record_index(rows)  # @invisible
    ecg_id = rows[i]["ecg_id"]                                       # @inspect ecg_id
    record = torch.from_numpy(np.array(x[i:i+1]))  # @invisible
    I, II, III = (physical(record)[0, LEADS.index(l)].numpy() for l in ("I", "II", "III"))  # @invisible

    # Einthoven: lead III equals II minus I.
    einthoven_residual = round(relative_error(II - I, III), 3)       # @inspect einthoven_residual

    # S2 keeps only I and II; zero-filling turns lead III into a flat line.
    available = CONFIGS["S2"]                                        # @inspect available
    III_zero = physical(zero_fill(record, available))[0, LEADS.index("III")].numpy()
    zero_fill_residual = round(relative_error(III_zero, III), 3)     # @inspect zero_fill_residual

    # Derive the missing limb leads from I and II instead (E10).
    III_derived = physical(derive_limb_leads(record, available))[0, LEADS.index("III")].numpy()
    derived_residual = round(relative_error(III_derived, III), 3)    # @inspect derived_residual

    plot(lead_chart({"III recorded": III, "III zero-filled (S2)": III_zero, "III derived from I and II": III_derived}, title=f"Lead III of test ECG {ecg_id}"))  # @stepover
    text(f"""
        Under S2 the network sees four limb leads that are flat lines, although those leads are
        determined by the two it does see. The recorded relation holds to within
        {100*einthoven_residual:.1f}%; zero-filling breaks it completely ({100*zero_fill_residual:.0f}%
        error); deriving the leads restores it ({100*derived_residual:.1f}% error). So S2 and S6 carry
        nearly the same information, and a model that treats them differently is not limited by
        information.
        """)

    text("## State 2 — inside the forward pass: BatchNorm sees a different distribution")
    A = load_model("A")  # @invisible
    B = load_model("B")  # @invisible

    # How far each BatchNorm layer's input mean drifts from its running mean (in running SDs).
    shift_A_complete = bn_shift(A, x, CONFIGS["S12"])                # @inspect shift_A_complete
    shift_A_S2 = bn_shift(A, x, CONFIGS["S2"])                       # @inspect shift_A_S2
    shift_B_S2 = bn_shift(B, x, CONFIGS["S2"])                       # @inspect shift_B_S2
    plot(bn_shift_chart({"A, complete input": shift_A_complete, "A, S2": shift_A_S2, "B, S2": shift_B_S2}))  # @stepover

    # The consequence at the output: per-label AUROC on the full test set under S2.
    p_A = predict(A, x, CONFIGS["S2"])  # @invisible
    p_B = predict(B, x, CONFIGS["S2"])  # @invisible
    auroc_A = {c: round(auroc(y[:, j], p_A[:, j]), 3) for j, c in enumerate(CLASSES)}  # @inspect auroc_A
    auroc_B = {c: round(auroc(y[:, j], p_B[:, j]), 3) for j, c in enumerate(CLASSES)}  # @inspect auroc_B
    below_chance = [c for c, v in auroc_A.items() if v < 0.5]       # @inspect below_chance
    text(f"""
        With complete input, A's BatchNorm statistics fit its activations (shift at most
        {max(shift_A_complete):.2f} SD). Under S2 they drift by {min(shift_A_S2):.2f}–{max(shift_A_S2):.2f}
        SD in every layer, and A's macro-AUROC falls to {macro_auroc(y, p_A):.3f}; for
        {', '.join(below_chance)} the ranking is inverted (AUROC below 0.5). B, whose statistics
        were estimated on mini-batches that already mix complete and masked inputs, drifts less
        ({min(shift_B_S2):.2f}–{max(shift_B_S2):.2f}) and keeps {macro_auroc(y, p_B):.3f}.
        """)

    text("## State 3 — repair A at test time, without changing a weight")
    train_x = train_signals()  # @invisible

    # Derive the missing limb leads, then re-estimate every BatchNorm mean and variance on
    # 4,096 training records presented the same way (E11). The weights are untouched.
    A_repaired = reestimate_batchnorm(A, train_x, CONFIGS["S2"])     # @stepover
    p_repaired = predict(A_repaired, x, CONFIGS["S2"], derive=True)  # @invisible
    macro_zero = round(macro_auroc(y, p_A), 3)                       # @inspect macro_zero
    macro_repaired = round(macro_auroc(y, p_repaired), 3)            # @inspect macro_repaired

    # The same comparison over all four configurations and three seeds.
    share_closed = gap_closed()                                      # @inspect share_closed
    table(ladder_rows(), caption="Mean macro-AUROC over S6/S4/S3/S2, three seeds, S12-only selection.")  # @stepover
    gap = results()["ladder"]["B_minus_repaired_s12"]  # @invisible
    text(f"""
        On seed 17 under S2, the repair lifts A from {macro_zero:.3f} to {macro_repaired:.3f}.
        Over all configurations and seeds it closes {pct(share_closed['both'])} of the gap: B still
        leads by {signed(gap['estimate'])} {ci(gap['ci'])}, and models trained for each
        configuration are about 0.02 above B. Most of A's failure was inconsistent inputs and
        shifted statistics, both repairable at test time.
        """)

    text("## State 4 — the key result, and where masked training still matters")
    r = results()  # @invisible

    # The prespecified test: B minus A over the four partial configurations, per paired seed.
    seed_effects = [round(e, 4) for e in r["primary"]["seed_effects"]]   # @inspect seed_effects
    primary = round(sum(r["primary"]["seed_effects"]) / 3, 4)            # @inspect primary
    table(config_rows(), caption=config_caption())                   # @stepover

    # The repair needs two limb leads. Group the 207 lead subsets by how many remain.
    gaps = gaps_by_limb_leads()  # @invisible
    n_subsets = {g: len(v) for g, v in gaps.items()}                 # @inspect n_subsets
    mean_gap = {g: round(float(np.mean(v)), 3) for g, v in gaps.items()}  # @inspect mean_gap
    image(fig("e12_subset_gap.png"), width=680)
    ratio = r["masking_ratio"]  # @invisible
    text(f"""
        **Result.** Against zero-filled complete-input training, masking gains
        {signed(r['primary']['estimate'])} macro-AUROC {ci(r['primary']['ci'])}, positive for every
        seed. Against the best training-free repair, the gain shrinks to about
        {mean_gap['two or more limb leads']:+.3f} whenever two limb leads remain, and stays large,
        {mean_gap['fewer than two']:+.3f}, when they do not. Masking more is better still: a masking
        probability of 1.0 reaches {ratio['p100']:.3f} against {ratio['B_s12_p050']:.3f} at 0.5,
        with no loss on complete input (E4).
        """)

    text(f"""
        ## Healthcare interpretation

        | Situation | Best option |
        | --- | --- |
        | lead combination cannot be predicted | one masked-training model |
        | one fixed device configuration, retraining possible | a dedicated model |
        | only an existing complete-input model | derive limb leads + re-estimate BatchNorm (needs two limb leads) |

        All three need an upstream check that detects unusable leads and zeroes them: when a lead
        carries a plausible but wrong signal, B loses {abs(r['fills']['other_record_cost_B']['estimate']):.3f}
        macro-AUROC (E9). The model has learned to ignore absent channels, not to recognize wrong ones.

        **The lesson I take from this project:** the size of an effect depends on the baseline.
        The same intervention looks like +0.197 against a naive baseline and close to zero against
        a cheap strong one on most lead sets. Robustness claims should be tested against the
        strongest cheap baseline and beyond the protocol configurations.
        """)
    text("""
        ## Limits

        One public cohort (PTB-XL), one backbone, three seeds; exploratory analyses reuse the test
        set. Lead loss is simulated by zeroing, not recorded from failing electrodes. Labels are
        diagnostic superclasses, not clinical decisions.
        """)
    note("End on the decision table. If asked about AI use: Codex wrote the pipeline, Claude Code proposed and checked analyses, and a simulated peer review found the stronger baselines.")


if __name__ == "__main__":
    main()
