import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Tuple, Optional

Mode = Literal["equal_priority", "female_priority"]

@dataclass
class YearStats:
    year: int
    n_male: int
    n_female: int
    mean_skill_all: float
    mean_skill_male: float
    mean_skill_female: float
    cand_male_share: float = float("nan")
    cand_female_share: float = float("nan")

def _draw_skills(n: int, mu: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=mu, scale=sigma, size=n)

def initialize_staff(P0: int, male_ratio_x: float, mu: float, sigma: float, seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    既存スタッフの性別と能力を初期化
    Returns:
        genders: (P0,) bool array (True=male, False=female)
        skills:  (P0,) float abilities
    """
    rng = np.random.default_rng(seed)
    n_male0 = int(round(P0 * male_ratio_x / 100.0))
    n_female0 = P0 - n_male0
    genders = np.array([True]*n_male0 + [False]*n_female0, dtype=bool)
    rng.shuffle(genders)
    skills = _draw_skills(P0, mu, sigma, rng)
    # Re-center sample mean to exactly match mu at t=0
    m0 = float(skills.mean()) if P0 > 0 else 0.0
    skills = skills - m0 + mu
    return genders, skills

def simulate(
    T: int,
    P0: int,
    x: float,            # 初期男性割合 [%]
    y: float,            # 新人の男性割合（基準値）[%]
    r: int,              # 毎年の退職/公募数
    mu: float = 0.0,     # 能力分布の平均（男女共通）
    sigma: float = 1.0,  # 能力分布の標準偏差（男女共通）
    seed: Optional[int] = None,
    mode: Mode = "equal_priority",
    backfill_if_short: bool = True,  # 女性限定で人数不足の際に男女混合で穴埋めするか
    feedback_strength: float = 0.0,  # 応募者の性別比を現職構成にブレンド（0=no feedback, 1=mirror staff）
    max_candidate_years: int = 10,   # 候補者として残れる最大年数（デフォルト10年）
    applicants_multiplier: float = 1.0,  # 応募倍率：毎年の新規応募者数 = round(r * k)
    stop_female_priority_at: float = 50.0,  # 在職女性比がこの%に達したら female_priority を停止
) -> Dict[str, List[YearStats]]:
    """
    Returns:
        {"stats": [YearStats, ...]}
    """
    rng = np.random.default_rng(seed)

    genders, skills = initialize_staff(P0, x, mu, sigma, seed=rng.integers(1<<30))

    def collect_stats(year: int) -> YearStats:
        n_m = int(genders.sum())
        n_f = int((~genders).sum())
        mean_all = float(skills.mean()) if skills.size else np.nan
        mean_m = float(skills[genders].mean()) if n_m > 0 else np.nan
        mean_f = float(skills[~genders].mean()) if n_f > 0 else np.nan
        return YearStats(year, n_m, n_f, mean_all, mean_m, mean_f)

    stats: List[YearStats] = [collect_stats(0)]

    # Rolling candidate pool: genders(bool), skills(float), ages(int years in pool)
    pool_genders = np.zeros(0, dtype=bool)
    pool_skills = np.zeros(0, dtype=float)
    pool_ages = np.zeros(0, dtype=int)

    for year in range(1, T + 1):
        # --- 1) 退職（毎年 r 名を一様ランダムに離職）
        if r > P0:
            raise ValueError("r must be <= P0 (vacancies cannot exceed total posts)")
        leave_idx = rng.choice(P0, size=r, replace=False)
        stay_mask = np.ones(P0, dtype=bool)
        stay_mask[leave_idx] = False
        genders = genders[stay_mask]
        skills  = skills[stay_mask]
        assert genders.size == P0 - r

        # --- 2) 応募者プール（前年度からの持ち越し + 毎年 r 名の新人）
        # 応募者の男性割合にフィードバックを適用
        prev = stats[-1]
        denom = (prev.n_male + prev.n_female)
        male_share_staff = (prev.n_male / denom) if denom > 0 else 0.5
        y_eff = (1.0 - feedback_strength) * y + feedback_strength * (100.0 * male_share_staff)
        y_eff = float(np.clip(y_eff, 0.0, 100.0))

        # female priority の停止条件（在職女性比が閾値以上なら equal priority に切替）
        female_share_staff_prev = (prev.n_female / denom) * 100.0 if denom > 0 else 50.0
        use_female_priority = (
            (mode == "female_priority") and (female_share_staff_prev < stop_female_priority_at)
        )

        n_new = int(round(max(0.0, applicants_multiplier) * r))
        n_male_cand = int(round(n_new * y_eff / 100.0))
        n_fem_cand  = n_new - n_male_cand
        new_genders = np.array([True]*n_male_cand + [False]*n_fem_cand, dtype=bool)
        rng.shuffle(new_genders)
        new_skills  = _draw_skills(n_new, mu, sigma, rng)
        new_ages    = np.zeros(n_new, dtype=int)

        # Age existing pool, drop those exceeding max_candidate_years-1 after increment
        if pool_ages.size:
            pool_ages = pool_ages + 1
            keep = pool_ages < max_candidate_years
            pool_genders = pool_genders[keep]
            pool_skills = pool_skills[keep]
            pool_ages = pool_ages[keep]

        # Add new entrants to pool
        pool_genders = np.concatenate([pool_genders, new_genders])
        pool_skills = np.concatenate([pool_skills, new_skills])
        pool_ages = np.concatenate([pool_ages, new_ages])
        # For plotting, store the continuous effective applicant shares (not discretized counts)
        cand_male_share = y_eff if r > 0 else float("nan")
        cand_female_share = 100.0 - y_eff if r > 0 else float("nan")

        # --- 3) 採用（モード別）
        if not use_female_priority:  # equal priority
            # 性別無関係（equal priority）に能力上位 r 名（候補者プール全体から）
            if pool_skills.size > 0:
                order = np.argsort(-pool_skills)[:min(r, pool_skills.size)]
            else:
                order = np.array([], dtype=int)
            hire_g = pool_genders[order]
            hire_s = pool_skills[order]

        elif use_female_priority:
            # まず女性のみで上位を選抜（候補者プール全体から）
            female_idx = np.where(~pool_genders)[0]
            if female_idx.size >= r:
                chosen_f = female_idx[np.argsort(-pool_skills[female_idx])[:r]]
                hire_g = pool_genders[chosen_f]
                hire_s = pool_skills[chosen_f]
            else:
                # 女性が不足：女性を全員採用し、残りをどうするか
                chosen_f = female_idx
                need = r - chosen_f.size
                if backfill_if_short and need > 0:
                    # 残りは性別不問の上位から（ただし既に選んだ女性を除外）
                    all_idx = np.argsort(-pool_skills)
                    # 既採用を除いた順序で need 名
                    mask_exclude = np.zeros(pool_skills.size, dtype=bool)
                    mask_exclude[chosen_f] = True
                    fill = [i for i in all_idx if not mask_exclude[i]][:need]
                    chosen = np.concatenate([chosen_f, np.array(fill, dtype=int)])
                else:
                    # 穴埋めしない（欠員のまま＝能力・性別のない空スロットを作る）
                    chosen = chosen_f
                    # 欠員分はダミーで埋める（能力=-inf, 性別はNone代替→Falseで置く）
                    if need > 0:
                        # 欠員の扱い：ここでは「不在」を表すために後で埋める
                        pass
                hire_g = pool_genders[chosen]
                hire_s = pool_skills[chosen]

                if not backfill_if_short and chosen_f.size < r:
                    # 欠員分は、能力/性別の要素を持たない空きを後で追加
                    # ここでは簡単に「次年度まで空席にせず」、平均能力に影響しない 0 能力・男女比にも影響しない
                    # という扱いは不自然なので、ポスト数一定の前提に合わせて「空席なし（埋めない）」にはできない。
                    # したがって、欠員を許さない前提では backfill_if_short=True を使うことを推奨。
                    # ここでは安全のため、欠員は男性で埋めるが能力は非常に低い候補がいたと仮定して補充する：
                    need = r - chosen_f.size
                    if need > 0:
                        # 候補群以外からの臨時補充（非常に低スコア）という近似
                        extra_g = np.array([True]*need, dtype=bool)  # 性別は慣例上 True=male
                        extra_s = np.full(need, -1e9)
                        hire_g = np.concatenate([hire_g, extra_g])
                        hire_s = np.concatenate([hire_s, extra_s])
        else:
            raise ValueError("mode must be 'equal_priority' or 'female_priority'")

        # --- 4) 候補者プールから採用者を削除
        if pool_skills.size > 0 and hire_g.size > 0:
            keep_mask = np.ones(pool_skills.size, dtype=bool)
            # Identify indices used in pool (order or chosen). Recompute indices relative to pool
            if not use_female_priority:
                idx_hired = order
            else:
                # mode == female_priority
                if 'chosen' in locals():
                    idx_hired = chosen
                else:
                    idx_hired = np.array([], dtype=int)
            keep_mask[idx_hired] = False
            pool_genders = pool_genders[keep_mask]
            pool_skills = pool_skills[keep_mask]
            pool_ages = pool_ages[keep_mask]

        # --- 5) 補充して総ポスト数を P0 に戻す
        # ここまでで在籍 P0 - r、人採用 len(hire_g) 人（通常 r）。不足があれば臨時補充する。
        if hire_g.size < r:
            need = r - hire_g.size
            # 臨時補充（極端に低スコアのダミー）—ポスト数一定を守るための保険
            tmp_g = np.array([True]*need, dtype=bool)
            tmp_s = np.full(need, -1e9)
            hire_g = np.concatenate([hire_g, tmp_g])
            hire_s = np.concatenate([hire_s, tmp_s])

        genders = np.concatenate([genders, hire_g])
        skills  = np.concatenate([skills,  hire_s])
        assert genders.size == P0 and skills.size == P0

        ys = collect_stats(year)
        ys.cand_male_share = cand_male_share
        ys.cand_female_share = cand_female_share
        stats.append(ys)

    return {"stats": stats}

# ===== Visualization utilities =====
def _stats_to_arrays(stats: List[YearStats]) -> Dict[str, np.ndarray]:
    years = np.array([s.year for s in stats], dtype=int)
    n_m = np.array([s.n_male for s in stats], dtype=int)
    n_f = np.array([s.n_female for s in stats], dtype=int)
    mu_all = np.array([s.mean_skill_all for s in stats], dtype=float)
    mu_m = np.array([s.mean_skill_male for s in stats], dtype=float)
    mu_f = np.array([s.mean_skill_female for s in stats], dtype=float)
    return {
        "year": years,
        "n_m": n_m,
        "n_f": n_f,
        "mu_all": mu_all,
        "mu_m": mu_m,
        "mu_f": mu_f,
        "cand_male_share": np.array([s.cand_male_share for s in stats], dtype=float),
        "cand_female_share": np.array([s.cand_female_share for s in stats], dtype=float),
    }

def plot_results(results: List[Dict[str, List[YearStats]]], labels: List[str], title: Optional[str] = None, save_path: Optional[str] = None) -> None:
    """Plot comparative results.
    Args:
        results: [{"stats": [YearStats, ...]}, ...]
        labels:  legend labels for each result
        title:   figure title
        save_path: path to save (None to show)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[plot] matplotlib not found. Please 'pip install matplotlib'.")
        return

    arrays = [_stats_to_arrays(r["stats"]) for r in results]
    years = arrays[0]["year"]

    # Consistent colors per label across all panels
    cmap = None
    try:
        import matplotlib.pyplot as plt  # already imported above, but safe
        cmap = plt.get_cmap('tab10')
    except Exception:
        cmap = None
    colors = {lab: (cmap(i) if cmap else None) for i, lab in enumerate(labels)}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    if title:
        fig.suptitle(title)

    # (1) Female share (staff vs applicants)
    ax = axes[0, 0]
    for arr, lab in zip(arrays, labels):
        col = colors.get(lab)
        share_f_staff = arr["n_f"] / (arr["n_f"] + arr["n_m"]) * 100.0
        ax.plot(years, share_f_staff, label=f"{lab} staff", color=col)
        ax.plot(years, arr["cand_female_share"], linestyle=":", alpha=0.9, label=f"{lab} applicants", color=col)
    ax.set_title("Female share (%)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Female (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    # (2) Headcount by gender
    ax = axes[0, 1]
    for arr, lab in zip(arrays, labels):
        col = colors.get(lab)
        ax.plot(years, arr["n_m"], linestyle="-", alpha=0.9, label=f"M ({lab})", color=col)
        ax.plot(years, arr["n_f"], linestyle="--", alpha=0.9, label=f"F ({lab})", color=col)
    ax.set_title("Headcount by gender")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    # (3) Mean ability (overall)
    ax = axes[1, 0]
    for arr, lab in zip(arrays, labels):
        col = colors.get(lab)
        ax.plot(years, arr["mu_all"], label=lab, color=col)
    ax.set_title("Mean ability (overall)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean score")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (4) Mean ability (by gender)
    ax = axes[1, 1]
    for arr, lab in zip(arrays, labels):
        col = colors.get(lab)
        ax.plot(years, arr["mu_m"], linestyle="-", label=f"M ({lab})", color=col)
        ax.plot(years, arr["mu_f"], linestyle="--", label=f"F ({lab})", color=col)
    ax.set_title("Mean ability (by gender)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean score")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()

# ===== CLI example =====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hiring simulation (Equal priority vs Female priority)")
    parser.add_argument("--T", type=int, default=30, help="Years")
    parser.add_argument("--P0", type=int, default=200, help="Initial number of posts")
    parser.add_argument("--x", type=float, default=80.0, help="Initial male share (%)")
    parser.add_argument("--y", type=float, default=70.0, help="Male share among applicants (%)")
    parser.add_argument("--r", type=int, default=20, help="Annual turnover / vacancies")
    parser.add_argument("--mu", type=float, default=0.0, help="Mean of ability distribution")
    parser.add_argument("--sigma", type=float, default=1.0, help="Std. dev. of ability distribution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-backfill", action="store_true", help="Do not backfill shortfall in female-priority mode")
    parser.add_argument("--save", type=str, default=None, help="Path to save figure (show if omitted)")
    parser.add_argument("--feedback-strength", type=float, default=0.0,
                        help="Blend applicants' male share toward current staff (0=no feedback, 1=mirror staff)")
    parser.add_argument("--max-candidate-years", type=int, default=10,
                        help="Maximum years an applicant can remain in the pool (default 10)")
    parser.add_argument("--applicants-multiplier", type=float, default=1.0,
                        help="Applicants per year = round(r * k). k>1 increases selectivity (default 1.0)")
    parser.add_argument("--stop-female-priority-at", type=float, default=50.0,
                        help="Stop using female-priority once staff female share reaches this % (default 50.0)")
    args = parser.parse_args()

    # (A) Equal-priority
    res_equal = simulate(
        args.T, args.P0, args.x, args.y, args.r, args.mu, args.sigma, args.seed,
        mode="equal_priority", feedback_strength=args.feedback_strength,
        max_candidate_years=args.max_candidate_years,
        applicants_multiplier=args.applicants_multiplier,
        stop_female_priority_at=args.stop_female_priority_at
    )
    # (B) Female-priority
    res_fonly = simulate(
        args.T, args.P0, args.x, args.y, args.r, args.mu, args.sigma, args.seed,
        mode="female_priority", backfill_if_short=not args.no_backfill,
        feedback_strength=args.feedback_strength,
        max_candidate_years=args.max_candidate_years,
        applicants_multiplier=args.applicants_multiplier,
        stop_female_priority_at=args.stop_female_priority_at
    )

    # Quick summary (start and end)
    a0, b0 = res_equal["stats"][0], res_fonly["stats"][0]
    aL, bL = res_equal["stats"][-1], res_fonly["stats"][-1]
    print("[start] equal  : M%3d F%3d  mu=%.3f" % (a0.n_male, a0.n_female, a0.mean_skill_all))
    print("[start] female : M%3d F%3d  mu=%.3f" % (b0.n_male, b0.n_female, b0.mean_skill_all))
    print("[end]   equal  : M%3d F%3d  mu=%.3f" % (aL.n_male, aL.n_female, aL.mean_skill_all))
    print("[end]   female : M%3d F%3d  mu=%.3f" % (bL.n_male, bL.n_female, bL.mean_skill_all))

    plot_results(
        [res_equal, res_fonly],
        labels=["Equal priority", "Female priority" + (" (no backfill)" if args.no_backfill else "")],
        title=f"Hiring modes comparison (feedback={args.feedback_strength:.2f})",
        save_path=args.save,
    )
