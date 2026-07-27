"""
PubMed-based literature evidence concordance analysis for methamphetamine-related
candidate brain regions.

Outputs are written to ./pubmed_concordance_output by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OUT_DIR = Path("outputs/pubmed_concordance")
DEFAULT_ML_RANKING = Path("outputs/extended_resampling/freq_top10_Ours_consensus_dedup.csv")


METH_QUERY = (
    '("Methamphetamine"[Mesh] OR methamphetamine*[Title/Abstract] OR '
    '"methamphetamine addiction"[Title/Abstract] OR '
    '"methamphetamine dependence"[Title/Abstract] OR '
    '"methamphetamine use disorder"[Title/Abstract] OR '
    '"methamphetamine self-administration"[Title/Abstract] OR '
    '"methamphetamine seeking"[Title/Abstract] OR '
    '"methamphetamine relapse"[Title/Abstract] OR '
    '"methamphetamine withdrawal"[Title/Abstract])'
)


BRAIN_REGIONS = {
    "ACC": {
        "name": "anterior cingulate cortex",
        "terms": [
            '"anterior cingulate cortex"[Title/Abstract]',
            '"anterior cingulate"[Title/Abstract]',
            '"cingulate cortex"[Title/Abstract]',
        ],
    },
    "NAco": {
        "name": "nucleus accumbens core",
        "terms": [
            '"nucleus accumbens core"[Title/Abstract]',
            '"accumbens core"[Title/Abstract]',
            '"NAc core"[Title/Abstract]',
            '"core of the nucleus accumbens"[Title/Abstract]',
        ],
    },
    "CLA": {
        "name": "claustrum",
        "terms": [
            'claustrum[Title/Abstract]',
            'claustral[Title/Abstract]',
        ],
    },
    "VTA": {
        "name": "ventral tegmental area",
        "terms": [
            '"ventral tegmental area"[Title/Abstract]',
            '"ventral tegmentum"[Title/Abstract]',
            'VTA[Title/Abstract]',
        ],
    },
    "DMS": {
        "name": "dorsomedial striatum",
        "terms": [
            '"dorsomedial striatum"[Title/Abstract]',
            '"dorso-medial striatum"[Title/Abstract]',
            '"medial dorsal striatum"[Title/Abstract]',
        ],
    },
    "CeA": {
        "name": "central amygdala",
        "terms": [
            '"central amygdala"[Title/Abstract]',
            '"central nucleus of the amygdala"[Title/Abstract]',
            '"central amygdaloid nucleus"[Title/Abstract]',
            'CeA[Title/Abstract]',
        ],
    },
    "BLA": {
        "name": "basolateral amygdala",
        "terms": [
            '"basolateral amygdala"[Title/Abstract]',
            '"basolateral amygdaloid nucleus"[Title/Abstract]',
            'BLA[Title/Abstract]',
        ],
    },
    "PL": {
        "name": "prelimbic cortex",
        "terms": [
            '"prelimbic cortex"[Title/Abstract]',
            '"prelimbic medial prefrontal cortex"[Title/Abstract]',
            '"prelimbic area"[Title/Abstract]',
        ],
    },
    "VP": {
        "name": "ventral pallidum",
        "terms": [
            '"ventral pallidum"[Title/Abstract]',
            '"ventral pallidal"[Title/Abstract]',
        ],
    },
    "OFC": {
        "name": "orbitofrontal cortex",
        "terms": [
            '"orbitofrontal cortex"[Title/Abstract]',
            '"orbito-frontal cortex"[Title/Abstract]',
            '"orbital frontal cortex"[Title/Abstract]',
        ],
    },
    "NAsh": {
        "name": "nucleus accumbens shell",
        "terms": [
            '"nucleus accumbens shell"[Title/Abstract]',
            '"accumbens shell"[Title/Abstract]',
            '"NAc shell"[Title/Abstract]',
            '"shell of the nucleus accumbens"[Title/Abstract]',
        ],
    },
    "GP": {
        "name": "globus pallidus",
        "terms": [
            '"globus pallidus"[Title/Abstract]',
            '"globus pallidal"[Title/Abstract]',
            '"pallidal"[Title/Abstract]',
        ],
    },
    "RSC": {
        "name": "retrosplenial cortex",
        "terms": [
            '"retrosplenial cortex"[Title/Abstract]',
            '"retrosplenial"[Title/Abstract]',
        ],
    },
    "CM": {
        "name": "centromedial amygdala",
        "terms": [
            '"centromedial amygdala"[Title/Abstract]',
            '"central medial amygdala"[Title/Abstract]',
            '"central-medial amygdala"[Title/Abstract]',
        ],
    },
    "IL": {
        "name": "infralimbic cortex",
        "terms": [
            '"infralimbic cortex"[Title/Abstract]',
            '"infralimbic medial prefrontal cortex"[Title/Abstract]',
            '"infralimbic area"[Title/Abstract]',
        ],
    },
    "MS": {
        "name": "medial septum",
        "terms": [
            '"medial septum"[Title/Abstract]',
            '"medial septal"[Title/Abstract]',
        ],
    },
    "LH": {
        "name": "lateral hypothalamus",
        "terms": [
            '"lateral hypothalamus"[Title/Abstract]',
            '"lateral hypothalamic"[Title/Abstract]',
        ],
    },
    "AM": {
        "name": "anteromedial thalamic nucleus",
        "terms": [
            '"anteromedial thalamic nucleus"[Title/Abstract]',
            '"anteromedial thalamus"[Title/Abstract]',
            '"anterior medial thalamus"[Title/Abstract]',
        ],
    },
}


EVIDENCE_PATTERNS = {
    "intervention": [
        "optogenetic", "chemogenetic", "lesion", "inactivation", "microinjection",
        "antagonist", "agonist", "blockade", "stimulation", "inhibition",
        "knockdown", "knockout", "silencing",
    ],
    "circuit": [
        "projection", "tracing", "connectivity", "circuit", "pathway",
        "afferent", "efferent", "synaptic", "glutamatergic", "gabaergic",
    ],
    "activity": [
        "c-fos", "cfos", "fos", "fmri", "imaging", "activation", "activity",
        "electrophysiology", "expression", "dopamine", "neurotransmission",
    ],
    "behavioral": [
        "self-administration", "conditioned place preference", "cpp", "seeking",
        "reinstatement", "relapse", "withdrawal", "craving", "choice",
    ],
}


def ncbi_get(endpoint: str, params: dict, retries: int = 5) -> dict | str:
    url = f"{NCBI_BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = response.read().decode("utf-8", errors="replace")
            break
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = exc
            wait = min(30, 2 ** attempt)
            print(f"NCBI request failed on attempt {attempt}/{retries}: {exc}. Retrying in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"NCBI request failed after {retries} attempts: {url}") from last_error
    if params.get("retmode") == "json":
        return json.loads(payload)
    return payload


def esearch(query: str, retmax: int = 500) -> tuple[int, list[str]]:
    payload = ncbi_get(
        "esearch.fcgi",
        {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
        },
    )
    result = payload["esearchresult"]
    return int(result["count"]), result.get("idlist", [])


def efetch(pmids: list[str]) -> list[dict]:
    records = []
    for start in range(0, len(pmids), 100):
        batch = pmids[start : start + 100]
        xml_text = ncbi_get(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
            },
        )
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            records.append(parse_article(article))
        time.sleep(0.34)
    return records


def text_of(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def parse_article(article: ET.Element) -> dict:
    medline = article.find("./MedlineCitation")
    pmid = text_of(medline.find("./PMID")) if medline is not None else ""
    art = medline.find("./Article") if medline is not None else None
    title = text_of(art.find("./ArticleTitle")) if art is not None else ""
    journal = text_of(art.find("./Journal/Title")) if art is not None else ""
    year = ""
    if art is not None:
        year = text_of(art.find("./Journal/JournalIssue/PubDate/Year"))
        if not year:
            medline_date = text_of(art.find("./Journal/JournalIssue/PubDate/MedlineDate"))
            year = medline_date[:4] if medline_date else ""
    abstract = " ".join(text_of(x) for x in art.findall("./Abstract/AbstractText")) if art is not None else ""
    publication_types = [
        text_of(x) for x in article.findall(".//PublicationTypeList/PublicationType")
    ]
    mesh_terms = [text_of(x.find("./DescriptorName")) for x in article.findall(".//MeshHeading")]
    authors = article.findall(".//AuthorList/Author")
    first_author = ""
    if authors:
        last = text_of(authors[0].find("./LastName"))
        fore = text_of(authors[0].find("./ForeName"))
        first_author = " ".join(x for x in [last, fore] if x)
    return {
        "pmid": pmid,
        "title": title,
        "journal": journal,
        "pub_year": int(year) if year.isdigit() else None,
        "abstract": abstract,
        "publication_types": "; ".join(publication_types),
        "mesh_terms": "; ".join(mesh_terms),
        "first_author": first_author,
        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
    }


def classify_evidence(text: str) -> str:
    lower = text.lower()
    hits = []
    for label, patterns in EVIDENCE_PATTERNS.items():
        if any(pattern in lower for pattern in patterns):
            hits.append(label)
    return ";".join(hits) if hits else "unclassified"


def classify_species(text: str) -> str:
    lower = text.lower()
    species = []
    if any(x in lower for x in ["human", "patient", "participant", "volunteer", "users"]):
        species.append("human")
    if any(x in lower for x in ["rat", "rats"]):
        species.append("rat")
    if any(x in lower for x in ["mouse", "mice", "murine"]):
        species.append("mouse")
    if any(x in lower for x in ["monkey", "macaque", "primate"]):
        species.append("primate")
    return ";".join(species) if species else "not coded"


def is_original_study(publication_types: str) -> bool:
    pt = publication_types.lower()
    excluded = ["review", "editorial", "letter", "comment", "news", "meta-analysis"]
    return not any(x in pt for x in excluded)


def load_ml_ranking(ranking_path: Path = DEFAULT_ML_RANKING) -> pd.DataFrame:
    ranking_path = Path(ranking_path)
    if ranking_path.exists():
        df = pd.read_csv(ranking_path)
        df = df.rename(columns={"BrainRegion": "brain_region", "freq": "top10_frequency", "freq_rate": "stability"})
        df = df[df["brain_region"].isin(BRAIN_REGIONS)].copy()
        df["ml_rank"] = df["top10_frequency"].rank(ascending=False, method="min").astype(int)
        return df[["brain_region", "ml_rank", "top10_frequency", "stability"]].sort_values("ml_rank")
    fallback = pd.DataFrame({"brain_region": list(BRAIN_REGIONS), "ml_rank": range(1, len(BRAIN_REGIONS) + 1)})
    fallback["top10_frequency"] = None
    fallback["stability"] = None
    return fallback


def rank_from_score(values: pd.Series, ascending: bool = False) -> pd.Series:
    return values.rank(ascending=ascending, method="min").astype(int)


def permutation_pvalue(ml_rank: list[float], lit_rank: list[float], observed: float, n_perm: int = 10000) -> float:
    import random

    if math.isnan(observed):
        return float("nan")
    more_extreme = 0
    shuffled = lit_rank[:]
    for _ in range(n_perm):
        random.shuffle(shuffled)
        rho = stats.spearmanr(ml_rank, shuffled).statistic
        if abs(rho) >= abs(observed):
            more_extreme += 1
    return (more_extreme + 1) / (n_perm + 1)


def plot(summary: pd.DataFrame, metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), dpi=180)
    x = summary["stability"].fillna(1 / summary["ml_rank"])
    y = summary["normalized_cooccurrence_strength"]
    sizes = 80 + summary["direct_original_studies"] * 18
    colors = summary["concordance_class"].map(
        {
            "convergent": "#2f7d59",
            "novel_candidate": "#d98b2b",
            "literature_supported_lower_ml": "#3b6ea8",
            "limited_evidence": "#7a7f87",
        }
    )
    ax.scatter(x, y, s=sizes, c=colors, edgecolor="#222222", linewidth=0.9, alpha=0.78)
    for _, row in summary.iterrows():
        ax.annotate(row["brain_region"], (row["stability"], row["normalized_cooccurrence_strength"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9, weight="bold")
    ax.set_xlabel("Machine-learning stability (Top-10 frequency rate)")
    ax.set_ylabel("PubMed normalized co-occurrence strength")
    ax.set_title("Literature Evidence Concordance for Methamphetamine-Related Brain Regions")
    ax.grid(True, alpha=0.22)
    note = (
        f"Spearman rho={metrics['spearman_rho']:.3f}, permutation p={metrics['permutation_p']:.4f}\n"
        f"Kendall tau={metrics['kendall_tau']:.3f}; Top-3 overlap={metrics['top3_overlap']:.0%}; "
        f"Top-5 overlap={metrics['top5_overlap']:.0%}"
    )
    ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"})
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_concordance_scatter.png", dpi=300)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append(["" if pd.isna(value) else str(value) for value in row.tolist()])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        escaped = [cell.replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description="PubMed literature-evidence concordance analysis")
    parser.add_argument("--ml_ranking", type=Path, default=DEFAULT_ML_RANKING,
                        help="Top-10 stability-frequency CSV from the extended resampling analysis")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ml = load_ml_ranking(args.ml_ranking)
    ml.to_csv(OUT_DIR / "ml_ranking_all_candidate_regions.csv", index=False, encoding="utf-8-sig")

    meth_count, broad_pmids = esearch(METH_QUERY, retmax=500)
    region_counts = {}
    region_total_counts = {}
    region_pmids = {}
    queries = []

    for code, meta in BRAIN_REGIONS.items():
        region_query = "(" + " OR ".join(meta["terms"]) + ")"
        meth_region_query = f"{METH_QUERY} AND {region_query}"
        count, pmids = esearch(meth_region_query, retmax=500)
        total_count, _ = esearch(region_query, retmax=0)
        region_counts[code] = count
        region_total_counts[code] = total_count
        region_pmids[code] = pmids
        queries.append({
            "brain_region": code,
            "region_name": meta["name"],
            "pubmed_query": meth_region_query,
            "pubmed_count": count,
            "region_background_count": total_count,
            "retrieved_pmids": len(pmids),
        })
        print(f"{code}: PubMed count={count}, retrieved={len(pmids)}")
        time.sleep(0.34)

    with open(OUT_DIR / "pubmed_queries.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "search_date": date.today().isoformat(),
                "database": "PubMed",
                "methamphetamine_query": METH_QUERY,
                "methamphetamine_pubmed_count": meth_count,
                "region_queries": queries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    all_pmids = sorted({pmid for pmids in region_pmids.values() for pmid in pmids})
    records = {record["pmid"]: record for record in efetch(all_pmids)}

    rows = []
    for code, pmids in region_pmids.items():
        for pmid in pmids:
            rec = records.get(pmid)
            if not rec:
                continue
            text = " ".join([rec["title"], rec["abstract"], rec["mesh_terms"]])
            rows.append({
                "brain_region": code,
                "region_name": BRAIN_REGIONS[code]["name"],
                "pmid": pmid,
                **rec,
                "direct_meth_study_auto": True,
                "original_study_auto": is_original_study(rec["publication_types"]),
                "evidence_type_auto": classify_evidence(text),
                "species_auto": classify_species(text),
            })

    literature = pd.DataFrame(rows)
    literature.to_csv(OUT_DIR / "pubmed_region_records_long.csv", index=False, encoding="utf-8-sig")
    unique_records = pd.DataFrame(records.values()).sort_values(["pub_year", "pmid"], na_position="last")
    unique_records.to_csv(OUT_DIR / "pubmed_unique_records.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for code in BRAIN_REGIONS:
        subset = literature[literature["brain_region"] == code]
        original = subset[subset["original_study_auto"]]
        intervention_circuit = original[
            original["evidence_type_auto"].str.contains("intervention|circuit", regex=True, na=False)
        ]
        direct_count = int(len(original))
        specificity = region_counts[code] / region_total_counts[code] if region_total_counts[code] else 0
        cooccurrence = region_counts[code] / math.sqrt(max(meth_count, 1) * max(region_total_counts[code], 1))
        years = [int(y) for y in original["pub_year"].dropna().tolist()]
        summary_rows.append({
            "brain_region": code,
            "region_name": BRAIN_REGIONS[code]["name"],
            "pubmed_region_meth_count": region_counts[code],
            "retrieved_records": len(subset),
            "direct_original_studies": direct_count,
            "intervention_or_circuit_studies_auto": int(len(intervention_circuit)),
            "activity_studies_auto": int(original["evidence_type_auto"].str.contains("activity", na=False).sum()),
            "behavioral_studies_auto": int(original["evidence_type_auto"].str.contains("behavioral", na=False).sum()),
            "region_background_count": region_total_counts[code],
            "methamphetamine_background_count": meth_count,
            "specificity_ratio": specificity,
            "normalized_cooccurrence_strength": cooccurrence,
            "first_pub_year": min(years) if years else None,
            "latest_pub_year": max(years) if years else None,
            "mean_pub_year": round(mean(years), 1) if years else None,
        })

    summary = pd.DataFrame(summary_rows).merge(ml, on="brain_region", how="left")
    summary["literature_rank_by_count"] = rank_from_score(summary["direct_original_studies"], ascending=False)
    summary["literature_rank_by_specificity"] = rank_from_score(summary["specificity_ratio"], ascending=False)
    summary["literature_rank_by_cooccurrence"] = rank_from_score(summary["normalized_cooccurrence_strength"], ascending=False)
    summary["literature_composite_rank"] = summary[
        ["literature_rank_by_count", "literature_rank_by_specificity", "literature_rank_by_cooccurrence"]
    ].mean(axis=1).rank(method="min").astype(int)

    merged = summary.dropna(subset=["ml_rank", "literature_composite_rank"]).copy()
    spearman = stats.spearmanr(merged["ml_rank"], merged["literature_composite_rank"])
    kendall = stats.kendalltau(merged["ml_rank"], merged["literature_composite_rank"])
    ml_top3 = set(merged.nsmallest(3, "ml_rank")["brain_region"])
    lit_top3 = set(merged.nsmallest(3, "literature_composite_rank")["brain_region"])
    ml_top5 = set(merged.nsmallest(5, "ml_rank")["brain_region"])
    lit_top5 = set(merged.nsmallest(5, "literature_composite_rank")["brain_region"])
    metrics = {
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "kendall_tau": float(kendall.statistic),
        "kendall_p": float(kendall.pvalue),
        "top3_overlap": len(ml_top3 & lit_top3) / 3,
        "top5_overlap": len(ml_top5 & lit_top5) / 5,
    }
    metrics["permutation_p"] = permutation_pvalue(
        merged["ml_rank"].tolist(),
        merged["literature_composite_rank"].tolist(),
        metrics["spearman_rho"],
    )

    ml_median = merged["ml_rank"].median()
    lit_median = merged["literature_composite_rank"].median()
    summary["concordance_class"] = "limited_evidence"
    summary.loc[
        (summary["ml_rank"] <= ml_median) & (summary["literature_composite_rank"] <= lit_median),
        "concordance_class",
    ] = "convergent"
    summary.loc[
        (summary["ml_rank"] <= ml_median) & (summary["literature_composite_rank"] > lit_median),
        "concordance_class",
    ] = "novel_candidate"
    summary.loc[
        (summary["ml_rank"] > ml_median) & (summary["literature_composite_rank"] <= lit_median),
        "concordance_class",
    ] = "literature_supported_lower_ml"

    summary = summary.sort_values("ml_rank")
    summary.to_csv(OUT_DIR / "literature_evidence_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "concordance_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    plot(summary, metrics)

    report_lines = [
        "# PubMed文献证据一致性分析报告",
        "",
        f"检索日期：{date.today().isoformat()}",
        f"数据库：PubMed",
        f"甲基苯丙胺主题检索总量：{meth_count}",
        f"候选脑区数量：{len(BRAIN_REGIONS)}",
        f"按候选脑区检索并去重后的PubMed记录数：{len(unique_records)}",
        "",
        "## 一致性指标",
        "",
        f"- Spearman rho = {metrics['spearman_rho']:.3f}, p = {metrics['spearman_p']:.4f}, permutation p = {metrics['permutation_p']:.4f}",
        f"- Kendall tau = {metrics['kendall_tau']:.3f}, p = {metrics['kendall_p']:.4f}",
        f"- Top-3 overlap = {metrics['top3_overlap']:.0%}",
        f"- Top-5 overlap = {metrics['top5_overlap']:.0%}",
        "",
        "## 主要解释",
        "",
        "本分析用于评价机器学习筛选脑区与既有甲基苯丙胺研究之间的文献证据一致性，不用于证明模型准确性或脑区因果作用。"
        "文献排名综合考虑脑区-甲基苯丙胺共现原始研究数量、相对于脑区总体研究量的特异性比例，以及标准化共现强度。",
        "",
        "## 脑区汇总表",
        "",
        dataframe_to_markdown(summary[[
            "brain_region", "ml_rank", "top10_frequency", "stability", "direct_original_studies",
            "intervention_or_circuit_studies_auto", "specificity_ratio",
            "normalized_cooccurrence_strength", "literature_composite_rank", "concordance_class",
        ]]),
        "",
        "## 方法文字草稿",
        "",
        "To evaluate whether data-driven prioritization of candidate brain regions was concordant with the existing methamphetamine literature, "
        "we performed a PubMed-based literature evidence concordance analysis. All brain regions entering the original screening procedure were queried, "
        "rather than only the final high-priority regions. Searches combined methamphetamine-related terms with the full names and common synonyms of each "
        "candidate region. Retrieved records were parsed for bibliographic information, publication type, abstract text, and MeSH terms. Reviews, editorials, "
        "letters, comments, news items, and meta-analyses were excluded from the primary original-study count by automated publication-type screening. "
        "For each region, we calculated the number of direct methamphetamine-related original studies, the number of records containing intervention or circuit "
        "evidence keywords, a specificity ratio defined as methamphetamine-region records divided by all PubMed records for that region query, and a normalized "
        "co-occurrence strength. Literature-based ranks were compared with machine-learning ranks using Spearman correlation, Kendall tau, Top-k overlap, and "
        "a 10,000-iteration permutation test. The analysis was interpreted as external literature concordance rather than independent proof of predictive "
        "accuracy or causal relevance.",
    ]
    (OUT_DIR / "PubMed文献证据一致性分析报告.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved outputs in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
