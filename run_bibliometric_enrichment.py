"""
Offline bibliometric enrichment for the PubMed concordance analysis.

This script reads the already retrieved PubMed CSV files and adds classical
bibliometric outputs: annual trends, journal distribution, MeSH frequencies,
brain-region by evidence-type matrices, region co-occurrence, and a manuscript
ready Chinese method/results draft.
"""

from __future__ import annotations

import itertools
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("outputs/pubmed_concordance")


STOP_MESH = {
    "Animals", "Humans", "Male", "Female", "Adult", "Aged", "Middle Aged",
    "Adolescent", "Young Adult", "Rats", "Mice", "Rats, Sprague-Dawley",
    "Rats, Wistar", "Time Factors", "Dose-Response Relationship, Drug",
    "Research Support, Non-U.S. Gov't", "Research Support, U.S. Gov't, P.H.S.",
}


def split_semicolon(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for value in row.tolist():
            cell = "" if pd.isna(value) else str(value)
            cells.append(cell.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, outfile: Path, horizontal: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    if horizontal:
        ax.barh(df[x], df[y], color="#4878a8")
        ax.invert_yaxis()
        ax.set_xlabel(y)
        ax.set_ylabel("")
    else:
        ax.bar(df[x], df[y], color="#4878a8")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(title)
    ax.grid(axis="y" if not horizontal else "x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def save_annual_trend(unique_records: pd.DataFrame) -> pd.DataFrame:
    years = unique_records.dropna(subset=["pub_year"]).copy()
    years["pub_year"] = years["pub_year"].astype(int)
    annual = years.groupby("pub_year").size().reset_index(name="records")
    annual.to_csv(OUT_DIR / "bibliometric_annual_publication_trend.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=180)
    ax.plot(annual["pub_year"], annual["records"], marker="o", color="#2f7d59", linewidth=2)
    ax.fill_between(annual["pub_year"], annual["records"], color="#2f7d59", alpha=0.15)
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Number of PubMed records")
    ax.set_title("Annual PubMed Publications on Methamphetamine-Related Candidate Brain Regions")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_annual_publication_trend.png", dpi=300)
    plt.close(fig)
    return annual


def save_journal_frequency(unique_records: pd.DataFrame) -> pd.DataFrame:
    journals = (
        unique_records["journal"].fillna("")
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(20)
        .reset_index()
    )
    journals.columns = ["journal", "records"]
    journals.to_csv(OUT_DIR / "bibliometric_top_journals.csv", index=False, encoding="utf-8-sig")
    save_bar(journals.head(15), "journal", "records", "Top Journals", OUT_DIR / "figure_top_journals.png", horizontal=True)
    return journals


def save_mesh_frequency(unique_records: pd.DataFrame) -> pd.DataFrame:
    counter = Counter()
    for value in unique_records["mesh_terms"]:
        for term in split_semicolon(value):
            if term not in STOP_MESH and not term.startswith("Research Support"):
                counter[term] += 1
    mesh = pd.DataFrame(counter.most_common(50), columns=["mesh_term", "records"])
    mesh.to_csv(OUT_DIR / "bibliometric_top_mesh_terms.csv", index=False, encoding="utf-8-sig")
    save_bar(mesh.head(20), "mesh_term", "records", "Top MeSH Terms After Generic-Term Filtering",
             OUT_DIR / "figure_top_mesh_terms.png", horizontal=True)
    return mesh


def save_evidence_matrix(long_records: pd.DataFrame) -> pd.DataFrame:
    labels = ["activity", "behavioral", "circuit", "intervention", "unclassified"]
    rows = []
    for region, subset in long_records.groupby("brain_region"):
        row = {"brain_region": region, "records": len(subset)}
        for label in labels:
            row[label] = int(subset["evidence_type_auto"].fillna("").str.contains(label, regex=False).sum())
        rows.append(row)
    matrix = pd.DataFrame(rows).sort_values("records", ascending=False)
    matrix.to_csv(OUT_DIR / "bibliometric_evidence_type_by_region.csv", index=False, encoding="utf-8-sig")

    plot_df = matrix.set_index("brain_region")[["activity", "behavioral", "circuit", "intervention"]]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    bottom = None
    colors = ["#4878a8", "#d98b2b", "#2f7d59", "#8f5aa8"]
    for color, col in zip(colors, plot_df.columns):
        ax.bar(plot_df.index, plot_df[col], bottom=bottom, label=col, color=color)
        bottom = plot_df[col] if bottom is None else bottom + plot_df[col]
    ax.set_xlabel("Brain region")
    ax.set_ylabel("Records with evidence keywords")
    ax.set_title("Automated Evidence-Type Coding by Brain Region")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_evidence_type_by_region.png", dpi=300)
    plt.close(fig)
    return matrix


def save_region_cooccurrence(long_records: pd.DataFrame) -> pd.DataFrame:
    pmid_regions = (
        long_records.groupby("pmid")["brain_region"]
        .apply(lambda values: sorted(set(values)))
        .reset_index()
    )
    edge_counter = Counter()
    for regions in pmid_regions["brain_region"]:
        for source, target in itertools.combinations(regions, 2):
            edge_counter[(source, target)] += 1
    edges = pd.DataFrame(
        [{"source": source, "target": target, "shared_records": count}
         for (source, target), count in edge_counter.most_common()]
    )
    edges.to_csv(OUT_DIR / "bibliometric_region_cooccurrence_edges.csv", index=False, encoding="utf-8-sig")
    return edges


def save_mesh_cooccurrence(unique_records: pd.DataFrame, top_terms: pd.DataFrame) -> pd.DataFrame:
    selected = set(top_terms.head(30)["mesh_term"])
    counter = Counter()
    for value in unique_records["mesh_terms"]:
        terms = sorted({term for term in split_semicolon(value) if term in selected})
        for source, target in itertools.combinations(terms, 2):
            counter[(source, target)] += 1
    edges = pd.DataFrame(
        [{"source": source, "target": target, "shared_records": count}
         for (source, target), count in counter.most_common(200)]
    )
    edges.to_csv(OUT_DIR / "bibliometric_mesh_cooccurrence_edges.csv", index=False, encoding="utf-8-sig")
    return edges


def save_species_distribution(long_records: pd.DataFrame) -> pd.DataFrame:
    counter = Counter()
    for value in long_records["species_auto"]:
        labels = split_semicolon(str(value).replace("not coded", "not coded"))
        for label in labels or ["not coded"]:
            counter[label] += 1
    species = pd.DataFrame(counter.most_common(), columns=["species_auto", "region_records"])
    species.to_csv(OUT_DIR / "bibliometric_species_distribution.csv", index=False, encoding="utf-8-sig")
    save_bar(species, "species_auto", "region_records", "Automated Species Coding",
             OUT_DIR / "figure_species_distribution.png", horizontal=False)
    return species


def save_bibliometric_report(
    unique_records: pd.DataFrame,
    long_records: pd.DataFrame,
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    journals: pd.DataFrame,
    mesh: pd.DataFrame,
    evidence: pd.DataFrame,
    species: pd.DataFrame,
) -> None:
    with open(OUT_DIR / "concordance_metrics.json", "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with open(OUT_DIR / "pubmed_queries.json", "r", encoding="utf-8") as handle:
        queries = json.load(handle)

    original_unique = unique_records[
        ~unique_records["publication_types"].fillna("").str.lower().str.contains(
            "review|editorial|letter|comment|news|meta-analysis", regex=True
        )
    ]
    first_year = int(unique_records["pub_year"].dropna().astype(int).min())
    latest_year = int(unique_records["pub_year"].dropna().astype(int).max())
    peak = annual.sort_values("records", ascending=False).iloc[0]

    top_region = summary.sort_values("direct_original_studies", ascending=False).iloc[0]
    top_specific = summary.sort_values("specificity_ratio", ascending=False).iloc[0]

    report = [
        "# PubMed文献计量学与脑区证据一致性分析",
        "",
        "## 数据概况",
        "",
        f"- 检索日期：{queries.get('search_date')}",
        "- 数据库：PubMed",
        f"- 甲基苯丙胺主题检索总量：{queries.get('methamphetamine_pubmed_count')}",
        f"- 候选脑区数量：{summary['brain_region'].nunique()}",
        f"- 脑区-文献长表记录：{len(long_records)}",
        f"- 去重后PubMed记录：{len(unique_records)}",
        f"- 自动排除综述、社论、评论、新闻和Meta分析后的去重原始研究：{len(original_unique)}",
        f"- 文献年份范围：{first_year}-{latest_year}",
        f"- 年发文峰值：{int(peak['pub_year'])}年，{int(peak['records'])}篇",
        "",
        "## 主要文献计量学结果",
        "",
        f"- 文献数量最高的候选脑区为 {top_region['brain_region']}，自动识别原始研究 {int(top_region['direct_original_studies'])} 条。",
        f"- 文献特异性最高的候选脑区为 {top_specific['brain_region']}，specificity ratio = {float(top_specific['specificity_ratio']):.4f}。",
        f"- 机器学习排序与文献综合排序呈中等一致：Spearman rho = {metrics['spearman_rho']:.3f}，置换检验 p = {metrics['permutation_p']:.4f}；Top-5重叠率为 {metrics['top5_overlap']:.0%}。",
        "- 该结果支持“文献证据一致性/互补性”这一定位，但不能表述为文献计量学证明模型准确或证明脑区因果作用。",
        "",
        "## 候选脑区证据汇总",
        "",
        markdown_table(summary[[
            "brain_region", "ml_rank", "direct_original_studies",
            "intervention_or_circuit_studies_auto", "specificity_ratio",
            "normalized_cooccurrence_strength", "literature_composite_rank",
            "concordance_class",
        ]]),
        "",
        "## Top 15期刊",
        "",
        markdown_table(journals.head(15)),
        "",
        "## Top 20 MeSH主题词",
        "",
        markdown_table(mesh.head(20)),
        "",
        "## 证据类型分布",
        "",
        markdown_table(evidence),
        "",
        "## 物种自动编码分布",
        "",
        markdown_table(species),
        "",
        "## 方法部分可用文字",
        "",
        "为系统评价机器学习筛选脑区与既有甲基苯丙胺研究之间的关系，本研究在PubMed中开展文献计量学与文献证据一致性分析。"
        "分析对象为原始机器学习流程中参与筛选的全部候选脑区，而非仅限最终入选脑区。检索式由甲基苯丙胺相关主题词"
        "（包括methamphetamine、methamphetamine addiction、dependence、use disorder、self-administration、seeking、relapse和withdrawal）"
        "与各脑区全称及常用同义名称组合构成。检索所得记录经PMID去重，并提取题名、期刊、发表年份、摘要、Publication type和MeSH词。"
        "主要文献计量学指标包括年度发文趋势、期刊分布、MeSH主题词频次、脑区共现关系和证据类型分布。进一步地，针对每个候选脑区统计"
        "甲基苯丙胺相关原始研究数量、干预或环路证据关键词记录数、脑区文献特异性比例以及标准化共现强度，并构建文献综合排序。"
        "文献综合排序与机器学习排序之间采用Spearman秩相关、Kendall tau、Top-k重叠率和10,000次置换检验进行比较。该分析旨在评价"
        "数据驱动脑区优先级与既有文献证据的一致性及互补性，而不作为模型预测准确性或脑区因果作用的独立证明。",
        "",
        "## 结果部分可用文字",
        "",
        f"PubMed检索共获得{len(unique_records)}条去重记录，覆盖{summary['brain_region'].nunique()}个候选脑区，发表年份范围为{first_year}年至{latest_year}年。"
        f"脑区层面分析显示，{top_region['brain_region']}拥有最多甲基苯丙胺相关原始研究记录，而{top_specific['brain_region']}在脑区总体文献背景下表现出最高特异性。"
        f"机器学习排序与文献综合排序之间呈中等程度一致（Spearman rho = {metrics['spearman_rho']:.3f}，置换检验p = {metrics['permutation_p']:.4f}），"
        f"Top-5重叠率为{metrics['top5_overlap']:.0%}。这些结果提示，部分高优先级脑区与既有甲基苯丙胺研究形成收敛证据；与此同时，"
        "机器学习排序较高但文献积累较少的脑区可被解释为具有潜在新颖性的候选脑区，值得在后续独立数据集或动物/环路实验中进一步验证。",
    ]
    (OUT_DIR / "文献计量学增强分析报告.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description="Offline bibliometric enrichment")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.out_dir
    unique_records = pd.read_csv(OUT_DIR / "pubmed_unique_records.csv")
    long_records = pd.read_csv(OUT_DIR / "pubmed_region_records_long.csv")
    summary = pd.read_csv(OUT_DIR / "literature_evidence_summary.csv")

    annual = save_annual_trend(unique_records)
    journals = save_journal_frequency(unique_records)
    mesh = save_mesh_frequency(unique_records)
    evidence = save_evidence_matrix(long_records)
    save_region_cooccurrence(long_records)
    save_mesh_cooccurrence(unique_records, mesh)
    species = save_species_distribution(long_records)
    save_bibliometric_report(unique_records, long_records, summary, annual, journals, mesh, evidence, species)
    print(f"Bibliometric enrichment saved in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
