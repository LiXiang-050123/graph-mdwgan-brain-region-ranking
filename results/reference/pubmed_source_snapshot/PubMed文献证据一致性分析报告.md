# PubMed文献证据一致性分析报告

检索日期：2026-07-22
数据库：PubMed
甲基苯丙胺主题检索总量：18910
候选脑区数量：18
按候选脑区检索并去重后的PubMed记录数：608

## 一致性指标

- Spearman rho = 0.536, p = 0.0218, permutation p = 0.0228
- Kendall tau = 0.388, p = 0.0252
- Top-3 overlap = 33%
- Top-5 overlap = 60%

## 主要解释

本分析用于评价机器学习筛选脑区与既有甲基苯丙胺研究之间的文献证据一致性，不用于证明模型准确性或脑区因果作用。文献排名综合考虑脑区-甲基苯丙胺共现原始研究数量、相对于脑区总体研究量的特异性比例，以及标准化共现强度。

## 脑区汇总表

| brain_region | ml_rank | top10_frequency | stability | direct_original_studies | intervention_or_circuit_studies_auto | specificity_ratio | normalized_cooccurrence_strength | literature_composite_rank | concordance_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACC | 1 | 1250 | 0.9999999999999992 | 97 | 47 | 0.0035380599065677385 | 0.0043899082045339675 | 7 | convergent |
| NAco | 2 | 1231 | 0.9847999999999992 | 80 | 65 | 0.04293193717277487 | 0.013644304542164101 | 1 | convergent |
| CLA | 3 | 1226 | 0.9807999999999992 | 9 | 6 | 0.005802707930367505 | 0.0016618462679065156 | 13 | novel_candidate |
| VTA | 4 | 1203 | 0.9623999999999991 | 189 | 150 | 0.018087408048463868 | 0.01413889752311569 | 2 | convergent |
| DMS | 5 | 1126 | 0.9007999999999994 | 20 | 16 | 0.02886002886002886 | 0.0055248133946802155 | 4 | convergent |
| CeA | 6 | 1106 | 0.8847999999999994 | 25 | 15 | 0.0007932077910631923 | 0.0010642155570722777 | 15 | novel_candidate |
| BLA | 7 | 1096 | 0.8767999999999994 | 46 | 30 | 0.0021829916476841307 | 0.002304407014091219 | 10 | novel_candidate |
| PL | 8 | 900 | 0.7199999999999995 | 28 | 23 | 0.018704074816299265 | 0.005262612182501679 | 4 | convergent |
| VP | 9 | 711 | 0.5687999999999996 | 10 | 8 | 0.00764525993883792 | 0.0020107142860579536 | 11 | novel_candidate |
| OFC | 10 | 584 | 0.4671999999999996 | 56 | 34 | 0.006402604449267498 | 0.004469496252164351 | 6 | literature_supported_lower_ml |
| NAsh | 11 | 524 | 0.4191999999999997 | 60 | 52 | 0.02564102564102564 | 0.00916891141455478 | 3 | literature_supported_lower_ml |
| GP | 12 | 500 | 0.3999999999999997 | 48 | 36 | 0.004073150457190358 | 0.003248759047159494 | 8 | literature_supported_lower_ml |
| RSC | 13 | 295 | 0.2359999999999998 | 5 | 3 | 0.0019825535289452814 | 0.0007240218933119707 | 16 | limited_evidence |
| CM | 14 | 262 | 0.2095999999999998 | 0 | 0 | 0.0 | 0.0 | 18 | limited_evidence |
| IL | 15 | 175 | 0.1399999999999999 | 15 | 11 | 0.01642710472279261 | 0.0037281616933943686 | 8 | literature_supported_lower_ml |
| MS | 16 | 150 | 0.1199999999999999 | 3 | 3 | 0.0009609224855861628 | 0.0003904442585373418 | 17 | limited_evidence |
| LH | 17 | 108 | 0.0863999999999999 | 22 | 15 | 0.0034129692832764505 | 0.002037437725740252 | 12 | limited_evidence |
| AM | 18 | 53 | 0.0423999999999999 | 1 | 1 | 0.012195121951219513 | 0.0008030587332138194 | 14 | limited_evidence |

## 方法文字草稿

To evaluate whether data-driven prioritization of candidate brain regions was concordant with the existing methamphetamine literature, we performed a PubMed-based literature evidence concordance analysis. All brain regions entering the original screening procedure were queried, rather than only the final high-priority regions. Searches combined methamphetamine-related terms with the full names and common synonyms of each candidate region. Retrieved records were parsed for bibliographic information, publication type, abstract text, and MeSH terms. Reviews, editorials, letters, comments, news items, and meta-analyses were excluded from the primary original-study count by automated publication-type screening. For each region, we calculated the number of direct methamphetamine-related original studies, the number of records containing intervention or circuit evidence keywords, a specificity ratio defined as methamphetamine-region records divided by all PubMed records for that region query, and a normalized co-occurrence strength. Literature-based ranks were compared with machine-learning ranks using Spearman correlation, Kendall tau, Top-k overlap, and a 10,000-iteration permutation test. The analysis was interpreted as external literature concordance rather than independent proof of predictive accuracy or causal relevance.