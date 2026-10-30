# MOSAIC Final Report

## Major Research Results

- A scalable preprocessing pipeline was developed to integrate and standardize single-cell RNA sequencing data from CELLxGENE and the Curated Cancer Cell Atlas (3CA), establishing a unified corpus of approximately 31 million cells and 35,210 genes. Building on this resource, DeepSC, a dual-stream Transformer foundation model, was developed and pretrained using self-supervised learning. Evaluation on downstream tasks, including cell type annotation and perturbation prediction, demonstrated competitive or improved performance compared with established baseline models. The source code, data processing scripts, and trained models have been made openly available to support reproducibility, reuse, and further development by the research community.

- The methodology and experimental findings have been documented in the master’s thesis DeepSC: Deep Transcriptomic Foundation Models for Single Cell RNA-Sequencing Data, covering multi-source data integration, the dual-stream Transformer architecture, self-supervised pretraining, and evaluation on cell type annotation and perturbation prediction. A preprint on arXiv makes these findings accessible to the wider research community and provides a basis for scientific exchange and further investigation.

- A journal manuscript presenting the methodological contributions and a comprehensive evaluation of DeepSC is currently in preparation, with submission planned to a high-impact journal. Additional experiments are ongoing, including extended comparisons with established foundation models, systematic ablation studies of architectural components and preprocessing strategies, and analyses of pretraining effectiveness and transferability across downstream tasks. These experiments aim to strengthen the evidence for the model’s performance and robustness and clarify the contributions of individual design choices.

## Problems or Changes of Objectives

- Integrating heterogeneous datasets required addressing inconsistent gene identifiers, incompatible feature spaces, and differences in expression distributions. These challenges were addressed through a unified preprocessing workflow, although broader gene coverage and more diverse evaluation datasets remain important areas for improvement.

- The work documented to date focuses on data integration, foundation model development, and validation through cell type annotation and perturbation prediction. The broader objectives of a cross-cancer malignant cell atlas, clinical outcome prediction, and a user-facing application platform require further development and validation.

## Future Perspective of Research Topics

- Future research will extend the model towards cancer-specific applications, including characterization of malignant cell states, investigation of tumor heterogeneity, and assessment of their relationships with clinical outcomes.

- Further methodological work will investigate biological prior knowledge, such as gene regulatory networks and pathways, larger and more diverse pretraining datasets, and parameter-efficient fine-tuning.

- The results provide a basis for scientific publications and follow-up third-party funding proposals on AI-driven single-cell analysis and precision oncology. Further development of reusable analytical workflows would support adoption by biomedical researchers.

## Impact on RWTH Research Strategy

- MOSAIC brings together DBIS, DSMA, and the Hayat Lab at the Department of Medicine II, University Hospital RWTH Aachen, combining expertise in artificial intelligence, research data management, and single-cell biology.

- The project strengthens interdisciplinary research at the interface of computer science and medicine by establishing methods for integrating large-scale biological datasets and learning transferable representations of cellular states.

- The resulting data processing methods, model architecture, and experimental findings provide a foundation for further collaborative research and publications, supporting RWTH’s visibility in computational life sciences and AI-driven biomedical research.
