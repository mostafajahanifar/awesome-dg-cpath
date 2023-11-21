# Awesome Domain Generalization for Computational Pathology

## Datasets
Publicly available datasets for DG experiments in CPath. Column `DS` represents the type domain shift that can be studies with each dataset (`1: Covariate Shift`, `2: Prior Shift`, `3: Posterior Shift`, and `4: Class-Conditional Shift`).

| Dataset             | Application/Task                                        | DS | Domains                                  |
|---------------------|---------------------------------------------------------|----|------------------------------------------|
|                     |                   **Detection**                         |    |                                          |
| ATYPIA14 [[paper](http://ludo17.free.fr/mitos_atypia_2014/icpr2014_MitosAtypia_DataDescription.pdf)][[download](https://mitos-atypia-14.grand-challenge.org/)]           | Mitosis detection in breast cancer                       | 1  | 2 scanners                               |
| Crowdsource [[paper](https://www.worldscientific.com/doi/epdf/10.1142/9789814644730_0029)]        | Nuclei detection in renal cell carcinoma                | 3  | 6 annotators                             |
| TUPAC-Aux [[paper](https://www.sciencedirect.com/science/article/pii/S1361841518305231)][[download](https://tupac.grand-challenge.org/)]           | Mitosis detection in breast cancer                       | 1  | 3 centers                                |
| DigestPath [[paper](https://www.sciencedirect.com/science/article/pii/S1361841522001323)][[download](https://digestpath2019.grand-challenge.org/)]          | Signet ring cell detection in colon cancer               | 1  | 4 centers                                |
| TiGER-Cells [[paper](https://arxiv.org/abs/2206.11943)][[download](https://tiger.grand-challenge.org/)]        | TILs detection in breast cancer                          | 1  | 3 sources                                |
| MIDOG [[paper]()][[download]()]               | Mitosis detection in multiple cancer types               | 1, 2, 3 | 7 tumors, 2 species                  |
|  |                               **Classification**                           |    |                                          |
| TUPAC-Mitosis [[paper](https://www.sciencedirect.com/science/article/pii/S1361841518305231)][[download](https://tupac.grand-challenge.org/)]       | BC proliferation scoring based on mitosis score          | 1  | 3 centers                                |
| Camelyon16 [[paper](https://jamanetwork.com/journals/jama/article-abstract/2665774)][[download](https://camelyon16.grand-challenge.org/)]           | Lymph node WSI classification for BC metastasis          | 1  | 2 centers                                |
| PatchCamelyon [[paper](https://link.springer.com/chapter/10.1007/978-3-030-00934-2_24)][[download](https://patchcamelyon.grand-challenge.org/)]        | BC tumor classification based on Camelyon16              | 1  | 2 centers                                |
| Camelyon17 [[paper](https://ieeexplore.ieee.org/abstract/document/8447230)][[download](https://camelyon17.grand-challenge.org/)]           | BC metastasis detection and pN-stage estimation          | 1  | 5 centers                                |
| LC25000 [[paper](https://arxiv.org/abs/1912.12142)][[download](https://github.com/tampapath/lung_colon_image_set)]              | Lung and colon tumor classification                     | 4  | 2 organs                                 |
| Kather 100K [[paper](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002730)][[download](https://zenodo.org/records/1214456)]        | Colon cancer tissue phenotype classification            | 1  | 3 centers                                |
| WILDS [[paper](https://arxiv.org/abs/2012.07421)][[download](https://github.com/p-lambda/wilds)]              | BC tumor classification based on Camelyon17              | 1  | 5 centers                                |
| PANDA [[paper](https://www.nature.com/articles/s41591-021-01620-2)][[download](https://panda.grand-challenge.org/)]              | ISUP and Gleason grading of prostate cancer             | 1, 2, 3 | 2 centers                          |
|      |                                   **Regression**                       |    |                                          |
| TUPAC-PAM50 [[paper](https://www.sciencedirect.com/science/article/pii/S1361841518305231)][[download](https://tupac.grand-challenge.org/)]         | BC proliferation scoring based on PAM50                  | 1  | 3 centers                                |
| LYSTO [[paper](https://arxiv.org/abs/2301.06304)][[download](https://lysto.grand-challenge.org/)]              | Lymphocyte assessment (counting) in IHC images           | 1  | 3 cancers, 9 centers                     |
| CoNIC (Lizard) [[paper](https://warwick.ac.uk/fac/cross_fac/tia/data/)][[download](https://arxiv.org/abs/2108.11195)]      | Cellular composition in colon cancer                     | 1, 3 | 6 sources                             |
| TiGER-TILs [[paper](https://arxiv.org/abs/2206.11943)][[download](https://tiger.grand-challenge.org/)]          | TIL score estimation in breast cancer                    | 1  | 3 sources                                |
|     |                            **Segmentation**                             |    |                                          |
| Crowdsource [[paper](https://www.worldscientific.com/doi/epdf/10.1142/9789814644730_0029)]         | Nuclear segmentation in renal cell carcinoma             | 3  | 6 annotators                             |
| Camelyon [[paper]()][[download]()]            | BC metastasis segmentation in lymph node WSIs            | 1  | 2 and 5 centers                          |
| DS Bowl 2018 [[paper](https://www.nature.com/articles/s41592-019-0612-7)][[download](https://www.kaggle.com/c/data-science-bowl-2018)]        | Nuclear instance segmentation                           | 1, 4 | 31 sets, 5 modalities                   |
| CPM [[paper](https://arxiv.org/abs/1810.13230)][[download](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK)]                 | Nuclear instance segmentation                            | 1, 4 | 4 cancers                                |
| BCSS [[paper](https://academic.oup.com/bioinformatics/article/35/18/3461/5307750)][[download](https://bcsegmentation.grand-challenge.org/)]                | Semantic tissue segmentation in BC (from TCGA)           | 1  | 20 centers                               |
| AIDPATH [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0169260719311381)]            | Glomeruli segmentation in Kidney biopsies               | 1  | 3 centers                                |
| PanNuke [[paper](https://arxiv.org/abs/2003.10778)][[download](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)]             | Nuclear instance segmentation and classification         | 1, 2, 4 | 19 organs                           |
| MoNuSeg [[paper](https://ieeexplore.ieee.org/document/8880654)][[download](https://monuseg.grand-challenge.org/)]             | Nuclear instance segmentation in H&E images              | 1  | 9 organs, 18 centers                     |
| CryoNuSeg [[paper](https://www.sciencedirect.com/science/article/pii/S0010482521001438)][[download](https://www.kaggle.com/datasets/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images)]           | Nuclear segmentation in cryosectioned H&E                | 1, 3 | 10 organs, 3 annotations                 |
| MoNuSAC [[paper](https://ieeexplore.ieee.org/abstract/document/9446924)][[download](https://monusac-2020.grand-challenge.org/)]             | Nuclear instance segmentation and classification         | 1, 2 | 37 centers, 4 organs                     |
| Lizard [[paper](https://warwick.ac.uk/fac/cross_fac/tia/data/)][[download](https://arxiv.org/abs/2108.11195)]              | Nuclear instance segmentation and classification         | 1, 3 | 6 sources                                |
| MetaHistoSeg [[paper](https://arxiv.org/abs/2109.14754)][[download](https://github.com/salesforce/MetaHistoSeg)]        | Multiple segmentation tasks in various cancers            | 1  | 5 sources/tasks                          |
| PANDA [[paper](https://www.nature.com/articles/s41591-021-01620-2)][[download](https://panda.grand-challenge.org/)]               | Tissue segmentation in prostate cancer                   | 1, 2 | 2 centers                                |
| TiGER-BCSS [[paper](https://academic.oup.com/bioinformatics/article/35/18/3461/5307750)][[download](https://bcsegmentation.grand-challenge.org/)]        | Tissue segmentation in BC (BCSS extension)               | 1  | 3 sources                                |
| DigestPath [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841522001323)][[download](https://digestpath2019.grand-challenge.org/)]          | Colon tissue segmentation                                | 1  | 4 centers                                |
| NuInsSeg [[paper](https://arxiv.org/abs/2308.01760)][[download](https://www.kaggle.com/datasets/ipateam/nuinsseg)]            | Nuclear instance segmentation pan-cancer/species         | 1,4 | 31 organs, 2 species                     |
| |         **Survival and gene expression prediction**                |    |                                          |
| TCGA                | Pan-cancer survival and gene expression prediction       | 1, 2, 4 | 33 cancers, 20 centers             |
| CPTAC               | Pan-cancer survival and gene expression prediction       | 1, 2 | 10 cancers, 11 centers                  |

## Code bases

| Reference | DG Method | Title |
|-----------|-----------|-------|
| **Pretraining** | | |
| Yang *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S1361841522001864)][[code](https://github.com/easonyang1996/CS-CO)] | Minimizing Contrastive Loss | CS-CO: A Hybrid Self-Supervised Visual Representation Learning Method for H&E-stained Histopathological Images |
| Li *et al*. [[paper](https://conferences.miccai.org/2022/papers/293-Paper1939.html)][[code](https://github.com/junl21/lacl)] \cite{6_li2022cpath} | Minimizing Contrastive Loss | Lesion-Aware Contrastive Representation Learning For Histopathology Whole Slide Images Analysis |
| Galdran *et al*. [[paper]()][[code]()] \cite{86_galdran2022cpath} | Unsupervised/Self-supervised learning | Test Time Transform Prediction for Open Set Histopathological Image Recognition |
| Bozorgtabar *et al*. [[paper]()][[code]()] \cite{87_bozorgtabar2021cpath} | Unsupervised/Self-supervised learning | SOoD: Self-Supervised Out-of-Distribution Detection Under Domain Shift for Multi-Class Colorectal Cancer Tissue Types |
| Koohbanani *et al*. [[paper]()][[code]()] \cite{89_koohbanani2021cpath} | Multiple Pretext Tasks | Self Path: Self Supervision for Classification of Histology Images with Limited Budget of Annotation |
| Abbet *et al*. [[paper]()][[code]()] \cite{91_abbet2022cpath} | Unsupervised/Self-supervised learning | Self-rule to multi-adapt: Generalized multi-source feature learning using unsupervised domain adaptation for colorectal cancer tissue detection |
| Cho *et al*. [[paper]()][[code]()]\cite{117_cho2021cpath} | Unsupervised/Self-supervised learning | Cell Detection in Domain Shift Problem Using Pseudo-Cell-Position Heatmap |
| Chikontwe *et al*. [[paper]()][[code]()]\cite{123_chikontwe2022cpath} | Unsupervised/Self-supervised learning | Weakly supervised segmentation on neural compressed histopathology with self-equivariant regularization |
| Tran *et al*. [[paper]()][[code]()]\cite{142_tran2022cpath} | Minimizing Contrastive Loss | S5CL: Unifying Fully-Supervised, Self-Supervised, and Semi-Supervised Learning Through Hierarchical Contrastive Learning |
| Sikaroudi *et al*. [[paper]()][[code]()]\cite{154_sikaroudi2020cpath} | Unsupervised/Self-supervised learning | Supervision and Source Domain Impact on Representation Learning: A Histopathology Case Study |
| Wang *et al*. [[paper]()][[code]()]\cite{158_wang2022cpath} | Unsupervised/Self-supervised learning | Transformer-based unsupervised contrastive learning for histopathological image classification |
| Kang *et al*. [[paper]()][[code]()]\cite{159_kang2023cpath} | Unsupervised/Self-supervised learning | Benchmarking Self-Supervised Learning on Diverse Pathology Datasets |
| Lazard *et al*. [[paper]()][[code]()]\cite{178_lazard2023cpath} | Contrastive Learning | Giga-SSL: Self-Supervised Learning for Gigapixel Images |
| Vuong *et al*. [[paper]()][[code]()]\cite{180_vuong2023cpath} | Contrastive Learning | IMPaSh: A Novel Domain-Shift Resistant Representation for Colorectal Cancer Tissue Classification |
| Chen *et al*. [[paper]()][[code]()]\cite{194_chen2022cpath} | Unsupervised/Self-supervised learning | Fast and scalable search of whole-slide images via self-supervised deep learning |
| **Meta-Learning** | | |
| Sikaroudi *et al*. [[paper]()][[code]()]\cite{16_sikaroudi2021cpath} | Meta-learning | Magnification Generalization For Histopathology Image Embedding |
| Yuan *et al*. [[paper]()][[code]()]\cite{45_yuan2021cpath} | Meta-learning | MetaHistoSeg: A Python Framework for Meta Learning in Histopathology Image Segmentation |
| **Domain Alignment** | | |
| Sharma *et al*. [[paper]()][[code]()]\cite{19_sharma2022cpath} | Mutual Information | MaNi: Maximizing Mutual Information for Nuclei Cross-Domain Unsupervised Segmentation |
| Boyd *et al*. [[paper]()][[code]()]\cite{31_boyd2022cpath} | Generative Models | Region-guided CycleGANs for Stain Transfer in Whole Slide Images |
| Kather *et al*. [[paper]()][[code]()]\cite{79_kather2019cpath} | Stain Normalization | Deep learning can predict microsatellite instability directly from histology in gastrointestinal cancer |
| Zheng *et al*. [[paper]()][[code]()]\cite{203_zheng2019cpath} | Stain Normalization | Adaptive color deconvolution for histological WSI normalization |
| Sebai *et al*. [[paper]()][[code]()]\cite{80_sebai2020cpath} | Stain Normalization | MaskMitosis: a deep learning framework for fully supervised, weakly supervised, and unsupervised mitosis detection in histopathology images |
| Zhang *et al*. [[paper]()][[code]()]\cite{90_zhang2022cpath} | Minimizing Contrastive Loss | Stain Based Contrastive Co-training for Histopathological Image Analysis |
| Shahban *et al*. [[paper]()][[code]()]\cite{103_shaban2019cpath} | Generative Models | Staingan: Stain Style Transfer for Digital Histological Images |
| Wagner *et al*. [[paper]()][[code]()]\cite{113_wagner2022cpath} | Generative Models | Federated Stain Normalization for Computational Pathology|
| Quiros *et al*. [[paper]()][[code]()]\cite{143_quiros2021cpath} | Domain Adversarial Learning | Adversarial learning of cancer tissue representations |
| Salehi *et al*. [[paper]()][[code]()]\cite{147_salehi2022cpath} | Minimizing the KL Divergence | Unsupervised Cross-Domain Feature Extraction for Single Blood Cell Image Classification |
| Wilm *et al*. [[paper]()][[code]()]\cite{188_wilm2021cpath} | Domain-Adversarial Learning | Domain adversarial retinanet as a reference algorithm for the mitosis domain generalization (midog) challenge |
| Haan *et al*. [[paper]()][[code]()]\cite{193_haan2021cpath} | Generative models | Deep learning-based transformation of H&E stained tissues into special stains |
| Dawood *et al*. [[paper]()][[code]()]\cite{207_dawood2023cpath} | Stain Normalization | Do Tissue Source Sites leave identifiable Signatures in Whole Slide Images beyond staining? | 
| **Data Augmentation** | | |
| Pohjonen *et al*. [[paper]()][[code]()]\cite{29_pohjonen2022cpath} | Data augmentation | Augment like there’s no tomorrow: Consistently performing neural networks for medical imaging  |
| Chang *et al*. [[paper]()][[code]()]\cite{56_chang2021cpath} | Stain Augmentation | Stain Mix-up: Unsupervised Domain Generalization for Histopathology Images |
| Shen *et al*. [[paper]()][[code]()]\cite{57_shen2022cpath} | Stain Augmentation | RandStainNA: Learning Stain-Agnostic Features from Histology Slides by Bridging Stain Augmentation and Normalization |
| Koohbanani *et al*. [[paper]()][[code]()]\cite{59_alemi-koohbanani2020cpath} | Data augmentation | NuClick: A deep learning framework for interactive segmentation of microscopic images  |
| Wang *et al*. [[paper]()][[code]()]\cite{61_wang2023cpath} | Data augmentation |  A generalizable and robust deep learning algorithm for mitosis detection in multicenter breast histopathological images | 
| Lin *et al*. [[paper]()][[code]()]\cite{62_lin2022cpath} | Generative Models | InsMix: Towards Realistic Generative Data Augmentation for Nuclei Instance Segmentation |
| Zhang *et al*. [[paper]()][[code]()]\cite{85_zhang2022cpath} | Data augmentation | Benchmarking the Robustness of Deep Neural Networks to Common Corruptions in Digital Pathology |
| Yamashita *et al*. [[paper]()][[code]()]\cite{92_yamashita2021cpath} | Style Transfer Models | Learning domain-agnostic visual representation for computational pathology using medically-irrelevant style transfer augmentation |
| Falahkheirkhah *et al*. [[paper]()][[code]()]\cite{98_falahkheirkhah2023cpath} | Generative Models | Deepfake Histologic Images for Enhancing Digital Pathology |
| Scalbert *et al*. [[paper]()][[code]()]\cite{102_scalbert2022cpath} | Generative Models | Test-time image-to-image translation ensembling improves out-of-distribution generalization in histopathology |
| Mahmood *et al*. [[paper]()][[code]()]\cite{128_mahmood2020cpath} | Generative Models | Deep Adversarial Training for Multi-Organ Nuclei Segmentation in Histopathology Images |
| Fan *et al*. [[paper]()][[code]()]\cite{141_fan2022cpath} | Generative Models | Fast FF-to-FFPE Whole Slide Image Translation via Laplacian Pyramid and Contrastive Learning |
| Marini *et al*. [[paper]()][[code]()]\cite{148_marini2023cpath} | Stain Augmentation | Data-driven color augmentation for H&E stained images in computational pathology |
| Faryna *et al*. [[paper]()][[code]()]\cite{163_faryna2021cpath} | RandAugment for Histology | Tailoring automated data augmentation to H&E-stained histopathology |
| **Model Design** | | |
| Graham *et al*. [[paper]()][[code]()]\cite{37_graham2020cpath} | Model design | Dense Steerable Filter CNNs for Exploiting Rotational Symmetry in Histology Images |
| Lafarge *et al*. [[paper]()][[code]()]\cite{38_lafarge2021cpath} | Model design | Roto-translation equivariant convolutional networks: Application to histopathology image analysis |
| Zhang *et al*. [[paper]()][[code]()]\cite{40_zhang2022cpath} | Model design | DDTNet: A dense dual-task network for tumor-infiltrating lymphocyte detection and segmentation in histopathological images of breast cancer |
| Graham *et al*. \cite{119_graham2023cpath} | Model Design | One model is all you need: Multi-task learning enables simultaneous histology image segmentation and classification |
| Yu *et al*. \cite{130_yu2023cpath} | Model Design |  Prototypical multiple instance learning for predicting lymph node metastasis of breast cancer from whole-slide pathological images|
| Yaar *et al*. \cite{131_yaar2020cpath} | Model Design | Cross-Domain Knowledge Transfer for Prediction of Chemosensitivity in Ovarian Cancer Patients |
| Tang *et al*. \cite{145_tang2021cpath} | Model Design | Probeable DARTS with Application to Computational Pathology |
| Vuong *et al*. \cite{182_vuong2021cpath} | Model Design | Joint categorical and ordinal learning for cancer grading in pathology images |
| **Learning Disentangled Representations** | | |
| Wagner *et al*. \cite{49_wagner2021cpath} | Generative Models | HistAuGAN: Structure-Preserving Multi-Domain Stain Color Augmentation using Style-Transfer with Disentangled Representations |
| Chikontwe *et al*. \cite{107_chikontwe2022cpath} | Learning disentangled representations | Feature Re-calibration based Multiple Instance Learning for Whole Slide Image Classification |
| **Ensemble Learning** | | |
| Sohail *et al*. [[paper]()][[code]()]\cite{50_sohail2021cpath} | Ensemble learning | Mitotic nuclei analysis in breast cancer histopathology images using deep ensemble classifier |
| **Regularization Strategies** | | |
| Mehrtens *et al*. [[paper]()][[code]()]\cite{138_mehrtens2023cpath} | Regularization Strategies | Benchmarking common uncertainty estimation methods with histopathological images under domain shift and label noise |
| **Other** | | |
| Lu *et al*. [[paper]()][[code]()]\cite{33_lu2022cpath} | Other | Federated learning for computational pathology on gigapixel whole slide images |
| Aubreville *et al*. [[paper]()][[code]()]\cite{190_aubreville2021cpath} | Other | Quantifying the Scanner-Induced Domain Gap in Mitosis Detection |
| Sadafi *et al*. [[paper]()][[code]()]\cite{211_sadafi2023cpath} | Other | A Continual Learning Approach for Cross-Domain White Blood Cell Classification |
