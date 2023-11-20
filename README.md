# Awesome Domain Generalization for Computational Pathology

## Datasets
Publicly available datasets for DG experiments in CPath. Column `DS` represents the type domain shift that can be studies with each dataset (`1: Covariate Shift`, `2: Prior Shift`, `3: Posterior Shift`, and `4: Class-Conditional Shift`).

| Dataset             | Application/Task                                        | DS | Domains                                  |
|---------------------|---------------------------------------------------------|----|------------------------------------------|
|                     |                   **Detection**                         |    |                                          |
| ATYPIA14 [paper][download]           | Mitosis detection in breast cancer                       | 1  | 2 scanners                               |
| Crowdsource         | Nuclei detection in renal cell carcinoma                | 3  | 6 annotators                             |
| TUPAC-Aux           | Mitosis detection in breast cancer                       | 1  | 3 centers                                |
| DigestPath          | Signet ring cell detection in colon cancer               | 1  | 4 centers                                |
| TiGER-Cells         | TILs detection in breast cancer                          | 1  | 3 sources                                |
| MIDOG               | Mitosis detection in multiple cancer types               | 1, 2, 3 | 7 tumors, 2 species                  |
|  |                               **Classification**                           |    |                                          |
| TUPAC-Mitosis       | BC proliferation scoring based on mitosis score          | 1  | 3 centers                                |
| Camelyon16          | Lymph node WSI classification for BC metastasis          | 1  | 2 centers                                |
| PatchCamelyon       | BC tumor classification based on Camelyon16              | 1  | 2 centers                                |
| Camelyon17          | BC metastasis detection and pN-stage estimation          | 1  | 5 centers                                |
| LC25000             | Lung and colon tumor classification                     | 4  | 2 organs                                 |
| Kather 100K         | Colon cancer tissue phenotype classification            | 1  | 3 centers                                |
| WILDS               | BC tumor classification based on Camelyon17              | 1  | 5 centers                                |
| PANDA               | ISUP and Gleason grading of prostate cancer             | 1, 2, 3 | 2 centers                          |
|      |                                   **Regression**                       |    |                                          |
| TUPAC-PAM50         | BC proliferation scoring based on PAM50                  | 1  | 3 centers                                |
| LYSTO               | Lymphocyte assessment (counting) in IHC images           | 1  | 3 cancers, 9 centers                     |
| CoNIC (Lizard)      | Cellular composition in colon cancer                     | 1, 3 | 6 sources                             |
| TiGER-TILs          | TIL score estimation in breast cancer                    | 1  | 3 sources                                |
|     |                            **Segmentation**                             |    |                                          |
| Crowdsource         | Nuclear segmentation in renal cell carcinoma             | 3  | 6 annotators                             |
| Camelyon            | BC metastasis segmentation in lymph node WSIs            | 1  | 2 and 5 centers                          |
| DS Bowl 2018        | Nuclear instance segmentation                           | 1, 4 | 31 sets, 5 modalities                   |
| CPM                 | Nuclear instance segmentation                            | 1, 4 | 4 cancers                                |
| BCSS                | Semantic tissue segmentation in BC (from TCGA)           | 1  | 20 centers                               |
| AIDPATH             | Glomeruli segmentation in Kidney biopsies               | 1  | 3 centers                                |
| PanNuke             | Nuclear instance segmentation and classification         | 1, 2, 4 | 19 organs                           |
| MoNuSeg             | Nuclear instance segmentation in H&E images              | 1  | 9 organs, 18 centers                     |
| CryoNuSeg           | Nuclear segmentation in cryosectioned H&E                | 1, 3 | 10 organs, 3 annotations                 |
| MoNuSAC             | Nuclear instance segmentation and classification         | 1, 2 | 37 centers, 4 organs                     |
| Lizard              | Nuclear instance segmentation and classification         | 1, 3 | 6 sources                                |
| MetaHistoSeg        | Multiple segmentation tasks in various cancers            | 1  | 5 sources/tasks                          |
| PANDA               | Tissue segmentation in prostate cancer                   | 1, 2 | 2 centers                                |
| TiGER-BCSS          | Tissue segmentation in BC (BCSS extension)               | 1  | 3 sources                                |
| DigestPath          | Colon tissue segmentation                                | 1  | 4 centers                                |
| NuInsSeg            | Nuclear instance segmentation pan-cancer/species         | 1,4 | 31 organs, 2 species                     |
| |         **Survival and gene expression prediction**                |    |                                          |
| TCGA                | Pan-cancer survival and gene expression prediction       | 1, 2, 4 | 33 cancers, 20 centers             |
| CPTAC               | Pan-cancer survival and gene expression prediction       | 1, 2 | 10 cancers, 11 centers                  |

## Code bases

| Reference | DG Method | Title |
|-----------|-----------|-------|
| **Pretraining** | | |
| Yang *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S1361841522001864)][[code](https://github.com/easonyang1996/CS-CO)] \cite{5_yang2022cpath}| Minimizing Contrastive Loss | CS-CO: A Hybrid Self-Supervised Visual Representation Learning Method for H&E-stained Histopathological Images |
| Li *et al*. [[paper](https://conferences.miccai.org/2022/papers/293-Paper1939.html)][[code](https://github.com/junl21/lacl?utm_source=catalyzex.com)] \cite{6_li2022cpath} | Minimizing Contrastive Loss | Lesion-Aware Contrastive Representation Learning For Histopathology Whole Slide Images Analysis |
| Galdran *et al*. [[paper](https://conferences.miccai.org/2022/papers/500-Paper1738.html)][[code](https://github.com/agaldran/t3po?utm_source=catalyzex.com)] \cite{86_galdran2022cpath} | Unsupervised/Self-supervised learning | Test Time Transform Prediction for Open Set Histopathological Image Recognition |
| Bozorgtabar *et al*. [[paper](https://openaccess.thecvf.com/content/ICCV2021W/CVAMD/papers/Bozorgtabar_SOoD_Self-Supervised_Out-of-Distribution_Detection_Under_Domain_Shift_for_Multi-Class_Colorectal_ICCVW_2021_paper.pdf)][[code](https://github.com/BehzadBozorgtabar/SOoD?utm_source=catalyzex.com)] \cite{87_bozorgtabar2021cpath} | Unsupervised/Self-supervised learning | SOoD: Self-Supervised Out-of-Distribution Detection Under Domain Shift for Multi-Class Colorectal Cancer Tissue Types |
| Koohbanani *et al*. [[paper](https://ieeexplore.ieee.org/abstract/document/9343323)][[code](https://github.com/navidstuv/self_path)] \cite{89_koohbanani2021cpath} | Multiple Pretext Tasks | Self Path: Self Supervision for Classification of Histology Images with Limited Budget of Annotation |
| Abbet *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S1361841522001207)][[code](https://github.com/christianabbet/SRA)] \cite{91_abbet2022cpath} | Unsupervised/Self-supervised learning | Self-rule to multi-adapt: Generalized multi-source feature learning using unsupervised domain adaptation for colorectal cancer tissue detection |
| Cho *et al*. [[paper](https://link.springer.com/chapter/10.1007/978-3-030-87237-3_37)][[code](https://github.com/hyeonwoocho7/Cell_Detection-MICCAI)]\cite{117_cho2021cpath} | Unsupervised/Self-supervised learning | Cell Detection in Domain Shift Problem Using Pseudo-Cell-Position Heatmap |
| Chikontwe *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S1361841522001293)][[code](https://github.com/PhilipChicco/wsshisto)]\cite{123_chikontwe2022cpath} | Unsupervised/Self-supervised learning | Weakly supervised segmentation on neural compressed histopathology with self-equivariant regularization |
| Tran *et al*. [[paper](https://conferences.miccai.org/2022/papers/434-Paper0610.html)][[code](https://github.com/manuel-tran/s5cl)]\cite{142_tran2022cpath} | Minimizing Contrastive Loss | S5CL: Unifying Fully-Supervised, Self-Supervised, and Semi-Supervised Learning Through Hierarchical Contrastive Learning |
| Sikaroudi *et al*. [[paper](https://ieeexplore.ieee.org/document/9176279)][[code](https://github.com/bghojogh/Siamese-Network-Histopathology)]\cite{154_sikaroudi2020cpath} | Unsupervised/Self-supervised learning | Supervision and Source Domain Impact on Representation Learning: A Histopathology Case Study |
| Wang *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S1361841522002043)][[code](https://github.com/Xiyue-Wang/TransPath)]\cite{158_wang2022cpath} | Unsupervised/Self-supervised learning | Transformer-based unsupervised contrastive learning for histopathological image classification |
| Kang *et al*. [[paper](https://openaccess.thecvf.com/content/CVPR2023/html/Kang_Benchmarking_Self-Supervised_Learning_on_Diverse_Pathology_Datasets_CVPR_2023_paper.html)][[code](https://github.com/lunit-io/benchmark-ssl-pathology)]\cite{159_kang2023cpath} | Unsupervised/Self-supervised learning | Benchmarking Self-Supervised Learning on Diverse Pathology Datasets |
| Lazard *et al*. [[paper](https://openaccess.thecvf.com/content/CVPR2023W/CVMI/papers/Lazard_Giga-SSL_Self-Supervised_Learning_for_Gigapixel_Images_CVPRW_2023_paper.pdf)][[code](https://github.com/trislaz/gigassl)]\cite{178_lazard2023cpath} | Contrastive Learning | Giga-SSL: Self-Supervised Learning for Gigapixel Images |
| Vuong *et al*. [[paper](https://link.springer.com/chapter/10.1007/978-3-031-25066-8_31)][[code](https://github.com/trinhvg/IMPash)]\cite{180_vuong2023cpath} | Contrastive Learning | IMPaSh: A Novel Domain-Shift Resistant Representation for Colorectal Cancer Tissue Classification |
| Chen *et al*. [[paper](https://www.nature.com/articles/s41551-022-00929-8)][[code](https://github.com/mahmoodlab/SISH)]\cite{194_chen2022cpath} | Unsupervised/Self-supervised learning | Fast and scalable search of whole-slide images via self-supervised deep learning |
| **Meta-Learning** | | |
| Sikaroudi *et al*. [[paper](https://ieeexplore.ieee.org/document/9433978)][[code](https://github.com/bghojogh/Histopathology-Magnification-Generalization)]\cite{16_sikaroudi2021cpath} | Meta-learning | Magnification Generalization For Histopathology Image Embedding |
| Yuan *et al*. [[paper](https://link.springer.com/chapter/10.1007/978-3-030-88210-5_27)][[code](https://github.com/salesforce/MetaHistoSeg)]\cite{45_yuan2021cpath} | Meta-learning | MetaHistoSeg: A Python Framework for Meta Learning in Histopathology Image Segmentation |
| **Domain Alignment** | | |
| Sharma *et al*. [[paper](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_34)][[code](https://github.com/YashSharma/MaNi)]\cite{19_sharma2022cpath} | Mutual Information | MaNi: Maximizing Mutual Information for Nuclei Cross-Domain Unsupervised Segmentation |
| Boyd *et al*. [[paper](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_35)][[code](https://github.com/jcboyd/miccai2022-roigan)]\cite{31_boyd2022cpath} | Generative Models | Region-guided CycleGANs for Stain Transfer in Whole Slide Images |
| Kather *et al*. [[paper](https://www.nature.com/articles/s41591-019-0462-y)][[code](https://github.com/jnkather/MSIfromHE/tree/master)]\cite{79_kather2019cpath} | Stain Normalization | Deep learning can predict microsatellite instability directly from histology in gastrointestinal cancer |
| Zheng *et al*. [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0169260718312161)][[code](https://github.com/Zhengyushan/adaptive_color_deconvolution)]\cite{203_zheng2019cpath} | Stain Normalization | Adaptive color deconvolution for histological WSI normalization |
| Sebai *et al*. [[paper](https://link.springer.com/article/10.1007/s11517-020-02175-z)][[code](https://github.com/MeriemSebai/MaskMitosis)]\cite{80_sebai2020cpath} | Stain Normalization | MaskMitosis: a deep learning framework for fully supervised, weakly supervised, and unsupervised mitosis detection in histopathology images |
| Zhang *et al*. [[paper](https://arxiv.org/abs/2206.12505)][[code](https://github.com/BzhangURU/Paper_2022_Co-training)]\cite{90_zhang2022cpath} | Minimizing Contrastive Loss | Stain Based Contrastive Co-training for Histopathological Image Analysis |
| Shahban *et al*. [[paper](https://arxiv.org/abs/1804.01601)][[code](https://github.com/xtarx/StainGAN)]\cite{103_shaban2019cpath} | Generative Models | Staingan: Stain Style Transfer for Digital Histological Images |
| Wagner *et al*. [[paper](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_2)][[code](https://github.com/meclabtuda/bottlegan)]\cite{113_wagner2022cpath} | Generative Models | Federated Stain Normalization for Computational Pathology|
| Quiros *et al*. [[paper](https://link.springer.com/chapter/10.1007/978-3-030-87237-3_58)][[code](https://github.com/AdalbertoCq/Adversarial-learning-of-cancer-tissue-representations)]\cite{143_quiros2021cpath} | Domain Adversarial Learning | Adversarial learning of cancer tissue representations |
| Salehi *et al*. [[paper](https://conferences.miccai.org/2022/papers/540-Paper2464.html)][[code](https://github.com/marrlab/AE-CFE)]\cite{147_salehi2022cpath} | Minimizing the KL Divergence | Unsupervised Cross-Domain Feature Extraction for Single Blood Cell Image Classification |
| Wilm *et al*. [[paper](https://link.springer.com/chapter/10.1007/978-3-030-97281-3_1)][[code](https://github.com/DeepMicroscopy/MIDOG)]\cite{188_wilm2021cpath} | Domain-Adversarial Learning | Domain adversarial retinanet as a reference algorithm for the mitosis domain generalization (midog) challenge |
| Haan *et al*. [[paper](https://www.nature.com/articles/s41467-021-25221-2)][[code](https://github.com/kevindehaan/stain-transformation)]\cite{193_haan2021cpath} | Generative models | Deep learning-based transformation of H&E stained tissues into special stains |
| Dawood *et al*. [[paper](https://openreview.net/forum?id=flfJ1OwD-FD)][[code](https://github.com/pkeller00/Src-Site-Pred)]\cite{207_dawood2023cpath} | Stain Normalization | Do Tissue Source Sites leave identifiable Signatures in Whole Slide Images beyond staining? | 
| **Data Augmentation** | | |
| Pohjonen *et al*. [[paper](https://arxiv.org/pdf/2206.15274.pdf)][[code](https://github.com/jopo666/StrongAugment)]\cite{29_pohjonen2022cpath} | Data augmentation | Augment like there’s no tomorrow: Consistently performing neural networks for medical imaging  |
| Chang *et al*. [[paper](https://miccai2021.org/openaccess/paperlinks/2021/09/01/453-Paper0297.html)][[code](https://github.com/aetherAI/stain-mixup)]\cite{56_chang2021cpath} | Stain Augmentation | Stain Mix-up: Unsupervised Domain Generalization for Histopathology Images |
| Shen *et al*. [[paper](https://conferences.miccai.org/2022/papers/406-Paper1231.html)][[code](https://github.com/yiqings/RandStainNA)]\cite{57_shen2022cpath} | Stain Augmentation | RandStainNA: Learning Stain-Agnostic Features from Histology Slides by Bridging Stain Augmentation and Normalization |
| Koohbanani *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S1361841520301353)][[code](https://github.com/mostafajahanifar/nuclick_torch)]\cite{59_alemi-koohbanani2020cpath} | Data augmentation | NuClick: A deep learning framework for interactive segmentation of microscopic images  |
| Wang *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S1361841522003310)][[code](https://github.com/Xiyue-Wang/1st-in-MICCAI-MIDOG-2021-challenge)]\cite{61_wang2023cpath} | Data augmentation |  A generalizable and robust deep learning algorithm for mitosis detection in multicenter breast histopathological images | 
| Lin *et al*. [[paper](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_14)][[code](https://github.com/hust-linyi/insmix)]\cite{62_lin2022cpath} | Generative Models | InsMix: Towards Realistic Generative Data Augmentation for Nuclei Instance Segmentation |
| Zhang *et al*. [[paper](https://rdcu.be/cVRrG)][[code](https://github.com/superjamessyx/robustness_benchmark)]\cite{85_zhang2022cpath} | Data augmentation | Benchmarking the Robustness of Deep Neural Networks to Common Corruptions in Digital Pathology |
| Yamashita *et al*. [[paper](https://ieeexplore.ieee.org/abstract/document/9503389)][[code](https://github.com/rikiyay/style-transfer-for-digital-pathology)]\cite{92_yamashita2021cpath} | Style Transfer Models | Learning domain-agnostic visual representation for computational pathology using medically-irrelevant style transfer augmentation |
| Falahkheirkhah *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S002368372200006X)][[code](https://github.com/kiakh93/Synthesizing-histological-images)]\cite{98_falahkheirkhah2023cpath} | Generative Models | Deepfake Histologic Images for Enhancing Digital Pathology |
| Scalbert *et al*. [[paper](https://conferences.miccai.org/2022/papers/503-Paper0733.html)][[code](https://gitlab.com/vitadx/articles/test-time-i2i-translation-ensembling)]\cite{102_scalbert2022cpath} | Generative Models | Test-time image-to-image translation ensembling improves out-of-distribution generalization in histopathology |
| Mahmood *et al*. [[paper](https://ieeexplore.ieee.org/abstract/document/8756037)][[code](https://github.com/mahmoodlab/NucleiSegmentation)]\cite{128_mahmood2020cpath} | Generative Models | Deep Adversarial Training for Multi-Organ Nuclei Segmentation in Histopathology Images |
| Fan *et al*. [[paper](https://conferences.miccai.org/2022/papers/209-Paper2730.html)][[code](https://github.com/hellodfan/fastFF2FFPE)]\cite{141_fan2022cpath} | Generative Models | Fast FF-to-FFPE Whole Slide Image Translation via Laplacian Pyramid and Contrastive Learning |
| Marini *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S2153353922007830)][[code](https://github.com/ilmaro8/Data_Driven_Color_Augmentation)]\cite{148_marini2023cpath} | Stain Augmentation | Data-driven color augmentation for H&E stained images in computational pathology |
| Faryna *et al*. [[paper](https://proceedings.mlr.press/v143/faryna21a.html)][[code](https://github.com/DIAGNijmegen/pathology-he-auto-augment)]\cite{163_faryna2021cpath} | RandAugment for Histology | Tailoring automated data augmentation to H&E-stained histopathology |
| **Model Design** | | |
| Graham *et al*. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9153847)][[code](https://github.com/simongraham/dsf-cnn)]\cite{37_graham2020cpath} | Model design | Dense Steerable Filter CNNs for Exploiting Rotational Symmetry in Histology Images |
| Lafarge *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S1361841520302139)][[code](https://github.com/tueimage/se2cnn)]\cite{38_lafarge2021cpath} | Model design | Roto-translation equivariant convolutional networks: Application to histopathology image analysis |
| Zhang *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S1361841522000676)][[code](https://github.com/ZhangXX54/DDTNet)]\cite{40_zhang2022cpath} | Model design | DDTNet: A dense dual-task network for tumor-infiltrating lymphocyte detection and segmentation in histopathological images of breast cancer |
| Graham *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S1361841522003139)][[code](https://github.com/TissueImageAnalytics/cerberus)]\cite{119_graham2023cpath} | Model Design | One model is all you need: Multi-task learning enables simultaneous histology image segmentation and classification |
| Yu *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S1361841523000099)][[code](https://github.com/Zero-We/PMIL)]\cite{130_yu2023cpath} | Model Design |  Prototypical multiple instance learning for predicting lymph node metastasis of breast cancer from whole-slide pathological images|
| Yaar *et al*. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9151061)][[code](https://github.com/asfandasfo/LUPI)]\cite{131_yaar2020cpath} | Model Design | Cross-Domain Knowledge Transfer for Prediction of Chemosensitivity in Ovarian Cancer Patients |
| Tang *et al*. [[paper](https://ieeexplore.ieee.org/document/9607807)][[code](https://github.com/mahdihosseini/DARTS-ADP)]\cite{145_tang2021cpath} | Model Design | Probeable DARTS with Application to Computational Pathology |
| Vuong *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S1361841521002516)][[code](https://github.com/trinhvg/JCO_Learning-pytorch)]\cite{182_vuong2021cpath} | Model Design | Joint categorical and ordinal learning for cancer grading in pathology images |
| **Learning Disentangled Representations** | | |
| Wagner *et al*. [[paper](https://link.springer.com/chapter/10.1007/978-3-030-87237-3_25)][[code](https://github.com/sophiajw/HistAuGAN)]\cite{49_wagner2021cpath} | Generative Models | HistAuGAN: Structure-Preserving Multi-Domain Stain Color Augmentation using Style-Transfer with Disentangled Representations |
| Chikontwe *et al*. [[paper](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_41)][[code](https://github.com/PhilipChicco/FRMIL)]cite{107_chikontwe2022cpath} | Learning disentangled representations | Feature Re-calibration based Multiple Instance Learning for Whole Slide Image Classification |
| **Ensemble Learning** | | |
| Sohail *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S1361841521001675)][[code](https://github.com/PRLAB21/Mitosis-Detection)]\cite{50_sohail2021cpath} | Ensemble learning | Mitotic nuclei analysis in breast cancer histopathology images using deep ensemble classifier |
| **Regularization Strategies** | | |
| Mehrtens *et al*. [[paper](https://arxiv.org/abs/2301.01054)][[code](https://github.com/DBO-DKFZ/uncertainty-benchmark)]\cite{138_mehrtens2023cpath} | Regularization Strategies | Benchmarking common uncertainty estimation methods with histopathological images under domain shift and label noise |
| **Other** | | |
| Lu *et al*. [[paper](https://www.sciencedirect.com/science/article/pii/S1361841521003431)][[code](https://github.com/mahmoodlab/HistoFL)]\cite{33_lu2022cpath} | Other | Federated learning for computational pathology on gigapixel whole slide images |
| Aubreville *et al*. [[paper](https://arxiv.org/abs/2103.16515)][[code](https://github.com/DeepMicroscopy/MIDOG)]\cite{190_aubreville2021cpath} | Other | Quantifying the Scanner-Induced Domain Gap in Mitosis Detection |
| Sadafi *et al*. [[paper](https://arxiv.org/abs/2308.12679)][[code](https://github.com/marrlab/UACL)]\cite{211_sadafi2023cpath} | Other | A Continual Learning Approach for Cross-Domain White Blood Cell Classification |
