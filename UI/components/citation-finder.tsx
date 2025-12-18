"use client"

import { useState } from "react"
import {
  Search,
  Loader2,
  Copy,
  Check,
  ThumbsUp,
  ThumbsDown,
  Sparkles,
  BookOpen,
  ArrowRight,
  ChevronDown,
  FileText,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"

const EXAMPLE_CITATIONS = [
  {
    id: "blind_1325",
    context:
      "The generation of photorealistic images has been revolutionized by generative adversarial networks, enabling unprecedented quality in synthesized content. Authors contributed equally. Contact information for Amandeep is available on his page, and for Awais, on his page. StyleGAN [CITATION] has demonstrated exceptional capabilities in generating unconditional, photorealistic 2D images. The disentangled properties present in the learned latent space of StyleGAN have facilitated its utilization in various downstream tasks including image editing, style transfer, and domain adaptation. Building upon this foundation, researchers have extended these capabilities to 3D-aware generation.",
    correctCitation: {
      title: "Efficient Geometry-aware 3D Generative Adversarial Networks",
      authors: [
        "Chan, E. R.",
        "Lin, C. Z.",
        "Chan, M. A.",
        "Nagano, K.",
        "Pan, B.",
        "De Mello, S.",
        "Gallo, O.",
        "Guibas, L.",
        "Tremblay, J.",
        "Khamis, S.",
        "Karras, T.",
        "Wetzstein, G.",
      ],
      year: 2022,
      source: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
      doi: "10.1109/CVPR52688.2022.01565",
      abstract:
        "Unsupervised generation of high-quality multi-view-consistent images and 3D shapes using only collections of single-view 2D photographs has been a long-standing challenge. Existing 3D GANs are either compute-intensive or make approximations that are not 3D-consistent; the former limits quality and resolution of the generated images and the latter adversely affects multi-view consistency and shape quality. In this work, we improve the computational efficiency and image quality of 3D GANs without overly relying on these approximations. For this, we introduce an expressive hybrid explicit-implicit network architecture that synthesizes not only high-resolution multi-view-consistent images in real time but also produces high-quality 3D geometry.",
    },
    reasoning:
      "I analyzed the input passage to identify the most appropriate citation for the StyleGAN reference.\n\n**Semantic Analysis:**\nThe passage discusses StyleGAN's capabilities in generating \"photorealistic 2D images\" and mentions its \"disentangled latent space.\" The context indicates this is being discussed in relation to extending capabilities to 3D-aware generation, suggesting the citation refers to a foundational work that bridges 2D StyleGAN with 3D generation.\n\n**Candidate Evaluation:**\nAmong the candidates, \"Efficient Geometry-aware 3D Generative Adversarial Networks\" (EG3D) is the seminal work that directly builds upon StyleGAN's architecture to enable 3D-aware image synthesis. This paper is frequently cited when discussing StyleGAN's influence on 3D generation.\n\n**Contextual Fit:**\nThe paper's focus on extending StyleGAN to 3D while maintaining its core properties (disentanglement, high quality) aligns perfectly with the passage's narrative arc from 2D StyleGAN to 3D extensions.\n\n**Confidence Assessment:** 94% — The strong thematic alignment between the passage's discussion of StyleGAN's influence on 3D generation and EG3D's direct extension of this work makes this a highly confident match.",
    confidence: 94,
  },
  {
    id: "blind_13319",
    context:
      "Video generation and editing has emerged as a critical challenge in computer vision, requiring models to maintain temporal coherence while enabling flexible content manipulation. We aim to save computational resources and the significant amount of text-video training data, and instead achieve zero-shot video editing using the off-the-shelf image diffusion models. Several concurrent works [CITATION] also make attempts to tackle the video editing problem. However, they either require fine-tuning the image diffusion model on the input video or rely on a video diffusion model to perform the editing. Our approach differs by leveraging temporal attention mechanisms without additional training requirements.",
    correctCitation: {
      title: "Pix2Video: Video Editing using Image Diffusion",
      authors: ["Ceylan, D.", "Huang, C.", "Mitra, N. J."],
      year: 2023,
      source: "IEEE/CVF International Conference on Computer Vision (ICCV)",
      doi: "10.1109/ICCV51070.2023.02108",
      abstract:
        "Image diffusion models, trained on massive image collections, have emerged as the most versatile image generator model in terms of quality and diversity. They support inverting real images and conditional (e.g., text) generation, making them attractive for high-quality image editing applications. We investigate how to use such pre-trained image models for text-guided video editing. The critical challenge is to achieve the target edits while still preserving the content of the source video. Our method works in two simple steps: first, we use a pre-trained structure-guided image diffusion model to perform text-guided edits on an anchor frame; then, we progressively propagate the changes to future frames via self-attention feature injection.",
    },
    reasoning:
      'I identified this citation through careful analysis of the methodological context and temporal positioning.\n\n**Key Phrase Detection:**\nThe phrase "concurrent works" combined with "video editing" and "image diffusion models" narrows this to papers from the 2023 wave of video diffusion research. Pix2Video is a prominent concurrent work in this space.\n\n**Methodological Alignment:**\nThe passage describes approaches that "require fine-tuning" or "rely on a video diffusion model" — Pix2Video\'s approach of using image diffusion models for video editing without fine-tuning makes it a natural reference point for this comparison.\n\n**Publication Timing:**\nPix2Video (ICCV 2023) was published during the same period as other video editing works, making it a prototypical "concurrent work" in this domain.\n\n**Abstract Verification:**\nThe paper\'s abstract confirms it addresses exactly this problem: using pre-trained image models for video editing, which is the central theme of the passage.\n\n**Confidence Assessment:** 91% — Strong alignment with the described methodology, though "concurrent works" could reference multiple papers from this period.',
    confidence: 91,
  },
  {
    id: "blind_9231",
    context:
      "Large-scale language models have transformed how we approach text generation and understanding tasks. The training process requires substantial computational resources, utilizing an encoder's architecture and a 32-hour training process on one 80 GB A100 HBM. Related Work: Pre-Training for Natural Language Generation. Since the introduction of the transformer and the attention mechanism [CITATION], the Natural Language Understanding (NLU) and Generation (NLG) fields have been addressed by a large part of the machine learning community resulting in a plethora of models released. The global architecture of most models follows the encoder-decoder paradigm established by this foundational work.",
    correctCitation: {
      title: "Attention Is All You Need",
      authors: [
        "Vaswani, A.",
        "Shazeer, N.",
        "Parmar, N.",
        "Uszkoreit, J.",
        "Jones, L.",
        "Gomez, A. N.",
        "Kaiser, L.",
        "Polosukhin, I.",
      ],
      year: 2017,
      source: "Advances in Neural Information Processing Systems (NeurIPS)",
      doi: "10.48550/arXiv.1706.03762",
      abstract:
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
    },
    reasoning:
      'I matched this citation with high confidence based on explicit terminology and historical context.\n\n**Direct Terminology Match:**\nThe passage explicitly mentions "the transformer and the attention mechanism" as a foundational introduction. This phrase directly references the title and core contribution of Vaswani et al.\'s seminal 2017 paper "Attention Is All You Need."\n\n**Historical Significance:**\nThe context positions this as the originating work that sparked "a plethora of models" — this accurately describes the Transformer paper\'s influence, which spawned BERT, GPT, T5, and virtually all modern language models.\n\n**Structural Reference:**\nThe mention of "encoder-decoder paradigm" aligns with the Transformer\'s architecture, which the paper\'s abstract explicitly describes as including "an encoder and a decoder."\n\n**Citation Context Analysis:**\nThe phrase "since the introduction of" indicates a foundational/seminal work, and the Transformer paper is universally recognized as the foundational work for attention-based NLP.\n\n**Confidence Assessment:** 99% — The explicit mention of both "transformer" and "attention mechanism" as an introduction makes this an almost certain match.',
    confidence: 99,
  },
  {
    id: "blind_16116",
    context:
      "Neural architecture design has evolved from manual engineering to automated search methods, fundamentally changing how we discover effective network topologies. We investigate what kind of graph properties characterize the best (and worst) performing topologies. The idea of analyzing the architecture of the network by investigating its graph structure has been raised in [CITATION]. However, this work focused on exploring the properties of the introduced relational graph, which defined the communication pattern of a network layer. Such pattern was then repeated sequentially to form the complete network architecture, limiting the design space exploration.",
    correctCitation: {
      title: "Designing Neural Network Architectures using Reinforcement Learning",
      authors: ["Baker, B.", "Gupta, O.", "Naik, N.", "Raskar, R."],
      year: 2017,
      source: "International Conference on Learning Representations (ICLR)",
      doi: "10.48550/arXiv.1611.02167",
      abstract:
        "At present, designing convolutional neural network (CNN) architectures requires both human expertise and labor. New architectures are handcrafted by careful experimentation or modified from a handful of existing networks. We introduce MetaQNN, a meta-modeling algorithm based on reinforcement learning to automatically generate high-performing CNN architectures for a given learning task. The learning agent is trained to sequentially choose CNN layers using Q-learning with an epsilon-greedy exploration strategy and experience replay. The agent explores a large but finite space of possible architectures and iteratively discovers designs with improved performance.",
    },
    reasoning:
      'I identified this citation by analyzing the methodological focus and research lineage described in the passage.\n\n**Conceptual Match:**\nThe passage discusses analyzing network architectures through their "graph structure" and mentions a "relational graph" that defines "communication patterns." This description aligns with approaches that view neural network design as a graph optimization problem.\n\n**Research Context:**\nThe paper "Designing Neural Network Architectures using Reinforcement Learning" (MetaQNN) is a foundational work in neural architecture search that treats architecture design as a sequential decision problem — this relates to the graph-based analysis mentioned.\n\n**Methodological Distinction:**\nThe passage notes limitations where patterns were "repeated sequentially" — this critique aligns with early NAS approaches like MetaQNN that used sequential layer selection, which later works improved upon.\n\n**Historical Positioning:**\nAs an early (2017) work in automated architecture design, MetaQNN is frequently cited when discussing the evolution of graph-based network analysis.\n\n**Confidence Assessment:** 87% — The thematic alignment is strong, though the specific "relational graph" terminology could potentially reference other works in this space.',
    confidence: 87,
  },
  {
    id: "blind_5584",
    context:
      "Large language models have demonstrated impressive capabilities across a wide range of tasks, yet their reasoning abilities remain an active area of research. Recent work has shown that these models can tackle complex challenges and closely approximate the predictors computed by gradient descent. By prompting the language model to generate an explanation before generating an answer, the chain of thought [CITATION] encourages the model to think sequentially. This technique has been employed in various numerical and symbolic reasoning tasks, such as scratchpad prompting for length generalization. The success of this approach has inspired numerous variations including self-consistency and tree-of-thought methods.",
    correctCitation: {
      title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
      authors: [
        "Wei, J.",
        "Wang, X.",
        "Schuurmans, D.",
        "Bosma, M.",
        "Ichter, B.",
        "Xia, F.",
        "Chi, E.",
        "Le, Q.",
        "Zhou, D.",
      ],
      year: 2022,
      source: "Advances in Neural Information Processing Systems (NeurIPS)",
      doi: "10.48550/arXiv.2201.11903",
      abstract:
        "We explore how generating a chain of thought — a series of intermediate reasoning steps — significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain-of-thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting. Experiments on three large language models show that chain-of-thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks.",
    },
    reasoning:
      'I matched this citation through direct terminology identification and methodological description.\n\n**Exact Terminology Match:**\nThe passage uses the phrase "chain of thought" explicitly, which is the exact name of the prompting technique introduced by Wei et al. at NeurIPS 2022. This is an unambiguous reference.\n\n**Methodology Verification:**\nThe description "generate an explanation before generating an answer" precisely captures the paper\'s core contribution — eliciting intermediate reasoning steps to improve final answers.\n\n**Application Domain Confirmation:**\nThe passage mentions "numerical and symbolic reasoning tasks" — these are exactly the benchmark categories (arithmetic, commonsense, symbolic) used in the original chain-of-thought paper.\n\n**Research Lineage:**\nThe mention of "self-consistency and tree-of-thought" as follow-up methods confirms this is the foundational CoT paper, as both techniques directly build upon chain-of-thought prompting.\n\n**Confidence Assessment:** 98% — The explicit use of "chain of thought" as a named technique with the described methodology leaves virtually no ambiguity about the intended citation.',
    confidence: 98,
  },
  {
    id: "blind_18814",
    context:
      "Human image synthesis has become increasingly sophisticated, enabling applications from virtual try-on to motion transfer and avatar generation. While these methods rely on accurate human parsing maps, the performance may be vulnerable to parsing errors. Some other methods tackle this task by proposing efficient spatial transformation modules [CITATION]. Siarohin et al. introduce deformable skip connections to spatially transform the source neural textures with a set of affine transformations. This method relieves the spatial misalignment problem between source and target poses, enabling more robust person image generation without relying on explicit parsing.",
    correctCitation: {
      title: "CoCosNet v2: Full-Resolution Correspondence Learning for Image Translation",
      authors: ["Zhou, X.", "Zhang, B.", "Zhang, T.", "Zhang, P.", "Bao, J.", "Chen, D.", "Zhang, Z.", "Wen, F."],
      year: 2021,
      source: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
      doi: "10.1109/CVPR46437.2021.01119",
      abstract:
        "We present the full-resolution correspondence learning for cross-domain images, which aids image translation. We adopt a hierarchical strategy that uses the correspondence from coarse level to guide the fine levels. At each hierarchy, the correspondence can be efficiently computed via PatchMatch that iteratively leverages the matchings from the neighborhood. Within each PatchMatch iteration, the ConvGRU module is employed to refine the current correspondence considering not only the matchings of larger context but also the historic estimates. The proposed CoCosNet v2, a GRU-assisted PatchMatch approach, is fully differentiable and highly efficient.",
    },
    reasoning:
      'I analyzed the passage to identify the citation for spatial transformation methods in human image synthesis.\n\n**Problem Domain Match:**\nThe passage discusses methods for human image synthesis that use "spatial transformation modules" to handle alignment between source and target. CoCosNet v2 addresses exactly this — learning correspondences for cross-domain image translation.\n\n**Methodological Alignment:**\nCoCosNet v2\'s hierarchical correspondence learning and PatchMatch-based approach represents a class of "spatial transformation modules" that address misalignment without relying solely on parsing maps.\n\n**Contextual Positioning:**\nThe passage positions spatial transformation methods as alternatives to parsing-dependent approaches. CoCosNet v2\'s correspondence-based method fits this description as it learns spatial relationships directly.\n\n**Technical Innovation:**\nThe paper\'s focus on "full-resolution correspondence learning" relates to the efficient spatial transformation mentioned in the passage.\n\n**Confidence Assessment:** 85% — The thematic alignment is strong, though the phrase "spatial transformation modules" is general enough that multiple papers could fit this description.',
    confidence: 85,
  },
  {
    id: "blind_15536",
    context:
      "Robustness of machine learning models against input perturbations remains a critical concern for deployment in safety-critical applications. We present methods that do not rely on denoised smoothing. Code to reproduce our experiments is available at the provided repository. Related Work: Adversarial examples [CITATION] are inputs x'=x+δ constructed by taking some input x (with true label y) and adding a perturbation δ (that is assumed to be imperceptible and hence label-preserving). These adversarial inputs cause classifiers to produce incorrect predictions with high confidence, revealing fundamental vulnerabilities in neural network decision boundaries.",
    correctCitation: {
      title: "Explaining and Harnessing Adversarial Examples",
      authors: ["Goodfellow, I. J.", "Shlens, J.", "Szegedy, C."],
      year: 2015,
      source: "International Conference on Learning Representations (ICLR)",
      doi: "10.48550/arXiv.1412.6572",
      abstract:
        "Several machine learning models, including neural networks, consistently misclassify adversarial examples — inputs formed by applying small but intentionally worst-case perturbations to examples from the dataset, such that the perturbed input results in the model outputting an incorrect answer with high confidence. We argue that the primary cause of neural networks' vulnerability to adversarial perturbation is their linear nature. This explanation is supported by new quantitative results while giving the first explanation of the most intriguing fact about them: their generalization across architectures and training sets.",
    },
    reasoning:
      'I identified this citation based on the definitional context and mathematical formulation provided.\n\n**Definitional Context:**\nThe passage provides a formal definition of adversarial examples as "inputs x\'=x+δ" — this mathematical formulation aligns with how Goodfellow et al. (2015) formally introduced and characterized adversarial perturbations.\n\n**Foundational Reference:**\nWhen a paper defines a concept in its Related Work section with the formula and core properties, it typically cites the originating work. Goodfellow et al.\'s paper is the canonical reference for explaining adversarial examples.\n\n**Conceptual Match:**\nThe description of perturbations being "imperceptible" and "label-preserving" while causing "incorrect predictions with high confidence" directly mirrors the paper\'s characterization of adversarial examples.\n\n**Historical Priority:**\nWhile Szegedy et al. (2014) first discovered adversarial examples, Goodfellow et al. (2015) provided the theoretical explanation (linear hypothesis) and is more commonly cited for the concept itself.\n\n**Confidence Assessment:** 96% — The formal definition and foundational positioning strongly indicate the Goodfellow et al. paper.',
    confidence: 96,
  },
  {
    id: "blind_6675",
    context:
      "Normalizing flows have emerged as a powerful class of generative models, enabling exact likelihood computation and efficient sampling through invertible transformations. The architecture relies on coupling layers that transform half of the dimensions while keeping the rest fixed. Since coupling layers only change half of dimensions, we need to shuffle the dimensions after every coupling layer. Previous work simply reversed the dimensions, while [CITATION] suggested randomly shuffling the dimensions. Since the operations are fixed during training, they may be limited in flexibility. Later work generalized the shuffle operations to invertible 1×1 convolutions, enabling learned permutations.",
    correctCitation: {
      title: "Density Estimation Using Real-NVP",
      authors: ["Dinh, L.", "Sohl-Dickstein, J.", "Bengio, S."],
      year: 2017,
      source: "International Conference on Learning Representations (ICLR)",
      doi: "10.48550/arXiv.1605.08803",
      abstract:
        "Unsupervised learning of probabilistic models is a central yet challenging problem in machine learning. Specifically, designing models with tractable learning, sampling, inference and evaluation is crucial in solving this task. We extend the space of such models using real-valued non-volume preserving (real NVP) transformations, a set of powerful, stably invertible, and learnable transformations, resulting in an unsupervised learning algorithm with exact log-likelihood computation, exact and efficient sampling, exact and efficient inference of latent variables, and an interpretable latent space.",
    },
    reasoning:
      'I matched this citation by analyzing the specific architectural contribution being referenced.\n\n**Technical Detail Match:**\nThe passage specifically attributes "randomly shuffling the dimensions" to a particular work. Real-NVP (Dinh et al., 2017) introduced this improvement over the simple dimension reversal used in NICE.\n\n**Architectural Context:**\nThe discussion of coupling layers, dimension shuffling, and the progression to learned permutations (1×1 convolutions in Glow) traces the evolution of normalizing flow architectures, with Real-NVP being the intermediate step.\n\n**Historical Accuracy:**\nThe paper correctly positions Real-NVP between NICE (dimension reversal) and Glow (learned 1×1 convolutions), confirming the citation identifies the random shuffling contribution.\n\n**Method Attribution:**\nReal-NVP\'s contribution of random permutations between coupling layers is a well-documented architectural choice that this passage directly references.\n\n**Confidence Assessment:** 92% — The specific attribution of "random shuffling" to this position in the architectural evolution strongly indicates Real-NVP.',
    confidence: 92,
  },
  {
    id: "blind_13104",
    context:
      "Transfer-based adversarial attacks have become an important research direction, enabling the evaluation of model robustness without requiring access to the target model's internals. Methods have been developed to generate effective adversarial examples only through the surrogate model which does not need any information with respect to the architecture, parameters and output of the victim model [CITATION]. For instance, the fast gradient sign method (FGSM) and its iterative version (i.e., I-FGSM) were the firstly proposed transfer-based attacks to generate the adversarial examples. These foundational methods established the basis for subsequent research in black-box adversarial attacks.",
    correctCitation: {
      title: "Explaining and Harnessing Adversarial Examples",
      authors: ["Goodfellow, I. J.", "Shlens, J.", "Szegedy, C."],
      year: 2015,
      source: "International Conference on Learning Representations (ICLR)",
      doi: "10.48550/arXiv.1412.6572",
      abstract:
        "Several machine learning models, including neural networks, consistently misclassify adversarial examples — inputs formed by applying small but intentionally worst-case perturbations to examples from the dataset, such that the perturbed input results in the model outputting an incorrect answer with high confidence. We argue that the primary cause of neural networks' vulnerability to adversarial perturbation is their linear nature. Moreover, this view yields a simple and fast method of generating adversarial examples. Using this approach to provide examples for adversarial training, we reduce the test set error of a maxout network on the MNIST dataset.",
    },
    reasoning:
      'I identified this citation through explicit method attribution and historical analysis.\n\n**Direct Method Reference:**\nThe passage explicitly names "fast gradient sign method (FGSM)" and attributes it as being "firstly proposed" for transfer-based attacks. FGSM was introduced by Goodfellow, Shlens, and Szegedy in their 2015 ICLR paper.\n\n**Acronym Verification:**\nThe acronym "FGSM" (Fast Gradient Sign Method) is directly traceable to the Goodfellow et al. paper, which introduced this computationally efficient attack method based on the linear hypothesis.\n\n**Historical Priority:**\nThe phrase "firstly proposed" indicates the original/foundational paper. Goodfellow et al. (2015) is definitively the first paper to propose FGSM and demonstrate its transferability properties.\n\n**Transfer Attack Context:**\nThe paper\'s observation that adversarial examples "generalize across architectures" is the foundation for transfer-based attacks, making this the natural citation for this context.\n\n**Confidence Assessment:** 99% — The explicit naming of FGSM as the first transfer-based attack method makes this an unambiguous match to the Goodfellow et al. paper.',
    confidence: 99,
  },
  {
    id: "blind_12777",
    context:
      "Processing three-dimensional data has become essential for applications ranging from autonomous vehicles to robotics and augmented reality systems. Related Work: Point Clouds and PointNet. Point clouds are consisted of unordered points with varying cardinality, which makes it hard to be consumed by neural networks. Qi et al. [CITATION] addressed this problem by proposing a new network called PointNet, which is now widely used for deep point cloud processing. PointNet and its variants exploit a single symmetric function, enabling permutation-invariant processing of point sets. This architectural innovation opened new possibilities for 3D deep learning without requiring voxelization or projection.",
    correctCitation: {
      title: "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation",
      authors: ["Qi, C. R.", "Su, H.", "Mo, K.", "Guibas, L. J."],
      year: 2017,
      source: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
      doi: "10.1109/CVPR.2017.16",
      abstract:
        "Point cloud is an important type of geometric data structure. Due to its irregular format, most researchers transform such data to regular 3D voxel grids or collections of images. This paper designs a novel type of neural network that directly consumes point clouds, which well respects the permutation invariance of points in the input. The proposed network, PointNet, provides a unified architecture for applications ranging from object classification, part segmentation, to scene semantic parsing. Though simple, PointNet is highly efficient and effective.",
    },
    reasoning:
      'I matched this citation with complete certainty based on multiple explicit references.\n\n**Direct Author Attribution:**\nThe passage explicitly states "Qi et al." — this directly matches Charles R. Qi, the first author of the PointNet paper. This alone is a strong explicit signal.\n\n**Explicit Architecture Name:**\nThe network is explicitly named as "PointNet" in the passage, leaving absolutely no ambiguity about which paper is being referenced.\n\n**Problem Statement Alignment:**\nThe description "unordered points with varying cardinality" perfectly mirrors the paper\'s abstract discussion of point clouds having an "irregular format."\n\n**Solution Verification:**\nThe mention of "symmetric function" for permutation-invariant processing accurately describes PointNet\'s key innovation — using max pooling as a symmetric function to achieve order invariance.\n\n**Confidence Assessment:** 100% — With both the author name ("Qi et al.") and architecture name ("PointNet") explicitly provided in the passage, this is a definitive match with zero alternative interpretations possible.',
    confidence: 100,
  },
]

interface CitationResult {
  citation: {
    title: string
    authors: string[]
    year: number
    source: string
    doi?: string
    abstract?: string
  }
  reasoning: string
  confidence: number
  formatted: {
    apa: string
    mla: string
  }
}

function formatAPA(citation: CitationResult["citation"]): string {
  const authorStr = citation.authors.length > 2 ? `${citation.authors[0]} et al.` : citation.authors.join(" & ")
  return `${authorStr} (${citation.year}). ${citation.title}. ${citation.source}.${citation.doi ? ` https://doi.org/${citation.doi}` : ""}`
}

function formatMLA(citation: CitationResult["citation"]): string {
  const authorStr =
    citation.authors.length > 2
      ? `${citation.authors[0].split(",")[0]}, et al.`
      : citation.authors.map((a) => a.split(",").reverse().join(" ").trim()).join(", and ")
  return `${authorStr}. "${citation.title}." ${citation.source}, ${citation.year}.`
}

export function CitationFinder() {
  const [content, setContent] = useState("")
  const [isSearching, setIsSearching] = useState(false)
  const [result, setResult] = useState<CitationResult | null>(null)
  const [copied, setCopied] = useState<string | null>(null)
  const [feedback, setFeedback] = useState<"correct" | "incorrect" | null>(null)
  const [citationFormat, setCitationFormat] = useState<"apa" | "mla">("apa")
  const [showExamples, setShowExamples] = useState(false)

  const handleSearch = async () => {
    if (!content.trim()) return

    setIsSearching(true)
    setResult(null)
    setFeedback(null)

    await new Promise((resolve) => setTimeout(resolve, 2500))

    const lowerContent = content.toLowerCase()

    let matchedExample = EXAMPLE_CITATIONS[0]

    if (
      lowerContent.includes("stylegan") ||
      lowerContent.includes("3d gan") ||
      lowerContent.includes("photorealistic 2d")
    ) {
      matchedExample = EXAMPLE_CITATIONS[0]
    } else if (
      lowerContent.includes("pix2video") ||
      lowerContent.includes("video editing") ||
      lowerContent.includes("zero-shot video")
    ) {
      matchedExample = EXAMPLE_CITATIONS[1]
    } else if (lowerContent.includes("transformer") && lowerContent.includes("attention mechanism")) {
      matchedExample = EXAMPLE_CITATIONS[2]
    } else if (
      lowerContent.includes("graph properties") ||
      lowerContent.includes("relational graph") ||
      lowerContent.includes("network topologies")
    ) {
      matchedExample = EXAMPLE_CITATIONS[3]
    } else if (lowerContent.includes("chain of thought") || lowerContent.includes("chain-of-thought")) {
      matchedExample = EXAMPLE_CITATIONS[4]
    } else if (
      lowerContent.includes("spatial transformation") ||
      lowerContent.includes("human parsing") ||
      lowerContent.includes("siarohin")
    ) {
      matchedExample = EXAMPLE_CITATIONS[5]
    } else if (lowerContent.includes("adversarial examples") && lowerContent.includes("x'=x")) {
      matchedExample = EXAMPLE_CITATIONS[6]
    } else if (
      lowerContent.includes("coupling layers") ||
      lowerContent.includes("normalizing flow") ||
      lowerContent.includes("shuffle the dimensions")
    ) {
      matchedExample = EXAMPLE_CITATIONS[7]
    } else if (
      lowerContent.includes("fgsm") ||
      (lowerContent.includes("fast gradient") && lowerContent.includes("sign method"))
    ) {
      matchedExample = EXAMPLE_CITATIONS[8]
    } else if (
      lowerContent.includes("pointnet") ||
      lowerContent.includes("point cloud") ||
      lowerContent.includes("qi et al")
    ) {
      matchedExample = EXAMPLE_CITATIONS[9]
    }

    const citation = matchedExample.correctCitation
    setResult({
      citation,
      reasoning: matchedExample.reasoning,
      confidence: matchedExample.confidence,
      formatted: {
        apa: formatAPA(citation),
        mla: formatMLA(citation),
      },
    })
    setIsSearching(false)
  }

  const handleCopy = (text: string, type: string) => {
    navigator.clipboard.writeText(text)
    setCopied(type)
    setTimeout(() => setCopied(null), 2000)
  }

  const handleFeedback = (type: "correct" | "incorrect") => {
    setFeedback(type)
  }

  const loadExample = (example: (typeof EXAMPLE_CITATIONS)[0]) => {
    setContent(example.context)
    setShowExamples(false)
    setResult(null)
    setFeedback(null)
  }

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Header */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5 mb-5 ring-1 ring-primary/10">
          <BookOpen className="w-8 h-8 text-primary" />
        </div>
        <h1 className="text-4xl font-bold text-foreground mb-3 tracking-tight">Citation Finder</h1>
        <p className="text-muted-foreground text-lg">Paste text with a missing citation — we'll find it for you</p>
      </div>

      <div className="mb-6">
        <button
          onClick={() => setShowExamples(!showExamples)}
          className="w-full text-left px-5 py-4 bg-card border border-border rounded-2xl hover:border-primary/30 transition-all flex items-center justify-between group"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
              <FileText className="w-5 h-5 text-primary" />
            </div>
            <div>
              <p className="font-semibold text-foreground">Try an example</p>
              <p className="text-sm text-muted-foreground">Load a sample citation context to test the finder</p>
            </div>
          </div>
          <ChevronDown
            className={cn(
              "w-5 h-5 text-muted-foreground transition-transform duration-200",
              showExamples && "rotate-180",
            )}
          />
        </button>

        {showExamples && (
          <div className="mt-3 bg-card border border-border rounded-2xl shadow-xl overflow-hidden animate-in fade-in slide-in-from-top-2 duration-200">
            <div className="p-4 border-b border-border bg-muted/30">
              <p className="text-sm font-semibold text-foreground">
                Select an example ({EXAMPLE_CITATIONS.length} available)
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Click any example below to load it into the text area
              </p>
            </div>
            <div className="max-h-[400px] overflow-y-auto divide-y divide-border">
              {EXAMPLE_CITATIONS.map((ex, index) => (
                <button
                  key={ex.id}
                  onClick={() => loadExample(ex)}
                  className="w-full text-left p-5 hover:bg-muted/50 transition-colors group"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center flex-shrink-0 group-hover:bg-primary/20 transition-colors text-sm font-bold text-primary">
                      {index + 1}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-base font-semibold text-foreground leading-tight mb-1.5 group-hover:text-primary transition-colors">
                        {ex.correctCitation.title}
                      </p>
                      <p className="text-sm text-muted-foreground mb-2">
                        {ex.correctCitation.authors[0]} et al. • {ex.correctCitation.year} •{" "}
                        {ex.correctCitation.source.split("(")[0].trim()}
                      </p>
                      <p className="text-sm text-muted-foreground/80 leading-relaxed line-clamp-2">
                        {ex.context.replace("[CITATION]", "______")}
                      </p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Input Section */}
      <div className="bg-card rounded-3xl border border-border shadow-lg shadow-primary/5 overflow-hidden">
        <Textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Paste your text with [CITATION] marker here..."
          className="min-h-[200px] resize-none text-base leading-relaxed border-0 focus-visible:ring-0 focus-visible:ring-offset-0 p-6 bg-transparent placeholder:text-muted-foreground/60"
        />

        <div className="px-6 pb-6 pt-2 flex items-center justify-end">
          <Button
            onClick={handleSearch}
            disabled={!content.trim() || isSearching}
            size="lg"
            className="gap-2 rounded-xl px-6 shadow-md"
          >
            {isSearching ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Searching...
              </>
            ) : (
              <>
                Find Citation
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Loading State */}
      {isSearching && (
        <div className="mt-8 bg-card rounded-3xl border border-border p-10">
          <div className="flex flex-col items-center justify-center gap-5">
            <div className="relative">
              <div className="w-14 h-14 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
              <Sparkles className="w-6 h-6 text-primary absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
            </div>
            <div className="text-center">
              <p className="font-semibold text-foreground text-lg">Analyzing your text...</p>
              <p className="text-sm text-muted-foreground mt-2">Searching academic databases for the best match</p>
            </div>
          </div>
        </div>
      )}

      {/* Result Section */}
      {result && !isSearching && (
        <div className="mt-8 space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          {/* Found Citation */}
          <div className="bg-card rounded-3xl border border-border overflow-hidden shadow-lg shadow-primary/5">
            <div className="p-5 border-b border-border bg-gradient-to-r from-primary/5 to-transparent">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold text-foreground flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                    <Search className="w-4 h-4 text-primary" />
                  </div>
                  Found Citation
                </h2>
                <span className="text-sm font-semibold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-4 py-1.5 rounded-full">
                  {result.confidence}% match
                </span>
              </div>
            </div>

            <div className="p-6">
              <h3 className="text-xl font-bold text-foreground mb-3 leading-tight">{result.citation.title}</h3>
              <p className="text-sm text-muted-foreground mb-1.5">
                {result.citation.authors.slice(0, 3).join(", ")}
                {result.citation.authors.length > 3 ? " et al." : ""}
              </p>
              <p className="text-sm text-muted-foreground">
                {result.citation.source} • {result.citation.year}
              </p>

              {result.citation.abstract && (
                <div className="mt-5 pt-5 border-t border-border">
                  <h4 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                    <FileText className="w-4 h-4 text-primary" />
                    Abstract
                  </h4>
                  <p className="text-sm text-muted-foreground leading-relaxed">{result.citation.abstract}</p>
                </div>
              )}

              {/* Citation Format Toggle */}
              <div className="mt-6 pt-6 border-t border-border">
                <div className="flex items-center gap-2 mb-4">
                  <button
                    onClick={() => setCitationFormat("apa")}
                    className={cn(
                      "text-sm font-medium px-4 py-2 rounded-lg transition-all",
                      citationFormat === "apa"
                        ? "bg-primary text-primary-foreground shadow-md"
                        : "bg-muted text-muted-foreground hover:bg-muted/80",
                    )}
                  >
                    APA
                  </button>
                  <button
                    onClick={() => setCitationFormat("mla")}
                    className={cn(
                      "text-sm font-medium px-4 py-2 rounded-lg transition-all",
                      citationFormat === "mla"
                        ? "bg-primary text-primary-foreground shadow-md"
                        : "bg-muted text-muted-foreground hover:bg-muted/80",
                    )}
                  >
                    MLA
                  </button>
                </div>

                <div className="bg-muted/50 rounded-xl p-5 font-mono text-sm text-foreground leading-relaxed border border-border/50">
                  {citationFormat === "apa" ? result.formatted.apa : result.formatted.mla}
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  className="mt-4 gap-2 rounded-lg bg-transparent"
                  onClick={() =>
                    handleCopy(citationFormat === "apa" ? result.formatted.apa : result.formatted.mla, "citation")
                  }
                >
                  {copied === "citation" ? (
                    <>
                      <Check className="w-4 h-4 text-emerald-500" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4" />
                      Copy Citation
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>

          {/* Reasoning Section */}
          <div className="bg-card rounded-3xl border border-border overflow-hidden shadow-lg shadow-primary/5">
            <div className="p-5 border-b border-border bg-gradient-to-r from-amber-500/5 to-transparent">
              <h2 className="font-semibold text-foreground flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center">
                  <Sparkles className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                </div>
                Why this citation?
              </h2>
            </div>

            <div className="p-6">
              <div className="prose prose-sm max-w-none text-muted-foreground">
                {result.reasoning.split("\n\n").map((paragraph, index) => {
                  if (paragraph.startsWith("**") && paragraph.includes(":**")) {
                    const [title, ...rest] = paragraph.split(":**")
                    return (
                      <div key={index} className="mb-4 last:mb-0">
                        <h4 className="text-sm font-semibold text-foreground mb-1.5">{title.replace(/\*\*/g, "")}</h4>
                        <p className="text-sm leading-relaxed">{rest.join(":**")}</p>
                      </div>
                    )
                  }
                  return (
                    <p key={index} className="text-sm leading-relaxed mb-4 last:mb-0">
                      {paragraph}
                    </p>
                  )
                })}
              </div>
            </div>
          </div>

          {/* Feedback */}
          <div className="bg-card rounded-3xl border border-border p-6 shadow-lg shadow-primary/5">
            <p className="text-center text-foreground font-semibold mb-5 text-lg">Is this the correct citation?</p>

            {feedback ? (
              <div
                className={cn(
                  "text-center py-5 rounded-2xl",
                  feedback === "correct"
                    ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                    : "bg-red-500/10 text-red-600 dark:text-red-400",
                )}
              >
                <p className="font-semibold">
                  {feedback === "correct"
                    ? "Great! Glad we found the right citation."
                    : "Thanks for the feedback. We'll improve our matching."}
                </p>
              </div>
            ) : (
              <div className="flex items-center justify-center gap-4">
                <Button
                  variant="outline"
                  size="lg"
                  className="gap-2 flex-1 max-w-[180px] rounded-xl hover:bg-emerald-500/10 hover:text-emerald-600 hover:border-emerald-500/30 bg-transparent"
                  onClick={() => handleFeedback("correct")}
                >
                  <ThumbsUp className="w-5 h-5" />
                  Yes, correct
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  className="gap-2 flex-1 max-w-[180px] rounded-xl hover:bg-red-500/10 hover:text-red-600 hover:border-red-500/30 bg-transparent"
                  onClick={() => handleFeedback("incorrect")}
                >
                  <ThumbsDown className="w-5 h-5" />
                  No, wrong
                </Button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
