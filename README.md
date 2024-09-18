# vlm-benchmarks
[GitHub Link](https://github.com/aryankargwal/vlm-benchmarks)<br>
[Youtube Link](https://youtu.be/MwryGctpWrM)


In the fast-evolving world of AI, Vision-Language Models (VLMs) are breaking new ground. Today, we are diving into [Pixtral 12B](https://mistral.ai/news/pixtral-12b/), the latest VLM from Mistral AI, which I benchmarked against GPT-4 on multiple datasets. This technical blog will walk you through my benchmarking process and share insights on how Pixtral 12B fares against GPT-4v in various tasks.

Pixtral 12B is an exciting release, and it brings several innovations to the table, including a 400M parameter vision encoder and a massive 128K token context window. If you’re working on any image-to-text pipelines, this might be the model you need. Let’s dig into the details.

### What is Pixtral 12B?

Pixtral 12B, Mistral AI's latest VLM, is built for complex multimodal tasks, such as chart analysis, code generation from images, and multi-image inferences. Its unique architecture features a **400M parameter vision encoder** capable of processing images at their native resolution and aspect ratio, significantly reducing preprocessing efforts. Additionally, the **128K token context window** allows it to handle up to 2,000 images in one batch, streamlining image processing at scale.


![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/cu3nts1m9o8rpophyh4k.png)

The model is versatile across various tasks, especially in understanding visuals with intricate details, such as diagrams. It even supports multi-image inferences, a feature highly beneficial for complex scenarios like medical imaging or sequential image analysis.

### Datasets and Benchmarks

For this benchmarking exercise, I evaluated Pixtral 12B on three key datasets:

1. **ArxivQA**: A large collection of research paper-based question-answering tasks.
   - Dataset link: [ArxivQA on Hugging Face](https://huggingface.co/datasets/MMInstruction/ArxivQA?row=0)
2. **VisIT Benchmark**: A dataset for vision-language instruction tasks inspired by real-life scenarios.
   - Dataset link: [VisIT Bench on Hugging Face](https://huggingface.co/datasets/mlfoundations/VisIT-Bench?row=0)
3. **Flickr30K**: A long-standing image captioning dataset with both human-generated and GPT-4o captions.
   - Dataset link: [Flickr30K on Hugging Face](https://huggingface.co/datasets/mlfoundations/VisIT-Bench?row=0)

In addition to evaluating Pixtral 12B, I also used **GPT-4v** for comparison. The critical evaluation metric was **Cosine Similarity**, which measures the semantic similarity between the generated captions and the references. This metric gives us insights into **win rate** and **top-k scores** (top-1, top-5, and top-10) for the model-generated captions and GPT-4v outputs.

### Cosine Similarity with all-MiniLM-L6-v2

In this benchmarking process, I used **Cosine Similarity** to evaluate the quality of the model-generated captions and responses by comparing them with reference texts. Specifically, I leveraged the **all-MiniLM-L6-v2** model, a lightweight transformer model fine-tuned for sentence embedding tasks, to compute the embeddings of both the predicted and reference texts.

#### Why Cosine Similarity?
Cosine Similarity is an efficient and commonly used metric for measuring the **semantic similarity** between two pieces of text. Unlike traditional methods like BLEU or METEOR, which emphasize exact word matching, Cosine Similarity evaluates the **contextual alignment** between two text embeddings, making it ideal for tasks like image captioning and question answering where the meaning of the text matters more than the exact word sequence.

For each comparison, both the reference and predicted texts were transformed into **vector embeddings** using the all-MiniLM-L6-v2 model, and the cosine similarity score was calculated as:

\[
\text{Cosine Similarity} = \frac{\text{A} \cdot \text{B}}{\|\text{A}\| \|\text{B}\|}
\]

Where:
- **A** and **B** represent the embeddings of the predicted and reference texts, respectively.
- The result is a score between -1 and 1, where 1 indicates that the two vectors are perfectly aligned (high similarity), and -1 indicates they are diametrically opposed.

#### Why all-MiniLM-L6-v2?

I chose **all-MiniLM-L6-v2** because of its balance between **speed** and **performance**. The model, with just 22 million parameters, is capable of generating high-quality sentence embeddings that can efficiently compute similarity scores in real-time. Despite being compact, it retains much of the semantic understanding found in larger models, making it ideal for scenarios like benchmarking where large volumes of data need to be processed quickly.

Here’s why **all-MiniLM-L6-v2** was the perfect fit for this task:
- **Efficient Embeddings**: It generates high-quality embeddings that are lightweight yet semantically rich.
- **Scalability**: Due to its small size, it scales well with large datasets without compromising inference speed.
- **Accurate Semantic Representation**: It captures a strong semantic understanding, essential when comparing captions or answers where the meaning matters more than exact matches.

This embedding model enabled me to compute **cosine similarity** for various benchmarks like **ArxivQA**, **VisIT**, and **Flickr30K**, allowing for a more nuanced evaluation of how well Pixtral 12B and GPT-4v perform on these datasets.

### Evaluation Setup and Methodology

To evaluate Pixtral 12B’s performance, I used [**Tune Studio**](https://studio.tune.app/playground), which offers **unlimited API calls** and provides fast inference with **350+ instruction inferences/hour** and **500+ captioning inferences/hour**.


![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/o05xbnfir2rzxbuesvtt.png)

Each dataset was benchmarked as follows:

- **ArxivQA**: I sampled 1,000 randomly selected images from a pool of 100,000 research paper-based questions. The model had to select the correct answer from multiple options and provide a rationale.
  
** VisIT Benchmark **: I evaluated the model on 500+ images containing real-life VLM applications. The task required Pixtral 12B to generate instruction-based responses from the images, which were then compared against human and GPT-4-generated captions.

- **Flickr30K**: For this dataset, Pixtral 12B generated captions for 1,000 random images. These captions were compared with both human and GPT-4o-generated captions.

### Results

#### ArxivQA
On the **ArxivQA** dataset, Pixtral 12B faced the challenge of generating accurate answers for research-based questions. Compared to **GPT-4v**, Pixtral 12B’s multi-word responses lowered the **win rate**, but its **rationale score** remained high, showcasing its ability to reason through complex topics.

| Metric          | GPT-4v (Labels)  | GPT-4v (Rationale)     |
|-----------------|------------------|------------------------|
| **Win Rate**    | 20.2%            | 23.1%                  |
| **Top-1 Score** | 90.8%            | 94.2%                  |
| **Top-5 Score** | 84.3%            | 94.1%                  |
| **Top-10 Score**| 77.2%            | 92.66%                 |

#### VisIT Benchmark
The **VisIT Benchmark** focuses on real-world VLM tasks, making it a more practical measure of Pixtral 12B’s capabilities. Pixtral 12B performed well against **GPT-4’s** captions, showing improved instruction-following abilities, especially when dealing with more specific queries.

| Metric          | Human Captions    | GPT-4 (Captions)       |
|-----------------|-------------------|------------------------|
| **Win Rate**    | 9.1%              | 37.5%                  |
| **Top-1 Score** | 88.4%             | 95.1%                  |
| **Top-5 Score** | 85.8%             | 94.6%                  |
| **Top-10 Score**| 84.4%             | 93.4%                  |

#### Flickr30K
For **Flickr30K**, Pixtral 12B’s performance was close to **GPT-4v**, especially for machine-generated captions, though it scored lower when compared to human captions due to its more concise and objective outputs.

| Metric          | Data Captions     | GPT-4v (Captions)      |
|-----------------|-------------------|------------------------|
| **Win Rate**    | 0%                | 5.1%                   |
| **Top-1 Score** | 33.6%             | 98.7%                  |
| **Top-5 Score** | 27.9%             | 97.8%                  |
| **Top-10 Score**| 0%                | 96.6%                  |

### Conclusion
In conclusion, **Pixtral 12B** proves to be a formidable contender in the VLM space. While it may not fully outshine **GPT-4** in terms of creative reasoning, its **analytical** and **cognitive** capabilities make it a valuable tool for tasks involving structured visual data, like charts, diagrams, and instructional content. It’s faster, cheaper, and more scalable for applications that rely on image-to-text processing.

As I continue to explore Pixtral 12B and other models, I’ll be sharing code and updates on my [GitHub repo](https://github.com/aryankargwal/vlm-benchmarks). If you’re curious about Pixtral’s performance in other benchmarks or know of datasets I should test, feel free to reach out in the comments!
