---
layout: default
title: "How to Use This Book"
parent: "Context and Reader’s Guide"
nav_order: 3
has_children: false
---

# How to Use This Book

### About This Book

This book was written with a clear intent: to help readers understand how modern artificial intelligence workloads—particularly large-scale training—are executed, scaled, and optimized on supercomputing infrastructures. Its focus is not on proposing new algorithms or surveying model architectures, but on explaining how existing AI methods interact with hardware, system software, runtimes, and distributed execution environments.

The subtitle of this book—Foundations, Architectures, and Scaling Deep Learning Workloads—is a deliberate statement of scope and method. “Foundations” refers to the performance principles and reasoning habits used throughout the book to interpret measurements and avoid misleading conclusions. “Architectures” covers the hardware–software stack that makes large-scale training possible: compute nodes, system software, runtimes, and parallel execution models. “Scaling” focuses on making deep learning training workflows run efficiently across multiple GPUs and nodes, connecting high-level frameworks to the underlying execution and communication mechanisms.

The material originated from several university courses taught at the Universitat Politècnica de Catalunya (UPC), each with different scopes, depths, and prerequisites. As a result, the book was deliberately designed to support readers with diverse backgrounds and objectives. Rather than assuming a single, linear learning path, it provides multiple entry points and flexible routes through the content.

Some readers may approach the book from a high performance computing background and wish to understand how deep learning and large language models are trained at scale. Others may come from an AI or data science perspective and want to understand what happens beneath high-level frameworks when training is distributed across GPUs and nodes. This book is intended to support both perspectives, and to help bridge the conceptual gap between them.

This book is intended for readers who want a system-oriented understanding of how AI training runs in practice, including:

- Master’s and advanced undergraduate students seeking a practical view of AI workloads on modern supercomputing platforms;

- Researchers interested in performance, scalability, and distributed execution of large-scale training;

- Engineers and data scientists using deep learning frameworks who want to understand what happens beneath high-level APIs;

- HPC practitioners and system architects looking to connect AI requirements with infrastructure design choices;

- Instructors building reusable laboratory assignments and reproducible experiments for courses at the intersection of HPC and AI.

The emphasis throughout is on training workloads executed on supercomputing platforms. Inference, deployment, and edge scenarios are acknowledged where relevant, but they are not treated as primary optimization targets. This choice reflects the book’s central thesis: the most demanding and instructive challenges arise during large-scale training, where compute, memory, communication, and coordination costs interact most strongly.

### How to Read This Book

This book can be read in more than one way, and no single reading strategy is assumed.

It can be read as a conceptual and theoretical resource, focusing on architecture, execution models, and performance reasoning without executing any code. Readers following this approach may concentrate on the explanatory text, figures, and performance discussions, using the tasks as illustrative examples rather than as mandatory exercises.

Alternatively, it can be used as a hands-on guide, where understanding is built primarily through execution and experimentation. In this mode, readers are encouraged to run the tasks, modify parameters, observe performance behavior, and relate the results back to the conceptual models presented in the text.

Selective reading is not only acceptable; it is intentional. The book is structured around abstraction layers and self-contained chapters, making it possible to focus on specific parts without reading everything in sequence. For example:

- Readers with a strong supercomputing background may skim Parts I and II and jump directly to Parts III, IV or V.

- Readers already familiar with deep learning frameworks may skim Part III and focus on distributed training and scalability.

- Readers primarily interested in large language models may concentrate on Parts IV and V, using earlier chapters as reference material when needed.

Throughout the book, cross-references and recurring figures are used to maintain coherence across these different paths. The goal is not exhaustive coverage, but the development of a solid mental model that can be applied consistently across scales and technologies.

### How to Use This Book in Practical Laboratory Courses

A defining characteristic of this book is its task-based structure. Practical tasks are intentionally fine-grained, self-contained, and designed to be combined flexibly. They are not meant to be executed all at once, nor do they form a single mandatory sequence.

In instructor-led courses, laboratory assignments are typically constructed by selecting a subset of tasks aligned with specific learning objectives, available infrastructure, and time constraints. This approach allows the same book to support courses with very different profiles—for example, introductory HPC courses with an AI focus, advanced courses on distributed deep learning, or specialized seminars on large language models.

Independent readers are encouraged to adopt a similar strategy. Tasks should be treated as building blocks rather than as a checklist. Some tasks are exploratory and introductory, others consolidate core concepts, and a smaller number are intended for deeper experimentation and performance analysis. Readers may return to tasks multiple times as their understanding evolves.

Reproducibility is a central concern in the design of these tasks. Wherever possible, experiments are grounded in concrete execution environments, explicit job scripts, and well-defined software stacks. Examples are aligned with real systems—such as MareNostrum 5 supercomputer at the Barcelona Supercomputing Center—or with accessible platforms like Google Colab, allowing readers to move between environments while preserving the same execution model.

This modular, task-oriented approach reflects a pedagogical belief that depth of understanding is achieved through carefully chosen practical experiences, not through exhaustive coverage. The book is therefore intended to be reused, adapted, and revisited, both in formal courses and in independent study, as technologies and workloads continue to evolve.

### Typographical Conventions Used in This Book

Throughout this book, we use a set of typographical conventions to improve readability and clarity. These are summarized below for your reference:

- Text in Courier is used to indicate variable names in code, file names, URLs, and similar elements.

-  Text in *italics* is used to highlight important concepts within the book’s content, often when they are introduced for the first time.

- Code blocks are displayed using a monospaced font on a gray background, as shown below:

> 
>
>     #include <stdio.h>
>
>     int main(){
>
>         printf("Hello world!\n");
>
>     }

- Highlighted lines within code blocks—those that are referenced in the main text—appear in bold monospaced font on the same gray background:

> 
>
>     #include <stdio.h>
>
>     int main(){
>
>         printf("Hello world!\n");
>
>     }

- Command-line commands are shown in a monospaced font on a gray background, prefixed with a \$ symbol to indicate they are entered in a terminal session:

> 
>
>     $ module load intel
>
>     $ icx hello.c -o hello

- Standard output is shown in Courier New font:

> 
>
>     $ ./hello
>
> Hello world!

#### A Note on the Figures

Some figures in this book—particularly in the early chapters—are intentionally hand-drawn. They originate from the first editions of the material (in 2016) and have been deliberately preserved.

This is not a limitation, nor an oversight. It is a conscious editorial choice and, in part, a homage to the origins of this book and to a way of teaching and thinking about supercomputing that has proven remarkably durable.

More than ten years later, these figures still convey the same ideas with clarity. That persistence is precisely why they remain.
