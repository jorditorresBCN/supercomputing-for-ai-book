---
layout: default
title: Jupyter Notebook Basics
parent: "Appendices"
nav_order: 806
has_children: false
---

### Jupyter Notebook Basics

In various sections of this book, we use Jupyter Notebooks to explain important concepts interactively. This *appendix* offers a concise introduction for students unfamiliar with this tool. Jupyter Notebooks are widely used in both academia and industry, especially in machine learning and data science workflows. They allow researchers and engineers to combine code, explanations, and visualizations in the same document, making them ideal for teaching and experimentation.

#### What is a Jupyter Notebook?

A Jupyter Notebook is an open-source web application that lets you create and share documents containing:

- Live executable code (typically Python)

- Narrative text written in Markdown

- Mathematical equations using LaTeX

- Visualizations (e.g., plots and charts)

- Interactive widgets

Notebooks are composed of cells, which can be executed independently. This interactive design enables quick experimentation, immediate feedback, and step-by-step explanations—all of which are essential for iterative development and pedagogy.

Notebooks are stored as files with the .ipynb extension (*IPython Notebook*). These are plain-text JSON files that contain structured metadata, source code, outputs, and explanatory text. While they can be edited as raw text, they are best used within a Jupyter interface that provides a rich web-based experience.

We can use Jupyter Notebooks in different environments: Docker, Google Colab, and the MareNostrum 5 (MN5) supercomputer. Below, we describe how to launch Jupyter in each of these setups.

#### Launching Jupyter Notebooks in Docker

In Section 3.3, we use Docker to illustrate how to run a Jupyter Notebook server. We rely on the pre-built image jorditorresbcn/dl, which comes with many essential packages preinstalled, including:

- Python 3

- Jupyter Notebook

- NumPy, Pandas, Matplotlib

- TensorFlow, PyTorch

- Scikit-learn and other ML libraries

Once inside the Docker container, you can start the Jupyter server with the following command:


    jupyter notebook --ip=0.0.0.0 --allow-root

- jupyter notebook: starts the server.

- --ip=0.0.0.0: tells Jupyter to listen on all network interfaces (so it can be accessed externally).

- --allow-root: permits launching Jupyter as the root user (which is common in Docker containers).

By default, Jupyter listens on port 8888. If that port is busy, it will automatically increment until it finds a free one (8889, 8890, etc.).

Once the server is running, open your web browser and navigate to http://localhost:8888. You will either be redirected or asked for a token. The token is printed in the terminal and looks like this:


    http://localhost:8888/?token=abc123...

Paste that URL or token into your browser to access the Notebook Dashboard.

#### Using Google Colab

In Sections 7.6 and 14.2, we also use Jupyter Notebooks via Google Colab, a free cloud platform provided by Google that allows you to run Python code in your browser.

Google Colab (short for Colaboratory) provides:

- Cloud-based execution: No installation required; it works from any device with internet access.

- Free access to GPUs and TPUs: Useful for accelerating deep learning workloads.

- Seamless Google Drive integration: Notebooks are saved automatically and can be shared or synced.

- Real-time collaboration: Multiple users can edit the same notebook simultaneously.

- Preinstalled libraries: TensorFlow, PyTorch, NumPy, Pandas, Matplotlib, Scikit-learn, etc.

- Markdown and LaTeX support: For combining code with rich explanations and equations.

- AI-powered assistant: Google's Gemini AI can help explain code, debug, or suggest improvements.

Essentially, Google Colab is a cloud-hosted Jupyter environment tailored for fast prototyping, learning, and collaborative development—ideal for students and educators.

We suggest using several Jupyter notebooks (.ipynb files) throughout the book. A Jupyter notebook is an interactive document that combines live code, explanations, and visualizations in a single file. These notebooks are publicly hosted on the book’s GitHub repository. To open them in Google Colab, simply go to <https://colab.research.google.com>, click on the GitHub tab, and search for the corresponding repository to locate the desired .ipynb file.

The first time a notebook is introduced is in Section 7.6, where we explain in detail how to open and use it.

#### Running Jupyter Notebooks on MareNostrum 5 (MN5)

We can also use Jupyter Notebooks directly on MareNostrum 5, allowing us to run the same exercises as in Colab, but with access to HPC resources.

Here is the step-by-step guide to launch a Jupyter server on MN5:

***Step 1 – Connect to MN5 login node with X forwarding:***


    ssh -X <username>@alogin1.bsc.es

***Step 2 – Request an interactive job using SLURM:***


    salloc --account=<account>--qos=acc_debug --nodes=1 --ntasks=1 --cpus-per-task=4 --time=01:00:00

Once the job is assigned, you'll see:

salloc: Nodes as01r2b11 are ready for job

Take note of the compute node name (e.g., as01r2b11) — you'll need it later in step 5.

***Step 3 – Load Miniforge and activate the Jupyter environment:***


    module load miniforge

load MINIFORGE/24.3.0-0 (PATH)


    source activate jupyterhub-env

(jupyterhub-env) \$

***Step 4 – Launch the Jupyter Notebook server on a specific port:***


    (jupyterhub-env) $ jupyter notebook --no-browser --port=18888 --ip=0.0.0.0

You will see a URL with a token:

http://127.0.0.1:18888/tree?token=5d862a41...

Save the token.

***Step 5 – In a second terminal, create a port forwarding tunnel from your local machine:***


    ssh -L 18888:<node>:18888 <username>@alogin1.bsc.es

***Step 6 – Open your browser and go to:***


    http://localhost:18888

Enter the token (step 4) as the password when prompted. You now have access to Jupyter running directly on MN5. You can open or create notebooks, upload datasets, and run experiments on the HPC system.

***Step 7 – When finished:***

Stop the server with CTRL+C in the terminal where Jupyter is running amd close the browser tab.

This *appendix* provides a clear path to using Jupyter Notebooks effectively, whether on your local machine (via Docker), in the cloud (via Colab), or on a supercomputer (via MareNostrum 5).
