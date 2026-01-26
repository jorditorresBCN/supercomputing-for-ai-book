---
layout: default
title: SSH Public Key Authentication
parent: "Appendices"
nav_order: 802
has_children: false
---

### SSH Public Key Authentication

SSH public key authentication is a widely used and secure method for accessing remote systems without the need to type your password every time. This method is especially convenient when working frequently with remote HPC environments like MareNostrum 5.

This appendix outlines the steps to configure SSH public key authentication from a Linux or macOS system (or Windows using WSL).

#### Step 1 – Generate SSH Keys on Your Local Machine

On your local machine (e.g., your Linux laptop or WSL terminal), open a terminal and run the following command:


    ssh-keygen -t rsa

This command generates a pair of SSH keys:

- A private key (~/.ssh/id_rsa)

- A public key (~/.ssh/id_rsa.pub)

You can press Enter to accept the default path and filename. The system will also ask whether to use a passphrase for additional protection. For convenience, in this context we assume you skip it, so the authentication becomes fully passwordless.

IMPORTANT: Your private key must remain secure and never be shared. It should stay only on your local machine.

#### Step 2 – Copy the Public Key to MareNostrum 5

To upload your public key to MareNostrum 5, use the following command:


    ssh-copy-id <username>@glogin1.bsc.es

Replace \<username\> with your actual username on the cluster. You will be prompted to enter your password one last time. The command will add your public key to the appropriate location (~/.ssh/authorized_keys) on the remote system.

#### Step 3 – Connect Without a Password

Now that your public key is in place, you can log in to MareNostrum 5 without needing to type your password:


    ssh <username>@glogin1.bsc.es

If everything is correctly configured, the system will authenticate you automatically using your private key.

SSH key authentication is supported across all Linux and macOS systems. On Windows, this can be achieved either using:

- WSL (Windows Subsystem for Linux): allows you to use the standard Linux commands described above.

- PuTTY tools: for users who prefer native Windows solutions.

Due to the variety of Windows environments and tools available, detailed installation steps are beyond the scope of this book. We encourage interested users to consult the official documentation for their preferred setup.
