---
layout: default
title: Essential Linux Commands
parent: "Appendices"
nav_order: 801
has_children: false
---


### Essential Linux Commands

The following is a brief reference guide to essential Linux commands that will help you navigate the terminal, manage files, inspect data, and control processes in a typical HPC environment.

List files and directories in the current directory:


    ls 

Print the current working directory (i.e., your current location in the file system):


    pwd 

Change directory. Use it to navigate to a different directory:


    cd <directory_path> 

Create a new directory:


    mkdir <directory_name> 

Remove an empty directory:


    rmdir <directory_name> 

Remove files or directories. Be cautious as it is a permanent action:


    rm <file_name> 
    rm -r <directory_name> 
    # Use -r for recursive deletion of directories and their contents. 

Copy files or directories from one location to another:


    cp <source_file> <destination_path> 
    cp -r <source_directory> <destination_path> 
    # Use -r for recursive copying of directories and their contents. 

Move or rename files or directories:


    mv <source> <destination>

Display the beginning lines of a file:


    head <file_name> 

Display the last lines of a file:


    tail <file_name> 

Search for a pattern in a file or output:


    grep <pattern> <file_name> 

Access the manual pages for a command to get more information:


    man <command_name> 

Display information about active processes:


    ps aux 

Terminate a process using its PID (Process ID):


    kill <PID> 


