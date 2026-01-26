---
layout: default
title: C Language Basics
parent: "Appendices"
nav_order: 803
has_children: false
---

### C Language Basics

In this course, we assume that students already have some basic knowledge of the C programming language. However, since many of the practical tasks throughout the book involve writing or analyzing C code, this section offers a compact review of fundamental features that are particularly relevant for scientific computing and high performance programming.

#### Pointers and Addresses

One of the defining features of C is its ability to manipulate memory directly through pointers. A pointer is a variable that stores the memory address of another variable. The \* symbol is used to declare a pointer, while the & symbol is the address-of operator, which returns the memory address of a variable.


    int x = 100;
    int *ptr = &x;
    int value = *ptr;

In this example, ptr stores the address of x, and \*ptr dereferences the pointer to access the value stored at that address. This concept is central to many high performance programming techniques, especially when dealing with dynamic memory and arrays.

#### One-Dimensional Arrays to Simulate Two-Dimensional Arrays

C stores arrays in linear memory using row-major order, meaning that rows are laid out consecutively in memory. For matrix operations, it's common to simulate two-dimensional arrays using one-dimensional arrays and manual index calculations:


    M[i * ncols + j]

This indexing scheme gives full control over memory layout and is essential when implementing optimized matrix operations or interfacing with libraries and accelerators.

The following example demonstrates the use of this technique in a basic matrix multiplication program. Two 4×4 matrices are initialized and multiplied using nested loops, and the result is printed to the screen.


    #include <sys/time.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <assert.h>

    #define N 4
    double h_a[N][N], h_b[N][N], h_c[N][N];

    void cleanMatrices(int nrows, int ncols, double *M) {
        for (int i = 0; i < nrows; i++)
            for (int j = 0; j < ncols; j++)
                M[i * ncols + j] = 0;
    }

    void iniMatrices(int nrows, int ncols, double *M) {
        for (int i = 0; i < nrows; i++)
            for (int j = 0; j < ncols; j++)
                M[i * ncols + j] = 1;
    }

    void printMatrix(int nrows, int ncols, double *M) {
        printf("\n******** PRINT MATRIX ********\n");
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++)
                printf("| %f |", M[i * ncols + j]);
            printf("\n");
        }
        printf("******** END ********\n");
    }

    int main() {
        iniMatrices(N, N, (double *) h_a);
        iniMatrices(N, N, (double *) h_b);
        cleanMatrices(N, N, (double *) h_c);

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    h_c[i][j] += h_a[i][k] * h_b[k][j];

        printMatrix(N, N, (double *) h_c);
        return 0;
    }

This code demonstrates how to manipulate matrices in memory-efficient ways and shows how pointer casting can be used to treat statically declared arrays as linear data blocks.

#### **Static vs Dynamic Arrays**

For fixed-size problems, arrays can be declared statically, e.g., double A\[100\]. However, when the size depends on input or runtime conditions, you must allocate memory dynamically using malloc() and release it using free():


    double *matrix = (double *) malloc(N * N * sizeof(double));
    free(matrix);

#### **Passing Arrays to Functions**

In C, arrays are passed to functions by reference, meaning that changes inside the function affect the original data. This behavior is efficient and widely used in matrix and buffer operations.

#### Operators and Control Structures

C provides standard arithmetic and relational operators such as +, -, \*, /, ==, !=, \<, \>, and logical operators like &&, \|\|, and !. These are commonly used in if, while, and for statements.

#### **Structures**

A struct groups variables under one name and can contain multiple types. This is useful for storing complex data like matrix sizes, coordinates, or configurations.


    struct Point {
        int x;
        int y;
    };

#### **Assertions**

The assert() macro is used to verify assumptions during debugging. If the condition is false, the program stops and prints an error:


    assert(N > 0);

#### **Comments**

Use // for single-line comments and /\* \*/ for multi-line comments to make code more readable and maintainable.


    // Single-line comment

    /* Multi-line
       comment block */

###  
