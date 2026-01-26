---
layout: default
title: Taking Timing
parent: "Appendices"
nav_order: 804
has_children: false
---

### Taking Timing

In order to obtain performance metrics, we need to measure the execution time of the program under analysis. There are different approaches depending on the type of platform, as well as on the specific programming models used.

#### What Time?

One important thing to note is that we are usually not only interested in the total time elapsed between the program’s start and finish, but also in the execution time of specific parts of the program. For this reason, we need something more sophisticated than the Unix time command, which simply reports the total time taken to run a program from start to finish.

Another important point is that we are usually not interested in CPU time, because it does not include idle periods. For instance, when an MPI process calls a receive function, it may have to wait for the corresponding send to be issued. This idle time is not accounted for as CPU time, but it must be included as part of the total parallel execution time when computing performance metrics.

For this reason, we usually consider the wall clock time, which is the elapsed real time between the start and end of the code section under analysis. The timing function used will depend on the programming model. For example MPI provides MPI_Wtime() and OpenMP provides omp_get_wtime().

#### The POSIX gettimeofday Function

To measure execution time in serial programs, as well as in MPI or OpenMP programs, a very useful POSIX function is gettimeofday(), which returns the number of microseconds that have elapsed since a fixed point in time (as seen in an earlier section of this course). Being a standard POSIX function, it is portable and works on any Linux/Unix system without depending on MPI, OpenMP, or any high-level libraries. This makes it especially useful for measuring the execution time of code that combines multiple programming models (e.g., MPI + OpenMP + serial). It also offers high resolution, returning time in microseconds.

The gettimeofday() function returns the number of microseconds elapsed since some point in the past, and is defined as:


    #include <sys/time.h>
    int gettimeofday(struct timeval * tv, struct timezone * tz);

The tv argument is a pointer to a struct timeval, defined as:


    struct timeval {
        time_t tv_sec;     /* seconds */
        suseconds_t tv_usec; /* microseconds */
    };

The tz argument is a pointer to a struct timezone, defined as:


    struct timezone {
        int tz_minuteswest; /* minutes west of Greenwich */
        int tz_dsttime;     /* type of DST correction */
    };

However, the use of the timezone structure is obsolete; the tz argument should normally be set to NULL.

To simplify its use—since the details of the syntax are not critical for the goals of this course—we will use the following code pattern to time a code section:


    #include <sys/time.h>
    struct timeval start_time, end_time;

    gettimeofday(&start_time, NULL);

    // << code section to measure >>

    gettimeofday(&end_time, NULL);
    print_times();
    }

    print_times()
    {
        int total_usecs;
        float total_time;
        total_usecs = (end_time.tv_sec - start_time.tv_sec) * 1000000 +
                      (end_time.tv_usec - start_time.tv_usec);
        printf(" %.2f mSec \n", ((float) total_usecs) / 1000.0);
        total_time = ((float) total_usecs) / 1000000.0;
    }

An important issue related to timing is the resolution of the timer function—that is, the unit of measurement. In this case, we consider microsecond resolution (10⁻⁶ seconds).

#### Taking Time at MareNostrum 5

To test the code, you can use the following program, available on the GitHub repository:


    /* timesample.c */
    #include <stdio.h>
    #include <sys/time.h>
    #include <stdlib.h>

    #define SIZE 1000
    typedef double matrix[SIZE][SIZE];
    matrix m1;

    struct timeval start_time, end_time;

    static void foo(void) {
        int i, j;
        for (i = 0; i < SIZE; ++i)
            for (j = 0; j < SIZE; ++j)
                m1[i][j] = 1.0;
    }

    void print_times() {
        int total_usecs;
        float total_time;
        total_usecs = (end_time.tv_sec - start_time.tv_sec) * 1000000 +
                      (end_time.tv_usec - start_time.tv_usec);
        printf(" %.2f mSec \n", ((float) total_usecs) / 1000.0);
        total_time = ((float) total_usecs) / 1000000.0;
    }


    int main() {
        int i;

        gettimeofday(&start_time, NULL);

        for (i = 0; i < 10; ++i) {
            foo();
        }

        gettimeofday(&end_time, NULL);
        print_times();

        return 0;
    }

Example:


    $ gcc -O3 -o timesample timesample.c
    $ ./timesample

3.81 mSec

#### 

#### Variability in Timing

We must be aware of timing variability. When running a program several times, we may observe substantial differences in the measured times—even if we use the same input, the same number of processes, and the same system configuration for each run.

This variability is due to interactions between our program and the rest of the system, particularly the software stack, as discussed in previous chapters. Since these interactions will almost never make the program run faster, it is common practice to report the minimum execution time rather than the mean or median.

In parallel programs, each process may have its own timing. Ideally, all processes would start simultaneously, and we would measure the time until the last process finishes. However, in practice, we cannot guarantee perfect synchronization.

In some programming models, such as MPI, we can approximate this using a collective communication function like MPI_Barrier, which ensures that no process proceeds until all processes have reached the barrier. After the barrier, each process can measure its execution time. At the end, we can collect all timings using MPI_Reduce with the MPI_MAX operator to obtain the maximum (i.e., the effective total parallel time). For simpler performance analysis, it is often sufficient to time only the rank 0 process.

###  
