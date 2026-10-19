#include <iostream>
#include <chrono>
#include <vector>
#include <functional>
#include <atomic>
#include <mutex>
#include "ThreadPool.h"

// Global shared vector and mutex
std::vector<int> globalResults;
std::mutex globalMutex;

class ParallelTest {
public:
    ParallelTest(size_t numTasks, size_t workload)
        : numTasks(numTasks), workload(workload) {}

    // Simulated workload: returns an int result
    int task(size_t id) {
        volatile int dummy = 0;
        for (size_t i = 0; i < workload; ++i) {
            dummy += (i ^ id) % 7;  // arbitrary math to prevent optimization
        }
        return dummy;
    }

    // Sequential without global vector appending
    void runSequential() {
        auto start = std::chrono::high_resolution_clock::now();

        sequentialResult = 0;
        for (size_t i = 0; i < numTasks; ++i) {
            sequentialResult += task(i);
        }

        auto end = std::chrono::high_resolution_clock::now();
        sequentialDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    // Parallel with atomic result accumulation, no global vector
    void runParallelAtomic() {
        ThreadPool pool(std::thread::hardware_concurrency());
        std::atomic<size_t> completedTasks{0};
        std::atomic<int> parallelResult{0};

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < numTasks; ++i) {
            pool.enqueue([this, i, &completedTasks, &parallelResult] {
                int res = task(i);
                parallelResult.fetch_add(res, std::memory_order_relaxed);
                completedTasks.fetch_add(1, std::memory_order_relaxed);
            });
        }

        pool.wait();

        auto end = std::chrono::high_resolution_clock::now();
        parallelDurationAtomic = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        this->parallelResultAtomic = parallelResult.load(std::memory_order_relaxed);
    }

    // Parallel with global vector appending protected by mutex
    void runParallelMutex() {
        ThreadPool pool(std::thread::hardware_concurrency());
        std::atomic<size_t> completedTasks{0};

        // Clear global vector before starting
        {
            std::lock_guard<std::mutex> lock(globalMutex);
            globalResults.clear();
        }

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < numTasks; ++i) {
            pool.enqueue([this, i, &completedTasks] {
                int res = task(i);
                {
                    std::lock_guard<std::mutex> lock(globalMutex);
                    globalResults.push_back(res);
                }
                completedTasks.fetch_add(1, std::memory_order_relaxed);
            });
        }

        pool.wait();

        // Sum results from global vector
        int sum = 0;
        {
            std::lock_guard<std::mutex> lock(globalMutex);
            for (int val : globalResults) {
                sum += val;
            }
        }
        parallelResultMutex = sum;

        auto end = std::chrono::high_resolution_clock::now();
        parallelDurationMutex = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }

    void printResults() const {
        std::cout << "Sequential time: " << sequentialDuration << " ms\n";
        std::cout << "Parallel (atomic) time: " << parallelDurationAtomic << " ms\n";
        if (parallelDurationAtomic > 0)
            std::cout << "Speedup (atomic): " << static_cast<double>(sequentialDuration) / parallelDurationAtomic << "x\n";
        std::cout << "Parallel (mutex) time: " << parallelDurationMutex << " ms\n";
        if (parallelDurationMutex > 0)
            std::cout << "Speedup (mutex): " << static_cast<double>(sequentialDuration) / parallelDurationMutex << "x\n";

        std::cout << "Sequential result: " << sequentialResult << "\n";
        std::cout << "Parallel result (atomic): " << parallelResultAtomic << "\n";
        std::cout << "Parallel result (mutex): " << parallelResultMutex << "\n";

        if (sequentialResult == parallelResultAtomic)
            std::cout << "Atomic results match ✅\n";
        else
            std::cout << "Atomic results do NOT match ❌\n";

        if (sequentialResult == parallelResultMutex)
            std::cout << "Mutex results match ✅\n";
        else
            std::cout << "Mutex results do NOT match ❌\n";
    }

private:
    size_t numTasks;
    size_t workload;
    long long sequentialDuration = 0;
    long long parallelDurationAtomic = 0;
    long long parallelDurationMutex = 0;
    int sequentialResult = 0;
    int parallelResultAtomic = 0;
    int parallelResultMutex = 0;
};

int main() {
    ParallelTest test(1000, 100000);
    test.runSequential();
    test.runParallelAtomic();
    test.runParallelMutex();
    test.printResults();

    return 0;
}
