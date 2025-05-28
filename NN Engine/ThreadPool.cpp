#include "ThreadPool.h"

ThreadPool::ThreadPool(size_t numThreads) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back([this] { worker(); });
    }
}

ThreadPool::~ThreadPool() {
    stop = true;
    cv.notify_all();

    for (std::thread &worker : workers) {
        if(worker.joinable())
            worker.join();
    }
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        tasks.push(task);
    }
    cv.notify_one();
}

void ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(queueMutex);
    finishedCv.wait(lock, [this] { return tasks.empty() && workingThreads == 0; });
}

void ThreadPool::worker() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(queueMutex);

            cv.wait(lock, [this] { return stop || !tasks.empty(); });

            if (stop && tasks.empty())
                return;

            task = std::move(tasks.front());
            tasks.pop();

            ++workingThreads;
        }

        task();

        {
            std::unique_lock<std::mutex> lock(queueMutex);
            --workingThreads;
            if (tasks.empty() && workingThreads == 0)
                finishedCv.notify_one();
        }
    }
}
