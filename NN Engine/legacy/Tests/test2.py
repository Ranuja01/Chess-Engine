import multiprocessing
import random
import time
import os

# Constants
DATA_THRESHOLD = 10  # Define a threshold for the data size

# Data collection function
def collect_data(task_id, shared_data, data_lock, data_ready_event, processing_done_event):
    print("DDD")
    while True:
        # Simulate data collection
        new_data = random.randint(1, 100)

        with data_lock:
            shared_data.append(new_data)
            print(f"Task {task_id} collected data: {new_data} | Current size: {len(shared_data)}")

            # Check if data size has reached the threshold
            if len(shared_data) >= DATA_THRESHOLD:
                print(f"Task {task_id} pausing for data processing...")
                data_ready_event.set()  # Signal that data is ready
                data_lock.release()  # Release the lock before waiting
                processing_done_event.wait()  # Wait for the processing to be done
                data_lock.acquire()  # Reacquire the lock after processing
                processing_done_event.clear()  # Clear the processing done event

        time.sleep(random.random())  # Simulate varying data collection times

def process_data(shared_data, data_lock):
    print(f"Processing data in process ID: {os.getpid()}")
    with data_lock:
        for i in range(len(shared_data)):
            shared_data[i] *= 2  # Example processing: doubling the numbers
    print("Processed data:", shared_data)

if __name__ == "__main__":
    print(f"Main process ID: {os.getpid()}")

    # Use the manager within the main guard
    with multiprocessing.Manager() as manager:
        # Initialize shared data and lock
        shared_data = manager.list()  # Shared list for data collection
        data_lock = manager.Lock()  # Lock for synchronizing access
        data_ready_event = manager.Event()  # Event to signal data readiness
        processing_done_event = manager.Event()  # Event to signal processing completion
        print("AAA")
        # Create multiple data collection processes
        num_tasks = 3  # Define number of data collection tasks
        collection_processes = []
        
        
        # Create a data processing process
        processing_process = multiprocessing.Process(
            target=process_data, 
            args=(shared_data, data_lock)
        )
        processing_process.start()

        

        # Terminate the processing process when all collection is done
        processing_process.join()
        

        #for task_id in range(num_tasks):
        print("BBB")
        process = multiprocessing.Process(
            target=collect_data, 
            args=(0, shared_data, data_lock, data_ready_event, processing_done_event)
        )
        collection_processes.append(process)
        process.start()
        
        # Wait for all collection processes to finish (infinite loop in this case)
        #for process in collection_processes:
        print("CCC")
        process.join()
        
        

    print("Main process continues after all tasks are complete.")
