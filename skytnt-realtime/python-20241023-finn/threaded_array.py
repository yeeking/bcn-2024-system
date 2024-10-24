import threading

class FixedSizeArray:
    def __init__(self, size):
        self.size = size
        self.array = [None] * size
        self.index = 0
        self.lock = threading.Lock()

    def addEvent(self, event):
        with self.lock:
            # Add the event at the current index, using modulo to wrap around the array.
            self.array[self.index] = event
            # Increment the index, wrapping around using modulo with the array size.
            self.index = (self.index + 1) % self.size

    def getEvents(self):
        with self.lock:
            # Return a copy of the current array to avoid race conditions.
            return list(self.array)

# Example usage with multiple threads
if __name__ == "__main__":
    import time
    from threading import Thread

    # Create an instance of FixedSizeArray with size 5
    fixed_array = FixedSizeArray(10)

    def add_events_in_thread():
        for i in range(10):
            fixed_array.addEvent([i, i*2, i*3])  # Adding a list of events
            time.sleep(0.1)

    # Create multiple threads
    threads = [Thread(target=add_events_in_thread) for _ in range(100)]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Print the current events in the array
    print("Final Events in the Array:", fixed_array.getEvents())
