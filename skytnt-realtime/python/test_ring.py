# from gen_utils import RingBuffer
import threading

class RingBuffer:
    """
    circular / ring buffer
    """
    def __init__(self, size:int):
        self.size = size
        self.lock = threading.Lock()
        self.reset()

    def addEvent(self, event):
        with self.lock:
            # print(f"buffer got event {event}")
            self.array[self.index] = event
            self.index = (self.index + 1) % self.size

    def getEvents(self):
        with self.lock:
            return list(self.array)

    def getItemsInTimeFrame(self, max_age:int, age_ind:int):
        """
        returns from newest to up to 'max_age'ms old
        since the ringbuffer is generic and does not care what kind of data
        you put into it, this function needs to be told which idnex in the data
        in the buffer is the time value. it can then use that to filter anything 
        with age >  newest_item_age - max_age  (since higher 'age' means more recent as its longer since the start time)
        """
        with self.lock:
            items = []
            newest = self.array[self.index-1][age_ind] # index-1 as index always points at next write slot 
            oldest = newest - max_age
            # assert oldest > 0, f"Error: oldest age is less than zero"
            # print(f"newest is {newest} so oldest is {oldest}")
            for i,item in enumerate(self.array):
                if (item is not None) and (item[age_ind] > oldest):
                    items.append(item)
            return items 

    def getLatestItems(self, want_n):
        """
        returns items from last stored backwards by 'want_n' steps
        want_n is capped at len(self.array) so it does not repeat items if 
        you ask for too many 
        """
        with self.lock:
            items = []
            if len(self.array) < want_n: want_n = len(self.array)
            sub_ind = self.index - 1# index is always pointing at next memory write slot
            if sub_ind == -1: sub_ind = len(self.array) - 1 # edge case where index == 0
            for i in range(0, want_n):
                items.append(self.array[sub_ind])
                sub_ind = sub_ind - 1
                if sub_ind < 0: sub_ind = len(self.array)-1
            return list(items)

    def isFull(self):
        """
        return true if the index is pointing to the last position
        """
        if self.index == len(self.array) - 1:
            return True
        else:
            return False
        
    def reset(self):
        with self.lock:
            self.index = 0
            self.array = [None] * self.size



def test_timeframe():

    buff = RingBuffer(10)
    buff.addEvent(["note", 3000, 1])
    buff.addEvent(["note", 6000, 2])
    buff.addEvent(["note", 7000, 3])
    # now try pull some events out based on time frames
    events = buff.getItemsInTimeFrame(2000, 1) # should notes 2 and 3
    assert len(events) == 2, f"expected 2 events but got {len(events)}"
    assert events[0][1] == 6000, f"expected first event to be at time {6000}"
    assert events[1][1] == 7000, f"expected second event to be at time {7000}"
    
    

if __name__ == "__main__":
    # buff = RingBuffer(100)
    # for i in range(10): buff.addEvent(i)
    # print(buff.array)
    # res = buff.getLatestItems(5)
    # print("result", res)
    # assert len(res) == 5, f"Length wrong - should be 5 but is {len(res)} : {res}"
    # assert res[0] is not None, f"Got a none at 0-  : {res}"

    # buff = RingBuffer(20)
    # want_n = 10
    # for i in range(10): buff.addEvent(i)
    # print(buff.array)
    # res = buff.getLatestItems(want_n)
    # print("result", res)
    # assert len(res) == want_n, f"Length wrong - should be {want_n} but is {len(res)} : {res}"

    # buff = RingBuffer(6)
    # want_n = 10
    # for i in range(10): buff.addEvent(i)
    # print(buff.array)
    # res = buff.getLatestItems(want_n)
    # print("result", res)
    # assert len(res) == len(buff.array), f"Length wrong - should be {len(buff.array)} but is {len(res)} : {res}"

    # buff = RingBuffer(6)
    # want_n = 3
    # for i in range(6): buff.addEvent(i)
    # assert buff.index == 0, f"For this test, I want the index to be zero but its {buff.index}"
    
    # res = buff.getLatestItems(want_n)
    # print(buff.array)
    # print(res)
    # assert len(res) == want_n, f"Length wrong - should be {want_n} but is {len(res)} : {res}"


    buff = RingBuffer(10)
    for i in range(1000): buff.addEvent(["note", i*1000, 1])
   
    events = buff.getItemsInTimeFrame(2000, 1) # should notes 2 and 3

    