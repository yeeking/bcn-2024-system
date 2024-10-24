from gen_utils import RingBuffer



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

    buff = RingBuffer(6)
    want_n = 3
    for i in range(6): buff.addEvent(i)
    assert buff.index == 0, f"For this test, I want the index to be zero but its {buff.index}"
    
    res = buff.getLatestItems(want_n)
    print(buff.array)
    print(res)
    assert len(res) == want_n, f"Length wrong - should be {want_n} but is {len(res)} : {res}"

