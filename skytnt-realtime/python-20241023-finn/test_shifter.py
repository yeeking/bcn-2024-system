from gen_utils import ImproviserAgent

if __name__ == "__main__":
    data = [["note", i, 0, 0] for i in range(20)]
    tracks = ImproviserAgent.shift_events_to_multitrack(data)
    