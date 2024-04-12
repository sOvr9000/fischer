
from dataclasses import dataclass
# from enum import IntEnum



@dataclass
class StoragePrototype:
    def __init__(self, capacity: int, max_load: int, max_unload: int):
        self.capacity = capacity
        self.max_load = max_load
        self.max_unload = max_unload



class Storage:
    def __init__(self, prot: StoragePrototype):
        self.prot = prot
        self.contents: dict[int, int] = {}
        self.load_queue = list[tuple[int, int]] = []
    def get_load_queue_total_amount(self) -> int:
        # if len(self.load_queue) == 0:
        #     return 0
        return sum(v for _, v in self.load_queue)
    def get_contents_total_amount(self) -> int:
        return sum(v for _, v in self.contents.item())
    def load_step(self):
        queue_total = self.get_load_queue_total_amount()
        if queue_total == 0:
            return
        contents_total = self.get_contents_total_amount()
        if contents_total + queue_total > self.prot.capacity:
            partial = self.prot.capacity - contents_total
            
        else:
            for k, v in self.load_queue:
                if k not in self.contents:
                    self.contents = v
                else:
                    self.contents[k] += v



class Building:
    def __init__(self):
        self.storage = Storage()



class AbstractFactoryGame:
    def __init__(self):
        self.buildings = []


