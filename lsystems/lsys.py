

from typing import Optional


__all__ = ['LSystem']




class LSystem:
    '''
    Lindenmeyer system.  All matching patterns within the current string are iteratively replaced by substrings according to some rule and some initial "seed" string.
    
    Conventionally, capital letters from "A" to "Z" along with a couple special characters like "-" populate the strings, but the characters can be anything unicode.
    
    Multi-character pattern replacements are generally not defined for L-systems, so this class is not designed to handle those.
    '''
    def __init__(self, seed: str = 'A', patterns: Optional[dict[str, str]] = None):
        if patterns is None:
            patterns = {}
        self.seed = seed
        self.patterns = patterns
        self.reset()
    def reset(self):
        '''
        Reset the current string to the `LSystem.seed` string.
        '''
        self.current_str = self.seed
    def step(self, until_length: Optional[int] = None) -> str:
        '''
        Perform a single iteration of the substring replacements, according to the currently defined `LSystem.patterns`.
        
        The patterns are not required to remain unchanged across iterations, although L-systems are typically defined to have constant patterns ("rules" for a dynamical system).

        Return the newly modified string.
        '''
        if until_length is None:
            self.current_str = ''.join(self.patterns[c] if c in self.patterns else c for c in self.current_str)
            return self.current_str
        if not isinstance(until_length, int):
            raise TypeError(f'until_length is not an int, received type: {type(until_length)}')
        while len(self.current_str) < until_length:
            self.step()
        return self.current_str
    def __iter__(self):
        return self
    def __next__(self):
        if len(self.current_str) > 1024:
            raise StopIteration
        return self.step()
    def __str__(self) -> str:
        return self.current_str
    def __len__(self) -> int:
        return len(self.current_str)


if __name__ == '__main__':
    sys = LSystem(seed='A', patterns={
        'A': 'CA',
        'B': 'BC',
        'C': 'ACB-',
    })
    for cur in sys:
        print(cur)


