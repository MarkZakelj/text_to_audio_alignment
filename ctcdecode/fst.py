import openfst_python as openfst
from numba import jit


@jit(nopython=True, fastmath=True)
def binary_search(x, value):
    if not len(x):
        return 0

    low = 0
    high = len(x)
    while low < high:
        mid = (low + high) >> 1
        if x[mid] < value:
            low = mid + 1
        else:
            high = mid
    return low


class FST:
    TROPICAL_WEIGHT_ONE = openfst.Weight.One('tropical')

    def __init__(self, fst_path, fst=None):

        if not fst:
            fst = openfst.Fst.read(fst_path)

        self.fst = fst

        self._state = openfst.NO_STATE_ID
        self._eps_next_state = self._state = openfst.NO_STATE_ID

    @classmethod
    def from_vocab(cls, vocab, tokenizer):
        fst = openfst.Fst()

        def add_word(word):
            i_words = tokenizer.token2idx(word) + [tokenizer.space_idx]
            if not fst.num_states():
                initial_state = fst.add_state()
                assert initial_state == 0
                fst.set_start(initial_state)

            source_state = fst.start()
            dest_state = None
            for i in i_words:
                # The initial state of FST is state 0, hence the index of chars in
                # the FST should start from 1 to avoid the conflict with the initial
                # state, otherwise wrong decoding results would be given.
                i += 1

                dest_state = fst.add_state()
                fst.add_arc(source_state, openfst.Arc(i, i, 0, dest_state))
                source_state = dest_state

            fst.set_final(dest_state, openfst.Weight.One('tropical'))

        lexicon_size = 0
        for word in vocab:
            add_word(word)
            lexicon_size += 1

        # This gets rid of "epsilon" transitions in the FST.
        # These are transitions that don't require a string input to be taken.
        # Getting rid of them is necessary to make the FST determinisitc, but
        # can greatly increase the size of the FST
        fst.rmepsilon()

        # This makes the FST deterministic, meaning for any string input there's
        # only one possible state the FST could be in.  It is assumed our
        # dictionary is deterministic when using it.
        # (lest we'd have to check for multiple transitions at each state)
        fst = openfst.determinize(fst)

        # Finds the simplest equivalent fst. This is unnecessary but decreases
        # memory usage of the dictionary
        fst.minimize()

        return cls(fst_path=None, fst=fst)

    def set_state(self, state):
        if self._state == state:
            return

        self._arc_iter = self.fst.arcs(state)
        self._state = state

        self._eps_next_state = state

    def find(self, label):
        # openfst index 0 is reserved for eps
        label = label + 1

        label = 0 if label == openfst.NO_LABEL else label

        self._eps_loop = label == 0

        if self.linear_search(label):
            return True

        return self._eps_loop

    def linear_search(self, label):
        self._arc_iter.reset()
        labels = [arc.ilabel for arc in self._arc_iter]
        if label in labels:
            self._arc_iter.seek(labels.index(label))
            return True
        return False

    def binary_search(self, label):
        'Locate the leftmost value exactly equal to x'
        self._arc_iter.reset()
        labels = [arc.ilabel for arc in self._arc_iter]

        i = 0
        if len(labels):  # avoiding numba ValueError: cannot compute fingerprint of empty list
            i = binary_search(labels, label)

        if i != len(labels):
            self._arc_iter.seek(i)
            if self._arc_iter.value().ilabel == label:
                return True

        return False

    def is_final(self, state):
        return self.fst.final(state) == self.TROPICAL_WEIGHT_ONE

    @property
    def initial_state(self):
        return self.fst.start()

    @property
    def state(self):
        return self._state

    @property
    def next_state(self):
        if self._eps_loop:
            return self._eps_next_state

        return self._arc_iter.value().nextstate

    def save(self, output_path):
        self.fst.write(output_path)
