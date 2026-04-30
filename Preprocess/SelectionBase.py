from abc import ABC, abstractmethod

import awkward as ak


class RegionSelection(ABC):
    """Base class for cumulative region selections."""

    selection_name = None
    cut_descriptions = ()

    @property
    def start_description(self):
        return f'start of {self.selection_name} selection'

    @property
    def end_description(self):
        return f'end of {self.selection_name} selection'

    def get_flags(self, events: ak.Array):
        cuts = self.get_cuts(events)
        if len(cuts) != len(self.cut_descriptions):
            raise ValueError(
                f'{self.__class__.__name__} defines {len(self.cut_descriptions)} descriptions '
                f'for {len(cuts)} cuts.'
            )

        flags = {
            self.start_description: self.ones_like_events(events),
        }
        pass_filter = flags[self.start_description]

        for description, cut in zip(self.cut_descriptions, cuts):
            pass_filter = ak.fill_none(cut & pass_filter, False)
            flags[description] = pass_filter

        flags[self.end_description] = pass_filter
        return flags

    @abstractmethod
    def get_cuts(self, events: ak.Array):
        raise NotImplementedError

    @staticmethod
    def ones_like_events(events: ak.Array):
        return ak.ones_like(events['evtNumber'], dtype=bool)
