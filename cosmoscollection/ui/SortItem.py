#!/usr/bin/env python3
"""A table cell that sorts by a value rather than its text: '10.5 h' after
'2.0 h', 01:00 after 23:00, a blank magnitude after the known ones, 'Urgent'
above 'Low'. Optionally within a group: groups keep their order whichever way
a column is sorted (the Session Manager's open sessions above finished ones).

    table.setItem(row, col, SortItem(format_magnitude(mag), mag if mag is not None else 99))
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidgetItem


class SortItem(QTableWidgetItem):
    SORT_ROLE = Qt.UserRole + 1
    GROUP_ROLE = Qt.UserRole + 2

    def __init__(self, text="", sort_key=None, align=Qt.AlignCenter, group=None):
        """sort_key: what the cell sorts by (None: its text); align: None leaves
        the default alignment."""
        super().__init__(text)
        if sort_key is not None:
            self.setData(self.SORT_ROLE, sort_key)
        if group is not None:
            self.setData(self.GROUP_ROLE, group)
        if align is not None:
            self.setTextAlignment(align)

    def sort_key(self):
        return self.data(self.SORT_ROLE)

    def __lt__(self, other):
        group, other_group = self.data(self.GROUP_ROLE), other.data(self.GROUP_ROLE)
        if group is not None and other_group is not None and group != other_group:
            table = self.tableWidget()
            # Qt sorts descending by asking other < self - turn that around to keep groups in place
            if table is not None and table.horizontalHeader().sortIndicatorOrder() == Qt.DescendingOrder:
                return group > other_group
            return group < other_group
        key, other_key = self.data(self.SORT_ROLE), other.data(self.SORT_ROLE)
        if key is not None and other_key is not None:
            try:
                return key < other_key
            except TypeError:
                pass
        # Not super().__lt__(): PySide sends that back to this override
        return self.text() < other.text()
