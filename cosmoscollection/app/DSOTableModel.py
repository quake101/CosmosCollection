"""Table model behind the main window's DSO list."""

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QBrush, QColor

from cosmoscollection.ui.Theme import COLORS, theme_manager

from cosmoscollection.app.DSOData import SEARCH_RANK_EXACT, DSOSearch
from cosmoscollection.core.Formatting import format_dec, format_ra, natural_key


# --- Model for displaying DSO data in table ---
class DSOTableModel(QAbstractTableModel):
    def __init__(self, dso_data, parent=None):
        super().__init__(parent)
        self.dso_data = dso_data
        self.filtered_data = list(dso_data)
        self.headers = ["Catalog", "Designation", "RA (hms)", "Dec (dms)", "Images"]
        self.selected_catalog = None
        self.highlight_no_images = False
        # (search text, catalog, images only, type code, no images only)
        self._filters = ("", None, False, None, False)
        self._sort_column = None
        self._sort_order = Qt.AscendingOrder
        self._cached_formatted_data = {}  # (row, col) -> display text
        self._search = DSOSearch()  # Filters and search ranking, shared with the command line
        # Row colors come from the theme (alternating rows on the view); only the
        # "no images" highlight is painted here, so rebuild it when the theme changes.
        self._brush_no_images = None
        self._update_theme_brushes()
        theme_manager().theme_changed.connect(self._update_theme_brushes)

    def _update_theme_brushes(self):
        color = QColor(COLORS['error'])
        color.setAlpha(110)
        self._brush_no_images = QBrush(color)
        if self.highlight_no_images and self.rowCount() > 0:
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1),
                                  [Qt.ItemDataRole.BackgroundRole])

    def rowCount(self, index=QModelIndex()):
        return len(self.filtered_data)

    def columnCount(self, index=QModelIndex()):
        return 5

    def data(self, index, role):
        if not index.isValid():
            return None
        row = index.row()
        entry = self.filtered_data[row]

        if role == Qt.ItemDataRole.BackgroundRole:
            if self.highlight_no_images and entry["image_count"] == 0:
                return self._brush_no_images
            return None
        elif role == Qt.ItemDataRole.DisplayRole:
            cache_key = (row, index.column())
            result = self._cached_formatted_data.get(cache_key)
            if result is None:
                result = self._format_cell_data(entry, index.column())
                self._cached_formatted_data[cache_key] = result
            return result
        return None

    def _display_designation(self, entry):
        """(catalog, designation) shown for an entry: the search match if any, else
        its designation in the selected catalog, else its primary designation"""
        return self._search.display_designation(entry, self.selected_catalog)

    def _format_cell_data(self, entry, col):
        """Format cell data for display"""
        if col in (0, 1):
            return self._display_designation(entry)[col]
        elif col == 2:
            return format_ra(entry["ra_deg"])
        elif col == 3:
            return format_dec(entry["dec_deg"])
        elif col == 4:
            return str(entry["image_count"])
        return None

    def headerData(self, index, orientation, role):
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        return self.headers[index]

    def sort(self, column, order=Qt.AscendingOrder):
        """Sort the filtered rows by the specified column, keeping selections on the same objects"""
        self._sort_column = column
        self._sort_order = order

        self.layoutAboutToBeChanged.emit()
        old_indexes = self.persistentIndexList()
        old_ids = [self.filtered_data[index.row()]["dsodetailid"] for index in old_indexes]

        self._sort_filtered_data()
        self._cached_formatted_data.clear()

        new_rows = {entry["dsodetailid"]: row for row, entry in enumerate(self.filtered_data)}
        self.changePersistentIndexList(
            old_indexes,
            [self.index(new_rows[dso_id], index.column()) for dso_id, index in zip(old_ids, old_indexes)]
        )
        self.layoutChanged.emit()

    def _sort_filtered_data(self):
        """Apply the current column sort to filtered_data, best search matches first"""
        column = self._sort_column
        if column in (0, 1):
            # Sort on what's displayed, so a catalog filter sorts by that catalog's numbers
            def key(entry):
                catalog, designation = self._display_designation(entry)
                keys = (natural_key(catalog), natural_key(designation))
                return keys if column == 0 else keys[::-1]
        elif column in (2, 3):
            field = "ra_deg" if column == 2 else "dec_deg"

            def key(entry):
                value = entry[field]
                return value if value is not None else float("-inf")
        elif column == 4:
            def key(entry):
                return entry["image_count"]
        else:
            key = None

        if key is not None:
            self.filtered_data.sort(key=key, reverse=(self._sort_order == Qt.DescendingOrder))

        # Stable sort keeps the column order within each search rank
        if self._filters[0].strip():
            self.filtered_data.sort(key=lambda entry: entry.get("search_rank", SEARCH_RANK_EXACT))

    def filter_data(self, search_text, selected_catalog=None, show_images_only=False, selected_type=None, show_no_images_only=False):
        """Filter the data based on search text, catalog, image presence, and DSO type"""
        catalog = selected_catalog if selected_catalog and selected_catalog != "All Catalogs" else None
        self.selected_catalog = catalog
        self._filters = (search_text or "", catalog, show_images_only, selected_type, show_no_images_only)
        self._apply_filters()

    def set_data(self, dso_data):
        """Replace all rows, e.g. after objects were added to the database"""
        self.dso_data = dso_data
        self._apply_filters()

    def refresh_image_counts(self, counts):
        """Update image counts from {str(dsodetailid): count} and re-apply filters"""
        for entry in self.dso_data:
            entry["image_count"] = counts.get(str(entry["dsodetailid"]), 0)
        self._apply_filters()

    def _apply_filters(self):
        self.beginResetModel()
        self.filtered_data = self._build_filtered_data(*self._filters)
        self._sort_filtered_data()
        self._cached_formatted_data.clear()
        self.endResetModel()

    def _build_filtered_data(self, search_text, selected_catalog=None, show_images_only=False,
                             selected_type=None, show_no_images_only=False):
        """Apply all filters to the data, tagging search matches with their rank"""
        return self._search.filter(self.dso_data, search_text, selected_catalog, show_images_only,
                                   selected_type, show_no_images_only)

    def _search_keys(self, designations):
        """Return [(designation, normalized designation)], cached by designations string"""
        return self._search.search_keys(designations)

    def has_exact_search_match(self):
        """Whether the current search found an exact designation match"""
        return any(item.get("search_rank") == SEARCH_RANK_EXACT for item in self.filtered_data)

    def setHighlightNoImages(self, highlight):
        """Set whether to highlight objects without images"""
        self.highlight_no_images = highlight
        if self.rowCount() > 0:
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1),
                                  [Qt.ItemDataRole.BackgroundRole])
