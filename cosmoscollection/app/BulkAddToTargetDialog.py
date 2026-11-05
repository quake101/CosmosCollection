"""Dialog for bulk-adding DSOs without images to the target list."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from cosmoscollection.core.DatabaseManager import DatabaseManager


# --- Bulk Add to Target List Dialog ---
class BulkAddToTargetDialog(QDialog):
    """Dialog for bulk-adding DSOs without images to the target list"""

    def __init__(self, dso_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bulk Add to Target List")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(700, 500)

        from cosmoscollection.catalog.DSONames import display_name
        # Each entry under the name it'll be added as, "M 31 (Andromeda Galaxy)"
        self.dso_list = sorted(
            ({**e, 'name': display_name(e.get('name', ''), designations=e.get('designations'))}
             for e in dso_list),
            key=lambda e: e.get('name', ''))
        # Build a lookup by name for retrieval during save
        self._dso_by_name = {e['name']: e for e in self.dso_list}
        self.db_manager = DatabaseManager()

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Info label
        count = len(self.dso_list)
        label = QLabel(f"{count} object{'s' if count != 1 else ''} without images are not yet in your target list.")
        layout.addWidget(label)

        # Table
        self.table = QTableWidget(count, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Priority", "Scope"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 110)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.setColumnWidth(2, 170)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)

        # Load telescope options once
        telescope_options = [("Any", None)]
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, name FROM usertelescopes WHERE is_active = 1 ORDER BY name"
                )
                for tid, tname in cursor.fetchall():
                    telescope_options.append((tname, tid))
        except Exception:
            pass

        for row, dso in enumerate(self.dso_list):
            # Name (read-only)
            name_item = QTableWidgetItem(dso.get('name', ''))
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)

            # Priority combo
            priority_combo = QComboBox()
            priority_combo.addItems(["Low", "Medium", "High", "Urgent"])
            priority_combo.setCurrentText("Medium")
            self.table.setCellWidget(row, 1, priority_combo)

            # Scope combo
            scope_combo = QComboBox()
            for label, tid in telescope_options:
                scope_combo.addItem(label, tid)
            self.table.setCellWidget(row, 2, scope_combo)

        layout.addWidget(self.table)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        add_btn = QPushButton(f"Add {count} Object{'s' if count != 1 else ''} to Target List")
        add_btn.setDefault(True)
        add_btn.clicked.connect(self._add_all)
        btn_layout.addWidget(add_btn)

        layout.addLayout(btn_layout)

    def _add_all(self):
        """Insert all rows into the target list database"""
        from datetime import datetime
        date_added = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        inserted = 0

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                for row in range(self.table.rowCount()):
                    name_item = self.table.item(row, 0)
                    if name_item is None:
                        continue
                    name = name_item.text()
                    dso = self._dso_by_name.get(name, {})

                    priority = self.table.cellWidget(row, 1).currentText()
                    telescope_id = self.table.cellWidget(row, 2).currentData()

                    size_min = dso.get('size_min', 0) or 0
                    size_max = dso.get('size_max', 0) or 0
                    size_info = f"{size_min:.1f} x {size_max:.1f}" if (size_min or size_max) else ""

                    cursor.execute("""
                        INSERT INTO usertargetlist (
                            name, dso_type, constellation, ra_deg, dec_deg,
                            magnitude, size_info, priority, status, date_added, telescope_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Not Observed', ?, ?)
                    """, (
                        name,
                        dso.get('dso_type', ''),
                        dso.get('constellation', ''),
                        dso.get('ra_deg', 0),
                        dso.get('dec_deg', 0),
                        dso.get('magnitude', 0),
                        size_info,
                        priority,
                        date_added,
                        telescope_id,
                    ))
                    inserted += 1
                conn.commit()

            QMessageBox.information(
                self, "Success",
                f"{inserted} object{'s' if inserted != 1 else ''} added to your target list."
            )
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add objects: {str(e)}")
