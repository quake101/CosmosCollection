#!/usr/bin/env python3
"""
DSO Target List Manager
Allows users to manage their observing target list for deep sky objects
"""

import sys
import os
import calendar
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLabel, QTableWidget,
                               QTableWidgetItem, QGroupBox, QMessageBox,
                               QHeaderView, QTextEdit, QDialog, QComboBox,
                               QLineEdit, QCheckBox, QDateEdit, QSpinBox, QMenu)
from PySide6.QtGui import QFont

from DatabaseManager import DatabaseManager
from BestDSOTonight import BestDSOTonightWindow
from WindowPositionManager import WindowPositionMixin
from Theme import COLORS
from NINAIntegration import NINAIntegration
import logging

# Set up logging
logger = logging.getLogger(__name__)


class PriorityTableWidgetItem(QTableWidgetItem):
    """Custom QTableWidgetItem that sorts priorities correctly"""

    PRIORITY_ORDER = {"Urgent": 4, "High": 3, "Medium": 2, "Low": 1}

    def __init__(self, priority_text):
        super().__init__(priority_text)
        self.priority_value = self.PRIORITY_ORDER.get(priority_text, 0)
        self.setTextAlignment(Qt.AlignCenter)

    def __lt__(self, other):
        """Override less-than operator for proper sorting"""
        if isinstance(other, PriorityTableWidgetItem):
            return self.priority_value < other.priority_value
        return super().__lt__(other)


class AddTargetDialog(QDialog):
    """Dialog for adding a new target to the list"""
    
    def __init__(self, dso_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Target to List")
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.resize(500, 400)
        
        self.dso_data = dso_data
        self.db_manager = DatabaseManager()
        self.is_edit_mode = False  # Track if we're editing an existing target
        self.target_id = None  # Store the ID of the target being edited
        self._setup_ui()
        
        # Pre-fill with DSO data if provided
        if self.dso_data:
            self._populate_from_dso_data()
    
    def _setup_ui(self):
        """Set up the dialog UI"""
        layout = QVBoxLayout()
        
        # DSO Information Group
        dso_group = QGroupBox("DSO Information")
        dso_layout = QVBoxLayout()
        
        # Name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        name_layout.addWidget(self.name_edit)
        dso_layout.addLayout(name_layout)
        
        # Type and Constellation
        type_constellation_layout = QHBoxLayout()
        type_constellation_layout.addWidget(QLabel("Type:"))
        self.type_edit = QLineEdit()
        type_constellation_layout.addWidget(self.type_edit)
        
        type_constellation_layout.addWidget(QLabel("Constellation:"))
        self.constellation_edit = QLineEdit()
        type_constellation_layout.addWidget(self.constellation_edit)
        dso_layout.addLayout(type_constellation_layout)
        
        # Coordinates
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("RA (deg):"))
        self.ra_edit = QLineEdit()
        coord_layout.addWidget(self.ra_edit)
        
        coord_layout.addWidget(QLabel("Dec (deg):"))
        self.dec_edit = QLineEdit()
        coord_layout.addWidget(self.dec_edit)
        dso_layout.addLayout(coord_layout)
        
        # Magnitude and Size
        mag_size_layout = QHBoxLayout()
        mag_size_layout.addWidget(QLabel("Magnitude:"))
        self.magnitude_edit = QLineEdit()
        coord_layout.addWidget(self.magnitude_edit)
        
        mag_size_layout.addWidget(QLabel("Size ('):"))
        self.size_edit = QLineEdit()
        mag_size_layout.addWidget(self.size_edit)
        dso_layout.addLayout(mag_size_layout)
        
        dso_group.setLayout(dso_layout)
        layout.addWidget(dso_group)
        
        # Target Information Group
        target_group = QGroupBox("Target Information")
        target_layout = QVBoxLayout()
        
        # Priority
        priority_layout = QHBoxLayout()
        priority_layout.addWidget(QLabel("Priority:"))
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(["Low", "Medium", "High", "Urgent"])
        self.priority_combo.setCurrentText("Medium")
        priority_layout.addWidget(self.priority_combo)

        # Status
        priority_layout.addWidget(QLabel("Status:"))
        self.status_combo = QComboBox()
        self.status_combo.addItems(["Not Observed", "Observed", "Imaged", "Completed"])
        self.status_combo.setCurrentText("Not Observed")
        priority_layout.addWidget(self.status_combo)
        target_layout.addLayout(priority_layout)

        # Telescope
        telescope_layout = QHBoxLayout()
        telescope_layout.addWidget(QLabel("Telescope:"))
        self.telescope_combo = QComboBox()
        self._populate_telescope_combo()
        telescope_layout.addWidget(self.telescope_combo)
        telescope_layout.addStretch()
        target_layout.addLayout(telescope_layout)
        
        # Best months for observing
        months_layout = QHBoxLayout()
        months_layout.addWidget(QLabel("Best Months:"))
        self.months_edit = QLineEdit()
        self.months_edit.setPlaceholderText("e.g., Nov-Feb, Mar-Jun")
        months_layout.addWidget(self.months_edit)
        target_layout.addLayout(months_layout)
        
        # Notes
        notes_layout = QVBoxLayout()
        notes_layout.addWidget(QLabel("Notes:"))
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(100)
        self.notes_edit.setPlaceholderText("Observing notes, equipment recommendations, etc.")
        notes_layout.addWidget(self.notes_edit)
        target_layout.addLayout(notes_layout)
        
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)
        
        self.save_btn = QPushButton("Add to Target List")
        self.save_btn.clicked.connect(self._save_target)
        self.save_btn.setDefault(True)
        buttons_layout.addWidget(self.save_btn)
        
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
    
    def set_edit_mode(self, target_id):
        """Set the dialog to edit mode, changing the button text"""
        self.is_edit_mode = True
        self.target_id = target_id
        self.save_btn.setText("Save Changes")
    
    def _populate_telescope_combo(self):
        """Populate telescope dropdown with active telescopes"""
        self.telescope_combo.clear()
        self.telescope_combo.addItem("Any", None)  # First item for unassigned

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, aperture, focal_length
                    FROM usertelescopes
                    WHERE is_active = 1
                    ORDER BY name
                """)
                telescopes = cursor.fetchall()

                for telescope in telescopes:
                    tel_id, name, aperture, focal_length = telescope
                    # Calculate f/ratio if we have both values
                    if aperture and focal_length and aperture > 0:
                        f_ratio = focal_length / aperture
                        display_text = f"{name} ({int(aperture)}mm f/{f_ratio:.1f})"
                    elif aperture:
                        display_text = f"{name} ({int(aperture)}mm)"
                    else:
                        display_text = name
                    self.telescope_combo.addItem(display_text, tel_id)
        except Exception as e:
            logger.error(f"Error loading telescopes: {str(e)}")

    def _populate_from_dso_data(self):
        """Populate dialog fields with DSO data"""
        if not self.dso_data:
            return

        self.name_edit.setText(self.dso_data.get("name", ""))
        self.type_edit.setText(self.dso_data.get("dso_type", ""))
        self.constellation_edit.setText(self.dso_data.get("constellation", ""))

        # Handle numeric fields - only set if value is not None
        ra_deg = self.dso_data.get("ra_deg")
        if ra_deg is not None:
            self.ra_edit.setText(str(ra_deg))

        dec_deg = self.dso_data.get("dec_deg")
        if dec_deg is not None:
            self.dec_edit.setText(str(dec_deg))

        magnitude = self.dso_data.get("magnitude")
        if magnitude is not None:
            self.magnitude_edit.setText(str(magnitude))

        # Format size
        size_min = self.dso_data.get("size_min", 0)
        size_max = self.dso_data.get("size_max", 0)
        if size_min > 0 or size_max > 0:
            self.size_edit.setText(f"{size_min:.1f} x {size_max:.1f}")

        # Populate best months if available
        self.months_edit.setText(self.dso_data.get("best_months", ""))
    
    def _save_target(self):
        """Save the target to the database"""
        try:
            # Validate required fields
            if not self.name_edit.text().strip():
                QMessageBox.warning(self, "Validation Error", "Name is required.")
                return

            # Helper function to safely convert to float
            def safe_float(text):
                """Convert text to float, handling empty strings and 'None'"""
                text = text.strip()
                if not text or text.lower() == 'none':
                    return 0.0
                return float(text)

            # Create target data
            target_data = {
                "name": self.name_edit.text().strip(),
                "dso_type": self.type_edit.text().strip(),
                "constellation": self.constellation_edit.text().strip(),
                "ra_deg": safe_float(self.ra_edit.text()),
                "dec_deg": safe_float(self.dec_edit.text()),
                "magnitude": safe_float(self.magnitude_edit.text()),
                "size_info": self.size_edit.text().strip(),
                "priority": self.priority_combo.currentText(),
                "status": self.status_combo.currentText(),
                "best_months": self.months_edit.text().strip(),
                "notes": self.notes_edit.toPlainText().strip(),
                "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "telescope_id": self.telescope_combo.currentData()
            }

            # Save to database - either INSERT new or UPDATE existing
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                if self.is_edit_mode and self.target_id:
                    # Update existing record
                    cursor.execute("""
                        UPDATE usertargetlist SET
                            name = ?, dso_type = ?, constellation = ?, ra_deg = ?, dec_deg = ?,
                            magnitude = ?, size_info = ?, priority = ?, status = ?,
                            best_months = ?, notes = ?, telescope_id = ?
                        WHERE id = ?
                    """, (
                        target_data["name"], target_data["dso_type"], target_data["constellation"],
                        target_data["ra_deg"], target_data["dec_deg"], target_data["magnitude"],
                        target_data["size_info"], target_data["priority"], target_data["status"],
                        target_data["best_months"], target_data["notes"], target_data["telescope_id"],
                        self.target_id
                    ))
                    success_message = f"{target_data['name']} has been updated in your target list."
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO usertargetlist (
                            name, dso_type, constellation, ra_deg, dec_deg, magnitude,
                            size_info, priority, status, best_months, notes, date_added, telescope_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        target_data["name"], target_data["dso_type"], target_data["constellation"],
                        target_data["ra_deg"], target_data["dec_deg"], target_data["magnitude"],
                        target_data["size_info"], target_data["priority"], target_data["status"],
                        target_data["best_months"], target_data["notes"], target_data["date_added"],
                        target_data["telescope_id"]
                    ))
                    success_message = f"{target_data['name']} has been added to your target list."
                
                conn.commit()
            
            QMessageBox.information(self, "Success", success_message)
            self.accept()
            
        except ValueError as e:
            QMessageBox.warning(self, "Validation Error", "Please enter valid numeric values for coordinates and magnitude.")
        except Exception as e:
            logger.error(f"Error saving target: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save target: {str(e)}")


class DSOTargetListWindow(WindowPositionMixin, QMainWindow):
    WINDOW_POSITION_KEY = "DSOTargetList"
    """Main window for DSO target list management"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSO Target List - Cosmos Collection")
        self.resize(1210, 850)
        self.setup_window_position()

        self.db_manager = DatabaseManager()
        self.targets_data = []
        self._init_database()
        self._init_ui()
        self._load_targets()
    
    def _init_database(self):
        """Initialize the target list database table"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS usertargetlist (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        dso_type TEXT,
                        constellation TEXT,
                        ra_deg REAL,
                        dec_deg REAL,
                        magnitude REAL,
                        size_info TEXT,
                        priority TEXT DEFAULT 'Medium',
                        status TEXT DEFAULT 'Not Observed',
                        best_months TEXT,
                        notes TEXT,
                        date_added TEXT,
                        date_observed TEXT,
                        created_date TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                logger.debug("DSO target list table initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing target list database: {str(e)}")

    def _populate_telescope_filter(self):
        """Populate telescope filter dropdown with all telescopes (including inactive)"""
        self.telescope_filter.clear()
        self.telescope_filter.addItem("All", "all")
        self.telescope_filter.addItem("Unassigned", "unassigned")

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                # Include all telescopes (even inactive) since targets may reference them
                cursor.execute("""
                    SELECT id, name, is_active
                    FROM usertelescopes
                    ORDER BY name
                """)
                telescopes = cursor.fetchall()

                for telescope in telescopes:
                    tel_id, name, is_active = telescope
                    display_text = name if is_active else f"{name} (inactive)"
                    self.telescope_filter.addItem(display_text, tel_id)
        except Exception as e:
            logger.error(f"Error loading telescopes for filter: {str(e)}")

    def _init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_label = QLabel("DSO Target List")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(header_label)
        
        # Control panel
        control_group = QGroupBox("Target List Management")
        control_layout = QHBoxLayout()
        
        # Add target button
        add_target_btn = QPushButton("Add New Target")
        add_target_btn.clicked.connect(self._add_new_target)
        control_layout.addWidget(add_target_btn)
        
        # Edit target button
        self.edit_target_btn = QPushButton("Edit Selected")
        self.edit_target_btn.clicked.connect(self._edit_selected_target)
        self.edit_target_btn.setEnabled(False)
        control_layout.addWidget(self.edit_target_btn)
        
        # View details button
        self.view_details_btn = QPushButton("View Details")
        self.view_details_btn.clicked.connect(self._view_target_details)
        self.view_details_btn.setEnabled(False)
        self.view_details_btn.setToolTip("Open detailed view of selected target")
        control_layout.addWidget(self.view_details_btn)
        
        # Remove target button
        self.remove_target_btn = QPushButton("Remove Selected")
        self.remove_target_btn.clicked.connect(self._remove_selected_target)
        self.remove_target_btn.setEnabled(False)
        control_layout.addWidget(self.remove_target_btn)
        
        # Best DSO Tonight button
        best_tonight_btn = QPushButton("Best DSO Tonight")
        best_tonight_btn.clicked.connect(self._open_best_dso_tonight)
        best_tonight_btn.setToolTip("Open Best DSO Tonight window to find the best objects to observe tonight")
        control_layout.addWidget(best_tonight_btn)
        
        control_layout.addStretch()
        
        # Filter controls
        control_layout.addWidget(QLabel("Filter by Status:"))
        self.status_filter = QComboBox()
        self.status_filter.addItems(["All", "Not Observed", "Observed", "Imaged", "Completed"])
        self.status_filter.currentTextChanged.connect(self._filter_targets)
        control_layout.addWidget(self.status_filter)
        
        control_layout.addWidget(QLabel("Filter by Priority:"))
        self.priority_filter = QComboBox()
        self.priority_filter.addItems(["All", "Low", "Medium", "High", "Urgent"])
        self.priority_filter.currentTextChanged.connect(self._filter_targets)
        control_layout.addWidget(self.priority_filter)

        control_layout.addWidget(QLabel("Telescope:"))
        self.telescope_filter = QComboBox()
        self._populate_telescope_filter()
        self.telescope_filter.currentIndexChanged.connect(self._filter_targets)
        control_layout.addWidget(self.telescope_filter)

        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_targets)
        control_layout.addWidget(refresh_btn)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Targets table
        targets_group = QGroupBox("Target List")
        targets_layout = QVBoxLayout()
        
        self.targets_table = QTableWidget()
        self.targets_table.setColumnCount(11)
        self.targets_table.setHorizontalHeaderLabels([
            "Name", "Type", "Constellation", "Magnitude", "Size",
            "Priority", "Status", "Telescope", "Direction", "Best Months", "Date Added"
        ])

        # Enable sorting and disable editing
        self.targets_table.setSortingEnabled(True)

        # Set column widths - Allow manual resizing
        header = self.targets_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Name column autosizes to content
        for col in range(1, 11):
            header.setSectionResizeMode(col, QHeaderView.Interactive)  # Other columns allow manual resizing

        # Set initial default widths for manually resizable columns
        self.targets_table.setColumnWidth(1, 120)  # Type
        self.targets_table.setColumnWidth(2, 100)  # Constellation
        self.targets_table.setColumnWidth(3, 90)   # Magnitude
        self.targets_table.setColumnWidth(4, 80)   # Size
        self.targets_table.setColumnWidth(5, 90)   # Priority
        self.targets_table.setColumnWidth(6, 100)  # Status
        self.targets_table.setColumnWidth(7, 120)  # Telescope
        self.targets_table.setColumnWidth(8, 70)   # Direction
        self.targets_table.setColumnWidth(9, 150)  # Best Months
        self.targets_table.setColumnWidth(10, 100) # Date Added
        
        self.targets_table.setAlternatingRowColors(True)
        self.targets_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.targets_table.setEditTriggers(QTableWidget.NoEditTriggers)  # Disable cell editing
        self.targets_table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.targets_table.itemDoubleClicked.connect(self._edit_selected_target)

        # Enable context menu
        self.targets_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.targets_table.customContextMenuRequested.connect(self._show_context_menu)
        
        targets_layout.addWidget(self.targets_table)
        targets_group.setLayout(targets_layout)
        main_layout.addWidget(targets_group)
        
        # Status bar
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
    
    def _add_new_target(self):
        """Add a new target to the list"""
        dialog = AddTargetDialog(parent=self)
        if dialog.exec() == QDialog.Accepted:
            self._load_targets()
    
    def _edit_selected_target(self):
        """Edit the selected target"""
        current_row = self.targets_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a target to edit.")
            return

        # Get target data from the name item (column 0) to handle sorting
        name_item = self.targets_table.item(current_row, 0)
        if not name_item:
            return

        target_data = name_item.data(Qt.UserRole)
        dialog = AddTargetDialog(dso_data=target_data, parent=self)
        dialog.setWindowTitle("Edit Target")
        dialog.set_edit_mode(target_data["id"])  # Change button text to "Save Changes" and set target ID
        
        # Pre-populate with target data
        dialog.name_edit.setText(target_data.get("name", ""))
        dialog.type_edit.setText(target_data.get("dso_type", ""))
        dialog.constellation_edit.setText(target_data.get("constellation", ""))

        # Handle numeric fields - only set if value is not None
        ra_deg = target_data.get("ra_deg")
        if ra_deg is not None:
            dialog.ra_edit.setText(str(ra_deg))

        dec_deg = target_data.get("dec_deg")
        if dec_deg is not None:
            dialog.dec_edit.setText(str(dec_deg))

        magnitude = target_data.get("magnitude")
        if magnitude is not None:
            dialog.magnitude_edit.setText(str(magnitude))

        dialog.size_edit.setText(target_data.get("size_info", ""))
        dialog.priority_combo.setCurrentText(target_data.get("priority", "Medium"))
        dialog.status_combo.setCurrentText(target_data.get("status", "Not Observed"))
        dialog.months_edit.setText(target_data.get("best_months", ""))
        dialog.notes_edit.setPlainText(target_data.get("notes", ""))

        # Set telescope selection
        telescope_id = target_data.get("telescope_id")
        if telescope_id is not None:
            index = dialog.telescope_combo.findData(telescope_id)
            if index >= 0:
                dialog.telescope_combo.setCurrentIndex(index)
        else:
            dialog.telescope_combo.setCurrentIndex(0)  # "Any"

        if dialog.exec() == QDialog.Accepted:
            # Store the target ID to re-select after reload
            edited_target_id = target_data["id"]

            # Reload targets to reflect the changes (dialog already handles the database update)
            self._load_targets()

            # Re-select the edited target
            self._select_target_by_id(edited_target_id)

    def _select_target_by_id(self, target_id):
        """Find and select a target row by its ID"""
        for row in range(self.targets_table.rowCount()):
            name_item = self.targets_table.item(row, 0)
            if name_item:
                row_data = name_item.data(Qt.UserRole)
                if row_data and row_data.get("id") == target_id:
                    self.targets_table.selectRow(row)
                    self.targets_table.scrollToItem(name_item)
                    return

    def _remove_selected_target(self):
        """Remove the selected target from the list"""
        current_row = self.targets_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a target to remove.")
            return

        # Get target data from the name item (column 0) to handle sorting
        name_item = self.targets_table.item(current_row, 0)
        if not name_item:
            return

        target_data = name_item.data(Qt.UserRole)
        target_name = target_data.get("name", "Unknown")
        
        reply = QMessageBox.question(
            self, "Confirm Removal", 
            f"Are you sure you want to remove '{target_name}' from your target list?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM usertargetlist WHERE id = ?", (target_data["id"],))
                    conn.commit()
                
                QMessageBox.information(self, "Success", f"'{target_name}' has been removed from your target list.")
                self._load_targets()
                
            except Exception as e:
                logger.error(f"Error removing target: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to remove target: {str(e)}")
    
    def _on_selection_changed(self):
        """Handle selection changes in the table"""
        has_selection = self.targets_table.currentRow() >= 0
        self.edit_target_btn.setEnabled(has_selection)
        self.view_details_btn.setEnabled(has_selection)
        self.remove_target_btn.setEnabled(has_selection)

    def _open_best_dso_tonight(self):
        """Open the Best DSO Tonight window"""
        try:
            # Create and show the Best DSO Tonight window with target list auto-selected
            self.best_dso_window = BestDSOTonightWindow(use_target_list=True)
            self.best_dso_window.show()
            self.best_dso_window.raise_()
            self.best_dso_window.activateWindow()
            logger.debug("Best DSO Tonight window opened successfully with target list selected")
        except Exception as e:
            logger.error(f"Error opening Best DSO Tonight window: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Best DSO Tonight window: {str(e)}")

    def _view_target_details(self):
        """Open DSODetailWindow for the selected target"""
        current_row = self.targets_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a target to view details.")
            return

        # Get target data from the name item (column 0) to handle sorting
        name_item = self.targets_table.item(current_row, 0)
        if not name_item:
            return

        try:
            target_data = name_item.data(Qt.UserRole)
            target_name = target_data.get("name", "")
            
            # Import DSODetailWindow from main.py
            from main import DSODetailWindow
            
            # Try to find the complete DSO data in the main database
            detail_data = self._get_full_dso_data(target_name, target_data)
            
            if detail_data:
                # Create and show the DSODetailWindow with full data
                detail_window = DSODetailWindow(detail_data, self)
                detail_window.show()
            else:
                QMessageBox.warning(self, "Object Not Found", 
                                  f"Could not find complete information for {target_name} in the main DSO database.")
            
        except ImportError as e:
            QMessageBox.critical(self, "Error", "Could not import DSODetailWindow. Please ensure Main.py is available.")
            logger.error(f"Failed to import DSODetailWindow: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open target details: {str(e)}")
            logger.error(f"Error opening target details: {str(e)}")
    
    def _get_full_dso_data(self, target_name, target_data):
        """Get full DSO data from the main database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Parse the target name to get catalogue and designation
                name_parts = target_name.split()
                if len(name_parts) >= 2:
                    catalogue = name_parts[0]
                    designation = " ".join(name_parts[1:])
                else:
                    # If name doesn't have clear catalogue/designation, try to find by coordinates
                    return self._get_dso_data_by_coordinates(target_data)
                
                # Query the full DSO data using the same method as Main.py
                cursor.execute("""
                    WITH object_dsodetailid AS (
                        SELECT d.id 
                        FROM dsodetail d
                        JOIN cataloguenr c ON d.id = c.dsodetailid
                        WHERE c.catalogue = ? AND c.designation = ?
                    )
                    SELECT d.id, d.ra, d.dec, d.magnitude, d.surfacebrightness, 
                           CAST(d.sizemin/60.0 AS REAL) as sizemin,
                           CAST(d.sizemax/60.0 AS REAL) as sizemax,
                           d.constellation, d.dsotype, d.dsoclass,
                           GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', ' ORDER BY 
                               CASE c.catalogue 
                                   WHEN 'M' THEN 1
                                   WHEN 'NGC' THEN 2
                                   WHEN 'IC' THEN 3
                                   ELSE 4
                               END, c.designation) as designations,
                           ui.image_path, ui.integration_time, ui.equipment, ui.date_taken, ui.notes,
                           (SELECT COUNT(*) FROM userimages WHERE dsodetailid = d.id) as image_count
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    LEFT JOIN userimages ui ON d.id = ui.dsodetailid
                    WHERE d.id = (SELECT id FROM object_dsodetailid)
                    GROUP BY d.id
                """, (catalogue, designation))
                
                result = cursor.fetchone()
                
                if result:
                    return self._process_dso_query_result(result, target_data)
                else:
                    # If not found by name, try by coordinates
                    return self._get_dso_data_by_coordinates(target_data)
                    
        except Exception as e:
            logger.error(f"Error querying DSO database: {str(e)}")
            return None
    
    def _get_dso_data_by_coordinates(self, target_data):
        """Try to find DSO by coordinates (within reasonable tolerance)"""
        try:
            ra_deg = target_data.get("ra_deg", 0)
            dec_deg = target_data.get("dec_deg", 0)
            tolerance = 0.1  # degrees
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT d.id, d.ra, d.dec, d.magnitude, d.surfacebrightness, 
                           CAST(d.sizemin/60.0 AS REAL) as sizemin,
                           CAST(d.sizemax/60.0 AS REAL) as sizemax,
                           d.constellation, d.dsotype, d.dsoclass,
                           GROUP_CONCAT(c.catalogue || ' ' || c.designation, ', ' ORDER BY 
                               CASE c.catalogue 
                                   WHEN 'M' THEN 1
                                   WHEN 'NGC' THEN 2
                                   WHEN 'IC' THEN 3
                                   ELSE 4
                               END, c.designation) as designations,
                           ui.image_path, ui.integration_time, ui.equipment, ui.date_taken, ui.notes,
                           (SELECT COUNT(*) FROM userimages WHERE dsodetailid = d.id) as image_count
                    FROM dsodetail d
                    JOIN cataloguenr c ON d.id = c.dsodetailid
                    LEFT JOIN userimages ui ON d.id = ui.dsodetailid
                    WHERE ABS(d.ra - ?) < ? AND ABS(d.dec - ?) < ?
                    GROUP BY d.id
                    ORDER BY ABS(d.ra - ?) + ABS(d.dec - ?) ASC
                    LIMIT 1
                """, (ra_deg, tolerance, dec_deg, tolerance, ra_deg, dec_deg))
                
                result = cursor.fetchone()
                if result:
                    return self._process_dso_query_result(result, target_data)
                    
        except Exception as e:
            logger.error(f"Error querying DSO by coordinates: {str(e)}")
            
        return None
    
    def _process_dso_query_result(self, result, target_data):
        """Process database query result into DSODetailWindow format"""
        try:
            obj_id, ra, dec, magnitude, surface_brightness, size_min, size_max, \
                constellation, dso_type, dso_class, designations, image_path, integration_time, \
                equipment, date_taken, notes, image_count = result

            # Get the primary designation
            primary_designation = designations.split(',')[0].strip()
            
            # Handle size values
            size_min_arcmin = float(size_min) if size_min is not None else 0.0
            size_max_arcmin = float(size_max) if size_max is not None else 0.0

            # Format coordinates for display
            ra_str = self._format_ra_for_display(ra)
            dec_str = self._format_dec_for_display(dec)

            return {
                "name": primary_designation,
                "ra": ra_str,
                "dec": dec_str,
                "ra_deg": ra,
                "dec_deg": dec,
                "magnitude": magnitude,
                "surface_brightness": surface_brightness,
                "size_min": size_min_arcmin,
                "size_max": size_max_arcmin,
                "constellation": constellation,
                "dso_type": dso_type,
                "dso_class": dso_class,
                "designations": designations,
                "catalogue": primary_designation.split()[0] if " " in primary_designation else "",
                "id": " ".join(primary_designation.split()[1:]) if " " in primary_designation else primary_designation,
                "dsodetailid": obj_id,
                "image_path": image_path,
                "integration_time": integration_time,
                "equipment": equipment,
                "date_taken": date_taken,
                "notes": notes if notes else target_data.get("notes", ""),  # Use target notes if DB notes empty
                "image_count": image_count
            }
            
        except Exception as e:
            logger.error(f"Error processing DSO query result: {str(e)}")
            return None
    
    def _format_ra_for_display(self, ra_deg):
        """Format RA in degrees to HMS format for display"""
        ra_hours = ra_deg / 15.0
        ra_h = int(ra_hours)
        ra_remaining = (ra_hours - ra_h) * 60
        ra_m = int(ra_remaining)
        ra_s = (ra_remaining - ra_m) * 60
        return f"{ra_h:02d}h{ra_m:02d}m{ra_s:05.2f}s"
    
    def _format_dec_for_display(self, dec_deg):
        """Format Dec in degrees to DMS format for display"""
        dec_sign = '-' if dec_deg < 0 else '+'
        dec_abs = abs(dec_deg)
        dec_d = int(dec_abs)
        dec_remaining = (dec_abs - dec_d) * 60
        dec_m = int(dec_remaining)
        dec_s = (dec_remaining - dec_m) * 60
        return f"{dec_sign}{dec_d:02d}°{dec_m:02d}'{dec_s:04.1f}\""
    
    def _filter_targets(self):
        """Apply filters to the targets table"""
        status_filter = self.status_filter.currentText()
        priority_filter = self.priority_filter.currentText()
        telescope_filter_data = self.telescope_filter.currentData()  # Can be "all", "unassigned", or telescope_id

        for row in range(self.targets_table.rowCount()):
            show_row = True

            if status_filter != "All":
                status_item = self.targets_table.item(row, 6)  # Status column
                if not status_item or status_item.text() != status_filter:
                    show_row = False

            if priority_filter != "All" and show_row:
                priority_item = self.targets_table.item(row, 5)  # Priority column
                if not priority_item or priority_item.text() != priority_filter:
                    show_row = False

            if telescope_filter_data != "all" and show_row:
                telescope_item = self.targets_table.item(row, 7)  # Telescope column
                if telescope_item:
                    telescope_id = telescope_item.data(Qt.UserRole)
                    if telescope_filter_data == "unassigned":
                        # Show only targets with no telescope assigned
                        if telescope_id is not None:
                            show_row = False
                    else:
                        # Filter by specific telescope ID
                        if telescope_id != telescope_filter_data:
                            show_row = False

            self.targets_table.setRowHidden(row, not show_row)
    
    def _load_targets(self):
        """Load targets from the database and populate the table"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT t.id, t.name, t.dso_type, t.constellation, t.ra_deg, t.dec_deg, t.magnitude,
                           t.size_info, t.priority, t.status, t.best_months, t.notes, t.date_added,
                           t.telescope_id, tel.name as telescope_name
                    FROM usertargetlist t
                    LEFT JOIN usertelescopes tel ON t.telescope_id = tel.id
                    ORDER BY t.priority DESC, t.date_added DESC
                """)

                rows = cursor.fetchall()
                self.targets_data = []

                for row in rows:
                    target_data = {
                        "id": row[0],
                        "name": row[1],
                        "dso_type": row[2],
                        "constellation": row[3],
                        "ra_deg": row[4],
                        "dec_deg": row[5],
                        "magnitude": row[6],
                        "size_info": row[7],
                        "priority": row[8],
                        "status": row[9],
                        "best_months": row[10],
                        "notes": row[11],
                        "date_added": row[12],
                        "telescope_id": row[13],
                        "telescope_name": row[14]
                    }
                    self.targets_data.append(target_data)
            
            self._populate_table()
            self._filter_targets()
            
            self.status_label.setText(f"Loaded {len(self.targets_data)} targets")
            
        except Exception as e:
            logger.error(f"Error loading targets: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load targets: {str(e)}")
    
    def _populate_table(self):
        """Populate the targets table with loaded data"""
        # Disable sorting temporarily while populating
        self.targets_table.setSortingEnabled(False)

        self.targets_table.setRowCount(len(self.targets_data))

        for row, target in enumerate(self.targets_data):
            # Name - store target data in item
            name_item = QTableWidgetItem(target.get("name", ""))
            name_item.setData(Qt.UserRole, target)  # Store full target data
            self.targets_table.setItem(row, 0, name_item)

            # Type - use friendly name
            dso_type = target.get("dso_type", "")
            friendly_type = self._get_friendly_type_name(dso_type)
            self.targets_table.setItem(row, 1, QTableWidgetItem(friendly_type))

            # Constellation
            self.targets_table.setItem(row, 2, QTableWidgetItem(target.get("constellation", "")))

            # Magnitude - use numeric sorting
            magnitude = target.get("magnitude", 0)
            mag_item = QTableWidgetItem()
            mag_item.setData(Qt.DisplayRole, f"{magnitude:.1f}" if magnitude > 0 else "")
            mag_item.setData(Qt.UserRole, magnitude if magnitude > 0 else 999)  # Store numeric value for sorting
            mag_item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(row, 3, mag_item)

            # Size
            self.targets_table.setItem(row, 4, QTableWidgetItem(target.get("size_info", "")))

            # Priority - use custom PriorityTableWidgetItem for proper sorting
            priority = target.get("priority", "")
            priority_item = PriorityTableWidgetItem(priority)
            self.targets_table.setItem(row, 5, priority_item)

            # Status - use status order for sorting
            status = target.get("status", "")
            status_item = QTableWidgetItem(status)
            status_order = {"Not Observed": 1, "Observed": 2, "Imaged": 3, "Completed": 4}
            status_item.setData(Qt.UserRole, status_order.get(status, 0))  # Store numeric value for sorting
            status_item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(row, 6, status_item)

            # Telescope - display name or "Any" for unassigned
            telescope_name = target.get("telescope_name", "")
            telescope_id = target.get("telescope_id")
            telescope_display = telescope_name if telescope_name else "Any"
            telescope_item = QTableWidgetItem(telescope_display)
            telescope_item.setData(Qt.UserRole, telescope_id)  # Store telescope_id for filtering
            telescope_item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(row, 7, telescope_item)

            # Direction - calculate current direction
            ra_deg = target.get("ra_deg", 0)
            dec_deg = target.get("dec_deg", 0)
            if ra_deg and dec_deg:
                direction = self._calculate_current_direction(ra_deg, dec_deg)
            else:
                direction = "No coordinates"
            direction_item = QTableWidgetItem(direction)
            direction_item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(row, 8, direction_item)

            # Best Months
            self.targets_table.setItem(row, 9, QTableWidgetItem(target.get("best_months", "")))

            # Date Added - use date object for sorting
            date_added = target.get("date_added", "")
            if date_added:
                try:
                    # Format date for display
                    date_obj = datetime.strptime(date_added, "%Y-%m-%d %H:%M:%S")
                    formatted_date = date_obj.strftime("%Y-%m-%d")
                    date_timestamp = date_obj.timestamp()
                except:
                    formatted_date = date_added
                    date_timestamp = 0
            else:
                formatted_date = ""
                date_timestamp = 0

            date_item = QTableWidgetItem(formatted_date)
            date_item.setData(Qt.UserRole, date_timestamp)  # Store timestamp for sorting
            date_item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(row, 10, date_item)

        # Re-enable sorting
        self.targets_table.setSortingEnabled(True)

        # Set default sort by Priority (column 5) in descending order (Urgent first)
        self.targets_table.sortItems(5, Qt.DescendingOrder)
    
    def _calculate_best_months_for_all(self):
        """Calculate best viewing months for all targets based on user location"""
        try:
            # Get user location from database
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon FROM usersettings WHERE is_active = 1 LIMIT 1")
                location_row = cursor.fetchone()
                if not location_row:
                    cursor.execute("SELECT location_lat, location_lon FROM usersettings ORDER BY id DESC LIMIT 1")
                    location_row = cursor.fetchone()

                if not location_row:
                    QMessageBox.warning(self, "No Location Set", 
                        "Please set your observing location in Settings first.\n\n" +
                        "Go to Settings and enter your latitude and longitude coordinates.")
                    return
                
                lat, lon = location_row
                logger.debug(f"Using user location: lat={lat}, lon={lon}")
                
                # Update status
                self.status_label.setText("Calculating best months for all targets...")
                
                # Calculate best months for each target
                targets_updated = 0
                for target in self.targets_data:
                    if target.get("ra_deg") and target.get("dec_deg"):
                        best_months = self._calculate_best_months_for_target(
                            target["ra_deg"], target["dec_deg"], lat, lon
                        )
                        
                        if best_months:
                            # Update database
                            cursor.execute("""
                                UPDATE usertargetlist SET best_months = ? WHERE id = ?
                            """, (best_months, target["id"]))
                            targets_updated += 1
                
                conn.commit()
                
                # Reload the table to show updated months
                self._load_targets()
                
                QMessageBox.information(self, "Calculation Complete", 
                    f"Best viewing months calculated for {targets_updated} targets based on your location.")
                
        except Exception as e:
            logger.error(f"Error calculating best months: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to calculate best months: {str(e)}")
        finally:
            self.status_label.setText(f"Loaded {len(self.targets_data)} targets")
    
    def _calculate_best_months_for_target(self, ra_deg, dec_deg, lat, lon):
        """Calculate best viewing months for a single target using centralized calculator"""
        # Import required modules at the very top, outside any try blocks
        import numpy as np
        from datetime import datetime, timedelta
        
        try:
            # Import astronomy libraries
            from DSOVisibilityCalculator import DSOVisibilityCalculator
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            # Create calculator with user location
            calculator = DSOVisibilityCalculator(lat, lon)
            
            # Create coordinate object once since we have RA/Dec
            dso_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
            
            # Sample dates throughout the year (same approach as DSODetailWindow)
            current_year = datetime.now().year
            min_altitude = 30  # Use 30° minimum altitude
            
            sample_dates = []
            visibility_results = []
            
            for day_offset in range(0, 365, 15):  # Every 15 days like DSODetailWindow
                try:
                    test_date = datetime(current_year, 1, 1) + timedelta(days=day_offset)
                    date_str = test_date.strftime('%Y-%m-%d')
                    
                    # Use coordinate-based calculation instead of name-based
                    time_range, dso_altaz, sun_altaz = calculator.calculate_altaz_over_time(
                        dso_coord, date_str, 12)
                    
                    # Find optimal viewing times using same criteria
                    optimal_times = calculator.find_optimal_viewing_times(
                        dso_altaz, sun_altaz, min_altitude)
                    
                    results = {"optimal_times": optimal_times}
                    
                    is_visible = False
                    if "error" not in results and np.any(results.get("optimal_times", [])):
                        is_visible = True
                    
                    sample_dates.append(test_date)
                    visibility_results.append(is_visible)
                    
                except Exception as e:
                    logger.debug(f"Error checking date {day_offset}: {e}")
                    continue
            
            # Group visible periods into months
            if any(visibility_results):
                good_months = set()
                for date, visible in zip(sample_dates, visibility_results):
                    if visible:
                        good_months.add(date.month)
                
                # Convert month numbers to abbreviations
                month_abbrs = [calendar.month_abbr[month] for month in sorted(good_months)]
                
                # Format the result
                if month_abbrs:
                    return self._format_month_ranges(month_abbrs)
                else:
                    return "Not optimal from location"
            else:
                return "Not optimal from location"
                
        except ImportError:
            logger.error("Missing DSOVisibilityCalculator for best months calculation")
            return "Calculation unavailable"
        except Exception as e:
            logger.error(f"Error calculating best months for target: {str(e)}")
            return "Calculation error"
    
    def _format_month_ranges(self, months):
        """Format month list into ranges (e.g., 'Nov-Feb, Jun-Aug')"""
        if not months:
            return ""
        
        # Convert month abbreviations back to numbers for processing
        month_nums = []
        month_map = {calendar.month_abbr[i]: i for i in range(1, 13)}
        
        for month in months:
            if month in month_map:
                month_nums.append(month_map[month])
        
        if not month_nums:
            return ", ".join(months)
        
        month_nums.sort()
        
        # Find consecutive ranges
        ranges = []
        start = month_nums[0]
        end = month_nums[0]
        
        for i in range(1, len(month_nums)):
            if month_nums[i] == end + 1:
                end = month_nums[i]
            else:
                # Add the range
                if start == end:
                    ranges.append(calendar.month_abbr[start])
                else:
                    ranges.append(f"{calendar.month_abbr[start]}-{calendar.month_abbr[end]}")
                start = month_nums[i]
                end = month_nums[i]
        
        # Don't forget the last range
        if start == end:
            ranges.append(calendar.month_abbr[start])
        else:
            ranges.append(f"{calendar.month_abbr[start]}-{calendar.month_abbr[end]}")
        
        return ", ".join(ranges)

    def _get_preferred_catalog_name(self, designations):
        """Extract the most common/preferred catalog name from designations string

        Priority: M > NGC > IC > others

        Args:
            designations: String of catalog designations (e.g., "M 42, NGC 1976, LBN 974")

        Returns:
            The preferred catalog name (e.g., "M 42")
        """
        if not designations:
            return ""

        # Split designations by comma
        designation_list = [d.strip() for d in designations.split(',')]

        # Priority order for catalogs
        priority_catalogs = ['M', 'NGC', 'IC']

        # Search for each priority catalog in order
        for catalog in priority_catalogs:
            for designation in designation_list:
                # Check if this designation starts with the catalog name
                if designation.startswith(catalog + ' '):
                    return designation

        # If no priority catalog found, return the first designation
        return designation_list[0] if designation_list else ""

    def _get_friendly_type_name(self, dso_type):
        """Convert DSO type code to user-friendly name"""
        type_mapping = {
            "GALXY": "Galaxy",
            "DRKNB": "Dark Nebula",
            "OPNCL": "Open Cluster",
            "PLNNB": "Planetary Nebula",
            "BRTNB": "Bright Nebula",
            "SNREM": "Supernova Remnant",
            "GALCL": "Galaxy Cluster",
            "GLOCL": "Globular Cluster",
            "CL+NB": "Cluster + Nebula",
            "GX+DN": "Galaxy + Dark Nebula",
            "ASTER": "Asterism",
            "2STAR": "Double Star",
            "3STAR": "Triple Star",
            "4STAR": "Quadruple Star",
            "1STAR": "Single Star",
            "QUASR": "Quasar",
            "NONEX": "Non-existent",
            "LMCCN": "LMC Cluster/Nebula",
            "LMCDN": "LMC Dark Nebula",
            "LMCGC": "LMC Globular Cluster",
            "LMCOC": "LMC Open Cluster",
            "SMCCN": "SMC Cluster/Nebula",
            "SMCDN": "SMC Dark Nebula",
            "SMCGC": "SMC Globular Cluster",
            "SMCOC": "SMC Open Cluster"
        }
        return type_mapping.get(dso_type, dso_type)  # Return original if not found

    def azimuth_to_direction(self, az):
        """Convert azimuth to cardinal direction"""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        idx = int((az + 11.25) / 22.5) % 16
        return directions[idx]

    def _calculate_current_direction(self, ra_deg, dec_deg):
        """Calculate current direction for a DSO based on current time and user location"""
        try:
            # Get user location
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT location_lat, location_lon, timezone FROM usersettings WHERE is_active = 1 LIMIT 1")
                location_row = cursor.fetchone()
                if not location_row:
                    cursor.execute("SELECT location_lat, location_lon, timezone FROM usersettings ORDER BY id DESC LIMIT 1")
                    location_row = cursor.fetchone()

                if not location_row:
                    return "Location not set"

                lat, lon, timezone_str = location_row
                if lat is None or lon is None:
                    return "Location not set"

            # Use DSOVisibilityCalculator to get current azimuth
            from DSOVisibilityCalculator import DSOVisibilityCalculator
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            from datetime import datetime
            import pytz

            # Create calculator
            calculator = DSOVisibilityCalculator(lat, lon)

            # Create coordinate object
            dso_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)

            # Get current time in user's timezone
            if timezone_str:
                try:
                    user_tz = pytz.timezone(timezone_str)
                    current_time = datetime.now(user_tz)
                except:
                    current_time = datetime.now()
            else:
                current_time = datetime.now()

            # Calculate current position
            date_str = current_time.strftime('%Y-%m-%d')
            time_range, dso_altaz, sun_altaz = calculator.calculate_altaz_over_time(
                dso_coord, date_str, 0.25)  # Just get current position

            if len(dso_altaz.az.deg) > 0:
                current_azimuth = dso_altaz.az.deg[0]
                return self.azimuth_to_direction(current_azimuth)
            else:
                return "Calculation error"

        except Exception as e:
            logger.debug(f"Error calculating direction: {e}")
            return "Not available"

    def _show_context_menu(self, position):
        """Show context menu when right-clicking on the table"""
        # Get the item at the clicked position
        item = self.targets_table.itemAt(position)
        if not item:
            return  # No item at this position

        # Get the row number
        row = item.row()
        if row < 0:
            return

        # Create context menu
        context_menu = QMenu(self)

        # Add menu actions
        details_action = context_menu.addAction("View DSO Details")
        details_action.triggered.connect(lambda: self._context_view_details(row))

        visibility_action = context_menu.addAction("Visibility Calculator")
        visibility_action.triggered.connect(lambda: self._context_open_visibility(row))

        aladin_action = context_menu.addAction("FOV Simulator")
        aladin_action.triggered.connect(lambda: self._context_open_aladin(row))

        if NINAIntegration.is_enabled():
            nina_action = context_menu.addAction("Send to NINA Framing Assistant")
            nina_action.triggered.connect(lambda: self._context_send_to_nina(row))

        context_menu.addSeparator()

        edit_action = context_menu.addAction("Edit Target")
        edit_action.triggered.connect(lambda: self._context_edit_target(row))

        remove_action = context_menu.addAction("Remove Target")
        remove_action.triggered.connect(lambda: self._context_remove_target(row))

        # Show the menu at the clicked position
        context_menu.exec(self.targets_table.mapToGlobal(position))

    def _context_view_details(self, row):
        """View DSO details from context menu"""
        # Set the table selection to this row and call existing method
        self.targets_table.selectRow(row)
        self._view_target_details()

    def _context_open_visibility(self, row):
        """Open DSO Visibility Calculator from context menu"""
        try:
            # Get target data from the name item (column 0) to handle sorting
            name_item = self.targets_table.item(row, 0)
            if not name_item:
                return

            target_data = name_item.data(Qt.UserRole)
            target_name = target_data.get("name", "")
            ra_deg = target_data.get("ra_deg", 0)
            dec_deg = target_data.get("dec_deg", 0)

            if not target_name:
                QMessageBox.warning(self, "Error", "No target name available")
                return

            logger.debug(f"Opening DSO Visibility Calculator for: {target_name} at RA {ra_deg}° Dec {dec_deg}°")

            # Import and open DSO Visibility Calculator
            from DSOVisibilityCalculator import DSOVisibilityApp

            # Store reference to prevent garbage collection
            self.visibility_window = DSOVisibilityApp()

            # Set DSO name in input field for display and title
            if hasattr(self.visibility_window, 'dso_input'):
                self.visibility_window.dso_input.setText(target_name)
                logger.debug(f"Set DSO name in input field: {target_name}")
            else:
                logger.warning("DSO input field not found in visibility window")

            # Use coordinates for accurate calculation
            if hasattr(self.visibility_window, 'set_dso_coordinates'):
                self.visibility_window.set_dso_coordinates(ra_deg, dec_deg)
                logger.debug(f"Set coordinates: RA {ra_deg}° Dec {dec_deg}°")

            # Show the window immediately
            self.visibility_window.show()
            self.visibility_window.raise_()
            self.visibility_window.activateWindow()

            # Automatically trigger calculation after a short delay to allow window to fully initialize
            if hasattr(self.visibility_window, 'calculate_visibility'):
                QTimer.singleShot(500, self.visibility_window.calculate_visibility)
                logger.debug("Triggered automatic visibility calculation")
            else:
                logger.warning("Calculate visibility method not found in visibility window")

            logger.debug("DSO Visibility Calculator window opened successfully")

        except Exception as e:
            logger.error(f"Error opening DSO Visibility Calculator: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open DSO Visibility Calculator: {str(e)}")

    def _context_open_aladin(self, row):
        """Open Aladin Lite from context menu"""
        try:
            # Get target data from the name item (column 0) to handle sorting
            name_item = self.targets_table.item(row, 0)
            if not name_item:
                return

            target_data = name_item.data(Qt.UserRole)
            target_name = target_data.get("name", "")

            # Get full DSO data for Aladin Lite
            detail_data = self._get_full_dso_data(target_name, target_data)
            if not detail_data:
                QMessageBox.warning(self, "Error", f"Could not find detailed data for {target_name}")
                return

            # Import and open Aladin Lite window
            from main import AladinLiteWindow
            aladin_window = AladinLiteWindow(detail_data, self)
            aladin_window.show()

        except Exception as e:
            logger.error(f"Error opening Aladin Lite: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open Aladin Lite: {str(e)}")

    def _context_send_to_nina(self, row):
        """Send target coordinates to NINA Framing Assistant"""
        name_item = self.targets_table.item(row, 0)
        if not name_item:
            return

        target_data = name_item.data(Qt.UserRole)
        NINAIntegration.send_to_framing_assistant(
            target_data.get("ra_deg"), target_data.get("dec_deg"),
            target_data.get("name", "Unknown"), self
        )

    def _context_edit_target(self, row):
        """Edit target from context menu"""
        # Set the table selection to this row and call existing method
        self.targets_table.selectRow(row)
        self._edit_selected_target()

    def _context_remove_target(self, row):
        """Remove target from context menu"""
        # Set the table selection to this row and call existing method
        self.targets_table.selectRow(row)
        self._remove_selected_target()


    def add_target_from_dso(self, dso_data):
        """Add a target from DSO data (called from DSODetailWindow)"""
        # If designations are available, use the preferred catalog name
        if "designations" in dso_data and dso_data["designations"]:
            preferred_name = self._get_preferred_catalog_name(dso_data["designations"])
            if preferred_name:
                dso_data["name"] = preferred_name

        dialog = AddTargetDialog(dso_data=dso_data, parent=self)
        if dialog.exec() == QDialog.Accepted:
            self._load_targets()

    def is_dso_in_target_list(self, dso_name):
        """Check if a DSO is in the target list by name

        Args:
            dso_name: Name of the DSO to check

        Returns:
            bool: True if the DSO is in the target list, False otherwise
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM usertargetlist WHERE name = ?", (dso_name,))
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            logger.error(f"Error checking if DSO is in target list: {str(e)}")
            return False

    def open_and_select_target(self, dso_name):
        """Open the target list window and select the target with the given name

        Args:
            dso_name: Name of the DSO to select

        Returns:
            bool: True if target was found and selected, False otherwise
        """
        try:
            # Ensure the window is visible
            if not self.isVisible():
                self.show()
            self.raise_()
            self.activateWindow()

            # Reload targets to ensure we have current data
            self._load_targets()

            # Find the row with the matching DSO name
            for row in range(self.targets_table.rowCount()):
                name_item = self.targets_table.item(row, 0)
                if name_item and name_item.text() == dso_name:
                    # Select the row
                    self.targets_table.selectRow(row)
                    self.targets_table.scrollToItem(name_item)

                    # Open the edit dialog to show the notes
                    target_data = name_item.data(Qt.UserRole)
                    if target_data:
                        self._edit_target_with_data(target_data)
                    return True

            return False
        except Exception as e:
            logger.error(f"Error opening and selecting target: {str(e)}")
            return False

    def _edit_target_with_data(self, target_data):
        """Open the edit target dialog with the given target data

        Args:
            target_data: Dictionary containing target information
        """
        try:
            dialog = AddTargetDialog(dso_data=target_data, parent=self)
            dialog.setWindowTitle("Edit Target")
            dialog.set_edit_mode(target_data["id"])  # Enable edit mode and set target ID

            # Pre-populate with target data
            dialog.name_edit.setText(target_data.get("name", ""))
            dialog.type_edit.setText(target_data.get("dso_type", ""))
            dialog.constellation_edit.setText(target_data.get("constellation", ""))

            # Handle numeric fields - only set if value is not None
            ra_deg = target_data.get("ra_deg")
            if ra_deg is not None:
                dialog.ra_edit.setText(str(ra_deg))

            dec_deg = target_data.get("dec_deg")
            if dec_deg is not None:
                dialog.dec_edit.setText(str(dec_deg))

            magnitude = target_data.get("magnitude")
            if magnitude is not None:
                dialog.magnitude_edit.setText(str(magnitude))

            dialog.size_edit.setText(target_data.get("size_info", ""))
            dialog.priority_combo.setCurrentText(target_data.get("priority", "Medium"))
            dialog.status_combo.setCurrentText(target_data.get("status", "Not Observed"))
            dialog.months_edit.setText(target_data.get("best_months", ""))
            dialog.notes_edit.setPlainText(target_data.get("notes", ""))

            # Set telescope selection
            telescope_id = target_data.get("telescope_id")
            if telescope_id is not None:
                index = dialog.telescope_combo.findData(telescope_id)
                if index >= 0:
                    dialog.telescope_combo.setCurrentIndex(index)
            else:
                dialog.telescope_combo.setCurrentIndex(0)  # "Any"

            if dialog.exec() == QDialog.Accepted:
                # Store the target ID to re-select after reload
                edited_target_id = target_data["id"]

                # Reload targets to reflect the changes
                self._load_targets()

                # Re-select the edited target
                self._select_target_by_id(edited_target_id)
        except Exception as e:
            logger.error(f"Error opening edit target dialog: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to open target details: {str(e)}")


def main():
    """Main entry point for the application"""
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = DSOTargetListWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()