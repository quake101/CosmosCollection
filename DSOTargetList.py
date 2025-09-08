#!/usr/bin/env python3
"""
DSO Target List Manager
Allows users to manage their observing target list for deep sky objects
"""

import sys
import os
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QLabel, QTableWidget, 
                               QTableWidgetItem, QGroupBox, QMessageBox, 
                               QHeaderView, QTextEdit, QDialog, QComboBox,
                               QLineEdit, QCheckBox, QDateEdit, QSpinBox)
from PySide6.QtGui import QFont

from DatabaseManager import DatabaseManager
import logging

# Set up logging
logger = logging.getLogger(__name__)


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
        
        save_btn = QPushButton("Add to Target List")
        save_btn.clicked.connect(self._save_target)
        save_btn.setDefault(True)
        buttons_layout.addWidget(save_btn)
        
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
    
    def _populate_from_dso_data(self):
        """Populate dialog fields with DSO data"""
        if not self.dso_data:
            return
            
        self.name_edit.setText(self.dso_data.get("name", ""))
        self.type_edit.setText(self.dso_data.get("dso_type", ""))
        self.constellation_edit.setText(self.dso_data.get("constellation", ""))
        self.ra_edit.setText(str(self.dso_data.get("ra_deg", "")))
        self.dec_edit.setText(str(self.dso_data.get("dec_deg", "")))
        self.magnitude_edit.setText(str(self.dso_data.get("magnitude", "")))
        
        # Format size
        size_min = self.dso_data.get("size_min", 0)
        size_max = self.dso_data.get("size_max", 0)
        if size_min > 0 or size_max > 0:
            self.size_edit.setText(f"{size_min:.1f} x {size_max:.1f}")
    
    def _save_target(self):
        """Save the target to the database"""
        try:
            # Validate required fields
            if not self.name_edit.text().strip():
                QMessageBox.warning(self, "Validation Error", "Name is required.")
                return
            
            # Create target data
            target_data = {
                "name": self.name_edit.text().strip(),
                "dso_type": self.type_edit.text().strip(),
                "constellation": self.constellation_edit.text().strip(),
                "ra_deg": float(self.ra_edit.text()) if self.ra_edit.text() else 0.0,
                "dec_deg": float(self.dec_edit.text()) if self.dec_edit.text() else 0.0,
                "magnitude": float(self.magnitude_edit.text()) if self.magnitude_edit.text() else 0.0,
                "size_info": self.size_edit.text().strip(),
                "priority": self.priority_combo.currentText(),
                "status": self.status_combo.currentText(),
                "best_months": self.months_edit.text().strip(),
                "notes": self.notes_edit.toPlainText().strip(),
                "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to database
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO usertargetlist (
                        name, dso_type, constellation, ra_deg, dec_deg, magnitude, 
                        size_info, priority, status, best_months, notes, date_added
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    target_data["name"], target_data["dso_type"], target_data["constellation"],
                    target_data["ra_deg"], target_data["dec_deg"], target_data["magnitude"],
                    target_data["size_info"], target_data["priority"], target_data["status"],
                    target_data["best_months"], target_data["notes"], target_data["date_added"]
                ))
                conn.commit()
            
            QMessageBox.information(self, "Success", f"{target_data['name']} has been added to your target list.")
            self.accept()
            
        except ValueError as e:
            QMessageBox.warning(self, "Validation Error", "Please enter valid numeric values for coordinates and magnitude.")
        except Exception as e:
            logger.error(f"Error saving target: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save target: {str(e)}")


class DSOTargetListWindow(QMainWindow):
    """Main window for DSO target list management"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSO Target List - Cosmos Collection")
        self.setGeometry(100, 100, 1200, 700)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #353535;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #0078d4;
                border: none;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QComboBox, QLineEdit, QTextEdit {
                background-color: #404040;
                border: 1px solid #666666;
                padding: 5px;
                border-radius: 3px;
                color: #ffffff;
            }
            QTableWidget {
                background-color: #404040;
                border: 1px solid #666666;
                color: #ffffff;
                gridline-color: #666666;
                selection-background-color: #0078d4;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #555555;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
            }
            QHeaderView::section {
                background-color: #555555;
                color: #ffffff;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
            QLabel {
                color: #ffffff;
            }
        """)
        
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
        
        # Remove target button
        self.remove_target_btn = QPushButton("Remove Selected")
        self.remove_target_btn.clicked.connect(self._remove_selected_target)
        self.remove_target_btn.setEnabled(False)
        control_layout.addWidget(self.remove_target_btn)
        
        # Calculate best months button
        calc_months_btn = QPushButton("Calculate Best Months")
        calc_months_btn.clicked.connect(self._calculate_best_months_for_all)
        calc_months_btn.setToolTip("Calculate best viewing months for all targets based on your location")
        control_layout.addWidget(calc_months_btn)
        
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
        self.targets_table.setColumnCount(9)
        self.targets_table.setHorizontalHeaderLabels([
            "Name", "Type", "Constellation", "Magnitude", "Size", 
            "Priority", "Status", "Best Months", "Date Added"
        ])
        
        # Set column widths
        header = self.targets_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)  # Name
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Type
        header.setSectionResizeMode(2, QHeaderView.Fixed)  # Constellation
        header.setSectionResizeMode(3, QHeaderView.Fixed)  # Magnitude
        header.setSectionResizeMode(4, QHeaderView.Fixed)  # Size
        header.setSectionResizeMode(5, QHeaderView.Fixed)  # Priority
        header.setSectionResizeMode(6, QHeaderView.Fixed)  # Status
        header.setSectionResizeMode(7, QHeaderView.Stretch)  # Best Months
        header.setSectionResizeMode(8, QHeaderView.Fixed)  # Date Added
        
        self.targets_table.setColumnWidth(0, 120)
        self.targets_table.setColumnWidth(2, 100)
        self.targets_table.setColumnWidth(3, 70)
        self.targets_table.setColumnWidth(4, 80)
        self.targets_table.setColumnWidth(5, 70)
        self.targets_table.setColumnWidth(6, 100)
        self.targets_table.setColumnWidth(8, 100)
        
        self.targets_table.setAlternatingRowColors(True)
        self.targets_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.targets_table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        self.targets_table.itemDoubleClicked.connect(self._edit_selected_target)
        
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
        if current_row < 0 or current_row >= len(self.targets_data):
            QMessageBox.warning(self, "No Selection", "Please select a target to edit.")
            return
        
        target_data = self.targets_data[current_row]
        dialog = AddTargetDialog(dso_data=target_data, parent=self)
        dialog.setWindowTitle("Edit Target")
        
        # Pre-populate with target data
        dialog.name_edit.setText(target_data.get("name", ""))
        dialog.type_edit.setText(target_data.get("dso_type", ""))
        dialog.constellation_edit.setText(target_data.get("constellation", ""))
        dialog.ra_edit.setText(str(target_data.get("ra_deg", "")))
        dialog.dec_edit.setText(str(target_data.get("dec_deg", "")))
        dialog.magnitude_edit.setText(str(target_data.get("magnitude", "")))
        dialog.size_edit.setText(target_data.get("size_info", ""))
        dialog.priority_combo.setCurrentText(target_data.get("priority", "Medium"))
        dialog.status_combo.setCurrentText(target_data.get("status", "Not Observed"))
        dialog.months_edit.setText(target_data.get("best_months", ""))
        dialog.notes_edit.setPlainText(target_data.get("notes", ""))
        
        if dialog.exec() == QDialog.Accepted:
            # Update the existing target instead of creating new one
            self._update_target(target_data["id"], dialog)
    
    def _update_target(self, target_id, dialog):
        """Update an existing target in the database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE usertargetlist SET
                        name = ?, dso_type = ?, constellation = ?, ra_deg = ?, 
                        dec_deg = ?, magnitude = ?, size_info = ?, priority = ?, 
                        status = ?, best_months = ?, notes = ?
                    WHERE id = ?
                """, (
                    dialog.name_edit.text().strip(),
                    dialog.type_edit.text().strip(),
                    dialog.constellation_edit.text().strip(),
                    float(dialog.ra_edit.text()) if dialog.ra_edit.text() else 0.0,
                    float(dialog.dec_edit.text()) if dialog.dec_edit.text() else 0.0,
                    float(dialog.magnitude_edit.text()) if dialog.magnitude_edit.text() else 0.0,
                    dialog.size_edit.text().strip(),
                    dialog.priority_combo.currentText(),
                    dialog.status_combo.currentText(),
                    dialog.months_edit.text().strip(),
                    dialog.notes_edit.toPlainText().strip(),
                    target_id
                ))
                conn.commit()
            
            QMessageBox.information(self, "Success", "Target updated successfully.")
            self._load_targets()
            
        except Exception as e:
            logger.error(f"Error updating target: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to update target: {str(e)}")
    
    def _remove_selected_target(self):
        """Remove the selected target from the list"""
        current_row = self.targets_table.currentRow()
        if current_row < 0 or current_row >= len(self.targets_data):
            QMessageBox.warning(self, "No Selection", "Please select a target to remove.")
            return
        
        target_data = self.targets_data[current_row]
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
        self.remove_target_btn.setEnabled(has_selection)
    
    def _filter_targets(self):
        """Apply filters to the targets table"""
        status_filter = self.status_filter.currentText()
        priority_filter = self.priority_filter.currentText()
        
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
            
            self.targets_table.setRowHidden(row, not show_row)
    
    def _load_targets(self):
        """Load targets from the database and populate the table"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, dso_type, constellation, ra_deg, dec_deg, magnitude,
                           size_info, priority, status, best_months, notes, date_added
                    FROM usertargetlist
                    ORDER BY priority DESC, date_added DESC
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
                        "date_added": row[12]
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
        self.targets_table.setRowCount(len(self.targets_data))
        
        for row, target in enumerate(self.targets_data):
            # Name
            self.targets_table.setItem(row, 0, QTableWidgetItem(target.get("name", "")))
            
            # Type
            self.targets_table.setItem(row, 1, QTableWidgetItem(target.get("dso_type", "")))
            
            # Constellation
            self.targets_table.setItem(row, 2, QTableWidgetItem(target.get("constellation", "")))
            
            # Magnitude
            magnitude = target.get("magnitude", 0)
            mag_item = QTableWidgetItem(f"{magnitude:.1f}" if magnitude > 0 else "")
            mag_item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(row, 3, mag_item)
            
            # Size
            self.targets_table.setItem(row, 4, QTableWidgetItem(target.get("size_info", "")))
            
            # Priority
            priority_item = QTableWidgetItem(target.get("priority", ""))
            priority_item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(row, 5, priority_item)
            
            # Status
            status_item = QTableWidgetItem(target.get("status", ""))
            status_item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(row, 6, status_item)
            
            # Best Months
            self.targets_table.setItem(row, 7, QTableWidgetItem(target.get("best_months", "")))
            
            # Date Added
            date_added = target.get("date_added", "")
            if date_added:
                try:
                    # Format date for display
                    date_obj = datetime.strptime(date_added, "%Y-%m-%d %H:%M:%S")
                    formatted_date = date_obj.strftime("%Y-%m-%d")
                except:
                    formatted_date = date_added
            else:
                formatted_date = ""
            
            date_item = QTableWidgetItem(formatted_date)
            date_item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(row, 8, date_item)
    
    def _calculate_best_months_for_all(self):
        """Calculate best viewing months for all targets based on user location"""
        try:
            # Get user location from database
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
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
        """Calculate best viewing months for a single target"""
        try:
            # Import astronomy libraries
            from astroplan import Observer, FixedTarget
            from astropy import units as u
            from astropy.coordinates import SkyCoord
            from astropy.time import Time
            from datetime import date, timedelta
            import calendar
            
            # Create observer and target
            observer = Observer(latitude=lat * u.deg, longitude=lon * u.deg, name="User Location")
            target = FixedTarget(coord=SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg), name="Target")
            
            # Check visibility for each month
            good_months = []
            today = date.today()
            
            for month in range(1, 13):
                # Check the 15th of each month as representative
                try:
                    check_date = date(today.year, month, 15)
                    midnight_time = Time(f"{check_date.year}-{check_date.month:02d}-{check_date.day:02d} 00:00:00", 
                                       location=observer.location)
                    
                    # Get sun and target positions at midnight
                    sun_altaz = observer.sun_altaz(midnight_time)
                    target_altaz = observer.altaz(midnight_time, target)
                    
                    # Target is good for this month if:
                    # 1. Sun is down (astronomical twilight or darker)
                    # 2. Target is reasonably high (>30 degrees altitude)
                    if sun_altaz.alt.degree < -12 and target_altaz.alt.degree > 30:
                        good_months.append(calendar.month_abbr[month])
                        
                except Exception as e:
                    logger.debug(f"Error checking month {month}: {e}")
                    continue
            
            # Format the result
            if good_months:
                # Group consecutive months for cleaner display
                return self._format_month_ranges(good_months)
            else:
                return "Not optimal from location"
                
        except ImportError:
            logger.error("Missing astronomy libraries for best months calculation")
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

    def add_target_from_dso(self, dso_data):
        """Add a target from DSO data (called from ObjectDetailWindow)"""
        dialog = AddTargetDialog(dso_data=dso_data, parent=self)
        if dialog.exec() == QDialog.Accepted:
            self._load_targets()


def main():
    """Main entry point for the application"""
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = DSOTargetListWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()