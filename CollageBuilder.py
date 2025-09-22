import os
from PIL import Image, ImageDraw, ImageFont
import math
from typing import List, Tuple, Optional


class CollageBuilder:
    def __init__(self, grid_width: int = 3, grid_height: int = 3,
                 cell_size: Tuple[int, int] = (300, 300),
                 spacing: int = 10, background_color: str = "white",
                 show_labels: bool = True, label_position: str = "Bottom Center"):
        """
        Initialize the CollageBuilder.

        Args:
            grid_width: Number of columns in the grid
            grid_height: Number of rows in the grid
            cell_size: Size of each cell (width, height) in pixels
            spacing: Spacing between images in pixels
            background_color: Background color of the collage
            show_labels: Whether to show DSO name labels on images
            label_position: Position of labels (Bottom Center, Top Center, etc.)
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.spacing = spacing
        self.background_color = background_color
        self.show_labels = show_labels
        self.label_position = label_position
        self.images = []
        self.image_paths = []
        self.dso_names = []
        
    def add_image(self, image_path: str, dso_name: str = "Unknown DSO") -> bool:
        """
        Add an image to the collage.

        Args:
            image_path: Path to the image file
            dso_name: Name of the DSO (Deep Sky Object) associated with this image

        Returns:
            True if image was successfully added, False otherwise
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return False
            
        if len(self.images) >= self.grid_width * self.grid_height:
            print(f"Error: Grid is full. Maximum {self.grid_width * self.grid_height} images allowed.")
            return False
            
        try:
            image = Image.open(image_path)
            # Resize image to fit cell while maintaining aspect ratio
            image = self._resize_image_to_fit(image, self.cell_size)
            self.images.append(image)
            self.image_paths.append(image_path)
            self.dso_names.append(dso_name)
            print(f"Added image: {os.path.basename(image_path)} ({dso_name})")
            return True
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return False
    
    def add_images_from_folder(self, folder_path: str, extensions: List[str] = None) -> int:
        """
        Add all images from a folder.
        
        Args:
            folder_path: Path to the folder containing images
            extensions: List of file extensions to include (default: common image formats)
            
        Returns:
            Number of images successfully added
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
            
        if not os.path.exists(folder_path):
            print(f"Error: Folder not found: {folder_path}")
            return 0
            
        added_count = 0
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in extensions):
                if self.add_image(os.path.join(folder_path, filename)):
                    added_count += 1
                    
        return added_count

    def add_images_from_collage_data(self, collage_images_list) -> int:
        """
        Add images from the collage images data structure.

        Args:
            collage_images_list: List of image data dictionaries from CollageBuilder UI

        Returns:
            Number of images successfully added
        """
        added_count = 0

        for image_data in collage_images_list:
            image_path = image_data.get('image_path', '')
            dso_name = image_data.get('dso_name', 'Unknown DSO')

            # Use integration_time as part of DSO identifier if available
            integration_time = image_data.get('integration_time', '')
            equipment = image_data.get('equipment', '')

            # Create a more descriptive name if we have additional info
            if integration_time or equipment:
                details = []
                if equipment:
                    details.append(equipment)
                if integration_time:
                    details.append(f"{integration_time}")
                if details:
                    dso_name = f"{dso_name} ({', '.join(details)})"

            if self.add_image(image_path, dso_name):
                added_count += 1

        return added_count

    def remove_image(self, index: int) -> bool:
        """
        Remove an image from the collage by index.
        
        Args:
            index: Index of the image to remove
            
        Returns:
            True if image was successfully removed, False otherwise
        """
        if 0 <= index < len(self.images):
            removed_path = self.image_paths.pop(index)
            removed_name = self.dso_names.pop(index) if index < len(self.dso_names) else "Unknown"
            self.images.pop(index)
            print(f"Removed image: {os.path.basename(removed_path)}")
            return True
        else:
            print(f"Error: Invalid index {index}. Valid range: 0-{len(self.images)-1}")
            return False
    
    def clear_images(self):
        """Clear all images from the collage."""
        self.images.clear()
        self.image_paths.clear()
        self.dso_names.clear()
        print("All images cleared from collage")

    def _group_images_by_path(self) -> dict:
        """
        Group DSO names by their image paths to identify shared images.

        Returns:
            Dictionary where keys are image paths and values are lists of DSO names
        """
        image_groups = {}
        for i, image_path in enumerate(self.image_paths):
            if i < len(self.dso_names):
                dso_name = self.dso_names[i]
                if image_path not in image_groups:
                    image_groups[image_path] = []
                image_groups[image_path].append(dso_name)
        return image_groups

    def _calculate_merged_layout(self) -> list:
        """
        Calculate layout with merged cells for shared images.

        Returns:
            List of dictionaries containing layout information for each cell.
            Each dict contains: 'image_path', 'dso_names', 'start_col', 'start_row', 'span_cols', 'span_rows'
        """
        image_groups = self._group_images_by_path()
        layout = []
        processed_paths = set()

        current_row = 0
        current_col = 0

        for i, image_path in enumerate(self.image_paths):
            if image_path in processed_paths:
                continue

            dso_names = image_groups[image_path]
            num_dsos = len(dso_names)

            # Determine cell span based on number of DSOs sharing the image
            if num_dsos == 1:
                span_cols = 1
                span_rows = 1
            elif num_dsos == 2:
                span_cols = 2
                span_rows = 1
            elif num_dsos <= 4:
                span_cols = 2
                span_rows = 2
            elif num_dsos <= 6:
                span_cols = 3
                span_rows = 2
            else:
                span_cols = 3
                span_rows = 3

            # Check if the span fits in current row
            if current_col + span_cols > self.grid_width:
                current_row += 1
                current_col = 0

            # Check if we have enough vertical space
            if current_row + span_rows > self.grid_height:
                # If we can't fit, use smaller span
                available_rows = self.grid_height - current_row
                available_cols = self.grid_width - current_col
                span_rows = min(span_rows, available_rows)
                span_cols = min(span_cols, available_cols)

            layout.append({
                'image_path': image_path,
                'dso_names': dso_names,
                'start_col': current_col,
                'start_row': current_row,
                'span_cols': span_cols,
                'span_rows': span_rows
            })

            processed_paths.add(image_path)

            # Move to next position
            current_col += span_cols
            if current_col >= self.grid_width:
                current_row += 1
                current_col = 0

        return layout

    def _resize_image_to_fit(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize image to fit within target size while maintaining aspect ratio.
        
        Args:
            image: PIL Image object
            target_size: Target size (width, height)
            
        Returns:
            Resized PIL Image object
        """
        # Calculate scaling factor to fit image within target size
        scale_x = target_size[0] / image.width
        scale_y = target_size[1] / image.height
        scale = min(scale_x, scale_y)
        
        # Calculate new size
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        # Resize image
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def create_collage(self, output_path: str = "collage.jpg", quality: int = 95) -> bool:
        """
        Create the collage and save it to a file.
        
        Args:
            output_path: Path where the collage will be saved
            quality: JPEG quality (1-100, only applies to JPEG format)
            
        Returns:
            True if collage was successfully created, False otherwise
        """
        if not self.images:
            print("Error: No images added to the collage")
            return False
            
        # Calculate canvas size
        canvas_width = (self.cell_size[0] * self.grid_width) + (self.spacing * (self.grid_width + 1))
        canvas_height = (self.cell_size[1] * self.grid_height) + (self.spacing * (self.grid_height + 1))
        
        # Create canvas
        canvas = Image.new('RGB', (canvas_width, canvas_height), self.background_color)
        draw = ImageDraw.Draw(canvas)

        # Try to load a font for labels
        try:
            if os.name == 'nt':  # Windows
                font = ImageFont.truetype("arial.ttf", 16)
            else:  # Linux/Mac
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Place images on canvas
        for i, image in enumerate(self.images):
            if i >= self.grid_width * self.grid_height:
                break

            # Calculate grid position
            col = i % self.grid_width
            row = i // self.grid_width

            # Calculate position on canvas (centered within cell)
            x = self.spacing + (col * (self.cell_size[0] + self.spacing))
            y = self.spacing + (row * (self.cell_size[1] + self.spacing))

            # Center image within cell
            x_offset = (self.cell_size[0] - image.width) // 2
            y_offset = (self.cell_size[1] - image.height) // 2

            canvas.paste(image, (x + x_offset, y + y_offset))

            # Add DSO name label if enabled
            if self.show_labels and i < len(self.dso_names):
                dso_name = self.dso_names[i]

                # Get text bounding box for positioning calculations
                bbox = draw.textbbox((0, 0), dso_name, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Calculate text position based on label_position setting
                if self.label_position == "Bottom Center":
                    text_x = x + x_offset + (image.width // 2) - (text_width // 2)
                    text_y = y + y_offset + image.height + 5
                elif self.label_position == "Top Center":
                    text_x = x + x_offset + (image.width // 2) - (text_width // 2)
                    text_y = y + y_offset - text_height - 5
                elif self.label_position == "Bottom Left":
                    text_x = x + x_offset + 5
                    text_y = y + y_offset + image.height + 5
                elif self.label_position == "Bottom Right":
                    text_x = x + x_offset + image.width - text_width - 5
                    text_y = y + y_offset + image.height + 5
                elif self.label_position == "Top Left":
                    text_x = x + x_offset + 5
                    text_y = y + y_offset - text_height - 5
                elif self.label_position == "Top Right":
                    text_x = x + x_offset + image.width - text_width - 5
                    text_y = y + y_offset - text_height - 5
                elif self.label_position == "Center Overlay":
                    text_x = x + x_offset + (image.width // 2) - (text_width // 2)
                    text_y = y + y_offset + (image.height // 2) - (text_height // 2)
                else:
                    # Default to bottom center
                    text_x = x + x_offset + (image.width // 2) - (text_width // 2)
                    text_y = y + y_offset + image.height + 5

                # Ensure text stays within cell bounds
                min_x = x + 2
                max_x = x + self.cell_size[0] - text_width - 2
                text_x = max(min_x, min(text_x, max_x))

                min_y = y + 2
                max_y = y + self.cell_size[1] - text_height - 2
                text_y = max(min_y, min(text_y, max_y))

                # Draw semi-transparent background box for better text visibility
                padding = 4
                box_x1 = text_x - padding
                box_y1 = text_y - padding
                box_x2 = text_x + text_width + padding
                box_y2 = text_y + text_height + padding

                # Create semi-transparent background
                box_overlay = Image.new('RGBA', (box_x2 - box_x1, box_y2 - box_y1), (0, 0, 0, 128))
                canvas.paste(box_overlay, (box_x1, box_y1), box_overlay)

                # Use white text with black outline for maximum visibility
                text_color = "white"
                outline_color = "black"

                # Draw text with outline for better visibility
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((text_x + dx, text_y + dy), dso_name, font=font, fill=outline_color)

                # Draw main text
                draw.text((text_x, text_y), dso_name, font=font, fill=text_color)
        
        # Save collage with appropriate format
        try:
            file_ext = output_path.lower().split('.')[-1]

            if file_ext in ['jpg', 'jpeg']:
                canvas.save(output_path, 'JPEG', quality=quality)
            elif file_ext == 'png':
                canvas.save(output_path, 'PNG')
            elif file_ext in ['tif', 'tiff']:
                canvas.save(output_path, 'TIFF')
            else:
                # Fallback to format detection
                canvas.save(output_path)

            print(f"Collage saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving collage: {str(e)}")
            return False

    def create_merged_collage(self, output_path: str = "collage.jpg", quality: int = 95) -> bool:
        """
        Create a collage with merged cells for DSOs sharing the same image.

        Args:
            output_path: Path where the collage will be saved
            quality: JPEG quality (1-100, only applies to JPEG format)

        Returns:
            True if collage was successfully created, False otherwise
        """
        if not self.images:
            print("Error: No images added to the collage")
            return False

        # Calculate merged layout
        layout = self._calculate_merged_layout()
        if not layout:
            print("Error: No valid layout calculated")
            return False

        # Calculate canvas size
        canvas_width = (self.cell_size[0] * self.grid_width) + (self.spacing * (self.grid_width + 1))
        canvas_height = (self.cell_size[1] * self.grid_height) + (self.spacing * (self.grid_height + 1))

        # Create canvas
        canvas = Image.new('RGB', (canvas_width, canvas_height), self.background_color)
        draw = ImageDraw.Draw(canvas)

        # Try to load fonts for labels (base and larger sizes)
        try:
            if os.name == 'nt':  # Windows
                base_font = ImageFont.truetype("arial.ttf", 16)
                large_font = ImageFont.truetype("arial.ttf", 20)
            else:  # Linux/Mac
                base_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                large_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            base_font = ImageFont.load_default()
            large_font = ImageFont.load_default()

        # Place images on canvas with merged cells
        for layout_item in layout:
            image_path = layout_item['image_path']
            dso_names = layout_item['dso_names']
            start_col = layout_item['start_col']
            start_row = layout_item['start_row']
            span_cols = layout_item['span_cols']
            span_rows = layout_item['span_rows']

            # Load and process the image
            try:
                image = Image.open(image_path)

                # Calculate merged cell size
                merged_width = (self.cell_size[0] * span_cols) + (self.spacing * (span_cols - 1))
                merged_height = (self.cell_size[1] * span_rows) + (self.spacing * (span_rows - 1))

                # Resize image to fit the merged cell
                image = self._resize_image_to_fit(image, (merged_width, merged_height))

                # Calculate position on canvas
                x = self.spacing + (start_col * (self.cell_size[0] + self.spacing))
                y = self.spacing + (start_row * (self.cell_size[1] + self.spacing))

                # Center image within merged cell
                x_offset = (merged_width - image.width) // 2
                y_offset = (merged_height - image.height) // 2

                canvas.paste(image, (x + x_offset, y + y_offset))

                # Add DSO name labels if enabled
                if self.show_labels and dso_names:
                    # Choose font size based on merged cell size
                    total_cells = span_cols * span_rows
                    if total_cells >= 4:  # Large merged cells get larger font
                        font = large_font
                    else:
                        font = base_font
                    # Create combined label for multiple DSOs
                    if len(dso_names) == 1:
                        label_text = dso_names[0]
                    else:
                        # For multiple DSOs, display all names for better visibility
                        if len(dso_names) == 2:
                            # For 2 DSOs, join with " & " for better readability
                            label_text = " & ".join(dso_names)
                        elif len(dso_names) <= 4:
                            # For 3-4 DSOs, use multi-line format
                            label_text = "\n".join(dso_names)
                        else:
                            # For 5+ DSOs, show first few and count
                            first_names = dso_names[:3]
                            remaining_count = len(dso_names) - 3
                            label_text = "\n".join(first_names) + f"\n+ {remaining_count} more"

                    # Get text bounding box for positioning calculations
                    bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Calculate text position based on label_position setting
                    # For merged cells with multiple DSOs, adjust positioning for better visibility
                    cell_width = span_cols * self.cell_size[0] + (span_cols - 1) * self.spacing
                    cell_height = span_rows * self.cell_size[1] + (span_rows - 1) * self.spacing

                    if self.label_position == "Bottom Center":
                        text_x = x + (cell_width // 2) - (text_width // 2)
                        text_y = y + cell_height + 5
                    elif self.label_position == "Top Center":
                        text_x = x + (cell_width // 2) - (text_width // 2)
                        text_y = y - text_height - 5
                    elif self.label_position == "Bottom Left":
                        text_x = x + 5
                        text_y = y + cell_height + 5
                    elif self.label_position == "Bottom Right":
                        text_x = x + cell_width - text_width - 5
                        text_y = y + cell_height + 5
                    elif self.label_position == "Top Left":
                        text_x = x + 5
                        text_y = y - text_height - 5
                    elif self.label_position == "Top Right":
                        text_x = x + cell_width - text_width - 5
                        text_y = y - text_height - 5
                    elif self.label_position == "Center Overlay":
                        text_x = x + (cell_width // 2) - (text_width // 2)
                        text_y = y + (cell_height // 2) - (text_height // 2)
                    else:
                        # Default to bottom center
                        text_x = x + (cell_width // 2) - (text_width // 2)
                        text_y = y + cell_height + 5

                    # Ensure text stays within merged cell bounds
                    min_x = x + 2
                    max_x = x + cell_width - text_width - 2
                    text_x = max(min_x, min(text_x, max_x))

                    min_y = y + 2
                    max_y = y + cell_height - text_height - 2
                    text_y = max(min_y, min(text_y, max_y))

                    # Draw semi-transparent background box for better text visibility
                    padding = 4
                    box_x1 = text_x - padding
                    box_y1 = text_y - padding
                    box_x2 = text_x + text_width + padding
                    box_y2 = text_y + text_height + padding

                    # Create semi-transparent background
                    box_overlay = Image.new('RGBA', (box_x2 - box_x1, box_y2 - box_y1), (0, 0, 0, 128))
                    canvas.paste(box_overlay, (box_x1, box_y1), box_overlay)

                    # Use white text with black outline for maximum visibility
                    text_color = "white"
                    outline_color = "black"

                    # Add text outline for better visibility
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                draw.text((text_x + dx, text_y + dy), label_text, font=font, fill=outline_color)

                    # Draw main text
                    draw.text((text_x, text_y), label_text, font=font, fill=text_color)

            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                continue

        # Save collage with appropriate format
        try:
            file_ext = output_path.lower().split('.')[-1]

            if file_ext in ['jpg', 'jpeg']:
                canvas.save(output_path, 'JPEG', quality=quality)
            elif file_ext == 'png':
                canvas.save(output_path, 'PNG')
            elif file_ext in ['tif', 'tiff']:
                canvas.save(output_path, 'TIFF')
            else:
                canvas.save(output_path)

            print(f"Merged collage saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving merged collage: {str(e)}")
            return False

    def preview_layout(self):
        """Print a preview of the current layout."""
        print(f"\nCollage Layout ({self.grid_width}x{self.grid_height}):")
        print(f"Cell size: {self.cell_size[0]}x{self.cell_size[1]} pixels")
        print(f"Spacing: {self.spacing} pixels")
        print(f"Images added: {len(self.images)}/{self.grid_width * self.grid_height}")
        
        if self.images:
            print("\nImages:")
            for i, path in enumerate(self.image_paths):
                row = i // self.grid_width
                col = i % self.grid_width
                print(f"  [{row},{col}] {os.path.basename(path)}")
        else:
            print("No images added yet")

    def preview_merged_layout(self):
        """Print a preview of the merged cell layout."""
        print(f"\nMerged Collage Layout ({self.grid_width}x{self.grid_height}):")
        print(f"Cell size: {self.cell_size[0]}x{self.cell_size[1]} pixels")
        print(f"Spacing: {self.spacing} pixels")
        print(f"Images added: {len(self.images)}")

        if not self.images:
            print("No images added yet")
            return

        # Show image groupings
        image_groups = self._group_images_by_path()
        print(f"\nImage groups: {len(image_groups)}")
        for image_path, dso_names in image_groups.items():
            print(f"  {os.path.basename(image_path)}: {len(dso_names)} DSO(s) - {', '.join(dso_names)}")

        # Show calculated layout
        layout = self._calculate_merged_layout()
        print(f"\nMerged layout: {len(layout)} cells")
        for i, layout_item in enumerate(layout):
            print(f"  Cell {i+1}: {os.path.basename(layout_item['image_path'])}")
            print(f"    Position: ({layout_item['start_col']}, {layout_item['start_row']})")
            print(f"    Span: {layout_item['span_cols']}x{layout_item['span_rows']}")
            print(f"    DSOs: {', '.join(layout_item['dso_names'])}")


# Import required Qt classes
try:
    from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                                   QLabel, QListWidget, QSpinBox, QGroupBox,
                                   QCheckBox, QSplitter, QWidget, QScrollArea,
                                   QColorDialog, QMessageBox, QInputDialog, QFileDialog,
                                   QProgressDialog, QGridLayout, QLineEdit, QComboBox)
    from PySide6.QtCore import Qt, QThread, Signal, QTimer
    from PySide6.QtGui import QColor, QDrag, QPixmap
    from PySide6.QtCore import QMimeData
    from PySide6.QtWidgets import QApplication

    # Import DatabaseManager (assuming it's available in the main application)
    try:
        from DatabaseManager import DatabaseManager
    except ImportError:
        # Fallback if DatabaseManager is not available as a separate module
        try:
            from Main import DatabaseManager
        except ImportError:
            DatabaseManager = None

    # Import logger
    try:
        import logging
        logger = logging.getLogger(__name__)
    except ImportError:
        # Create a simple logger fallback
        class SimpleLogger:
            def debug(self, msg): print(f"DEBUG: {msg}")
            def error(self, msg, exc_info=None): print(f"ERROR: {msg}")
        logger = SimpleLogger()

except ImportError as e:
    print(f"Warning: Could not import required Qt modules: {e}")


# --- Thumbnail Worker Thread ---
class ThumbnailWorker(QThread):
    """Worker thread for generating thumbnails in background"""

    # Signal emitted when a thumbnail is ready
    thumbnail_ready = Signal(int, int, QPixmap)  # row, col, pixmap
    thumbnail_error = Signal(int, int, str)      # row, col, error_message

    def __init__(self, image_requests, cache=None):
        super().__init__()
        self.image_requests = image_requests  # List of (row, col, image_path) tuples
        self.cache = cache  # ThumbnailCache instance
        self.cancelled = False

    def cancel(self):
        """Cancel the thumbnail generation"""
        self.cancelled = True

    def _load_fits_thumbnail(self, fits_path, colormap='gray'):
        """Load a FITS file and convert to QPixmap thumbnail with RGB color mapping - same as ObjectDetailWindow"""
        try:
            # Import required libraries
            from astropy.io import fits
            from astropy.visualization import simple_norm
            import numpy as np
            from PySide6.QtGui import QImage

            # Open FITS file
            with fits.open(fits_path) as hdul:
                # Get the primary image data (usually the first HDU with data)
                image_data = None
                for hdu in hdul:
                    if hdu.data is not None and len(hdu.data.shape) >= 2:
                        image_data = hdu.data
                        break

                if image_data is None:
                    return None

                # Handle different dimensionalities
                is_rgb = False
                if len(image_data.shape) > 2:
                    # Check if this is an RGB image (3 color planes)
                    if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                        # This is an RGB FITS image
                        is_rgb = True
                    elif len(image_data.shape) == 3 and image_data.shape[0] == 3:
                        # RGB planes are in first dimension, transpose
                        image_data = np.transpose(image_data, (1, 2, 0))
                        is_rgb = True
                    elif len(image_data.shape) == 3:
                        # Take the first 2D slice if it's a cube
                        image_data = image_data[0]
                    elif len(image_data.shape) == 4:
                        image_data = image_data[0, 0]
                    else:
                        return None

                # Normalize the data for display (handle NaN values)
                image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

                if is_rgb:
                    # Handle RGB FITS data - normalize each channel separately
                    normalized_data = np.zeros_like(image_data)

                    for channel in range(3):
                        channel_data = image_data[:, :, channel]
                        # Apply normalization to each color channel
                        try:
                            norm = simple_norm(channel_data, stretch='linear', percent=99.5)
                            normalized_data[:, :, channel] = norm(channel_data)
                        except Exception:
                            # Fallback to simple min-max normalization per channel
                            data_min, data_max = np.percentile(channel_data, [0.5, 99.5])
                            if data_max > data_min:
                                normalized_data[:, :, channel] = (channel_data - data_min) / (data_max - data_min)
                            else:
                                normalized_data[:, :, channel] = channel_data

                    # Clip to valid range
                    normalized_data = np.clip(normalized_data, 0, 1)

                    # Convert directly to 8-bit RGB (no false color mapping needed)
                    rgb_data = (normalized_data * 255).astype(np.uint8)

                    # Ensure the array is C-contiguous for QImage
                    if not rgb_data.flags['C_CONTIGUOUS']:
                        rgb_data = np.ascontiguousarray(rgb_data)

                    # Create QImage from RGB array
                    height, width, channels = rgb_data.shape
                    bytes_per_line = width * channels

                    qimage = QImage(rgb_data.data, width, height, bytes_per_line, QImage.Format_RGB888)

                else:
                    # Handle grayscale FITS data
                    # Apply simple normalization (linear stretch between percentiles)
                    try:
                        norm = simple_norm(image_data, stretch='linear', percent=99.5)
                        normalized_data = norm(image_data)
                    except Exception:
                        # Fallback to simple min-max normalization
                        data_min, data_max = np.percentile(image_data, [0.5, 99.5])
                        if data_max > data_min:
                            normalized_data = (image_data - data_min) / (data_max - data_min)
                        else:
                            normalized_data = image_data
                        normalized_data = np.clip(normalized_data, 0, 1)

                    # For grayscale data, apply color mapping if specified
                    if colormap == 'gray' or colormap == 'grey':
                        # Display as grayscale
                        image_8bit = (normalized_data * 255).astype(np.uint8)

                        # Ensure the array is C-contiguous for QImage
                        if not image_8bit.flags['C_CONTIGUOUS']:
                            image_8bit = np.ascontiguousarray(image_8bit)

                        height, width = image_8bit.shape
                        bytes_per_line = width
                        qimage = QImage(image_8bit.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                    else:
                        # Apply color mapping for better visualization
                        try:
                            import matplotlib.pyplot as plt
                            import matplotlib.cm as cm

                            # Apply a color map for better astronomical visualization
                            try:
                                cmap = cm.get_cmap(colormap)
                            except ValueError:
                                cmap = cm.get_cmap('viridis')

                            colored_data = cmap(normalized_data)

                            # Convert to 8-bit RGB
                            rgb_data = (colored_data[:, :, :3] * 255).astype(np.uint8)

                            # Ensure the array is C-contiguous for QImage
                            if not rgb_data.flags['C_CONTIGUOUS']:
                                rgb_data = np.ascontiguousarray(rgb_data)

                            # Create QImage from RGB array
                            height, width, channels = rgb_data.shape
                            bytes_per_line = width * channels

                            qimage = QImage(rgb_data.data, width, height, bytes_per_line, QImage.Format_RGB888)

                        except ImportError:
                            # Matplotlib not available, fallback to grayscale
                            image_8bit = (normalized_data * 255).astype(np.uint8)

                            # Ensure the array is C-contiguous for QImage
                            if not image_8bit.flags['C_CONTIGUOUS']:
                                image_8bit = np.ascontiguousarray(image_8bit)

                            height, width = image_8bit.shape
                            bytes_per_line = width
                            qimage = QImage(image_8bit.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                        except Exception:
                            # Color mapping failed, fallback to grayscale
                            image_8bit = (normalized_data * 255).astype(np.uint8)

                            # Ensure the array is C-contiguous for QImage
                            if not image_8bit.flags['C_CONTIGUOUS']:
                                image_8bit = np.ascontiguousarray(image_8bit)

                            height, width = image_8bit.shape
                            bytes_per_line = width
                            qimage = QImage(image_8bit.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

                # Convert to QPixmap
                return QPixmap.fromImage(qimage)

        except ImportError:
            # astropy not available
            return None
        except Exception:
            # Other FITS loading error
            return None

    def run(self):
        """Generate thumbnails in background"""
        for row, col, image_path in self.image_requests:
            if self.cancelled:
                break

            try:
                # Check cache first
                if self.cache:
                    cached_pixmap = self.cache.get(image_path)
                    if cached_pixmap:
                        self.thumbnail_ready.emit(row, col, cached_pixmap)
                        continue

                if os.path.exists(image_path):
                    # Check file size first
                    file_size = os.path.getsize(image_path)
                    if file_size == 0:
                        self.thumbnail_error.emit(row, col, "Empty File")
                        continue

                    # Get file extension
                    _, ext = os.path.splitext(image_path.lower())

                    # Try different loading methods based on file type
                    pixmap = None

                    # For FITS files, use the same method as ObjectDetailWindow
                    if ext in ['.fits', '.fit', '.fts']:
                        pixmap = self._load_fits_thumbnail(image_path)
                        if pixmap is None:
                            self.thumbnail_error.emit(row, col, "FITS Load\nError")
                            continue
                    else:
                        # Load regular image formats with better error handling - same approach as ObjectDetailWindow
                        from PySide6.QtGui import QImageReader

                        # Increase maximum allocation limit for large images (in MB) - same as ObjectDetailWindow
                        QImageReader.setAllocationLimit(512)

                        # First try standard QPixmap loading
                        pixmap = QPixmap(image_path)

                        # If Qt fails to load, try with QImageReader for better error reporting
                        if pixmap.isNull():
                            try:
                                reader = QImageReader(image_path)

                                # Check if reader can read the file first
                                if reader.canRead():
                                    # Try setting explicit format
                                    if ext in ['.jpg', '.jpeg']:
                                        reader.setFormat(b"JPEG")
                                    elif ext == '.png':
                                        reader.setFormat(b"PNG")
                                    elif ext == '.bmp':
                                        reader.setFormat(b"BMP")
                                    elif ext in ['.tiff', '.tif']:
                                        reader.setFormat(b"TIFF")

                                    image = reader.read()
                                    if not image.isNull():
                                        pixmap = QPixmap.fromImage(image)
                                else:
                                    # Log the specific error from QImageReader
                                    import logging
                                    logger = logging.getLogger(__name__)
                                    logger.warning(f"QImageReader cannot read file {image_path}, error: {reader.errorString()}")
                            except Exception as e:
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.warning(f"Error with QImageReader for {image_path}: {e}")
                                pass  # Fall through to error handling below

                    if pixmap and not pixmap.isNull():
                        # Scale the image to fit while maintaining aspect ratio
                        scaled_pixmap = pixmap.scaled(120, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                        # Cache the generated thumbnail
                        if self.cache:
                            self.cache.put(image_path, scaled_pixmap)

                        self.thumbnail_ready.emit(row, col, scaled_pixmap)
                    else:
                        # Log more detailed error info
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Failed to load image thumbnail: {image_path} (extension: {ext}, size: {file_size} bytes)")
                        error_msg = f"Invalid {ext.upper()}\nFormat"
                        self.thumbnail_error.emit(row, col, error_msg)
                else:
                    self.thumbnail_error.emit(row, col, "File Not\nFound")

            except Exception as e:
                # More detailed error reporting
                error_msg = f"Error:\n{str(e)[:20]}..."
                self.thumbnail_error.emit(row, col, error_msg)

# --- Thumbnail Cache ---
class ThumbnailCache:
    """Cache for storing generated thumbnails to avoid regeneration"""

    def __init__(self, max_size=100):
        self._cache = {}  # image_path -> QPixmap
        self._max_size = max_size
        self._access_order = []  # Track access order for LRU eviction

    def get(self, image_path):
        """Get cached thumbnail for image path"""
        if image_path in self._cache:
            # Move to end (most recently used)
            if image_path in self._access_order:
                self._access_order.remove(image_path)
            self._access_order.append(image_path)
            return self._cache[image_path]
        return None

    def put(self, image_path, pixmap):
        """Store thumbnail in cache"""
        if image_path in self._cache:
            # Update existing entry
            if image_path in self._access_order:
                self._access_order.remove(image_path)
        elif len(self._cache) >= self._max_size:
            # Remove least recently used item
            if self._access_order:
                lru_path = self._access_order.pop(0)
                if lru_path in self._cache:
                    del self._cache[lru_path]

        self._cache[image_path] = pixmap
        self._access_order.append(image_path)

    def clear(self):
        """Clear all cached thumbnails"""
        self._cache.clear()
        self._access_order.clear()

    def size(self):
        """Return current cache size"""
        return len(self._cache)

# --- Draggable Cell Widget ---
class DraggableCell(QWidget):
    """A draggable cell widget for grid reordering"""

    def __init__(self, row, col, index, parent_window):
        super().__init__()
        self.row = row
        self.col = col
        self.index = index
        self.parent_window = parent_window
        self.image_data = None
        self.has_image = False

        self.setAcceptDrops(True)

    def set_image_data(self, image_data):
        """Set the image data for this cell"""
        self.image_data = image_data
        self.has_image = image_data is not None

    def mousePressEvent(self, event):
        """Handle mouse press for drag operations"""
        if event.button() == Qt.LeftButton and self.has_image:
            self.drag_start_position = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Start drag operation when mouse moves with button held"""
        if (not (event.buttons() & Qt.LeftButton) or
            not self.has_image or
            not hasattr(self, 'drag_start_position')):
            return

        if ((event.position().toPoint() - self.drag_start_position).manhattanLength() <
            QApplication.startDragDistance()):
            return

        # Start drag operation
        drag = QDrag(self)
        mimeData = QMimeData()

        # Store the source cell information
        mimeData.setText(f"{self.row},{self.col},{self.index}")
        drag.setMimeData(mimeData)

        # Create a simple drag pixmap
        pixmap = self.grab()
        drag.setPixmap(pixmap)
        drag.setHotSpot(self.drag_start_position)

        # Execute the drag
        result = drag.exec(Qt.MoveAction)

    def dragEnterEvent(self, event):
        """Accept drag events from other cells"""
        if event.mimeData().hasText():
            event.acceptProposedAction()
            # Visual feedback - highlight the drop target
            self.setStyleSheet(self.styleSheet() + "border: 3px solid #2196F3;")

    def dragLeaveEvent(self, event):
        """Remove visual feedback when drag leaves"""
        # Restore original styling
        self.parent_window._populate_collage_images()

    def dropEvent(self, event):
        """Handle drop event - swap images between cells"""
        if event.mimeData().hasText():
            # Parse source cell information
            source_info = event.mimeData().text().split(',')
            if len(source_info) == 3:
                source_row = int(source_info[0])
                source_col = int(source_info[1])
                source_index = int(source_info[2])

                # Don't do anything if dropping on the same cell
                if source_index == self.index:
                    event.acceptProposedAction()
                    return

                # Swap the images in the parent window's project data
                self.parent_window._swap_images(source_index, self.index)

            event.acceptProposedAction()

# --- Collage Generation Worker Thread ---
class CollageGenerationWorker(QThread):
    """Worker thread for generating collages without blocking the UI"""

    # Signals
    progress_updated = Signal(int, str)  # progress percentage, status message
    finished = Signal(bool, str, str)   # success, output_path, error_message

    def __init__(self, collage_images, project_settings, output_path, use_merged_cells=False):
        super().__init__()
        self.collage_images = collage_images
        self.project_settings = project_settings
        self.output_path = output_path
        self.use_merged_cells = use_merged_cells
        self.cancelled = False

    def cancel(self):
        """Cancel the generation process"""
        self.cancelled = True

    def run(self):
        """Run the collage generation in the background"""
        try:
            # Emit initial progress
            self.progress_updated.emit(0, "Initializing collage builder...")

            if self.cancelled:
                return

            # Create CollageBuilder instance
            collage_builder = CollageBuilder(
                grid_width=self.project_settings['grid_width'],
                grid_height=self.project_settings['grid_height'],
                cell_size=(self.project_settings['cell_size'], self.project_settings['cell_size']),
                spacing=self.project_settings['spacing'],
                background_color=self.project_settings['background_color'],
                show_labels=self.project_settings.get('show_labels', True),
                label_position=self.project_settings.get('label_position', 'Bottom Center')
            )

            # Add images with progress updates
            total_images = len(self.collage_images)
            max_images = self.project_settings['grid_width'] * self.project_settings['grid_height']
            images_to_process = min(total_images, max_images)

            # Use the new method to add images from collage data
            self.progress_updated.emit(20, "Processing collage images...")
            added_count = collage_builder.add_images_from_collage_data(self.collage_images[:max_images])

            # Update progress for the processing phase
            for i in range(min(len(self.collage_images), max_images)):
                if self.cancelled:
                    return
                progress = 20 + int((i * 60) / min(len(self.collage_images), max_images))
                filename = os.path.basename(self.collage_images[i].get('image_path', '')) if i < len(self.collage_images) else 'Unknown'
                self.progress_updated.emit(progress, f"Processing image {i+1}: {filename}")

            if self.cancelled:
                return

            if added_count == 0:
                self.finished.emit(False, "", "No valid images could be loaded for the collage.")
                return

            # Generate the collage
            self.progress_updated.emit(85, "Creating collage layout...")

            if self.cancelled:
                return

            # Create the collage
            self.progress_updated.emit(90, "Saving collage file...")
            if self.use_merged_cells:
                success = collage_builder.create_merged_collage(self.output_path, quality=95)
            else:
                success = collage_builder.create_collage(self.output_path, quality=95)

            if self.cancelled:
                return

            self.progress_updated.emit(100, "Collage generation complete!")

            if success:
                self.finished.emit(True, self.output_path, "")
            else:
                self.finished.emit(False, "", "Failed to create collage file.")

        except Exception as e:
            if not self.cancelled:
                error_msg = f"Error during collage generation: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.finished.emit(False, "", error_msg)

# --- Image Selection Dialog ---
class ImageSelectionDialog(QDialog):
    """Dialog for selecting images from the database"""

    def __init__(self, available_images, parent=None):
        super().__init__(parent)

        # Remove duplicates from available_images based on image_path
        seen_paths = set()
        unique_images = []
        for img in available_images:
            img_path = img.get('image_path')
            if img_path and img_path not in seen_paths:
                seen_paths.add(img_path)
                unique_images.append(img)

        self.available_images = unique_images
        self.selected_images = []


        # Also log the first few images to console for easier debugging
        logger.info(f"ImageSelectionDialog showing {len(unique_images)} images:")
        for i, img in enumerate(self.available_images[:10]):  # Show first 10
            dso_name = img.get('dso_name', 'Unknown')
            filename = os.path.basename(img.get('image_path', '')) or 'No file'
            logger.info(f"  {i+1}. {dso_name} - {filename}")

        self.setWindowTitle("Select Images")
        self.setModal(True)
        self.resize(600, 400)

        self._setup_ui()

    def _get_dso_name_for_image(self, image_data):
        """Get the DSO name for an image by looking up its dsodetailid"""
        try:
            # First try to get from cached data if available
            if 'dso_name' in image_data:
                cached_name = image_data['dso_name']
                logger.debug(f"Found cached DSO name '{cached_name}' for image ID {image_data.get('id')}")
                # TEMPORARY: Clear bad cache and force fresh lookup if cached name looks wrong
                if cached_name and not ' ' in cached_name and cached_name.isdigit():
                    logger.debug(f"Clearing bad cache value '{cached_name}' - forcing fresh lookup")
                    del image_data['dso_name']
                else:
                    return cached_name

            # Get dsodetailid from image
            image_id = image_data.get('id')
            if not image_id or not DatabaseManager:
                return "Unknown DSO"

            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                # Get dsodetailid from the image
                cursor.execute("SELECT dsodetailid FROM userimages WHERE id = ?", (image_id,))
                result = cursor.fetchone()

                if not result or not result[0]:
                    return "Unknown DSO"

                dsodetailid = result[0]

                # Get the primary DSO name using the same logic as shown in the selected lines
                cursor.execute("""
                    SELECT c.catalogue, c.designation
                    FROM cataloguenr c
                    WHERE c.dsodetailid = ?
                    ORDER BY CASE c.catalogue
                                   WHEN 'M' THEN 1
                                   WHEN 'NGC' THEN 2
                                   WHEN 'IC' THEN 3
                                   ELSE 4
                               END, c.designation
                    LIMIT 1
                """, (dsodetailid,))

                dso_result = cursor.fetchone()
                logger.debug(f"ImageSelection DSO query result for dsodetailid {dsodetailid}: {dso_result}")
                if dso_result:
                    catalogue, designation = dso_result
                    logger.debug(f"ImageSelection: Parsed - catalogue='{catalogue}', designation='{designation}'")
                    dso_name = f"{catalogue} {designation}"
                    logger.debug(f"ImageSelection: Formatted dso_name='{dso_name}'")
                    # Cache the result
                    image_data['dso_name'] = dso_name
                    return dso_name

                logger.debug(f"ImageSelection: No DSO cataloguenr found for dsodetailid {dsodetailid}")
                return "Unknown DSO"

        except Exception as e:
            logger.error(f"Error getting DSO name for image: {str(e)}", exc_info=True)
            return "Unknown DSO"

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Header
        self.header_label = QLabel(f"Select images to add to collage ({len(self.available_images)} available):")
        layout.addWidget(self.header_label)

        # Search and filter controls
        search_layout = QHBoxLayout()

        # Search field
        search_layout.addWidget(QLabel("Search:"))
        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Search by DSO name or filename...")
        self.search_field.textChanged.connect(self._filter_images)
        search_layout.addWidget(self.search_field)

        # Clear search button
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_search)
        clear_btn.setMaximumWidth(60)
        search_layout.addWidget(clear_btn)

        layout.addLayout(search_layout)

        # Filter controls
        filter_layout = QHBoxLayout()

        # Equipment filter
        filter_layout.addWidget(QLabel("Equipment:"))
        self.equipment_filter = QComboBox()
        self.equipment_filter.addItem("All Equipment")
        # Populate equipment options from available images
        equipment_set = set()
        for img in self.available_images:
            equipment = img.get('equipment', '').strip()
            if equipment:
                equipment_set.add(equipment)
        for equipment in sorted(equipment_set):
            self.equipment_filter.addItem(equipment)
        self.equipment_filter.currentTextChanged.connect(self._filter_images)
        filter_layout.addWidget(self.equipment_filter)

        # Date filter
        filter_layout.addWidget(QLabel("Date:"))
        self.date_filter = QComboBox()
        self.date_filter.addItem("All Dates")
        # Populate date options from available images
        date_set = set()
        for img in self.available_images:
            date_taken = img.get('date_taken', '').strip()
            if date_taken:
                # Extract year from date for grouping
                try:
                    if len(date_taken) >= 4:
                        year = date_taken[:4]
                        date_set.add(year)
                except:
                    pass
        for year in sorted(date_set, reverse=True):
            self.date_filter.addItem(year)
        self.date_filter.currentTextChanged.connect(self._filter_images)
        filter_layout.addWidget(self.date_filter)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Images list with checkboxes
        scroll_area = QScrollArea()
        self.images_widget = QWidget()
        self.images_layout = QVBoxLayout(self.images_widget)

        # Store checkboxes for filtering
        self.checkboxes = []

        # Create all checkboxes initially
        self._create_checkboxes()

        scroll_area.setWidget(self.images_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # Selection buttons
        selection_buttons = QHBoxLayout()

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        selection_buttons.addWidget(select_all_btn)

        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self._select_none)
        selection_buttons.addWidget(select_none_btn)

        selection_buttons.addStretch()
        layout.addLayout(selection_buttons)

        # Dialog buttons
        button_layout = QHBoxLayout()

        ok_btn = QPushButton("Add Selected")
        ok_btn.clicked.connect(self._accept_selection)
        ok_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def _create_checkboxes(self):
        """Create checkboxes for all images"""
        self.checkboxes = []

        for i, image_data in enumerate(self.available_images):
            image_path = image_data.get('image_path', '')
            filename = os.path.basename(image_path)

            # Get DSO name for this image
            dso_name = self._get_dso_name_for_image(image_data)

            # Debug: Check what we actually got
            logger.debug(f"ImageSelection: _get_dso_name_for_image returned: '{dso_name}' (type: {type(dso_name)})")

            # Add some metadata if available
            metadata = []
            if image_data.get('integration_time'):
                metadata.append(f"Int: {image_data['integration_time']}")
            if image_data.get('equipment'):
                metadata.append(f"Eq: {image_data['equipment']}")
            if image_data.get('date_taken'):
                metadata.append(f"Date: {image_data['date_taken']}")

            # Format display text with DSO name first, then filename
            display_text = f"{dso_name} - {filename}"
            if metadata:
                display_text += f" ({', '.join(metadata)})"

            # Debug: Log the lookup result
            logger.debug(f"Image {i}: DSO lookup result: '{dso_name}' for image ID {image_data.get('id')} ({filename})")

            checkbox = QCheckBox(display_text)
            checkbox.setProperty("image_index", i)
            checkbox.setProperty("image_data", image_data)
            checkbox.setProperty("dso_name", dso_name)
            checkbox.setProperty("filename", filename)
            checkbox.setToolTip(f"DSO: {dso_name}\nFile: {filename}\nPath: {image_path}")

            self.checkboxes.append(checkbox)
            self.images_layout.addWidget(checkbox)

    def _filter_images(self):
        """Filter images based on search text and filter selections"""
        search_text = self.search_field.text().lower()
        equipment_filter = self.equipment_filter.currentText()
        date_filter = self.date_filter.currentText()

        visible_count = 0

        for checkbox in self.checkboxes:
            image_data = checkbox.property("image_data")
            dso_name = checkbox.property("dso_name").lower()
            filename = checkbox.property("filename").lower()

            # Check search text
            search_match = (search_text == "" or
                          search_text in dso_name or
                          search_text in filename)

            # Check equipment filter
            equipment_match = (equipment_filter == "All Equipment" or
                             image_data.get('equipment', '') == equipment_filter)

            # Check date filter
            date_match = True
            if date_filter != "All Dates":
                image_date = image_data.get('date_taken', '')
                date_match = image_date.startswith(date_filter) if image_date else False

            # Show/hide checkbox based on all filters
            should_show = search_match and equipment_match and date_match
            checkbox.setVisible(should_show)

            if should_show:
                visible_count += 1

        # Update header with visible count
        self.header_label.setText(f"Select images to add to collage ({visible_count} of {len(self.available_images)} shown):")

    def _clear_search(self):
        """Clear the search field"""
        self.search_field.clear()

    def _select_all(self):
        """Select all visible images"""
        for checkbox in self.checkboxes:
            if checkbox.isVisible():
                checkbox.setChecked(True)

    def _select_none(self):
        """Deselect all visible images"""
        for checkbox in self.checkboxes:
            if checkbox.isVisible():
                checkbox.setChecked(False)

    def _accept_selection(self):
        """Accept the selection and close dialog"""
        self.selected_images = []

        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                image_index = checkbox.property("image_index")
                if image_index is not None and image_index < len(self.available_images):
                    self.selected_images.append(self.available_images[image_index])

        if not self.selected_images:
            QMessageBox.information(self, "No Selection", "Please select at least one image.")
            return

        self.accept()

    def get_selected_images(self):
        """Return the selected images"""
        return self.selected_images

# --- Collage Builder Window ---
class CollageBuilderWindow(QDialog):
    """Window for creating and managing multiple collages"""

    def __init__(self, user_images: list, dso_name: str, dsodetailid: int, parent=None):
        super().__init__(parent)

        # Remove duplicates from user_images based on image_path
        seen_paths = set()
        unique_images = []
        for img in user_images:
            img_path = img.get('image_path')
            if img_path and img_path not in seen_paths:
                seen_paths.add(img_path)
                unique_images.append(img)

        self.user_images = unique_images
        self.dso_name = dso_name
        self.dsodetailid = dsodetailid  # Store the dsodetailid
        self.collage_projects = {}  # Dict to store multiple collage projects
        self.current_project = None

        # Initialize worker thread and progress dialog variables
        self.worker_thread = None
        self.progress_dialog = None

        # Initialize thumbnail cache
        self.thumbnail_cache = ThumbnailCache(max_size=200)  # Cache up to 200 thumbnails

        # Initialize thumbnail worker thread
        self.thumbnail_worker = None
        self.thumbnail_labels = {}  # Dictionary to store thumbnail label references

        # Debug output
        if len(user_images) != len(unique_images):
            logger.debug(f"Removed {len(user_images) - len(unique_images)} duplicate images from user_images list")

        self.setWindowTitle(f"Collage Builder - CosmosCollection")
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.resize(1400, 900)
        self.setMinimumSize(800, 600)

        self._setup_ui()
        self._load_saved_projects()  # Load existing projects from database

        # Create default project if no projects exist
        if not self.collage_projects:
            self._create_new_project("Default Collage")

    def show(self):
        """Override show to refresh data when window becomes visible"""
        # Refresh data to ensure we have the latest image paths and project data
        if hasattr(self, 'collage_projects') and self.collage_projects:
            self.refresh_data()
        super().show()

    def _setup_ui(self):
        """Set up the user interface"""
        from PySide6.QtWidgets import (QSplitter, QListWidget, QSpinBox,
                                       QColorDialog, QScrollArea)

        main_layout = QVBoxLayout(self)

        # Top toolbar
        toolbar = QHBoxLayout()

        new_project_btn = QPushButton("New Collage")
        new_project_btn.clicked.connect(lambda: self._create_new_project())
        toolbar.addWidget(new_project_btn)

        rename_project_btn = QPushButton("Rename")
        rename_project_btn.clicked.connect(self._rename_project)
        toolbar.addWidget(rename_project_btn)

        delete_project_btn = QPushButton("Delete")
        delete_project_btn.clicked.connect(self._delete_project)
        toolbar.addWidget(delete_project_btn)

        toolbar.addStretch()

        # Merge cells checkbox
        self.merge_cells_checkbox = QCheckBox("Merge cells for shared images")
        self.merge_cells_checkbox.setToolTip("When checked, DSO objects that share the same image will be merged into larger cells")
        self.merge_cells_checkbox.setStyleSheet("QCheckBox { color: #ffffff; }")
        toolbar.addWidget(self.merge_cells_checkbox)

        generate_collage_btn = QPushButton("Generate Collage")
        generate_collage_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        generate_collage_btn.clicked.connect(self._generate_collage)
        generate_collage_btn.setToolTip("Create and save the collage image file")
        toolbar.addWidget(generate_collage_btn)


        main_layout.addLayout(toolbar)

        # Main content area with splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel - Project list and settings
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Project list
        left_layout.addWidget(QLabel("Collage Projects:"))
        self.project_list = QListWidget()
        self.project_list.currentItemChanged.connect(self._on_project_selected)
        left_layout.addWidget(self.project_list)

        # Settings group
        settings_group = QGroupBox("Collage Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Grid size
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("Grid Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 10)
        self.width_spin.setValue(3)
        self.width_spin.valueChanged.connect(self._update_current_project)
        grid_layout.addWidget(self.width_spin)

        grid_layout.addWidget(QLabel("Height:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 10)
        self.height_spin.setValue(3)
        self.height_spin.valueChanged.connect(self._update_current_project)
        grid_layout.addWidget(self.height_spin)
        settings_layout.addLayout(grid_layout)

        # Cell size
        cell_layout = QHBoxLayout()
        cell_layout.addWidget(QLabel("Cell Size:"))
        self.cell_size_spin = QSpinBox()
        self.cell_size_spin.setRange(100, 800)
        self.cell_size_spin.setValue(400)
        self.cell_size_spin.setSuffix(" px")
        self.cell_size_spin.valueChanged.connect(self._update_current_project)
        cell_layout.addWidget(self.cell_size_spin)
        settings_layout.addLayout(cell_layout)

        # Spacing
        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("Spacing:"))
        self.spacing_spin = QSpinBox()
        self.spacing_spin.setRange(0, 50)
        self.spacing_spin.setValue(20)
        self.spacing_spin.setSuffix(" px")
        self.spacing_spin.valueChanged.connect(self._update_current_project)
        spacing_layout.addWidget(self.spacing_spin)
        settings_layout.addLayout(spacing_layout)

        # Background color
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Background:"))
        self.color_button = QPushButton("Black")
        self.color_button.setStyleSheet("QPushButton { background-color: black; color: white; }")
        self.color_button.clicked.connect(self._choose_background_color)
        color_layout.addWidget(self.color_button)
        settings_layout.addLayout(color_layout)

        # Label settings
        label_layout = QVBoxLayout()

        # Enable/disable labels
        self.show_labels_checkbox = QCheckBox("Show DSO Labels")
        self.show_labels_checkbox.setChecked(True)  # Default to enabled
        self.show_labels_checkbox.toggled.connect(self._update_current_project)
        label_layout.addWidget(self.show_labels_checkbox)

        # Label position
        position_layout = QHBoxLayout()
        position_layout.addWidget(QLabel("Label Position:"))
        self.label_position_combo = QComboBox()
        self.label_position_combo.addItems([
            "Bottom Center",
            "Top Center",
            "Bottom Left",
            "Bottom Right",
            "Top Left",
            "Top Right",
            "Center Overlay"
        ])
        self.label_position_combo.currentTextChanged.connect(self._update_current_project)
        position_layout.addWidget(self.label_position_combo)
        label_layout.addLayout(position_layout)

        settings_layout.addLayout(label_layout)

        left_layout.addWidget(settings_group)

        # Save Project button
        save_project_btn = QPushButton("Save Project")
        save_project_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        save_project_btn.clicked.connect(self._save_current_collage)
        save_project_btn.setToolTip("Save collage project to database")
        left_layout.addWidget(save_project_btn)

        left_panel.setMaximumWidth(300)
        splitter.addWidget(left_panel)

        # Right panel - Collage Images Management
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        right_layout.addWidget(QLabel("Images in Collage:"))

        # Simple grid for collage arrangement
        self.scroll_area = QScrollArea()
        self.images_widget = QWidget()
        self.images_layout = QGridLayout(self.images_widget)  # Use grid layout instead of vertical
        self.images_layout.setSpacing(15)  # Increased spacing to show borders better
        self.scroll_area.setWidget(self.images_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_layout.addWidget(self.scroll_area)

        # Image management buttons
        img_buttons = QHBoxLayout()

        add_images_btn = QPushButton("Add Images...")
        add_images_btn.clicked.connect(self._add_images_from_database)
        add_images_btn.setToolTip("Add images from database to this collage")
        add_images_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        img_buttons.addWidget(add_images_btn)

        remove_selected_btn = QPushButton("Remove Selected")
        remove_selected_btn.clicked.connect(self._remove_selected_images)
        remove_selected_btn.setToolTip("Remove selected images from collage")
        remove_selected_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        img_buttons.addWidget(remove_selected_btn)

        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self._clear_all_images)
        clear_all_btn.setToolTip("Remove all images from collage")
        img_buttons.addWidget(clear_all_btn)

        right_layout.addLayout(img_buttons)

        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])

        main_layout.addWidget(splitter)

        self._populate_collage_images()

    def _populate_collage_images(self):
        """Populate the grid with images currently in the collage"""
        # Cancel any running thumbnail generation
        if self.thumbnail_worker and self.thumbnail_worker.isRunning():
            self.thumbnail_worker.cancel()
            self.thumbnail_worker.wait()

        # Clear thumbnail labels dictionary
        self.thumbnail_labels.clear()

        # Clear existing widgets
        for i in reversed(range(self.images_layout.count())):
            item = self.images_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setParent(None)

        if not self.current_project:
            return

        project = self.collage_projects[self.current_project]
        collage_images = project.get('collage_images', [])
        grid_width = project.get('grid_width', 3)
        grid_height = project.get('grid_height', 3)
        max_cells = grid_width * grid_height

        if not collage_images:
            no_images_label = QLabel("No images in collage")
            no_images_label.setStyleSheet("color: #888; font-style: italic; padding: 10px; font-size: 14px;")
            no_images_label.setAlignment(Qt.AlignCenter)
            self.images_layout.addWidget(no_images_label, 0, 0, 1, grid_width)
        else:
            # Create grid cells to match the collage layout
            for row in range(grid_height):
                for col in range(grid_width):
                    cell_index = row * grid_width + col

                    # Create a draggable cell widget
                    cell_widget = DraggableCell(row, col, cell_index, self)
                    cell_widget.setMinimumSize(180, 180)  # Make cells bigger for images
                    cell_widget.setMaximumSize(220, 220)  # Make cells bigger for images
                    cell_layout = QVBoxLayout(cell_widget)
                    cell_layout.setContentsMargins(6, 6, 6, 6)
                    cell_layout.setSpacing(3)

                    if cell_index < len(collage_images):
                        # Cell has an image
                        image_data = collage_images[cell_index]
                        image_path = image_data.get('image_path', '')
                        filename = os.path.basename(image_path)
                        dso_name = self._get_dso_name_for_image(image_data)

                        # Position and checkbox in one line
                        top_layout = QHBoxLayout()
                        pos_label = QLabel(f"[{row},{col}]")
                        pos_label.setStyleSheet("font-size: 10px; color: #888; font-weight: bold;")
                        checkbox = QCheckBox()
                        checkbox.setProperty("collage_index", cell_index)
                        top_layout.addWidget(pos_label)
                        top_layout.addStretch()
                        top_layout.addWidget(checkbox)
                        cell_layout.addLayout(top_layout)

                        # Image thumbnail - start with loading placeholder
                        image_label = QLabel()
                        image_label.setAlignment(Qt.AlignCenter)
                        image_label.setMinimumSize(100, 80)
                        image_label.setMaximumSize(120, 100)
                        image_label.setStyleSheet("border: 1px solid #666; background-color: #444; color: #888; font-size: 10px;")
                        image_label.setText("Loading...")

                        # Store reference to the label for later thumbnail updates
                        cell_key = f"{row},{col}"
                        self.thumbnail_labels[cell_key] = image_label

                        cell_layout.addWidget(image_label)

                        # Add small stretch to center the DSO name below the image
                        cell_layout.addStretch(1)

                        # DSO name - centered below the image
                        dso_label = QLabel(dso_name)
                        dso_label.setAlignment(Qt.AlignCenter)
                        dso_label.setStyleSheet("font-size: 11px; color: #FFFFFF; font-weight: bold;")
                        dso_label.setWordWrap(True)
                        dso_label.setToolTip(f"DSO: {dso_name}\nFile: {filename}\nPath: {image_path}")
                        cell_layout.addWidget(dso_label)

                        # Add stretch to push DSO name to center of remaining space
                        cell_layout.addStretch(1)

                        # Set image data and styling for occupied cells
                        cell_widget.set_image_data(image_data)
                        cell_widget.setStyleSheet("""
                            DraggableCell {
                                border: 3px solid #4CAF50;
                                border-radius: 8px;
                                background-color: #2e4a2e;
                                margin: 2px;
                            }
                        """)
                    else:
                        # Empty cell - simpler layout with dark theme colors
                        pos_label = QLabel(f"[{row},{col}]")
                        pos_label.setAlignment(Qt.AlignCenter)
                        pos_label.setStyleSheet("font-size: 11px; color: #888; font-weight: bold;")
                        cell_layout.addWidget(pos_label)

                        empty_label = QLabel("Empty")
                        empty_label.setAlignment(Qt.AlignCenter)
                        empty_label.setStyleSheet("font-size: 14px; color: #666; font-style: italic;")
                        cell_layout.addWidget(empty_label)

                        cell_layout.addStretch()  # Push content to top

                        # Set no image data and styling for empty cells
                        cell_widget.set_image_data(None)
                        cell_widget.setStyleSheet("""
                            DraggableCell {
                                border: 3px dashed #555;
                                border-radius: 8px;
                                background-color: #3a3a3a;
                                margin: 2px;
                            }
                        """)

                    # Add cell to grid
                    self.images_layout.addWidget(cell_widget, row, col)

        # Start background thumbnail generation for occupied cells
        self._start_thumbnail_generation(collage_images, grid_width, grid_height)

    def _start_thumbnail_generation(self, collage_images, grid_width, grid_height):
        """Start background thumbnail generation"""
        # Cancel any existing thumbnail worker
        if self.thumbnail_worker and self.thumbnail_worker.isRunning():
            self.thumbnail_worker.cancel()
            self.thumbnail_worker.wait()

        # Prepare list of images to load
        image_requests = []
        for i, image_data in enumerate(collage_images):
            if i >= grid_width * grid_height:
                break

            row = i // grid_width
            col = i % grid_width
            image_path = image_data.get('image_path', '')
            if image_path:
                image_requests.append((row, col, image_path))

        if image_requests:
            # Start the thumbnail worker with cache
            self.thumbnail_worker = ThumbnailWorker(image_requests, self.thumbnail_cache)
            self.thumbnail_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
            self.thumbnail_worker.thumbnail_error.connect(self._on_thumbnail_error)
            self.thumbnail_worker.start()

    def _on_thumbnail_ready(self, row, col, pixmap):
        """Handle when a thumbnail is ready"""
        cell_key = f"{row},{col}"
        if cell_key in self.thumbnail_labels:
            label = self.thumbnail_labels[cell_key]
            label.setPixmap(pixmap)
            label.setText("")  # Clear the "Loading..." text
            label.setStyleSheet("border: 1px solid #666; background-color: #444;")

    def _on_thumbnail_error(self, row, col, error_message):
        """Handle thumbnail loading errors"""
        cell_key = f"{row},{col}"
        if cell_key in self.thumbnail_labels:
            label = self.thumbnail_labels[cell_key]
            label.setText(error_message.replace(" ", "\n"))  # Add line break for better fit
            label.setStyleSheet("border: 1px solid #666; background-color: #444; color: #999; font-size: 10px;")

    def _swap_images(self, source_index, target_index):
        """Swap images between two grid positions"""
        if not self.current_project:
            return

        project = self.collage_projects[self.current_project]
        collage_images = project.get('collage_images', [])

        # Extend the list if needed to accommodate the indices
        max_index = max(source_index, target_index)
        while len(collage_images) <= max_index:
            collage_images.append(None)

        # Get the images at both positions (could be None)
        source_image = collage_images[source_index] if source_index < len(collage_images) else None
        target_image = collage_images[target_index] if target_index < len(collage_images) else None

        # Swap the images
        collage_images[source_index] = target_image
        collage_images[target_index] = source_image

        # Remove trailing None values to keep the list clean
        while collage_images and collage_images[-1] is None:
            collage_images.pop()

        # Update the display
        self._populate_collage_images()

    def _on_images_reordered(self, reordered_images):
        """Handle when images are reordered in the grid"""
        if not self.current_project:
            return

        project = self.collage_projects[self.current_project]

        # Update the project's image order based on grid positions
        # reordered_images is a list of image_data dicts in grid order (left-to-right, top-to-bottom)
        project['collage_images'] = list(reordered_images)  # Create copy to avoid reference issues

        logger.debug(f"Images reordered in project '{self.current_project}': {len(reordered_images)} images")

    def _scroll_images_to_top(self):
        """Scroll the images list to the top"""
        if hasattr(self, 'scroll_area') and self.scroll_area:
            # Immediately scroll to top
            v_scrollbar = self.scroll_area.verticalScrollBar()
            h_scrollbar = self.scroll_area.horizontalScrollBar()

            v_scrollbar.setValue(0)
            h_scrollbar.setValue(0)

            # Ensure the widget is properly sized first
            self.images_widget.adjustSize()
            self.scroll_area.update()

            # Multiple delayed attempts to ensure scrolling works
            QTimer.singleShot(10, lambda: v_scrollbar.setValue(0))
            QTimer.singleShot(50, lambda: v_scrollbar.setValue(0))
            QTimer.singleShot(100, lambda: v_scrollbar.setValue(0))

    def _get_dso_name_for_image(self, image_data):
        """Get the DSO name for an image by looking up its dsodetailid"""
        try:
            # First try to get from cached data if available
            if 'dso_name' in image_data:
                cached_name = image_data['dso_name']
                logger.debug(f"Found cached DSO name '{cached_name}' for image ID {image_data.get('id')}")
                # TEMPORARY: Clear bad cache and force fresh lookup if cached name looks wrong
                if cached_name and not ' ' in cached_name and cached_name.isdigit():
                    logger.debug(f"Clearing bad cache value '{cached_name}' - forcing fresh lookup")
                    del image_data['dso_name']
                else:
                    return cached_name

            # Get dsodetailid from image
            image_id = image_data.get('id')
            if not image_id or not DatabaseManager:
                return "Unknown DSO"

            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                # Get dsodetailid from the image
                cursor.execute("SELECT dsodetailid FROM userimages WHERE id = ?", (image_id,))
                result = cursor.fetchone()

                if not result or not result[0]:
                    return "Unknown DSO"

                dsodetailid = result[0]

                # Get the primary DSO name using the same logic as shown in the selected lines
                cursor.execute("""
                    SELECT c.catalogue, c.designation
                    FROM cataloguenr c
                    WHERE c.dsodetailid = ?
                    ORDER BY CASE c.catalogue
                                   WHEN 'M' THEN 1
                                   WHEN 'NGC' THEN 2
                                   WHEN 'IC' THEN 3
                                   ELSE 4
                               END, c.designation
                    LIMIT 1
                """, (dsodetailid,))

                dso_result = cursor.fetchone()
                if dso_result:
                    catalogue, designation = dso_result
                    dso_name = f"{catalogue} {designation}"
                    # Cache the result
                    image_data['dso_name'] = dso_name
                    return dso_name

                return "Unknown DSO"

        except Exception as e:
            logger.error(f"Error getting DSO name for image: {str(e)}")
            return "Unknown DSO"

    def refresh_data(self):
        """Refresh the user images and reload projects - call this when data might have changed"""
        logger.debug("Refreshing CollageBuilder data...")

        # Clear existing projects to force reload
        self.collage_projects.clear()
        self.project_list.clear()

        # Reload projects from database
        self._load_saved_projects()

        # If we had a current project selected, try to reselect it
        if hasattr(self, '_last_selected_project') and self._last_selected_project:
            for i in range(self.project_list.count()):
                if self.project_list.item(i).text() == self._last_selected_project:
                    self.project_list.setCurrentRow(i)
                    break

        logger.debug("CollageBuilder data refresh complete")

    def _load_saved_projects(self):
        """Load all saved collage projects from the database for this DSO"""
        if not DatabaseManager:
            logger.warning("DatabaseManager not available - cannot load saved projects")
            return

        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                # Debug: Log what we're searching for
                logger.debug(f"Loading saved collages for DSO: {self.dso_name}, dsodetailid: {self.dsodetailid}")

                # First, let's see ALL collages in the database for debugging
                cursor.execute("SELECT id, name, dsodetailid FROM usercollages ORDER BY modified_date DESC")
                all_collages = cursor.fetchall()
                logger.debug(f"All collages in database: {all_collages}")

                # Get all saved collages for this DSO
                if self.dsodetailid is not None:
                    cursor.execute("""
                        SELECT id, name, grid_width, grid_height, cell_size,
                               spacing, background_color, created_date, modified_date
                        FROM usercollages
                        WHERE dsodetailid = ?
                        ORDER BY modified_date DESC
                    """, (self.dsodetailid,))
                    logger.debug(f"Searching for collages with dsodetailid = {self.dsodetailid}")
                elif self.dso_name == "All DSO Images":
                    # Special case: Show ALL collages when viewing "All DSO Images"
                    cursor.execute("""
                        SELECT id, name, grid_width, grid_height, cell_size,
                               spacing, background_color, created_date, modified_date
                        FROM usercollages
                        ORDER BY modified_date DESC
                    """)
                    logger.debug("Searching for ALL collages (All DSO Images view)")
                else:
                    # Handle case where dsodetailid is None but not "All DSO Images"
                    cursor.execute("""
                        SELECT id, name, grid_width, grid_height, cell_size,
                               spacing, background_color, created_date, modified_date
                        FROM usercollages
                        WHERE dsodetailid IS NULL
                        ORDER BY modified_date DESC
                    """)
                    logger.debug("Searching for collages with dsodetailid = NULL")

                saved_collages = cursor.fetchall()
                logger.debug(f"Found {len(saved_collages)} matching collages: {[c[1] for c in saved_collages]}")

                for collage_data in saved_collages:
                    (collage_id, name, grid_width, grid_height, cell_size,
                     spacing, background_color, created_date, modified_date) = collage_data

                    logger.debug(f"Loading collage: {name} (ID: {collage_id})")

                    # Load images for this collage
                    cursor.execute("""
                        SELECT ui.id, ui.image_path, ui.integration_time, ui.equipment,
                               ui.date_taken, ui.notes, uci.position_index,
                               (SELECT c2.designation FROM cataloguenr c2
                                WHERE c2.dsodetailid = ui.dsodetailid
                                ORDER BY
                                    CASE c2.catalogue
                                        WHEN 'M' THEN 1
                                        WHEN 'NGC' THEN 2
                                        WHEN 'IC' THEN 3
                                        ELSE 4
                                    END,
                                    c2.designation
                                LIMIT 1) as dso_designation
                        FROM usercollageimages uci
                        JOIN userimages ui ON uci.userimage_id = ui.id
                        WHERE uci.collage_id = ?
                        ORDER BY uci.position_index
                    """, (collage_id,))

                    image_data = cursor.fetchall()
                    collage_images = []
                    logger.debug(f"Found {len(image_data)} images for collage {name}")

                    for img_row in image_data:
                        (img_id, img_path, integration_time, equipment,
                         date_taken, notes, position_index, dso_designation) = img_row

                        # Check if image file exists and log status
                        file_exists = img_path and os.path.exists(img_path)
                        if not file_exists:
                            logger.warning(f"Missing image file for ID {img_id}: {img_path}")
                        else:
                            logger.debug(f"Found image file for ID {img_id}: {os.path.basename(img_path)}")

                        image_data_dict = {
                            'id': img_id,
                            'image_path': img_path,
                            'integration_time': integration_time,
                            'equipment': equipment,
                            'date_taken': date_taken,
                            'notes': notes,
                            'dso_name': dso_designation or 'Unknown DSO'
                        }

                        # Add flag for missing files to help with debugging
                        if not file_exists:
                            image_data_dict['missing_file'] = True

                        collage_images.append(image_data_dict)

                    # Create project data
                    project_data = {
                        'name': name,
                        'grid_width': grid_width,
                        'grid_height': grid_height,
                        'cell_size': cell_size,
                        'spacing': spacing,
                        'background_color': background_color,
                        'collage_images': collage_images,
                        'database_id': collage_id,  # Store database ID for updates
                        'created_date': created_date,
                        'modified_date': modified_date
                    }

                    # Debug: Log what we loaded from database
                    logger.debug(f"Loaded project '{name}': grid_width={grid_width}, grid_height={grid_height}, cell_size={cell_size}")

                    # Add to projects and UI
                    self.collage_projects[name] = project_data
                    self.project_list.addItem(name)
                    logger.debug(f"Added project to UI: {name}")

                if saved_collages:
                    logger.debug(f"Loaded {len(saved_collages)} saved collage projects for {self.dso_name}")
                    # Select the most recently modified project
                    self.project_list.setCurrentRow(0)
                else:
                    logger.debug(f"No saved collage projects found for {self.dso_name} with dsodetailid {self.dsodetailid}")

        except Exception as e:
            logger.error(f"Error loading saved collage projects: {str(e)}", exc_info=True)
            # Don't show error to user - just continue with empty projects list

    def _create_new_project(self, name=None):
        """Create a new collage project"""
        from PySide6.QtWidgets import QInputDialog

        if name is None:
            name, ok = QInputDialog.getText(self, "New Collage Project",
                                            "Enter collage name:", text=f"Collage {len(self.collage_projects) + 1}")
            if not ok or not name.strip():
                return
            name = name.strip()

        # Ensure name is a string (in case it was passed as something else)
        name = str(name).strip()

        if not name:  # If name is empty after stripping
            return

        # Check if name already exists
        if name in self.collage_projects:
            QMessageBox.warning(self, "Name Exists", "A collage with this name already exists.")
            return

        # Create new project
        project_data = {
            'name': name,
            'grid_width': 3,
            'grid_height': 3,
            'cell_size': 400,
            'spacing': 20,
            'background_color': 'black',
            'show_labels': True,
            'label_position': 'Bottom Center',
            'collage_images': []  # List of image data dictionaries
        }

        self.collage_projects[name] = project_data
        self.project_list.addItem(name)

        # Select the new project
        items = self.project_list.findItems(name, Qt.MatchExactly)
        if items:
            self.project_list.setCurrentItem(items[0])

    def _rename_project(self):
        """Rename the current project"""
        if not self.current_project:
            return

        from PySide6.QtWidgets import QInputDialog

        new_name, ok = QInputDialog.getText(self, "Rename Collage",
                                            "Enter new name:", text=self.current_project)
        if not ok or not new_name.strip():
            return

        new_name = new_name.strip()
        if new_name == self.current_project:
            return

        if new_name in self.collage_projects:
            QMessageBox.warning(self, "Name Exists", "A collage with this name already exists.")
            return

        # Update database if project has been saved
        project = self.collage_projects[self.current_project]
        if project.get('database_id'):
            try:
                with DatabaseManager().get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE usercollages SET name = ? WHERE id = ?",
                                 (new_name, project['database_id']))
                    conn.commit()
            except Exception as e:
                QMessageBox.critical(self, "Database Error", f"Failed to rename project in database: {str(e)}")
                return

        # Rename project in memory
        self.collage_projects[new_name] = self.collage_projects.pop(self.current_project)
        self.collage_projects[new_name]['name'] = new_name

        # Update list
        current_row = self.project_list.currentRow()
        self.project_list.takeItem(current_row)
        self.project_list.insertItem(current_row, new_name)
        self.project_list.setCurrentRow(current_row)

        self.current_project = new_name

    def _delete_project(self):
        """Delete the current project"""
        if not self.current_project or len(self.collage_projects) <= 1:
            QMessageBox.information(self, "Cannot Delete", "Cannot delete the last project.")
            return

        reply = QMessageBox.question(self, "Delete Project",
                                     f"Are you sure you want to delete '{self.current_project}'?")
        if reply != QMessageBox.Yes:
            return

        # Delete from database if project has been saved
        project = self.collage_projects[self.current_project]
        if project.get('database_id'):
            try:
                with DatabaseManager().get_connection() as conn:
                    cursor = conn.cursor()
                    # Delete related images first (foreign key constraint)
                    cursor.execute("DELETE FROM usercollageimages WHERE collage_id = ?", (project['database_id'],))
                    # Delete the collage project
                    cursor.execute("DELETE FROM usercollages WHERE id = ?", (project['database_id'],))
                    conn.commit()
            except Exception as e:
                QMessageBox.critical(self, "Database Error", f"Failed to delete project from database: {str(e)}")
                return

        # Remove project from memory
        del self.collage_projects[self.current_project]

        # Remove from list and select another
        current_row = self.project_list.currentRow()
        self.project_list.takeItem(current_row)

        if self.project_list.count() > 0:
            self.project_list.setCurrentRow(0)
        else:
            self.current_project = None

    def _on_project_selected(self, current, previous):
        """Handle project selection change"""
        if current is None:
            return

        project_name = current.text()
        self.current_project = project_name
        self._last_selected_project = project_name  # Remember selection for refresh
        project = self.collage_projects[project_name]

        # Update UI with project settings
        logger.debug(f"Setting UI for project '{project_name}': width={project['grid_width']}, height={project['grid_height']}, cell_size={project['cell_size']}")

        # Temporarily disconnect signals to prevent interference during loading
        self.width_spin.valueChanged.disconnect()
        self.height_spin.valueChanged.disconnect()
        self.cell_size_spin.valueChanged.disconnect()
        self.spacing_spin.valueChanged.disconnect()
        self.show_labels_checkbox.toggled.disconnect()
        self.label_position_combo.currentTextChanged.disconnect()

        # Set the values
        self.width_spin.setValue(project['grid_width'])
        self.height_spin.setValue(project['grid_height'])
        self.cell_size_spin.setValue(project['cell_size'])
        self.spacing_spin.setValue(project['spacing'])
        self._update_color_button(project['background_color'])

        # Set label settings with defaults
        self.show_labels_checkbox.setChecked(project.get('show_labels', True))
        label_position = project.get('label_position', 'Bottom Center')
        index = self.label_position_combo.findText(label_position)
        if index >= 0:
            self.label_position_combo.setCurrentIndex(index)

        # Reconnect signals
        self.width_spin.valueChanged.connect(self._update_current_project)
        self.height_spin.valueChanged.connect(self._update_current_project)
        self.cell_size_spin.valueChanged.connect(self._update_current_project)
        self.spacing_spin.valueChanged.connect(self._update_current_project)
        self.show_labels_checkbox.toggled.connect(self._update_current_project)
        self.label_position_combo.currentTextChanged.connect(self._update_current_project)

        # Debug: Verify what values were actually set
        logger.debug(f"UI values after setting: width={self.width_spin.value()}, height={self.height_spin.value()}, cell_size={self.cell_size_spin.value()}")

        # Refresh the collage images display
        try:
            self._populate_collage_images()
            logger.debug("_populate_collage_images returned successfully")
        except Exception as e:
            logger.error(f"Error in _populate_collage_images: {str(e)}", exc_info=True)
            raise

        # Scroll to top of the images list
        try:
            logger.debug("About to call _scroll_images_to_top")
            self._scroll_images_to_top()
            logger.debug("_scroll_images_to_top completed")
        except Exception as e:
            logger.error(f"Error in _scroll_images_to_top: {str(e)}", exc_info=True)
            raise

    def _update_current_project(self):
        """Update current project settings"""
        if not self.current_project:
            return

        project = self.collage_projects[self.current_project]
        old_width = project.get('grid_width', 3)
        old_height = project.get('grid_height', 3)

        project['grid_width'] = self.width_spin.value()
        project['grid_height'] = self.height_spin.value()
        project['cell_size'] = self.cell_size_spin.value()
        project['spacing'] = self.spacing_spin.value()
        project['show_labels'] = self.show_labels_checkbox.isChecked()
        project['label_position'] = self.label_position_combo.currentText()

        # Update images layout if grid size changed
        new_width = project['grid_width']
        new_height = project['grid_height']
        if old_width != new_width or old_height != new_height:
            # Repopulate to adjust image positions for the new grid size
            self._populate_collage_images()

    def _choose_background_color(self):
        """Choose background color"""
        color = QColorDialog.getColor(QColor("black"), self, "Choose Background Color")
        if color.isValid():
            color_name = color.name()
            if self.current_project:
                self.collage_projects[self.current_project]['background_color'] = color_name
            self._update_color_button(color_name)

    def _update_color_button(self, color_name):
        """Update the color button appearance"""
        self.color_button.setText(color_name.title())
        text_color = "white" if color_name in ["black", "#000000"] else "black"
        self.color_button.setStyleSheet(f"QPushButton {{ background-color: {color_name}; color: {text_color}; }}")

    def _add_images_from_database(self):
        """Show dialog to select images from the userimages database"""
        if not self.current_project:
            QMessageBox.warning(self, "No Project", "Please select a project first.")
            return

        # Get current collage images to filter them out (by DSO Detail ID, not image path)
        project = self.collage_projects[self.current_project]
        current_dso_detail_ids = set()

        # Get DSO Detail IDs of images already in collage
        for img in project.get('collage_images', []):
            img_id = img.get('id')
            if img_id:
                try:
                    with DatabaseManager().get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT dsodetailid FROM userimages WHERE id = ?", (img_id,))
                        result = cursor.fetchone()
                        if result and result[0]:
                            current_dso_detail_ids.add(result[0])
                except Exception as e:
                    logger.error(f"Error getting DSO Detail ID for image {img_id}: {e}")

        # Get fresh image data from database instead of using cached self.user_images
        fresh_user_images = []
        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                # Get the full data
                cursor.execute("""
                    SELECT ui.id, ui.image_path, ui.integration_time, ui.equipment,
                           ui.date_taken, ui.notes,
                           (SELECT
                                CASE
                                    WHEN c2.catalogue = 'M' THEN 'M ' || c2.designation
                                    WHEN c2.catalogue = 'NGC' THEN 'NGC ' || c2.designation
                                    WHEN c2.catalogue = 'IC' THEN 'IC ' || c2.designation
                                    WHEN c2.catalogue = 'Sh2' THEN 'Sh2-' || c2.designation
                                    ELSE c2.catalogue || ' ' || c2.designation
                                END
                            FROM cataloguenr c2
                            WHERE c2.dsodetailid = ui.dsodetailid
                            ORDER BY
                                CASE c2.catalogue
                                    WHEN 'M' THEN 1
                                    WHEN 'NGC' THEN 2
                                    WHEN 'IC' THEN 3
                                    WHEN 'Sh2' THEN 4
                                    ELSE 5
                                END,
                                CAST(c2.designation AS INTEGER)
                            LIMIT 1) as dso_designation
                    FROM userimages ui
                    ORDER BY ui.id DESC
                """)
                rows = cursor.fetchall()

                for row in rows:
                    # Get DSO Detail ID for this image
                    img_id = row[0]
                    cursor.execute("SELECT dsodetailid FROM userimages WHERE id = ?", (img_id,))
                    dso_result = cursor.fetchone()
                    dso_detail_id = dso_result[0] if dso_result else None

                    fresh_user_images.append({
                        'id': row[0],
                        'image_path': row[1],
                        'integration_time': row[2] or '',
                        'equipment': row[3] or '',
                        'date_taken': row[4] or '',
                        'notes': row[5] or '',
                        'dso_name': row[6] or 'Unknown DSO',
                        'dso_detail_id': dso_detail_id
                    })
        except Exception as e:
            logger.error(f"Error fetching fresh image data: {str(e)}")
            QMessageBox.critical(self, "Database Error", f"Failed to load current images: {str(e)}")
            return

        # Filter available images (exclude DSOs already in collage, but allow same image file)
        available_images = []
        for image_data in fresh_user_images:
            dso_detail_id = image_data.get('dso_detail_id')
            if dso_detail_id not in current_dso_detail_ids:
                available_images.append(image_data)

        if not available_images:
            QMessageBox.information(self, "No Images",
                f"No additional DSOs available to add.\n\n"
                f"Total images in database: {len(fresh_user_images)}\n"
                f"DSOs already in collage: {len(current_dso_detail_ids)}\n"
                f"DSOs available to add: {len(available_images)}")
            return

        # Create selection dialog
        dialog = ImageSelectionDialog(available_images, self)
        if dialog.exec() == QDialog.Accepted:
            selected_images = dialog.get_selected_images()
            if selected_images:
                # Add selected images to current collage (create copies to avoid reference issues)
                for img in selected_images:
                    project['collage_images'].append(dict(img))  # Create a copy
                self._populate_collage_images()
                self._scroll_images_to_top()  # Scroll to see the new images
                QMessageBox.information(self, "Images Added",
                                      f"Added {len(selected_images)} images to the collage.")

    def _remove_selected_images(self):
        """Remove selected images from the collage"""
        if not self.current_project:
            return

        project = self.collage_projects[self.current_project]
        collage_images = project.get('collage_images', [])

        # Get selected checkboxes from grid cells
        selected_indices = []
        for i in range(self.images_layout.count()):
            item = self.images_layout.itemAt(i)
            if item:
                cell_widget = item.widget()
                if cell_widget:
                    # Look for checkbox in the cell widget
                    for child in cell_widget.findChildren(QCheckBox):
                        if child.isChecked():
                            collage_index = child.property("collage_index")
                            if collage_index is not None:
                                selected_indices.append(collage_index)

        if not selected_indices:
            QMessageBox.information(self, "No Selection", "Please select images to remove by checking the boxes.")
            return

        # Remove selected images (in reverse order to maintain indices)
        for index in sorted(selected_indices, reverse=True):
            if 0 <= index < len(collage_images):
                collage_images.pop(index)

        # Update the display
        self._populate_collage_images()
        self._scroll_images_to_top()
        QMessageBox.information(self, "Images Removed",
                              f"Removed {len(selected_indices)} images from the collage.")

    def _clear_all_images(self):
        """Remove all images from the collage"""
        if not self.current_project:
            return

        project = self.collage_projects[self.current_project]
        collage_images = project.get('collage_images', [])

        if not collage_images:
            QMessageBox.information(self, "No Images", "Collage is already empty.")
            return

        reply = QMessageBox.question(self, "Clear All Images",
                                   "Are you sure you want to remove all images from this collage?")
        if reply == QMessageBox.Yes:
            project['collage_images'] = []
            self._populate_collage_images()
            self._scroll_images_to_top()  # Scroll to top after clearing
            QMessageBox.information(self, "Images Cleared", "All images removed from the collage.")


    def _generate_collage(self):
        """Generate and save the actual collage image file using a background thread"""
        if not self.current_project:
            QMessageBox.warning(self, "No Project", "Please select a project to generate.")
            return

        project = self.collage_projects[self.current_project]
        collage_images = project.get('collage_images', [])

        if not collage_images:
            QMessageBox.warning(self, "No Images", "Please add some images to the collage before generating.")
            return

        # Get output file path from user
        output_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Collage As",
            f"{project['name']}.jpg",
            "JPEG (*.jpg);;JPEG (*.jpeg);;PNG (*.png);;TIFF (*.tif)"
        )

        if not output_path:
            return  # User cancelled

        # Start the threaded generation process
        use_merged_cells = self.merge_cells_checkbox.isChecked()
        self._start_generation_thread(collage_images, project, output_path, use_merged_cells)

    def _start_generation_thread(self, collage_images, project_settings, output_path, use_merged_cells=False):
        """Start the collage generation in a background thread with progress dialog"""

        # Create and configure progress dialog
        self.progress_dialog = QProgressDialog("Initializing...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Generating Collage")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)  # Show immediately
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)

        # Create worker thread
        self.worker_thread = CollageGenerationWorker(collage_images, project_settings, output_path, use_merged_cells)

        # Connect signals
        self.worker_thread.progress_updated.connect(self._on_progress_updated)
        self.worker_thread.finished.connect(self._on_generation_finished)
        self.progress_dialog.canceled.connect(self._on_generation_cancelled)

        # Start the worker thread
        self.worker_thread.start()

        # Show progress dialog
        self.progress_dialog.show()

    def _on_progress_updated(self, percentage, message):
        """Handle progress updates from the worker thread"""
        if self.progress_dialog and not self.progress_dialog.wasCanceled():
            try:
                self.progress_dialog.setValue(percentage)
                self.progress_dialog.setLabelText(message)
            except (AttributeError, RuntimeError):
                # Dialog was destroyed or became invalid
                pass

    def _on_generation_cancelled(self):
        """Handle cancellation of the generation process"""
        # First disconnect signals to prevent further updates
        if self.worker_thread:
            try:
                self.worker_thread.progress_updated.disconnect()
                self.worker_thread.finished.disconnect()
            except (AttributeError, RuntimeError):
                pass

        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.cancel()
            self.worker_thread.wait(3000)  # Wait up to 3 seconds for thread to finish
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()  # Force terminate if needed

        # Clean up progress dialog
        if self.progress_dialog:
            try:
                self.progress_dialog.close()
            except (AttributeError, RuntimeError):
                pass
            self.progress_dialog = None

    def _on_generation_finished(self, success, output_path, error_message):
        """Handle completion of the collage generation"""
        # Close progress dialog safely
        if self.progress_dialog:
            try:
                self.progress_dialog.close()
            except (AttributeError, RuntimeError):
                pass
            self.progress_dialog = None

        # Clean up worker thread
        if self.worker_thread:
            try:
                self.worker_thread.deleteLater()
            except (AttributeError, RuntimeError):
                pass
            self.worker_thread = None

        if success:
            # Show success message with details
            project = self.collage_projects[self.current_project]
            collage_images = project.get('collage_images', [])
            max_images = project['grid_width'] * project['grid_height']
            images_used = min(len(collage_images), max_images)

            info_text = f"Collage created successfully!\n\n"
            info_text += f"File: {os.path.basename(output_path)}\n"
            info_text += f"Grid: {project['grid_width']}×{project['grid_height']}\n"
            info_text += f"Images: {images_used}/{len(collage_images)} used\n"
            info_text += f"Cell Size: {project['cell_size']}×{project['cell_size']} px"

            # Ask if user wants to open the file location
            reply = QMessageBox.information(
                self, "Collage Created", info_text,
                QMessageBox.Open | QMessageBox.Ok,
                QMessageBox.Ok
            )

            if reply == QMessageBox.Open:
                self._open_file_location(output_path)

            logger.debug(f"Generated collage: {output_path}")

        else:
            # Show error message
            QMessageBox.critical(self, "Generation Failed",
                               f"Failed to generate collage:\n\n{error_message}")

    def _open_file_location(self, file_path):
        """Open the file location in the system's file explorer"""
        try:
            import subprocess
            import platform

            folder_path = os.path.dirname(file_path)
            if platform.system() == "Windows":
                subprocess.run(["explorer", folder_path])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", folder_path])
            else:  # Linux
                subprocess.run(["xdg-open", folder_path])
        except Exception as e:
            logger.error(f"Could not open file location: {e}")
            QMessageBox.information(self, "File Location",
                                  f"Collage saved to:\n{file_path}")

    def closeEvent(self, event):
        """Handle window close event - clean up any running threads"""
        if self.worker_thread and self.worker_thread.isRunning():
            # Cancel the worker thread
            self.worker_thread.cancel()
            self.worker_thread.wait(2000)  # Wait up to 2 seconds
            if self.worker_thread.isRunning():
                self.worker_thread.terminate()

        if self.progress_dialog:
            self.progress_dialog.close()

        super().closeEvent(event)

    def _add_debug_button(self):
        """Add debug button to show all collages (temporary)"""
        debug_btn = QPushButton("Debug: Show All Collages")
        debug_btn.clicked.connect(self._debug_show_all_collages)
        debug_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")

        # Find the toolbar layout and add the button
        if hasattr(self, 'layout'):
            # Try to find the first layout with buttons
            for i in range(self.layout().count()):
                item = self.layout().itemAt(i)
                if hasattr(item, 'layout') and isinstance(item.layout(), QHBoxLayout):
                    item.layout().addWidget(debug_btn)
                    break

    def _debug_show_all_collages(self):
        """Debug method to show all collages in database regardless of DSO"""
        if not DatabaseManager:
            QMessageBox.warning(self, "Debug", "DatabaseManager not available")
            return

        try:
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                # Get ALL collages in the database
                cursor.execute("""
                    SELECT id, name, dsodetailid, grid_width, grid_height, cell_size,
                           spacing, background_color, created_date, modified_date
                    FROM usercollages
                    ORDER BY modified_date DESC
                """)

                all_collages = cursor.fetchall()

                # Create debug info string
                debug_info = f"Total collages in database: {len(all_collages)}\n\n"
                debug_info += f"Current DSO: {self.dso_name} (dsodetailid: {self.dsodetailid})\n\n"

                for collage in all_collages:
                    (cid, name, dsoid, grid_w, grid_h, cell_size, spacing, bg_color, created, modified) = collage
                    debug_info += f"ID: {cid}, Name: '{name}', DSO ID: {dsoid}\n"
                    debug_info += f"  Grid: {grid_w}x{grid_h}, Cell: {cell_size}, Modified: {modified}\n\n"

                # Show in message box
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Debug: All Collages")
                msg_box.setText(debug_info)
                msg_box.setDetailedText("This shows all collages in the usercollages table.")
                msg_box.exec()

        except Exception as e:
            QMessageBox.critical(self, "Debug Error", f"Error querying database: {str(e)}")

    def _save_current_collage(self):
        """Save the current collage project to database"""
        from datetime import datetime

        if not self.current_project:
            QMessageBox.warning(self, "No Project", "Please select a project to save.")
            return

        project = self.collage_projects[self.current_project]
        collage_images = project.get('collage_images', [])

        if not collage_images:
            QMessageBox.warning(self, "No Images", "Please select some images for the project before saving.")
            return

        # Use the current project name for saving
        save_name = project['name']

        try:
            # Use the stored dsodetailid (can be None for general collages)
            dsodetailid = self.dsodetailid
            collage_images = project.get('collage_images', [])

            # Special handling for "All DSO Images" view
            if self.dso_name == "All DSO Images" and dsodetailid is None:
                # For "All DSO Images", we need to determine the dsodetailid from the images
                # Use the first image's dsodetailid, or allow user to choose
                if collage_images:
                    # Try to get dsodetailid from the first image
                    first_image_id = collage_images[0].get('id')
                    if first_image_id:
                        with DatabaseManager().get_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT dsodetailid FROM userimages WHERE id = ?", (first_image_id,))
                            result = cursor.fetchone()
                            if result:
                                dsodetailid = result[0]
                                logger.debug(f"Using dsodetailid {dsodetailid} from first image for All DSO Images collage")

            current_time = datetime.now().isoformat()

            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()

                # Check if this is an update to existing project or a new save
                existing_id = project.get('database_id')
                if existing_id and save_name == project['name']:
                    # This is an update to the existing project
                    collage_id = existing_id
                    cursor.execute("""
                        UPDATE usercollages
                        SET grid_width = ?, grid_height = ?, cell_size = ?,
                            spacing = ?, background_color = ?, modified_date = ?
                        WHERE id = ?
                    """, (project['grid_width'], project['grid_height'], project['cell_size'],
                          project['spacing'], project['background_color'], current_time, collage_id))

                    # Clear existing image associations
                    cursor.execute("DELETE FROM usercollageimages WHERE collage_id = ?", (collage_id,))

                else:
                    # Check if a project with this name already exists (for new saves or renames)
                    if dsodetailid is not None:
                        cursor.execute("""
                            SELECT id FROM usercollages
                            WHERE dsodetailid = ? AND name = ?
                        """, (dsodetailid, save_name))
                    else:
                        cursor.execute("""
                            SELECT id FROM usercollages
                            WHERE dsodetailid IS NULL AND name = ?
                        """, (save_name,))
                    existing = cursor.fetchone()

                    if existing:
                        reply = QMessageBox.question(self, "Project Exists",
                                                     f"A project named '{save_name}' already exists for this DSO. "
                                                     f"Do you want to update it?")
                        if reply != QMessageBox.Yes:
                            return

                        collage_id = existing[0]
                        # Update existing project
                        cursor.execute("""
                            UPDATE usercollages
                            SET grid_width = ?, grid_height = ?, cell_size = ?,
                                spacing = ?, background_color = ?, modified_date = ?
                            WHERE id = ?
                        """, (project['grid_width'], project['grid_height'], project['cell_size'],
                              project['spacing'], project['background_color'], current_time, collage_id))

                        # Clear existing image associations
                        cursor.execute("DELETE FROM usercollageimages WHERE collage_id = ?", (collage_id,))

                    else:
                        # Create new project
                        cursor.execute("""
                            INSERT INTO usercollages
                            (dsodetailid, name, grid_width, grid_height, cell_size,
                             spacing, background_color, created_date, modified_date)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (dsodetailid, save_name, project['grid_width'], project['grid_height'],
                              project['cell_size'], project['spacing'], project['background_color'],
                              current_time, current_time))

                        collage_id = cursor.lastrowid

                # Add image associations for the collage_images
                for position, image_data in enumerate(collage_images):
                    userimage_id = image_data.get('id')

                    # If image doesn't have an ID, save it to userimages table first
                    if not userimage_id:
                        cursor.execute("""
                            INSERT INTO userimages (dsodetailid, image_path, integration_time, equipment, date_taken, notes)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            dsodetailid,
                            image_data.get('image_path', ''),
                            image_data.get('integration_time', ''),
                            image_data.get('equipment', ''),
                            image_data.get('date_taken', ''),
                            image_data.get('notes', '')
                        ))
                        userimage_id = cursor.lastrowid
                        # Update the image_data with the new ID for future reference
                        image_data['id'] = userimage_id

                    cursor.execute("""
                        INSERT INTO usercollageimages (collage_id, userimage_id, position_index)
                        VALUES (?, ?, ?)
                    """, (collage_id, userimage_id, position))

                # Update the project data with database ID and save info
                project['database_id'] = collage_id
                project['modified_date'] = current_time
                if 'created_date' not in project:
                    project['created_date'] = current_time

                conn.commit()

            QMessageBox.information(self, "Project Saved",
                                    f"Project '{save_name}' has been saved successfully!\n\n"
                                    f"Grid: {project['grid_width']}×{project['grid_height']}\n"
                                    f"Images: {len(collage_images)} in collage")

            logger.debug(f"Saved collage project '{save_name}' with {len(collage_images)} images")

        except Exception as e:
            logger.error(f"Error saving project: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")







    def _load_saved_project(self):
        """Load a saved collage project from database"""
        from PySide6.QtWidgets import QInputDialog

        try:
            # Get dsodetailid for this DSO
            dsodetailid = None
            for image_data in self.user_images:
                if image_data.get('id'):
                    with DatabaseManager().get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT dsodetailid FROM userimages WHERE id = ?", (image_data['id'],))
                        result = cursor.fetchone()
                        if result:
                            dsodetailid = result[0]
                            break

            if not dsodetailid:
                QMessageBox.critical(self, "Error", "Could not determine DSO for loading projects.")
                return

            # Get available saved projects for this DSO
            with DatabaseManager().get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, grid_width, grid_height, cell_size, spacing, background_color, created_date
                    FROM usercollages
                    WHERE dsodetailid = ?
                    ORDER BY modified_date DESC
                """, (dsodetailid,))

                saved_projects = cursor.fetchall()

            if not saved_projects:
                QMessageBox.information(self, "No Projects", "No saved projects found for this DSO.")
                return

            # Show list of projects to user
            project_names = [f"{project[1]} ({project[7][:10]})" for project in saved_projects]
            project_name, ok = QInputDialog.getItem(self, "Load Project",
                                                   "Select a project to load:",
                                                   project_names, 0, False)

            if not ok:
                return

            # Find selected project
            selected_index = project_names.index(project_name)
            selected_project = saved_projects[selected_index]

            # Load the project (implementation would continue here)
            QMessageBox.information(self, "Load Project", "Project loading feature not yet implemented.")

        except Exception as e:
            logger.error(f"Error loading projects: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load projects: {str(e)}")

    def _clear_thumbnail_cache(self):
        """Clear the thumbnail cache (useful for debugging or memory management)"""
        if self.thumbnail_cache:
            cache_size = self.thumbnail_cache.size()
            self.thumbnail_cache.clear()
            logger.debug(f"Cleared thumbnail cache ({cache_size} items)")

    def _get_cache_info(self):
        """Get thumbnail cache information for debugging"""
        if self.thumbnail_cache:
            return f"Cache: {self.thumbnail_cache.size()} items"
        return "Cache: Not initialized"


def main():
    """Example usage of CollageBuilder."""
    # Create a 3x3 collage with 250x250 pixel cells
    collage = CollageBuilder(grid_width=3, grid_height=3, cell_size=(250, 250), spacing=15)

    # Preview initial layout
    collage.preview_layout()

    # Example: Add images from a folder
    # folder_path = "path/to/your/images"
    # added_count = collage.add_images_from_folder(folder_path)
    # print(f"Added {added_count} images from folder")

    # Example: Add individual images
    # collage.add_image("image1.jpg")
    # collage.add_image("image2.png")

    # Preview layout with images
    # collage.preview_layout()

    # Create and save the collage
    # collage.create_collage("my_collage.jpg", quality=90)


if __name__ == "__main__":
    main()
