#!/usr/bin/env python3
"""
An image that zooms with the mouse wheel (around the cursor) and pans by
dragging - Light Quality Review's preview, DSO Details' image and the NINA
dashboard's live view, image and live stack. Qt draws it at the zoom shown
(no rescaled copy of the image is made), so panning a big image stays smooth.

Where they differ is set by options:
- min_fit_ratio / max_fit_ratio / max_zoom: how far out and in it zooms
- toggle_on_double_click: fit <-> 100% (otherwise only double_clicked is sent)
- keep_zoom: a zoomed-in view carries over to the next image of the same size,
  so the same stars can be compared frame to frame
- upscale: whether fit enlarges an image smaller than the view
- dark: black background with light text (for camera images)
- show_zoom_text: the zoom % in the corner
- message_points: text size of messages ("Loading...", "No image")

The ImageViewer has its own drawing (annotations, rotation, export framing).
"""

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap, QTransform
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QLabel

from cosmoscollection.ui.Theme import COLORS, adapt_color, chart_font_size, themed_style


class ZoomableImageView(QGraphicsView):
    """An image (or a message instead) with wheel zoom, drag pan and fit."""

    zoom_changed = Signal()
    double_clicked = Signal()
    ZOOM_STEP = 1.25
    MAX_ZOOM = 16.0

    def __init__(self, placeholder="", parent=None, *, min_fit_ratio=0.5, max_fit_ratio=None, max_zoom=MAX_ZOOM,
                 toggle_on_double_click=True, keep_zoom=True, upscale=True, dark=False, show_zoom_text=False,
                 message_points=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        # Whole repaints: the overlay and zoom text are drawn over the viewport
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.min_fit_ratio = min_fit_ratio
        self.max_fit_ratio = max_fit_ratio
        self.max_zoom = max_zoom
        self.toggle_on_double_click = toggle_on_double_click
        self.keep_zoom = keep_zoom
        self.upscale = upscale
        self.dark = dark
        self.show_zoom_text = show_zoom_text
        self.message_points = message_points
        self._placeholder = placeholder
        self._item = None
        self._fit = True
        self._view_state = None  # (image size, transform, center) of the last image shown
        self._overlay = None  # (AnnotationData, layers) drawn over the image
        if dark:
            self.setBackgroundBrush(QColor("black"))
        self._loading = QLabel("Loading...", self.viewport())
        themed_style(self._loading, lambda: f"background: {COLORS['background']}; color: {COLORS['text']}; "
                                            f"padding: 3px 8px;")
        self._loading.move(8, 8)
        self._loading.hide()
        self.set_image(None)

    # --- What's shown ---

    def set_image(self, image, message=None, error=False):
        """Show a QImage or QPixmap - or, for None, message (default: the
        placeholder), in the error color when error."""
        if self._item is not None:
            self._view_state = (self._item.pixmap().size(), self.transform(),
                                self.mapToScene(self.viewport().rect().center()))
        self.scene().clear()
        self._item = None
        self.show_loading(False)
        if image is None or image.isNull():
            self._show_message(self._placeholder if message is None else message, error)
            return
        pixmap = image if isinstance(image, QPixmap) else QPixmap.fromImage(image)
        self._item = self.scene().addPixmap(pixmap)
        self.setSceneRect(self._item.boundingRect())
        state = self._view_state
        if self.keep_zoom and not self._fit and state and state[0] == pixmap.size():
            self.setTransform(state[1])
            self.centerOn(state[2])
            self._zoom_applied()
        else:
            self.fit()

    def _show_message(self, text, error):
        item = self.scene().addText(text)
        if self.message_points:
            font = QFont(item.font())
            font.setPointSizeF(chart_font_size(self.message_points))
            item.setFont(font)
        if error:
            color = COLORS['error']
        elif self.dark:
            color = adapt_color('#a0a0a0', on_dark=True)
        else:
            color = COLORS['text_secondary']
        item.setDefaultTextColor(QColor(color))
        # Lines centered on each other; the view centers the text
        option = item.document().defaultTextOption()
        option.setAlignment(Qt.AlignCenter)
        item.document().setDefaultTextOption(option)
        item.setTextWidth(item.document().idealWidth())
        self.setSceneRect(item.boundingRect())
        self.setTransform(QTransform())
        self.zoom_changed.emit()

    def set_placeholder(self, text):
        """The message shown while there's no image."""
        self._placeholder = text
        if self._item is None:
            self.set_image(None)

    def set_overlay(self, data, layers=None):
        """Draw annotations over the image: AnnotationData (solved for this
        image, at any size) and the layers to show. None removes them."""
        self._overlay = (data, layers or {}) if data is not None else None
        self.viewport().update()

    def show_loading(self, loading):
        self._loading.adjustSize()
        self._loading.setVisible(loading)

    def has_image(self):
        return self._item is not None

    def image_size(self):
        """Size of the image shown (empty when there's none)."""
        return self._item.pixmap().size() if self._item is not None else QSize()

    # --- Zoom ---

    def zoom(self):
        """Screen pixels per image pixel."""
        return self.transform().m11()

    def _fit_zoom(self):
        """The zoom fit() shows the image at (fitInView keeps 2 px clear on each side)."""
        size = self._item.pixmap().size()
        width, height = self.viewport().width() - 4, self.viewport().height() - 4
        zoom = min(width / max(size.width(), 1), height / max(size.height(), 1))
        return zoom if self.upscale else min(zoom, 1.0)

    def _zoom_limits(self):
        fit = self._fit_zoom()
        highest = self.max_zoom if self.max_fit_ratio is None else min(self.max_zoom, fit * self.max_fit_ratio)
        return fit * self.min_fit_ratio, max(highest, fit)

    def fit(self):
        """Show the whole image."""
        self._fit = True
        if self._item is not None:
            if self.upscale or self._fit_zoom() < 1.0:
                self.fitInView(self._item, Qt.KeepAspectRatio)
            else:
                self.setTransform(QTransform())
                self.centerOn(self._item)
        self._zoom_applied()

    def set_zoom(self, factor, center=None):
        """Zoom to factor (screen pixels per image pixel), kept within the limits,
        centered on center (scene point; default: the middle of the view)."""
        if self._item is None:
            return
        center = center if center is not None else self.mapToScene(self.viewport().rect().center())
        lowest, highest = self._zoom_limits()
        factor = min(max(factor, lowest), highest)
        self.setTransform(QTransform.fromScale(factor, factor))
        self.centerOn(center)
        self._fit = False
        self._zoom_applied()

    def zoom_by(self, step):
        self.set_zoom(self.zoom() * step)

    def _zoom_applied(self):
        # Smoothing hides the pixels a close look is for
        self.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom() < 2)
        self.viewport().update()
        self.zoom_changed.emit()

    # --- Mouse ---

    def wheelEvent(self, event):
        if self._item is None:
            return
        step = self.ZOOM_STEP if event.angleDelta().y() > 0 else 1 / self.ZOOM_STEP
        lowest, highest = self._zoom_limits()
        target = self.zoom() * step
        if target > highest * 1.0001 or target < lowest * 0.9999:
            target = min(max(target, lowest), highest)
            if abs(target - self.zoom()) < 1e-6:
                return
            step = target / self.zoom()
        self.scale(step, step)  # scale() keeps the point under the cursor in place
        self._fit = False
        self._zoom_applied()

    def mouseDoubleClickEvent(self, event):
        if self._item is None:
            return
        self.double_clicked.emit()
        if not self.toggle_on_double_click:
            return
        if self._fit or abs(self.zoom() - 1) > 1e-3:
            self.set_zoom(1.0, self.mapToScene(event.position().toPoint()))
        else:
            self.fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit and self._item is not None:
            self.fit()

    # --- Drawn over the image ---

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._item is None or (self._overlay is None and not self.show_zoom_text):
            return
        painter = QPainter(self.viewport())
        try:
            if self._overlay is not None:
                from cosmoscollection.imaging.AnnotationOverlay import paint_annotations, screen_name_px
                data, layers = self._overlay
                size = self._item.pixmap().size()
                source = QRectF(0, 0, size.width(), size.height())
                target = QRectF(self.mapFromScene(self._item.sceneBoundingRect()).boundingRect())
                paint_annotations(painter, data, layers, size.width(), source, target,
                                  font_px=screen_name_px(target))
            if self.show_zoom_text and not self._fit:
                painter.setPen(QColor(adapt_color('#ffffff', on_dark=True) if self.dark else COLORS['text']))
                painter.drawText(10, 20, f"{self.zoom() * 100:.0f}%")
        finally:
            painter.end()
