"""
PyQt5-based concept map editor supporting IHMC CXL files.

This application implements an interactive editor for concept maps
similar to IHMC CmapTools.  It can load and save CXL files,
create new maps, and allows the user to add, resize and connect
concepts (nodes) and linking phrases via an intuitive graphical
interface.

Features:

* **Double click** on empty canvas creates a new concept at that
  location with the default label "????".  Double clicking an existing
  concept or linking phrase allows editing its text inline.

* Each concept has a small **resize handle** in its bottom right
  corner.  Drag this handle to resize the concept; resizing is
  constrained so that the label text always fits within the
  bounding box.

* Each concept also has a **connection handle** at its top centre.
  Dragging this handle drags out a new concept that follows the cursor.
  Releasing over an existing concept creates a connection; releasing in
  empty space drops the new concept there.  If the **Shift** key is held
  during release, the connection is direct.  Otherwise, a linking phrase
  with default label "????" is inserted between the concepts.

* **Dragging** a concept moves it around the canvas; all attached
  connections update dynamically.

* **Clicking** on a connection displays small handles at its
  endpoints.  Drag these handles to re-anchor the connection to
  different positions on the concept (top, bottom, left or right).

* **Selection and style editing**: a style panel is always visible;
  select a concept or connection to enable editing of label, font,
  colours and border settings.  Press **Ctrl+E** to focus the panel.

* Supports loading existing CXL files in both the old style list
  format and the newer appearance-based format.  Saving will
  preserve positions, labels and style (global for concepts and
  linking phrases) and update connections.

This editor is intended as a robust and extensible foundation for
concept map creation and editing.  It does not cover every nuance
of IHMC's CmapTools but provides core interactive functionality
requested by the user.

Usage:
    python3 cxl_editor_pyqt.py

Dependencies:
    PyQt5 must be installed in the Python environment.
"""

import sys
import uuid
import xml.etree.ElementTree as ET
import copy
import json
import math
import os
from typing import Dict, List, Optional, Tuple, Set

from PyQt5.QtCore import (
    Qt,
    QRectF,
    QPointF,
    QPoint,
    QSize,
    pyqtSignal,
    QEvent,
)
from PyQt5.QtGui import (
    QColor,
    QPen,
    QBrush,
    QFont,
    QFontMetrics,
    QPainter,
    QPalette,
    QTextCursor,
    QKeySequence,
    QPixmap,
    QIcon,
)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsTextItem,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsObject,
    QGraphicsDropShadowEffect,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QColorDialog,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QAction,
    QMenu,
    QMessageBox,
    QFontComboBox,
    QCheckBox,
    QToolBar,
    QStyle,
    QGraphicsBlurEffect,
    QGesture,
    QPinchGesture,
)
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog


##############################
# Data model and CXL parsing #
##############################

class ConceptData:
    """Represents conceptual data for a concept or linking phrase."""

    def __init__(self, cid: str, label: str, x: float, y: float, width: float, height: float,
                 font_family: str = "Verdana", font_size: float = 12.0,
                 font_color: str = "0,0,0,255", fill_color: str = "237,244,246,255",
                 border_color: str = "0,0,0,255", border_width: float = 1.0,
                 font_bold: bool = False, font_italic: bool = False,
                 font_underline: bool = False, is_linker: bool = False) -> None:
        self.id = cid
        self.label = label
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.font_family = font_family
        self.font_size = font_size
        self.font_color = font_color
        self.fill_color = fill_color
        self.border_color = border_color
        self.border_width = border_width
        self.font_bold = font_bold
        self.font_italic = font_italic
        self.font_underline = font_underline
        self.is_linker = is_linker


class ConnectionData:
    """Represents connection data between two node ids."""

    def __init__(self, cid: str, from_id: str, to_id: str,
                 arrow_start: bool = False, arrow_end: bool = True,
                 from_anchor: int = 0, to_anchor: int = 0) -> None:
        self.id = cid
        self.from_id = from_id
        self.to_id = to_id
        self.arrow_start = arrow_start
        self.arrow_end = arrow_end
        self.from_anchor = from_anchor
        self.to_anchor = to_anchor


class CXLDocument:
    """
    Parse and store a CXL file.  Supports both appearance-based and
    old style-list formats.  Provides methods to load and save maps.
    """

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptData] = {}
        self.connections: List[ConnectionData] = []
        self.tree: Optional[ET.ElementTree] = None
        self.root: Optional[ET.Element] = None
        self.ns: Dict[str, str] = {"c": "http://cmap.ihmc.us/xml/cmap/"}
        self.appearance_mode: bool = False
        self.filepath: Optional[str] = None
        # Default styles extracted from style sheet (global for concepts and linkers)
        self.default_concept_style: Dict[str, str] = {}
        self.default_linker_style: Dict[str, str] = {}

    def load(self, filepath: str) -> None:
        """Load a CXL file from disk."""
        try:
            self.tree = ET.parse(filepath)
        except (ET.ParseError, FileNotFoundError) as exc:
            raise ValueError(f"Errore nel parse del file CXL: {exc}")
        self.root = self.tree.getroot()
        self.filepath = filepath
        # Clear existing data
        self.concepts.clear()
        self.connections.clear()
        # Determine namespace
        if self.root.tag.startswith("{"):
            ns_uri = self.root.tag[1:].split("}")[0]
            self.ns = {"c": ns_uri}
        # Find map
        map_elem = self.root.find("c:map", self.ns)
        if map_elem is None:
            map_elem = self.root.find("map")
        if map_elem is None:
            raise ValueError("Il file non contiene un elemento <map>.")
        # Determine format
        self.appearance_mode = map_elem.find("c:concept-appearance-list", self.ns) is not None
        # Parse concepts and linking phrases
        if self.appearance_mode:
            # Concept list
            for c_elem in map_elem.findall("c:concept-list/c:concept", self.ns):
                cid = c_elem.get("id") or ""
                label = c_elem.get("label") or ""
                self.concepts[cid] = ConceptData(cid, label, 100, 100, 120, 60)
                self.concepts[cid].is_linker = False
            # Linking phrase list
            for l_elem in map_elem.findall("c:linking-phrase-list/c:linking-phrase", self.ns):
                lid = l_elem.get("id") or ""
                label = l_elem.get("label") or ""
                self.concepts[lid] = ConceptData(lid, label, 100, 100, 90, 11, is_linker=True)
            # Parse positions (appearance lists)
            
            for app in map_elem.findall("c:concept-appearance-list/c:concept-appearance", self.ns):
                aid = app.get("id")
                node = self.concepts.get(aid)
                if node:
                    try:
                        node.x = float(app.get("x", node.x))
                        node.y = float(app.get("y", node.y))
                        node.width = float(app.get("width", node.width))
                        node.height = float(app.get("height", node.height))
                    except ValueError:
                        pass
                    # Optional per-node style overrides on appearance
                    bg = app.get("background-color")
                    if bg:
                        node.fill_color = self._parse_color(bg)
                    fc = app.get("font-color")
                    if fc:
                        node.font_color = self._parse_color(fc)
                    bc = app.get("border-color")
                    if bc:
                        node.border_color = self._parse_color(bc)
                    bw = app.get("border-thickness")
                    if bw:
                        try:
                            node.border_width = float(bw)
                        except ValueError:
                            pass
                    fn = app.get("font-name")
                    if fn:
                        node.font_family = fn
                    fs = app.get("font-size")
                    if fs:
                        try:
                            node.font_size = float(fs)
                        except ValueError:
                            pass
                    fstyle = app.get("font-style", "")
                    if fstyle:
                        s = fstyle.lower()
                        node.font_bold = "bold" in s
                        node.font_italic = "italic" in s
                        node.font_underline = "underline" in s

            
            for app in map_elem.findall("c:linking-phrase-appearance-list/c:linking-phrase-appearance", self.ns):
                aid = app.get("id")
                node = self.concepts.get(aid)
                if node:
                    try:
                        node.x = float(app.get("x", node.x))
                        node.y = float(app.get("y", node.y))
                        node.width = float(app.get("width", node.width))
                        node.height = float(app.get("height", node.height))
                    except ValueError:
                        pass
                    # Optional per-linker style overrides on appearance
                    bg = app.get("background-color")
                    if bg:
                        node.fill_color = self._parse_color(bg)
                    fc = app.get("font-color")
                    if fc:
                        node.font_color = self._parse_color(fc)
                    bc = app.get("border-color")
                    if bc:
                        node.border_color = self._parse_color(bc)
                    bw = app.get("border-thickness")
                    if bw:
                        try:
                            node.border_width = float(bw)
                        except ValueError:
                            pass
                    fn = app.get("font-name")
                    if fn:
                        node.font_family = fn
                    fs = app.get("font-size")
                    if fs:
                        try:
                            node.font_size = float(fs)
                        except ValueError:
                            pass
                    fstyle = app.get("font-style", "")
                    if fstyle:
                        s = fstyle.lower()
                        node.font_bold = "bold" in s
                        node.font_italic = "italic" in s
                        node.font_underline = "underline" in s

            # Parse connections
            for conn in map_elem.findall("c:connection-list/c:connection", self.ns):
                cid = conn.get("id") or str(uuid.uuid4())
                from_id = conn.get("from-id") or ""
                to_id = conn.get("to-id") or ""
                cdata = ConnectionData(cid, from_id, to_id)
                dst = self.concepts.get(to_id)
                if dst and dst.is_linker:
                    cdata.arrow_end = False
                self.connections.append(cdata)
            # Parse connection anchor positions
            conn_lookup = {c.id: c for c in self.connections}
            processed: Set[str] = set()
            for app in map_elem.findall("c:connection-appearance-list/c:connection-appearance", self.ns):
                aid = app.get("id")
                conn = conn_lookup.get(aid or "")
                if not conn:
                    continue
                processed.add(aid or "")
                src = self.concepts.get(conn.from_id)
                dst = self.concepts.get(conn.to_id)
                if src and dst:
                    fa = app.get("from-index")
                    ta = app.get("to-index")
                    if fa is not None and fa.isdigit():
                        conn.from_anchor = int(fa)
                    else:
                        conn.from_anchor = self._pos_to_anchor(app.get("from-pos", "center"), src, dst)
                    if ta is not None and ta.isdigit():
                        conn.to_anchor = int(ta)
                    else:
                        conn.to_anchor = self._pos_to_anchor(app.get("to-pos", "center"), dst, src)
            # Assign default anchors if no appearance information is provided
            for conn in self.connections:
                if conn.id not in processed:
                    src = self.concepts.get(conn.from_id)
                    dst = self.concepts.get(conn.to_id)
                    if src and dst:
                        conn.from_anchor = self._pos_to_anchor("center", src, dst)
                        conn.to_anchor = self._pos_to_anchor("center", dst, src)
            # Extract default styles
            ss = map_elem.find("c:style-sheet-list/c:style-sheet", self.ns)
            if ss is not None:
                c_style = ss.find("c:concept-style", self.ns)
                if c_style is not None:
                    self.default_concept_style = {
                        "font-name": c_style.get("font-name", "Verdana"),
                        "font-size": c_style.get("font-size", "12"),
                        "font-color": c_style.get("font-color", "0,0,0,255"),
                        "font-style": c_style.get("font-style", "plain"),
                        "background-color": c_style.get("background-color", "237,244,246,255"),
                        "border-color": c_style.get("border-color", "0,0,0,255"),
                        "border-thickness": c_style.get("border-thickness", "1"),
                    }
                l_style = ss.find("c:linking-phrase-style", self.ns)
                if l_style is not None:
                    self.default_linker_style = {
                        "font-name": l_style.get("font-name", "Verdana"),
                        "font-size": l_style.get("font-size", "12"),
                        "font-color": l_style.get("font-color", "0,0,0,255"),
                        "font-style": l_style.get("font-style", "plain"),
                        "background-color": l_style.get("background-color", "0,0,255,0"),
                        "border-color": l_style.get("border-color", "0,0,0,0"),
                        "border-thickness": l_style.get("border-thickness", "1"),
                    }
            # Apply default styles to nodes
            for node in self.concepts.values():
                style = self.default_linker_style if node.is_linker else self.default_concept_style
                node.font_family = style.get("font-name", node.font_family)
                try:
                    node.font_size = float(style.get("font-size", node.font_size))
                except ValueError:
                    pass
                node.font_color = style.get("font-color", node.font_color)
                node.fill_color = style.get("background-color", node.fill_color)
                node.border_color = style.get("border-color", node.border_color)
                try:
                    node.border_width = float(style.get("border-thickness", node.border_width))
                except ValueError:
                    pass
                style_str = style.get("font-style", "plain")
                node.font_bold = "bold" in style_str
                node.font_italic = "italic" in style_str
                node.font_underline = "underline" in style_str

            # Re-apply per-node appearance style overrides (take precedence over style-sheet defaults)
            # Concept appearance
            for app in map_elem.findall("c:concept-appearance-list/c:concept-appearance", self.ns):
                aid = app.get("id")
                node = self.concepts.get(aid or "")
                if not node:
                    continue
                bg = app.get("background-color"); fc = app.get("font-color")
                bc = app.get("border-color"); bw = app.get("border-thickness")
                fn = app.get("font-name"); fs = app.get("font-size")
                fstyle = app.get("font-style")
                if bg: node.fill_color = self._parse_color(bg)
                if fc: node.font_color = self._parse_color(fc)
                if bc: node.border_color = self._parse_color(bc)
                if bw:
                    try: node.border_width = float(bw)
                    except ValueError: pass
                if fn: node.font_family = fn
                if fs:
                    try: node.font_size = float(fs)
                    except ValueError: pass
                if fstyle:
                    s = fstyle.lower()
                    node.font_bold = "bold" in s
                    node.font_italic = "italic" in s
                    node.font_underline = "underline" in s
            # Linking phrase appearance
            for app in map_elem.findall("c:linking-phrase-appearance-list/c:linking-phrase-appearance", self.ns):
                aid = app.get("id")
                node = self.concepts.get(aid or "")
                if not node:
                    continue
                bg = app.get("background-color"); fc = app.get("font-color")
                bc = app.get("border-color"); bw = app.get("border-thickness")
                fn = app.get("font-name"); fs = app.get("font-size")
                fstyle = app.get("font-style")
                if bg: node.fill_color = self._parse_color(bg)
                if fc: node.font_color = self._parse_color(fc)
                if bc: node.border_color = self._parse_color(bc)
                if bw:
                    try: node.border_width = float(bw)
                    except ValueError: pass
                if fn: node.font_family = fn
                if fs:
                    try: node.font_size = float(fs)
                    except ValueError: pass
                if fstyle:
                    s = fstyle.lower()
                    node.font_bold = "bold" in s
                    node.font_italic = "italic" in s
                    node.font_underline = "underline" in s

        else:
            # Old style list format
            # Parse concepts
            for c_elem in map_elem.findall("concept-list/concept"):
                cid = c_elem.get("id") or ""
                label = c_elem.get("label") or ""
                self.concepts[cid] = ConceptData(cid, label, 100, 100, 120, 60, is_linker=False)
            # Parse linking phrases
            for l_elem in map_elem.findall("linking-phrase-list/linking-phrase"):
                lid = l_elem.get("id") or ""
                label = l_elem.get("label") or ""
                self.concepts[lid] = ConceptData(lid, label, 100, 100, 120, 40, is_linker=True)
            # Parse connections
            for conn_elem in map_elem.findall("connection-list/connection"):
                cid = conn_elem.get("id") or str(uuid.uuid4())
                from_id = conn_elem.get("from-id") or ""
                to_id = conn_elem.get("to-id") or ""
                cdata = ConnectionData(cid, from_id, to_id)
                dst = self.concepts.get(to_id)
                if dst and dst.is_linker:
                    cdata.arrow_end = False
                self.connections.append(cdata)
            # Parse style list
            style_lookup: Dict[str, ET.Element] = {}
            for style_elem in map_elem.findall("style-list/style"):
                sid = style_elem.get("object-id") or ""
                style_lookup[sid] = style_elem
            # Apply positions and style
            for cid, node in self.concepts.items():
                style = style_lookup.get(cid)
                if style is not None:
                    # geom
                    geom = style.find("geom")
                    if geom is not None:
                        try:
                            node.x = float(geom.get("x", node.x))
                            node.y = float(geom.get("y", node.y))
                            node.width = float(geom.get("width", node.width))
                            node.height = float(geom.get("height", node.height))
                        except ValueError:
                            pass
                    # text
                    text = style.find("text")
                    if text is not None:
                        node.font_family = text.get("font-name", node.font_family)
                        try:
                            node.font_size = float(text.get("font-size", node.font_size))
                        except ValueError:
                            pass
                        node.font_color = text.get("color", node.font_color)
                        style_str = text.get("style", "")
                        node.font_bold = "bold" in style_str
                        node.font_italic = "italic" in style_str
                        node.font_underline = "underline" in style_str
                    # shape
                    shape = style.find("shape")
                    if shape is not None:
                        node.fill_color = shape.get("fill", node.fill_color)
                        node.border_color = shape.get("border", node.border_color)
        # Convert colours to QColor strings for GUI
        for node in self.concepts.values():
            # Convert RGBA string to hex or keep hex if provided
            node.font_color = self._parse_color(node.font_color)
            node.fill_color = self._parse_color(node.fill_color)
            node.border_color = self._parse_color(node.border_color)
        # Map loaded

    @staticmethod
    def _parse_color(val: str) -> str:
        """Convert colour representations into hex #RRGGBB."""
        val = val.strip()
        # Already in hex form
        if val.startswith("#") and len(val) == 7:
            return val
        # RGBA or RGB with commas
        parts = val.split(',')
        if len(parts) >= 3:
            try:
                r = int(parts[0])
                g = int(parts[1])
                b = int(parts[2])
                return f"#{r:02X}{g:02X}{b:02X}"
            except ValueError:
                pass
        # Space separated values
        parts = val.split()
        if len(parts) >= 3:
            try:
                r = int(float(parts[0]))
                g = int(float(parts[1]))
                b = int(float(parts[2]))
                return f"#{r:02X}{g:02X}{b:02X}"
            except ValueError:
                pass
        # Fallback
        return "#000000"

    @staticmethod
    def _pos_to_anchor(pos: str, node: ConceptData, other: ConceptData) -> int:
        """Convert CXL anchor position strings to node anchor indices."""
        p = (pos or "center").lower()
        if p == "top":
            return 6  # top centre
        if p == "bottom":
            return 7  # bottom centre
        if p == "left":
            return 18  # left centre
        if p == "right":
            return 19  # right centre
        # For "center" or unknown, choose side based on relative position
        dx = (other.x + other.width / 2) - (node.x + node.width / 2)
        dy = (other.y + other.height / 2) - (node.y + node.height / 2)
        if abs(dx) > abs(dy):
            return 19 if dx >= 0 else 18
        return 7 if dy >= 0 else 6

    @staticmethod
    def _anchor_to_pos(anchor: int) -> str:
        """Convert an anchor index to a CXL position string."""
        if 0 <= anchor <= 13:
            return "top" if anchor % 2 == 0 else "bottom"
        if 14 <= anchor <= 23:
            return "left" if anchor % 2 == 0 else "right"
        return "center"

    def save(self, filepath: Optional[str] = None) -> None:
        """Save the current map back to a CXL file."""
        if self.root is None:
            raise ValueError("Nessun documento CXL caricato o creato.")
        map_elem = self.root.find("c:map", self.ns) or self.root.find("map")
        if map_elem is None:
            raise ValueError("Il documento non contiene un <map>.")
        # Update or create concept and linking phrase lists
        if self.appearance_mode:
            # Concept and linking phrase lists
            clist = map_elem.find("c:concept-list", self.ns)
            if clist is None:
                clist = ET.SubElement(map_elem, f"{{{self.ns['c']}}}concept-list")
            llist = map_elem.find("c:linking-phrase-list", self.ns)
            if llist is None:
                llist = ET.SubElement(map_elem, f"{{{self.ns['c']}}}linking-phrase-list")
            # Clear existing elements and rebuild
            clist.clear()
            llist.clear()
            for cid, node in self.concepts.items():
                if node.is_linker:
                    elem = ET.SubElement(llist, f"{{{self.ns['c']}}}linking-phrase")
                else:
                    elem = ET.SubElement(clist, f"{{{self.ns['c']}}}concept")
                elem.set("id", cid)
                elem.set("label", node.label)
            # Concept and linking phrase appearance lists
            calist = map_elem.find("c:concept-appearance-list", self.ns)
            if calist is None:
                calist = ET.SubElement(map_elem, f"{{{self.ns['c']}}}concept-appearance-list")
            calist.clear()
            lpalist = map_elem.find("c:linking-phrase-appearance-list", self.ns)
            if lpalist is None:
                lpalist = ET.SubElement(map_elem, f"{{{self.ns['c']}}}linking-phrase-appearance-list")
            lpalist.clear()
            
            for cid, node in self.concepts.items():
                if node.is_linker:
                    app = ET.SubElement(lpalist, f"{{{self.ns['c']}}}linking-phrase-appearance")
                else:
                    app = ET.SubElement(calist, f"{{{self.ns['c']}}}concept-appearance")
                app.set("id", cid)
                app.set("x", str(node.x))
                app.set("y", str(node.y))
                app.set("width", str(node.width))
                app.set("height", str(node.height))
                # Persist per-node style attributes on appearance as RGBA values
                app.set("background-color", self._color_to_rgba(node.fill_color))
                app.set("font-color", self._color_to_rgba(node.font_color))
                app.set("border-color", self._color_to_rgba(node.border_color))
                app.set("border-thickness", str(int(getattr(node, "border_width", 1))))
                app.set("font-name", getattr(node, "font_family", "Verdana"))
                app.set("font-size", str(int(getattr(node, "font_size", 12))))
                app.set("font-style", self._font_style_to_str(node))

            # Connection list
            conn_list = map_elem.find("c:connection-list", self.ns)
            if conn_list is None:
                conn_list = ET.SubElement(map_elem, f"{{{self.ns['c']}}}connection-list")
            conn_list.clear()
            for conn in self.connections:
                celem = ET.SubElement(conn_list, f"{{{self.ns['c']}}}connection")
                celem.set("id", conn.id)
                celem.set("from-id", conn.from_id)
                celem.set("to-id", conn.to_id)
            # Connection appearance list
            capplist = map_elem.find("c:connection-appearance-list", self.ns)
            if capplist is None:
                capplist = ET.SubElement(map_elem, f"{{{self.ns['c']}}}connection-appearance-list")
            capplist.clear()
            # Ensure a connection-style exists to reflect logical direction (arrow at 'to' end)
            ss_list = map_elem.find("c:style-sheet-list", self.ns)
            if ss_list is None:
                ss_list = ET.SubElement(map_elem, f"{{{self.ns['c']}}}style-sheet-list")
            ss = ss_list.find("c:style-sheet", self.ns)
            if ss is None:
                ss = ET.SubElement(ss_list, f"{{{self.ns['c']}}}style-sheet")
            if ss.find("c:connection-style", self.ns) is None:
                cs = ET.SubElement(ss, f"{{{self.ns['c']}}}connection-style")
                cs.set("color", "0,0,0,255")
                cs.set("style", "solid")
                cs.set("thickness", "1")
                cs.set("type", "straight")
                cs.set("arrowhead", "if-to-concept")
    
            for conn in self.connections:
                app = ET.SubElement(capplist, f"{{{self.ns['c']}}}connection-appearance")
                app.set("id", conn.id)
                app.set("from-pos", self._anchor_to_pos(conn.from_anchor))
                app.set("to-pos", self._anchor_to_pos(conn.to_anchor))
                app.set("from-index", str(conn.from_anchor))
                app.set("to-index", str(conn.to_anchor))
            # Update style sheet based on first concept/linker
            ss_list = map_elem.find("c:style-sheet-list", self.ns)
            if ss_list is None:
                ss_list = ET.SubElement(map_elem, f"{{{self.ns['c']}}}style-sheet-list")
            ss = ss_list.find("c:style-sheet", self.ns)
            if ss is None:
                ss = ET.SubElement(ss_list, f"{{{self.ns['c']}}}style-sheet")
            # Concept and linker style elements
            c_style = ss.find("c:concept-style", self.ns)
            if c_style is None:
                c_style = ET.SubElement(ss, f"{{{self.ns['c']}}}concept-style")
            l_style = ss.find("c:linking-phrase-style", self.ns)
            if l_style is None:
                l_style = ET.SubElement(ss, f"{{{self.ns['c']}}}linking-phrase-style")
            # Use first concept for concept-style and first linker for linker-style
            first_c = next((n for n in self.concepts.values() if not n.is_linker), None)
            if first_c is not None:
                c_style.set("font-name", first_c.font_family)
                c_style.set("font-size", str(int(first_c.font_size)))
                c_style.set("font-color", self._color_to_rgba(first_c.font_color))
                c_style.set("font-style", self._font_style_to_str(first_c))
                c_style.set("background-color", self._color_to_rgba(first_c.fill_color))
                c_style.set("border-color", self._color_to_rgba(first_c.border_color))
                c_style.set("border-thickness", str(int(first_c.border_width)))
            first_l = next((n for n in self.concepts.values() if n.is_linker), None)
            if first_l is not None:
                l_style.set("font-name", first_l.font_family)
                l_style.set("font-size", str(int(first_l.font_size)))
                l_style.set("font-color", self._color_to_rgba(first_l.font_color))
                l_style.set("font-style", self._font_style_to_str(first_l))
                l_style.set("background-color", self._color_to_rgba(first_l.fill_color))
                l_style.set("border-color", self._color_to_rgba(first_l.border_color))
                l_style.set("border-thickness", str(int(first_l.border_width)))
        else:
            # Old style list format: update concept list and style list
            # Update concept and linking phrase labels
            for c_elem in map_elem.findall("concept-list/concept"):
                cid = c_elem.get("id") or ""
                node = self.concepts.get(cid)
                if node:
                    c_elem.set("label", node.label)
            for l_elem in map_elem.findall("linking-phrase-list/linking-phrase"):
                lid = l_elem.get("id") or ""
                node = self.concepts.get(lid)
                if node:
                    l_elem.set("label", node.label)
            # Style list
            style_lookup: Dict[str, ET.Element] = {}
            for style_elem in map_elem.findall("style-list/style"):
                sid = style_elem.get("object-id") or ""
                style_lookup[sid] = style_elem
            # Update each style with position and colours
            for cid, node in self.concepts.items():
                style_elem = style_lookup.get(cid)
                if style_elem is None:
                    # Create new style element
                    slist = map_elem.find("style-list")
                    if slist is None:
                        slist = ET.SubElement(map_elem, "style-list")
                    style_elem = ET.SubElement(slist, "style")
                    style_elem.set("object-id", cid)
                # geom
                geom = style_elem.find("geom")
                if geom is None:
                    geom = ET.SubElement(style_elem, "geom")
                geom.set("x", str(node.x))
                geom.set("y", str(node.y))
                geom.set("width", str(node.width))
                geom.set("height", str(node.height))
                # text
                text_elem = style_elem.find("text")
                if text_elem is None:
                    text_elem = ET.SubElement(style_elem, "text")
                text_elem.set("font-name", node.font_family)
                text_elem.set("font-size", str(int(node.font_size)))
                text_elem.set("color", node.font_color)
                text_elem.set("style", self._font_style_to_str(node))
                # shape
                shape_elem = style_elem.find("shape")
                if shape_elem is None:
                    shape_elem = ET.SubElement(style_elem, "shape")
                shape_elem.set("fill", node.fill_color)
                shape_elem.set("border", node.border_color)
        # Write file
        outpath = filepath or self.filepath
        if not outpath:
            raise ValueError("Specificare un percorso per salvare.")
        self.tree.write(outpath, encoding="UTF-8", xml_declaration=True)

    @staticmethod
    def _color_to_rgba(hex_color: str) -> str:
        """Convert #RRGGBB to R,G,B,255"""
        if hex_color.startswith("#") and len(hex_color) == 7:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            return f"{r},{g},{b},255"
        return hex_color

    @staticmethod
    def _font_style_to_str(node: ConceptData) -> str:
        styles = []
        if getattr(node, "font_bold", False):
            styles.append("bold")
        if getattr(node, "font_italic", False):
            styles.append("italic")
        if getattr(node, "font_underline", False):
            styles.append("underline")
        return "-".join(styles) if styles else "plain"

    def new_map(self, width: int = 800, height: int = 600) -> None:
        """Create a new blank map with default style sheet."""
        ns_uri = self.ns.get("c", "http://cmap.ihmc.us/xml/cmap/")
        self.root = ET.Element(f"{{{ns_uri}}}cmap")
        map_elem = ET.SubElement(self.root, f"{{{ns_uri}}}map")
        map_elem.set("width", str(width))
        map_elem.set("height", str(height))
        ET.SubElement(map_elem, f"{{{ns_uri}}}concept-list")
        ET.SubElement(map_elem, f"{{{ns_uri}}}linking-phrase-list")
        ET.SubElement(map_elem, f"{{{ns_uri}}}connection-list")
        ET.SubElement(map_elem, f"{{{ns_uri}}}concept-appearance-list")
        ET.SubElement(map_elem, f"{{{ns_uri}}}linking-phrase-appearance-list")
        ET.SubElement(map_elem, f"{{{ns_uri}}}connection-appearance-list")
        ss_list = ET.SubElement(map_elem, f"{{{ns_uri}}}style-sheet-list")
        ss = ET.SubElement(ss_list, f"{{{ns_uri}}}style-sheet")
        ss.set("id", "_Default_")
        # Default concept style
        c_style = ET.SubElement(ss, f"{{{ns_uri}}}concept-style")
        c_style.set("font-name", "Verdana")
        c_style.set("font-size", "12")
        c_style.set("font-style", "plain")
        c_style.set("font-color", "0,0,0,255")
        c_style.set("background-color", "237,244,246,255")
        c_style.set("border-color", "0,0,0,255")
        c_style.set("border-thickness", "1")
        c_style.set("border-style", "solid")
        c_style.set("border-shape", "rounded-rectangle")
        c_style.set("border-shape-rrarc", "15.0")
        c_style.set("text-margin", "4")
        # Linking phrase style
        l_style = ET.SubElement(ss, f"{{{ns_uri}}}linking-phrase-style")
        l_style.set("font-name", "Verdana")
        l_style.set("font-size", "12")
        l_style.set("font-style", "plain")
        l_style.set("font-color", "0,0,0,255")
        l_style.set("background-color", "0,0,255,0")
        l_style.set("border-color", "0,0,0,0")
        l_style.set("border-thickness", "1")
        l_style.set("border-style", "solid")
        l_style.set("border-shape", "rectangle")
        l_style.set("border-shape-rrarc", "15.0")
        l_style.set("text-margin", "1")
        # Reset data
        self.tree = ET.ElementTree(self.root)
        self.concepts.clear()
        self.connections.clear()
        self.appearance_mode = True
        self.default_concept_style = {
            "font-name": c_style.get("font-name"),
            "font-size": c_style.get("font-size"),
            "font-color": c_style.get("font-color"),
            "font-style": c_style.get("font-style"),
            "background-color": c_style.get("background-color"),
            "border-color": c_style.get("border-color"),
            "border-thickness": c_style.get("border-thickness"),
        }
        self.default_linker_style = {
            "font-name": l_style.get("font-name"),
            "font-size": l_style.get("font-size"),
            "font-color": l_style.get("font-color"),
            "font-style": l_style.get("font-style"),
            "background-color": l_style.get("background-color"),
            "border-color": l_style.get("border-color"),
            "border-thickness": l_style.get("border-thickness"),
        }
        self.filepath = None


##############################################
# Graphics items for Node, Linking Phrase and Connection
##############################################

class AnchorHandle(QGraphicsEllipseItem):
    """Interactive handle used for repositioning connection anchors."""

    def __init__(self, connection: 'ConnectionItem', is_source: bool, radius: float = 6.0) -> None:
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.setBrush(QBrush(QColor("#FFFFFF")))
        self.setPen(QPen(QColor("#000000")))
        self.setZValue(1000)
        self.connection = connection
        self.is_source = is_source
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

    def mouseReleaseEvent(self, event):
        # On release, assign to nearest anchor on node and update connection
        if self.is_source:
            node = self.connection.source
        else:
            node = self.connection.dest
        pos = self.scenePos()
        # Find nearest anchor index (convert anchors to scene coordinates)
        min_dist = None
        best_index = 0
        for i, anchor in enumerate(node.anchor_positions()):
            anchor_scene = node.mapToScene(anchor)
            d = (anchor_scene - pos).manhattanLength()
            if min_dist is None or d < min_dist:
                min_dist = d
                best_index = i
        if self.is_source:
            self.connection.source_anchor = best_index
        else:
            self.connection.dest_anchor = best_index
        self.connection.update_position()
        # Remove handles
        self.connection.hide_anchor_handles()
        super().mouseReleaseEvent(event)


class EditableTextItem(QGraphicsTextItem):
    """Text item that commits edits back to its owning node."""

    def __init__(self, node: 'NodeItem') -> None:
        super().__init__(node.data.label, node)
        self.node = node
        self.setDefaultTextColor(QColor(node.data.font_color))
        font = QFont(node.data.font_family, int(node.data.font_size))
        if getattr(node.data, "font_bold", False):
            font.setBold(True)
        if getattr(node.data, "font_italic", False):
            font.setItalic(True)
        if getattr(node.data, "font_underline", False):
            font.setUnderline(True)
        self.setFont(font)
        # Start with text interaction disabled so that clicking the node
        # does not interact with the label directly.  Interaction is
        # temporarily enabled when the user edits the label.
        self.setTextInteractionFlags(Qt.NoTextInteraction)

    def focusOutEvent(self, event) -> None:
        # When editing finishes, revert the label item to a passive state so
        # it doesn't become an additional selected item in the scene.  If the
        # label remains selectable, the scene reports two selected items (the
        # node and its label) which confuses the selection logic.
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.node.finish_editing(self.toPlainText())
        super().focusOutEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.clearFocus()
        else:
            super().keyPressEvent(event)


class NodeItem(QGraphicsObject):
    """Graphical representation of a concept or linking phrase."""

    # Signals to inform editor of actions
    request_connection = pyqtSignal('PyQt_PyObject', 'PyQt_PyObject')  # (nodeItem, event)
    node_moved = pyqtSignal('PyQt_PyObject')  # Node moved
    node_selected = pyqtSignal('PyQt_PyObject')  # Node selected

    def __init__(self, data: ConceptData, scene: QGraphicsScene) -> None:
        QGraphicsObject.__init__(self)
        self.data = data
        self.scene_ref = scene
        # Set default flags for movement and selection
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        # Children: label item
        self.label_item = EditableTextItem(self)
        self.label_item.setZValue(1)
        # Prevent the internal label item from being selected independently
        # of its parent node.  When the label could be selected, the scene
        # reported multiple selected items (the node and its label), which
        # left the styling panel disabled.  Disabling selection and mouse
        # events on the label keeps the node as the sole selected item.
        self.label_item.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.label_item.setAcceptedMouseButtons(Qt.NoButton)
        # Resizing handle (bottom right)
        self.handle_size = 8
        self.resizing = False
        # Connection handle (top centre)
        self.connection_handle_size = 10
        self.dragging_connection = False
        # Hover tracking
        self.hovered = False
        self.setAcceptHoverEvents(True)
        # Connections list
        self.connections: List['ConnectionItem'] = []
        # Set initial position
        self.setPos(self.data.x, self.data.y)
        # Prepare bounding rect
        self._update_bounds()

    def _update_bounds(self) -> None:
        """Update the bounding rect based on label and data width/height."""
        # Adjust label width based on text
        font = QFont(self.data.font_family, int(self.data.font_size))
        if getattr(self.data, "font_bold", False):
            font.setBold(True)
        if getattr(self.data, "font_italic", False):
            font.setItalic(True)
        if getattr(self.data, "font_underline", False):
            font.setUnderline(True)
        metrics = QFontMetrics(font)
        text_rect = metrics.boundingRect(self.data.label)
        # Ensure width >= text width + padding
        pad = 10
        min_w = text_rect.width() + pad * 2
        min_h = text_rect.height() + pad * 2
        # Notify the scene before changing the geometry so that its internal
        # indexing remains valid.  Changing the width/height first caused Qt to
        # lose track of the item's previous bounds, which could result in
        # items "teleporting" to the origin when subsequently manipulated.
        self.prepareGeometryChange()
        if self.data.width < min_w:
            self.data.width = min_w
        if self.data.height < min_h:
            self.data.height = min_h
        # Update label text and appearance
        self.label_item.setPlainText(self.data.label)
        lbl_font = QFont(self.data.font_family, int(self.data.font_size))
        if getattr(self.data, "font_bold", False):
            lbl_font.setBold(True)
        if getattr(self.data, "font_italic", False):
            lbl_font.setItalic(True)
        if getattr(self.data, "font_underline", False):
            lbl_font.setUnderline(True)
        self.label_item.setFont(lbl_font)
        self.label_item.setDefaultTextColor(QColor(self.data.font_color))
        self._position_label()

    def _position_label(self) -> None:
        """Center the editable label within the node bounds."""
        label_rect = self.label_item.boundingRect()
        self.label_item.setPos(
            (self.data.width - label_rect.width()) / 2,
            (self.data.height - label_rect.height()) / 2,
        )

    def boundingRect(self) -> QRectF:
        # Add margin for the top connection handle so that it is included in the redraw region
        top_offset = self.connection_handle_size / 2
        return QRectF(0, -top_offset, self.data.width, self.data.height + top_offset)

    def paint(self, painter, option, widget=None):
        rect = QRectF(0, 0, self.data.width, self.data.height)
        # For linking phrases, do not draw any fill or border. Concepts retain their colours.
        if self.data.is_linker:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(Qt.NoPen)
            radius = 2
        else:
            painter.setBrush(QBrush(QColor(self.data.fill_color)))
            painter.setPen(QPen(QColor(self.data.border_color), self.data.border_width))
            radius = 8
        painter.drawRoundedRect(rect, radius, radius)
        # Highlight if selected
        if self.isSelected():
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#00AEEF"), 2))
            painter.drawRoundedRect(rect, radius, radius)
        if self.isSelected() or self.hovered:
            # Resize handle
            painter.setBrush(QBrush(QColor("#888888")))
            painter.setPen(Qt.NoPen)
            resize_rect = QRectF(
                rect.right() - self.handle_size,
                rect.bottom() - self.handle_size,
                self.handle_size,
                self.handle_size,
            )
            painter.drawRect(resize_rect)
            # Connection handle (top centre)
            painter.setBrush(QBrush(QColor("#6666FF")))
            ch_size = self.connection_handle_size
            ch_x = (rect.width() - ch_size) / 2
            ch_y = -ch_size / 2  # Half above the node
            painter.drawEllipse(
                QPointF(ch_x + ch_size / 2, ch_y + ch_size / 2),
                ch_size / 2,
                ch_size / 2,
            )

    def anchor_positions(self) -> List[QPointF]:
        """Return available anchor points on the node for connections.

        The node exposes a grid of anchor points along each edge so that
        connection lines can be attached with fine granularity.  We divide
        each side into six segments which yields seven anchor points along
        the top and bottom edges and five additional points along the left
        and right edges (excluding the corners which are already provided by
        the top/bottom loops).  In total this provides twenty-four possible
        anchors.
        """
        w = self.data.width
        h = self.data.height
        anchors: List[QPointF] = []
        divisions = 6  # number of segments per side

        # Top and bottom edges including corners
        for i in range(divisions + 1):
            x = w * i / divisions
            anchors.append(QPointF(x, 0))
            anchors.append(QPointF(x, h))

        # Left and right edges excluding corners (already added above)
        for i in range(1, divisions):
            y = h * i / divisions
            anchors.append(QPointF(0, y))
            anchors.append(QPointF(w, y))

        return anchors

    def show_anchor_points(self) -> None:
        """Display visual markers for all anchor positions."""
        if hasattr(self, "_anchor_markers") and self._anchor_markers:
            positions = self.anchor_positions()
            for marker, pos in zip(self._anchor_markers, positions):
                marker.setPos(pos)
                marker.show()
            return
        self._anchor_markers: List[QGraphicsEllipseItem] = []
        for pos in self.anchor_positions():
            marker = QGraphicsEllipseItem(-3, -3, 6, 6, self)
            marker.setBrush(QBrush(QColor("#00AEEF")))
            marker.setPen(QPen(Qt.NoPen))
            marker.setZValue(1000)
            marker.setPos(pos)
            self._anchor_markers.append(marker)

    def hide_anchor_points(self) -> None:
        """Hide previously shown anchor markers."""
        if not hasattr(self, "_anchor_markers"):
            return
        for marker in self._anchor_markers:
            marker.hide()

    def update_effect(self) -> None:
        """Update drop shadow glow based on selection/hover state."""
        if self.isSelected() or self.hovered:
            if not self.graphicsEffect():
                effect = QGraphicsDropShadowEffect()
                effect.setBlurRadius(15)
                effect.setColor(QColor("#00AEEF"))
                effect.setOffset(0)
                self.setGraphicsEffect(effect)
        else:
            self.setGraphicsEffect(None)

    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update_effect()
        self.show_anchor_points()
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update_effect()
        if not self.isSelected():
            self.hide_anchor_points()
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Determine if we clicked on resize handle
            pos = event.pos()
            rect = QRectF(0, 0, self.data.width, self.data.height)
            handle_rect = QRectF(rect.right() - self.handle_size, rect.bottom() - self.handle_size,
                                  self.handle_size, self.handle_size)
            ch_size = self.connection_handle_size
            ch_rect = QRectF((rect.width() - ch_size) / 2, -ch_size / 2, ch_size, ch_size)
            if handle_rect.contains(pos):
                self.resizing = True
                event.accept()
                return
            elif ch_rect.contains(pos):
                # Start connection drag
                self.dragging_connection = True
                self.request_connection.emit(self, event)
                event.accept()
                return
            elif event.modifiers() & Qt.ShiftModifier:
                # Toggle selection without affecting others
                self.setSelected(not self.isSelected())
                self.node_selected.emit(self)
                event.accept()
                return
        super().mousePressEvent(event)
        # Selection
        if event.button() == Qt.LeftButton:
            self.node_selected.emit(self)

    def mouseMoveEvent(self, event):
        if self.resizing:
            # Resize node; adjust width/height but not below text bounds
            new_w = max(event.pos().x(), 40.0)
            new_h = max(event.pos().y(), 20.0)
            # Use text bounding for minimal
            font = QFont(self.data.font_family, int(self.data.font_size))
            if getattr(self.data, "font_bold", False):
                font.setBold(True)
            if getattr(self.data, "font_italic", False):
                font.setItalic(True)
            if getattr(self.data, "font_underline", False):
                font.setUnderline(True)
            metrics = QFontMetrics(font)
            text_rect = metrics.boundingRect(self.data.label)
            min_w = text_rect.width() + 20
            min_h = text_rect.height() + 20
            if new_w < min_w:
                new_w = min_w
            if new_h < min_h:
                new_h = min_h
            # Inform the scene of an upcoming geometry change before mutating
            # the node's dimensions.  Without this call the scene's internal
            # bookkeeping becomes inconsistent, which manifested as items
            # jumping to the corner of the view during interactions.
            self.prepareGeometryChange()
            self.data.width = new_w
            self.data.height = new_h
            self.update()
            self._position_label()
            # Update connections
            for conn in self.connections:
                conn.update_position()
            return
        elif self.dragging_connection:
            # Inform editor of drag movement
            self.request_connection.emit(self, event)
            return
        else:
            super().mouseMoveEvent(event)
            # Update connections when moving
            for conn in self.connections:
                conn.update_position()

    def mouseReleaseEvent(self, event):
        if self.resizing:
            self.resizing = False
            event.accept()
            return
        elif self.dragging_connection:
            # End connection drag: let the scene handle creation
            self.dragging_connection = False
            self.request_connection.emit(self, event)
            event.accept()
            return
        else:
            super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            # Update model
            self.data.x = self.pos().x()
            self.data.y = self.pos().y()
            # Update connections
            for conn in self.connections:
                conn.update_position()
            # Notify editor
            self.node_moved.emit(self)
        elif change == QGraphicsItem.ItemSelectedHasChanged:
            self.update_effect()
            if bool(value):
                self.show_anchor_points()
            elif not self.hovered:
                self.hide_anchor_points()
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Allow the embedded label to receive mouse events only while the
            # user is actively editing the text.  This prevents the label from
            # being treated as a separately selectable item once editing is
            # finished, which previously caused the side styling panel to stay
            # disabled.
            self.label_item.setAcceptedMouseButtons(Qt.LeftButton)
            self.label_item.setTextInteractionFlags(Qt.TextEditorInteraction)
            self.label_item.setFlag(QGraphicsItem.ItemIsSelectable, False)
            self.label_item.setFocus(Qt.MouseFocusReason)
            cursor = self.label_item.textCursor()
            cursor.select(QTextCursor.Document)
            self.label_item.setTextCursor(cursor)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def finish_editing(self, text: str) -> None:
        old_text = self.data.label
        new_text = text.strip() or "????"
        if new_text != old_text:
            editor = self.scene().parent()
            if hasattr(editor, "push_undo_state"):
                editor.push_undo_state()
            if hasattr(editor, "set_modified"):
                editor.set_modified(True)
        self.data.label = new_text
        self._update_bounds()
        self.update()
        for conn in self.connections:
            conn.update_position()


def closest_anchors(source: NodeItem, dest: NodeItem) -> Tuple[int, int]:
    """Return the pair of anchor indices that best align the nodes.

    The original implementation snapped connections to the centre point of
    the side facing the other node.  This version keeps the same idea but
    allows the connection to attach to any of the multiple anchor points
    available along that side.  The chosen anchor is the one whose position
    along the edge is closest to the other node's centre.
    """

    # Scene positions of node centres
    s_center_scene = source.mapToScene(
        QPointF(source.data.width / 2, source.data.height / 2)
    )
    d_center_scene = dest.mapToScene(
        QPointF(dest.data.width / 2, dest.data.height / 2)
    )

    dx = d_center_scene.x() - s_center_scene.x()
    dy = d_center_scene.y() - s_center_scene.y()
    divisions = 6  # must match NodeItem.anchor_positions

    if abs(dx) >= abs(dy):
        # Nodes relate horizontally -> use left/right sides
        # Determine vertical ratios in local coordinates of each node
        d_in_source = source.mapFromScene(d_center_scene)
        s_in_dest = dest.mapFromScene(s_center_scene)
        r_src = max(0.0, min(1.0, d_in_source.y() / source.data.height))
        r_dst = max(0.0, min(1.0, s_in_dest.y() / dest.data.height))
        idx_src = int(round(r_src * divisions))
        idx_dst = int(round(r_dst * divisions))

        right_side = [12, 15, 17, 19, 21, 23, 13]
        left_side = [0, 14, 16, 18, 20, 22, 1]
        if dx >= 0:
            return right_side[idx_src], left_side[idx_dst]
        return left_side[idx_src], right_side[idx_dst]

    # Nodes relate vertically -> use top/bottom sides
    d_in_source = source.mapFromScene(d_center_scene)
    s_in_dest = dest.mapFromScene(s_center_scene)
    r_src = max(0.0, min(1.0, d_in_source.x() / source.data.width))
    r_dst = max(0.0, min(1.0, s_in_dest.x() / dest.data.width))
    idx_src = int(round(r_src * divisions))
    idx_dst = int(round(r_dst * divisions))

    bottom_side = [1, 3, 5, 7, 9, 11, 13]
    top_side = [0, 2, 4, 6, 8, 10, 12]
    if dy >= 0:
        return bottom_side[idx_src], top_side[idx_dst]
    return top_side[idx_src], bottom_side[idx_dst]


class ConnectionItem(QGraphicsLineItem):
    """Represents a connection line between two nodes."""

    def __init__(self, source: NodeItem, source_anchor: int, dest: NodeItem, dest_anchor: int) -> None:
        super().__init__()
        self.source = source
        self.dest = dest
        self.source_anchor = source_anchor
        self.dest_anchor = dest_anchor
        self.arrow_start = False
        self.arrow_end = True
        # Handles for anchor repositioning
        self.source_handle: Optional[AnchorHandle] = None
        self.dest_handle: Optional[AnchorHandle] = None
        # Update line
        self.update_position()
        # Connect to nodes
        source.connections.append(self)
        dest.connections.append(self)
        # Interactivity
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.hovered = False
        self.setAcceptHoverEvents(True)
        self.setZValue(-1)
        self.setPen(QPen(QColor("#000000"), 1))

    def update_position(self) -> None:
        """Recompute the endpoints of the line based on anchor positions."""
        # Notify the scene that the geometry will change; skipping this can
        # leave the scene's index inconsistent and has been observed to cause
        # connected concepts to jump to the origin during edits.
        self.prepareGeometryChange()
        s_anchor = self.source.anchor_positions()[self.source_anchor]
        d_anchor = self.dest.anchor_positions()[self.dest_anchor]
        # Map to scene coordinates
        s_point = self.source.mapToScene(s_anchor)
        d_point = self.dest.mapToScene(d_anchor)
        self.setLine(s_point.x(), s_point.y(), d_point.x(), d_point.y())

    def update_effect(self) -> None:
        if self.isSelected() or self.hovered:
            if not self.graphicsEffect():
                effect = QGraphicsDropShadowEffect()
                effect.setBlurRadius(15)
                effect.setColor(QColor("#00AEEF"))
                effect.setOffset(0)
                self.setGraphicsEffect(effect)
        else:
            self.setGraphicsEffect(None)

    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update_effect()
        self.show_anchor_handles()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update_effect()
        if not self.isSelected():
            self.hide_anchor_handles()
        super().hoverLeaveEvent(event)

    def show_anchor_handles(self):
        """Display anchor handles at each end."""
        if self.source_handle or self.dest_handle:
            return
        self.source_handle = AnchorHandle(self, True)
        self.dest_handle = AnchorHandle(self, False)
        scene = self.scene()
        if scene:
            scene.addItem(self.source_handle)
            scene.addItem(self.dest_handle)
        # Position handles at anchor points
        s_anchor = self.source.anchor_positions()[self.source_anchor]
        d_anchor = self.dest.anchor_positions()[self.dest_anchor]
        self.source_handle.setPos(self.source.mapToScene(s_anchor))
        self.dest_handle.setPos(self.dest.mapToScene(d_anchor))
        # Show all available anchor positions on both nodes for guidance
        self.source.show_anchor_points()
        self.dest.show_anchor_points()

    def hide_anchor_handles(self):
        """Remove anchor handles from scene."""
        if self.source_handle:
            scene = self.scene()
            if scene:
                scene.removeItem(self.source_handle)
                scene.removeItem(self.dest_handle)
            self.source_handle = None
            self.dest_handle = None
        # Hide anchor indicators
        self.source.hide_anchor_points()
        self.dest.hide_anchor_points()

    def paint(self, painter, option, widget=None):
        pen = QPen(QColor("#000000"), 1)
        if self.isSelected():
            pen = QPen(QColor("#00AEEF"), 2)
        painter.setPen(pen)
        line = self.line()
        painter.drawLine(line)
        angle = math.atan2(line.dy(), line.dx())
        arrow_size = 10
        if self.arrow_end:
            p = line.p2()
            dest_arrow = [
                p,
                QPointF(p.x() - arrow_size * math.cos(angle - math.pi / 6), p.y() - arrow_size * math.sin(angle - math.pi / 6)),
                QPointF(p.x() - arrow_size * math.cos(angle + math.pi / 6), p.y() - arrow_size * math.sin(angle + math.pi / 6)),
            ]
            painter.setBrush(pen.color())
            painter.drawPolygon(*dest_arrow)
        if self.arrow_start:
            p = line.p1()
            start_arrow = [
                p,
                QPointF(p.x() + arrow_size * math.cos(angle - math.pi / 6), p.y() + arrow_size * math.sin(angle - math.pi / 6)),
                QPointF(p.x() + arrow_size * math.cos(angle + math.pi / 6), p.y() + arrow_size * math.sin(angle + math.pi / 6)),
            ]
            painter.setBrush(pen.color())
            painter.drawPolygon(*start_arrow)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.update_effect()
            if bool(value):
                self.show_anchor_handles()
            elif not self.hovered:
                self.hide_anchor_handles()
        return super().itemChange(change, value)


###################################################
# Main editor window with scene, view and style UI
###################################################

class GraphicsView(QGraphicsView):
    """Graphics view that supports mouse-based panning and zooming."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._panning = False
        self._last_pos = QPoint()
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        # Enable gesture recognition for pinch-to-zoom
        self.viewport().grabGesture(Qt.PinchGesture)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.itemAt(event.pos()):
            scene = self.scene()
            if scene:
                scene.clearSelection()
            super().mousePressEvent(event)
            return
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = True
            self._last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Pan with trackpad scroll and zoom with a real mouse wheel."""
        if event.source() == Qt.MouseEventNotSynthesized:
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self.scale(factor, factor)
        else:
            delta = event.pixelDelta()
            if delta.manhattanLength() == 0:
                delta = event.angleDelta()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

    def event(self, event):
        if event.type() == QEvent.Gesture:
            return self._handle_gesture(event)
        return super().event(event)

    def _handle_gesture(self, event):
        pinch = event.gesture(Qt.PinchGesture)
        if isinstance(pinch, QPinchGesture):
            factor = pinch.scaleFactor() / (pinch.lastScaleFactor() or 1)
            self.scale(factor, factor)
            return True
        return False

class StyleDialog(QDialog):
    """Modal dialog for editing the style of one or more nodes or connections."""

    def __init__(self, nodes: Optional[List[NodeItem]] = None,
                 connections: Optional[List[ConnectionItem]] = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.nodes = nodes or []
        self.connections = connections or []
        # Convenience references to the first selected item
        self.node = self.nodes[0] if self.nodes else None
        self.connection = self.connections[0] if self.connections else None
        self.setWindowTitle("Modifica stile")
        layout = QVBoxLayout(self)
        self.label_edit = QLineEdit()
        layout.addWidget(QLabel("Etichetta:"))
        layout.addWidget(self.label_edit)
        self.font_combo = QFontComboBox()
        layout.addWidget(QLabel("Font:"))
        layout.addWidget(self.font_combo)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 72)
        layout.addWidget(QLabel("Dimensione font:"))
        layout.addWidget(self.font_size_spin)
        self.bold_check = QCheckBox("Grassetto")
        self.italic_check = QCheckBox("Corsivo")
        self.underline_check = QCheckBox("Sottolineato")
        layout.addWidget(self.bold_check)
        layout.addWidget(self.italic_check)
        layout.addWidget(self.underline_check)
        self.font_color_btn = QPushButton("Scegli colore testo")
        layout.addWidget(self.font_color_btn)
        self.fill_btn = QPushButton("Scegli colore riempimento")
        layout.addWidget(self.fill_btn)
        self.border_btn = QPushButton("Scegli colore bordo")
        layout.addWidget(self.border_btn)
        self.border_thick_spin = QSpinBox()
        self.border_thick_spin.setRange(0, 10)
        layout.addWidget(QLabel("Spessore bordo:"))
        layout.addWidget(self.border_thick_spin)
        self.arrow_start_btn = QCheckBox("Freccia iniziale")
        self.arrow_end_btn = QCheckBox("Freccia finale")
        layout.addWidget(self.arrow_start_btn)
        layout.addWidget(self.arrow_end_btn)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)
        button_box.accepted.connect(self.apply)
        button_box.rejected.connect(self._init_values)
        self.font_color_btn.clicked.connect(self.choose_font_color)
        self.fill_btn.clicked.connect(self.choose_fill_color)
        self.border_btn.clicked.connect(self.choose_border_color)
        self._init_values()

    def closeEvent(self, event):  # type: ignore[override]
        event.ignore()

    def update_selection(self, nodes: Optional[List[NodeItem]] = None,
                         connections: Optional[List[ConnectionItem]] = None) -> None:
        self.nodes = nodes or []
        self.connections = connections or []
        self.node = self.nodes[0] if self.nodes else None
        self.connection = self.connections[0] if self.connections else None
        self._init_values()

    def _init_values(self) -> None:
        if self.node:
            self.label_edit.show()
            self.font_combo.show()
            self.font_size_spin.show()
            self.bold_check.show()
            self.italic_check.show()
            self.underline_check.show()
            self.font_color_btn.show()
            self.fill_btn.show()
            self.border_btn.show()
            self.border_thick_spin.show()
            data = self.node.data
            self.label_edit.setText(data.label)
            self.font_combo.setCurrentFont(QFont(data.font_family))
            self.font_size_spin.setValue(int(data.font_size))
            self.bold_check.setChecked(getattr(data, "font_bold", False))
            self.italic_check.setChecked(getattr(data, "font_italic", False))
            self.underline_check.setChecked(getattr(data, "font_underline", False))
            self.font_color_btn.setStyleSheet(f"background-color: {data.font_color}")
            self.fill_btn.setStyleSheet(f"background-color: {data.fill_color}")
            self.border_btn.setStyleSheet(f"background-color: {data.border_color}")
            self.border_thick_spin.setValue(int(data.border_width))
            self.arrow_start_btn.hide()
            self.arrow_end_btn.hide()
        elif self.connection:
            self.label_edit.hide()
            self.font_combo.hide()
            self.font_size_spin.hide()
            self.bold_check.hide()
            self.italic_check.hide()
            self.underline_check.hide()
            self.font_color_btn.hide()
            self.fill_btn.hide()
            self.border_btn.hide()
            self.border_thick_spin.hide()
            self.arrow_start_btn.show()
            self.arrow_end_btn.show()
            self.arrow_start_btn.setChecked(self.connection.arrow_start)
            self.arrow_end_btn.setChecked(self.connection.arrow_end)
        else:
            self.label_edit.hide()
            self.font_combo.hide()
            self.font_size_spin.hide()
            self.bold_check.hide()
            self.italic_check.hide()
            self.underline_check.hide()
            self.font_color_btn.hide()
            self.fill_btn.hide()
            self.border_btn.hide()
            self.border_thick_spin.hide()
            self.arrow_start_btn.hide()
            self.arrow_end_btn.hide()

    def choose_fill_color(self) -> None:
        if not self.nodes:
            return
        colour = QColorDialog.getColor(QColor(self.nodes[0].data.fill_color), self, "Colore riempimento")
        if colour.isValid():
            self.fill_btn.setStyleSheet(f"background-color: {colour.name()}")
            for n in self.nodes:
                n.data.fill_color = colour.name()
                n.update()

    def choose_border_color(self) -> None:
        if not self.nodes:
            return
        colour = QColorDialog.getColor(QColor(self.nodes[0].data.border_color), self, "Colore bordo")
        if colour.isValid():
            self.border_btn.setStyleSheet(f"background-color: {colour.name()}")
            for n in self.nodes:
                n.data.border_color = colour.name()
                n.update()

    def choose_font_color(self) -> None:
        if not self.nodes:
            return
        colour = QColorDialog.getColor(QColor(self.nodes[0].data.font_color), self, "Colore testo")
        if colour.isValid():
            self.font_color_btn.setStyleSheet(f"background-color: {colour.name()}")
            for n in self.nodes:
                n.data.font_color = colour.name()
                n.update()

    def apply(self) -> None:
        parent = self.parent()
        if isinstance(parent, ConceptMapEditor):
            parent.push_undo_state()
        if self.nodes:
            text = self.label_edit.text()
            for node in self.nodes:
                if len(self.nodes) == 1 and text:
                    node.data.label = text
                node.data.font_family = self.font_combo.currentFont().family()
                node.data.font_size = self.font_size_spin.value()
                node.data.font_bold = self.bold_check.isChecked()
                node.data.font_italic = self.italic_check.isChecked()
                node.data.font_underline = self.underline_check.isChecked()
                node.data.border_width = self.border_thick_spin.value()
                node.update()
                node._update_bounds()
                for conn in node.connections:
                    conn.update_position()
        elif self.connections:
            arrow_start = self.arrow_start_btn.isChecked()
            arrow_end = self.arrow_end_btn.isChecked()
            for conn in self.connections:
                conn.arrow_start = arrow_start
                conn.arrow_end = arrow_end
                conn.update()
        if isinstance(parent, ConceptMapEditor):
            parent.update_model_from_scene()

class ConceptMapEditor(QMainWindow):
    """Main window for concept map editing."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CXL Concept Map Editor (PyQt)")
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "antmapicon.png")))
        self.resize(1024, 768)
        # Data model
        self.document = CXLDocument()
        # Scene and view
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QBrush(Qt.white))
        self.view = GraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setDragMode(QGraphicsView.RubberBandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setBackgroundBrush(QBrush(Qt.white))
        self.setCentralWidget(self.view)
        # Actions and menu
        self._create_actions()
        self._create_menu()
        self._create_toolbar()
        self.statusBar().showMessage("Pronto")
        self.style_panel = StyleDialog(parent=self)
        self.style_panel.setWindowModality(Qt.NonModal)
        self.style_panel.show()
        self._disable_style_panel()
        # Map state
        self.node_items: Dict[str, NodeItem] = {}
        self.connection_items: List[ConnectionItem] = []
        self.temp_new_node: Optional[NodeItem] = None
        self.temp_connection: Optional[ConnectionItem] = None
        # Scene events
        self.scene.mouseDoubleClickEvent = self.scene_double_click
        self.scene.selectionChanged.connect(self.selection_changed)
        # Undo/redo stacks
        self.undo_stack: List[CXLDocument] = []
        self.redo_stack: List[CXLDocument] = []
        self._dirty: bool = False
        self.push_undo_state(mark_dirty=False)
        self.update_window_title()

    def _create_actions(self) -> None:
        style = self.style()
        self.new_act = QAction(style.standardIcon(QStyle.SP_FileIcon), "Nuovo", self)
        self.open_act = QAction(style.standardIcon(QStyle.SP_DirOpenIcon), "Apri…", self)
        self.save_act = QAction(style.standardIcon(QStyle.SP_DialogSaveButton), "Salva", self)
        self.save_as_act = QAction(style.standardIcon(QStyle.SP_DialogSaveButton), "Salva come…", self)
        self.exit_act = QAction(style.standardIcon(QStyle.SP_DialogCloseButton), "Esci", self)
        self.zoom_in_act = QAction(style.standardIcon(QStyle.SP_ArrowUp), "Zoom in", self)
        self.zoom_in_act.setShortcut(QKeySequence.ZoomIn)
        self.zoom_out_act = QAction(style.standardIcon(QStyle.SP_ArrowDown), "Zoom out", self)
        self.zoom_out_act.setShortcut(QKeySequence.ZoomOut)
        self.autofit_act = QAction(style.standardIcon(QStyle.SP_BrowserReload), "Autofit", self)
        self.edit_style_act = QAction(style.standardIcon(QStyle.SP_DialogApplyButton), "Modifica stile", self)
        self.edit_style_act.setShortcut(QKeySequence("Ctrl+E"))
        self.edit_style_act.triggered.connect(self.edit_style)
        self.copy_act = QAction(style.standardIcon(QStyle.SP_FileDialogNewFolder), "Copia", self)
        self.copy_act.setShortcut(QKeySequence.Copy)
        self.copy_act.triggered.connect(self.copy_selection)
        self.paste_act = QAction(style.standardIcon(QStyle.SP_FileDialogContentsView), "Incolla", self)
        self.paste_act.setShortcut(QKeySequence.Paste)
        self.paste_act.triggered.connect(self.paste_selection)
        self.export_pdf_act = QAction(style.standardIcon(QStyle.SP_FileIcon), "Esporta PDF", self)
        self.export_pdf_act.triggered.connect(self.export_pdf)
        printer_pixmap = getattr(QStyle, "SP_PrinterIcon", QStyle.SP_FileIcon)
        self.print_act = QAction(style.standardIcon(printer_pixmap), "Stampa", self)
        self.print_act.triggered.connect(self.print_map)
        # Connect actions
        self.new_act.triggered.connect(self.new_file)
        self.open_act.triggered.connect(self.open_file)
        self.save_act.triggered.connect(self.save_file)
        self.save_as_act.triggered.connect(self.save_file_as)
        self.exit_act.triggered.connect(self.close)
        self.zoom_in_act.triggered.connect(self.zoom_in)
        self.zoom_out_act.triggered.connect(self.zoom_out)
        self.autofit_act.triggered.connect(self.autofit_view)

    def _create_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(self.new_act)
        file_menu.addAction(self.open_act)
        file_menu.addAction(self.save_act)
        file_menu.addAction(self.save_as_act)
        file_menu.addAction(self.export_pdf_act)
        file_menu.addAction(self.print_act)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_act)
        edit_menu = menubar.addMenu("Modifica")
        edit_menu.addAction(self.edit_style_act)
        edit_menu.addAction(self.copy_act)
        edit_menu.addAction(self.paste_act)
        view_menu = menubar.addMenu("Vista")
        view_menu.addAction(self.zoom_in_act)
        view_menu.addAction(self.zoom_out_act)
        view_menu.addAction(self.autofit_act)

    def _create_toolbar(self) -> None:
        toolbar = self.addToolBar("View")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setStyleSheet("QToolBar { background: #2d2d2d; border: none; }")
        toolbar.addAction(self.zoom_in_act)
        toolbar.addAction(self.zoom_out_act)
        toolbar.addAction(self.autofit_act)
        toolbar.addAction(self.save_act)
        toolbar.addAction(self.export_pdf_act)
        toolbar.addAction(self.print_act)

    def set_modified(self, modified: bool) -> None:
        """Update modified flag and window title."""
        self._dirty = modified
        self.update_window_title()

    def update_window_title(self) -> None:
        """Refresh window title, appending an asterisk if unsaved."""
        if self.document.filepath:
            base = self.document.filepath
        else:
            base = "Nuova mappa"
        title = f"{base} - Editor concetti (PyQt)"
        if getattr(self, "_dirty", False):
            title += " *"
        self.setWindowTitle(title)

    def maybe_save(self) -> bool:
        """Prompt to save if there are unsaved changes."""
        if not self._dirty:
            return True
        resp = QMessageBox.warning(
            self,
            "Modifiche non salvate",
            "Ci sono modifiche non salvate. Vuoi salvarle?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
        )
        if resp == QMessageBox.Yes:
            self.save_file()
            return not self._dirty
        return resp == QMessageBox.No

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.maybe_save():
            event.accept()
        else:
            event.ignore()

    def copy_selection(self) -> None:
        items = self.scene.selectedItems()
        node_items = [i for i in items if isinstance(i, NodeItem)]
        if not node_items:
            return
        selected_ids = {n.data.id for n in node_items}
        data = {
            "nodes": [
                {
                    "id": n.data.id,
                    "label": n.data.label,
                    "x": n.data.x,
                    "y": n.data.y,
                    "width": n.data.width,
                    "height": n.data.height,
                    "font_family": n.data.font_family,
                    "font_size": n.data.font_size,
                    "font_color": n.data.font_color,
                    "fill_color": n.data.fill_color,
                    "border_color": n.data.border_color,
                    "border_width": n.data.border_width,
                    "font_bold": n.data.font_bold,
                    "font_italic": n.data.font_italic,
                    "font_underline": n.data.font_underline,
                    "is_linker": n.data.is_linker,
                }
                for n in node_items
            ],
            "connections": [
                {
                    "from": c.source.data.id,
                    "to": c.dest.data.id,
                    "from_anchor": c.source_anchor,
                    "to_anchor": c.dest_anchor,
                }
                for c in self.connection_items
                if c.source.data.id in selected_ids and c.dest.data.id in selected_ids
            ],
        }
        QApplication.clipboard().setText(json.dumps(data))

    def paste_selection(self) -> None:
        try:
            data = json.loads(QApplication.clipboard().text())
        except json.JSONDecodeError:
            return
        if not data:
            return
        self.push_undo_state()
        id_map: Dict[str, str] = {}
        for node in data.get("nodes", []):
            new_id = str(uuid.uuid4())
            id_map[node["id"]] = new_id
            ndata = ConceptData(
                new_id,
                node.get("label", "????"),
                node.get("x", 0) + 20,
                node.get("y", 0) + 20,
                node.get("width", 120),
                node.get("height", 60),
                font_family=node.get("font_family", "Verdana"),
                font_size=float(node.get("font_size", 12)),
                font_color=node.get("font_color", "0,0,0,255"),
                fill_color=node.get("fill_color", "237,244,246,255"),
                border_color=node.get("border_color", "0,0,0,255"),
                border_width=float(node.get("border_width", 1)),
                font_bold=node.get("font_bold", False),
                font_italic=node.get("font_italic", False),
                font_underline=node.get("font_underline", False),
                is_linker=node.get("is_linker", False),
            )
            self.document.concepts[new_id] = ndata
            item = NodeItem(ndata, self.scene)
            item.request_connection.connect(self.handle_connection_request)
            item.node_moved.connect(self.node_moved)
            item.node_selected.connect(self.node_selected)
            self.scene.addItem(item)
            self.node_items[new_id] = item
        for conn in data.get("connections", []):
            sid = id_map.get(conn.get("from"))
            did = id_map.get(conn.get("to"))
            if sid and did:
                conn_item = ConnectionItem(
                    self.node_items[sid],
                    conn.get("from_anchor", 0),
                    self.node_items[did],
                    conn.get("to_anchor", 0),
                )
                self.scene.addItem(conn_item)
                self.connection_items.append(conn_item)
        self.update_model_from_scene()

    def export_pdf(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(self, "Esporta PDF", "", "PDF (*.pdf)")
        if not filepath:
            return
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(filepath)
        painter = QPainter(printer)
        self.view.render(painter)
        painter.end()

    def print_map(self) -> None:
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)
        if dialog.exec_() == QDialog.Accepted:
            painter = QPainter(printer)
            self.view.render(painter)
            painter.end()

    def edit_style(self) -> None:
        self.style_panel.raise_()
        self.style_panel.activateWindow()

    def zoom_in(self) -> None:
        """Zoom into the scene."""
        self.view.scale(1.2, 1.2)

    def zoom_out(self) -> None:
        """Zoom out of the scene."""
        self.view.scale(1 / 1.2, 1 / 1.2)

    def autofit_view(self) -> None:
        """Center and fit the view to show the entire map."""
        rect = self.scene.itemsBoundingRect()
        if rect.isNull():
            self.view.resetTransform()
            return
        self.view.fitInView(rect, Qt.KeepAspectRatio)
        self.view.centerOn(rect.center())

    def clear_scene(self) -> None:
        """Remove all items from scene and reset state."""
        self.scene.clear()
        self.node_items.clear()
        self.connection_items.clear()
        self.edit_style_act.setEnabled(False)

    def new_file(self) -> None:
        """Create a new blank map."""
        if not self.maybe_save():
            return
        self.document.new_map()
        self.clear_scene()
        self.set_modified(False)
        self.update_window_title()
        self.autofit_view()

    def open_file(self) -> None:
        if not self.maybe_save():
            return
        filepath, _ = QFileDialog.getOpenFileName(self, "Apri file CXL", "", "CXL (*.cxl);;Tutti i file (*)")
        if not filepath:
            return
        try:
            self.document.load(filepath)
        except Exception as exc:
            QMessageBox.critical(self, "Errore", f"Impossibile caricare il file:\n{exc}")
            return
        # Clear current map
        self.clear_scene()
        # Create NodeItems
        for cid, data in self.document.concepts.items():
            item = NodeItem(data, self.scene)
            item.request_connection.connect(self.handle_connection_request)
            item.node_moved.connect(self.node_moved)
            item.node_selected.connect(self.node_selected)
            self.scene.addItem(item)
            item.setPos(data.x, data.y)
            self.node_items[cid] = item
        # Create connections
        for conn in self.document.connections:
            src_item = self.node_items.get(conn.from_id)
            dst_item = self.node_items.get(conn.to_id)
            if not src_item or not dst_item:
                continue
            src_anchor = getattr(conn, "from_anchor", 0)
            dst_anchor = getattr(conn, "to_anchor", 0)
            connection_item = ConnectionItem(src_item, src_anchor, dst_item, dst_anchor)
            self.scene.addItem(connection_item)
            self.connection_items.append(connection_item)
        self.set_modified(False)
        self.update_window_title()
        self.autofit_view()

    def import_file(self) -> None:
        """Import a map. Currently behaves like :meth:`open_file`.

        This provides a dedicated entry point for future extensions where
        imported maps could be merged into an existing one instead of
        replacing the current scene."""
        self.open_file()

    def save_file(self) -> None:
        """Save current map."""
        if not self.document.filepath:
            self.save_file_as()
            return
        self.update_model_from_scene()
        try:
            self.document.save(self.document.filepath)
        except Exception as exc:
            QMessageBox.critical(self, "Errore", f"Errore durante il salvataggio:\n{exc}")
            return
        self.set_modified(False)
        QMessageBox.information(self, "Salvataggio", "File salvato con successo.")

    def save_file_as(self) -> None:
        """Save current map under a new name."""
        filepath, _ = QFileDialog.getSaveFileName(self, "Salva file CXL", "", "CXL (*.cxl)")
        if not filepath:
            return
        self.update_model_from_scene()
        try:
            self.document.save(filepath)
            self.document.filepath = filepath
            self.set_modified(False)
        except Exception as exc:
            QMessageBox.critical(self, "Errore", f"Errore durante il salvataggio:\n{exc}")

    def update_model_from_scene(self) -> None:
        """Push current scene state back to the document model."""
        # Update concept positions and dimensions
        for cid, item in self.node_items.items():
            self.document.concepts[cid].x = item.pos().x()
            self.document.concepts[cid].y = item.pos().y()
            self.document.concepts[cid].width = item.data.width
            self.document.concepts[cid].height = item.data.height
            self.document.concepts[cid].label = item.data.label
            self.document.concepts[cid].font_family = item.data.font_family
            self.document.concepts[cid].font_size = item.data.font_size
            self.document.concepts[cid].font_color = item.data.font_color
            self.document.concepts[cid].fill_color = item.data.fill_color
            self.document.concepts[cid].border_color = item.data.border_color
            self.document.concepts[cid].border_width = item.data.border_width
            self.document.concepts[cid].font_bold = getattr(item.data, "font_bold", False)
            self.document.concepts[cid].font_italic = getattr(item.data, "font_italic", False)
            self.document.concepts[cid].font_underline = getattr(item.data, "font_underline", False)
        # Update connections
        self.document.connections.clear()
        for conn_item in self.connection_items:
            conn = ConnectionData(
                str(uuid.uuid4()),
                conn_item.source.data.id,
                conn_item.dest.data.id,
                conn_item.arrow_start,
                conn_item.arrow_end,
                conn_item.source_anchor,
                conn_item.dest_anchor,
            )
            self.document.connections.append(conn)

    ###################
    # Scene interaction
    ###################

    def push_undo_state(self, mark_dirty: bool = True) -> None:
        self.undo_stack.append(copy.deepcopy(self.document))
        self.redo_stack.clear()
        if mark_dirty:
            self.set_modified(True)

    def rebuild_scene_from_document(self) -> None:
        self.clear_scene()
        for cid, data in self.document.concepts.items():
            item = NodeItem(data, self.scene)
            item.request_connection.connect(self.handle_connection_request)
            item.node_moved.connect(self.node_moved)
            item.node_selected.connect(self.node_selected)
            self.scene.addItem(item)
            item.setPos(data.x, data.y)
            self.node_items[cid] = item
        for conn in self.document.connections:
            src_item = self.node_items.get(conn.from_id)
            dst_item = self.node_items.get(conn.to_id)
            if src_item and dst_item:
                conn_item = ConnectionItem(src_item, conn.from_anchor, dst_item, conn.to_anchor)
                conn_item.arrow_start = conn.arrow_start
                conn_item.arrow_end = conn.arrow_end
                self.scene.addItem(conn_item)
                self.connection_items.append(conn_item)

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(copy.deepcopy(self.document))
        self.document = self.undo_stack.pop()
        self.rebuild_scene_from_document()
        self.set_modified(True)

    def redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(copy.deepcopy(self.document))
        self.document = self.redo_stack.pop()
        self.rebuild_scene_from_document()
        self.set_modified(True)

    def delete_selected(self) -> None:
        items = list(self.scene.selectedItems())
        if not items:
            return
        self.push_undo_state()
        for item in items:
            if isinstance(item, NodeItem):
                cid = item.data.id
                if item.data.is_linker:
                    related = [c for c in self.document.connections if c.from_id == cid or c.to_id == cid]
                    if len(related) == 2:
                        source_id = dest_id = None
                        for c in related:
                            if c.to_id == cid:
                                source_id = c.from_id
                            elif c.from_id == cid:
                                dest_id = c.to_id
                        self.document.connections = [c for c in self.document.connections if c not in related]
                        if source_id and dest_id:
                            self.document.connections.append(ConnectionData(str(uuid.uuid4()), source_id, dest_id))
                else:
                    self.document.connections = [c for c in self.document.connections if c.from_id != cid and c.to_id != cid]
                self.document.concepts.pop(cid, None)
            elif isinstance(item, ConnectionItem):
                self.document.connections = [
                    c for c in self.document.connections
                    if not (c.from_id == item.source.data.id and c.to_id == item.dest.data.id)
                ]
        self.rebuild_scene_from_document()

    def selection_changed(self) -> None:
        items = self.scene.selectedItems()
        node_items = [i for i in items if isinstance(i, NodeItem)]
        conn_items = [i for i in items if isinstance(i, ConnectionItem)]
        enable = (
            (len(node_items) >= 1 and not conn_items)
            or (len(conn_items) >= 1 and not node_items)
        )
        if enable:
            self.style_panel.update_selection(node_items, conn_items)
            self._enable_style_panel()
        else:
            self.style_panel.update_selection([], [])
            self._disable_style_panel()

    def _enable_style_panel(self) -> None:
        self.style_panel.setEnabled(True)
        self.style_panel.setGraphicsEffect(None)
        self.style_panel.setWindowOpacity(1.0)

    def _disable_style_panel(self) -> None:
        self.style_panel.setEnabled(False)
        blur = QGraphicsBlurEffect()
        blur.setBlurRadius(5)
        self.style_panel.setGraphicsEffect(blur)
        self.style_panel.setWindowOpacity(0.5)

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.Undo):
            self.undo()
            return
        if event.matches(QKeySequence.Redo):
            self.redo()
            return
        if event.key() == Qt.Key_Backspace:
            self.delete_selected()
            return
        super().keyPressEvent(event)
    def scene_double_click(self, event) -> None:
        """Handle double click on scene to create new concept."""
        # Determine if click on empty area
        pos = event.scenePos()
        item = self.scene.itemAt(pos, self.view.transform())
        if item is None:
            self.push_undo_state()
            # Create new concept data
            cid = str(uuid.uuid4())
            style = self.document.default_linker_style if False else self.document.default_concept_style
            data = ConceptData(
                cid, "????",
                pos.x(), pos.y(),
                120, 60,
                font_family=style.get("font-name", "Verdana"),
                font_size=float(style.get("font-size", "12")),
                font_color=self.document._parse_color(style.get("font-color", "0,0,0,255")),
                fill_color=self.document._parse_color(style.get("background-color", "237,244,246,255")),
                border_color=self.document._parse_color(style.get("border-color", "0,0,0,255")),
                border_width=float(style.get("border-thickness", "1")),
                font_bold="bold" in style.get("font-style", "plain"),
                font_italic="italic" in style.get("font-style", "plain"),
                font_underline="underline" in style.get("font-style", "plain"),
                is_linker=False
            )
            self.document.concepts[cid] = data
            # Create NodeItem
            item = NodeItem(data, self.scene)
            item.request_connection.connect(self.handle_connection_request)
            item.node_moved.connect(self.node_moved)
            item.node_selected.connect(self.node_selected)
            self.scene.addItem(item)
            item.setPos(pos)
            self.node_items[cid] = item
            self.update_model_from_scene()
            event.accept()
            return
        # Otherwise default
        QGraphicsScene.mouseDoubleClickEvent(self.scene, event)

    def handle_connection_request(self, source_node: NodeItem, event) -> None:
        """Handle interactive creation of connections and new concepts."""
        etype = event.type()
        if etype == QEvent.GraphicsSceneMousePress:
            # Reset any temporary items while ensuring existing nodes remain
            if self.temp_new_node is not None and self.temp_new_node not in self.node_items.values():
                try:
                    if self.temp_new_node.scene() is not None:
                        self.scene.removeItem(self.temp_new_node)
                except RuntimeError:
                    pass
            self.temp_new_node = None
            if self.temp_connection is not None:
                try:
                    if self.temp_connection.scene() is not None:
                        self.scene.removeItem(self.temp_connection)
                except RuntimeError:
                    pass
            self.temp_connection = None
            for n in list(self.node_items.values()):
                n.hide_anchor_points()
            return
        if etype == QEvent.GraphicsSceneMouseMove:
            pos = event.scenePos()
            if self.temp_new_node is None:
                # Start dragging a new concept
                style = self.document.default_concept_style
                data = ConceptData(
                    "temp", "????",
                    pos.x(), pos.y(),
                    120, 60,
                    font_family=style.get("font-name", "Verdana"),
                    font_size=float(style.get("font-size", "12")),
                    font_color=self.document._parse_color(style.get("font-color", "0,0,0,255")),
                    fill_color=self.document._parse_color(style.get("background-color", "237,244,246,255")),
                    border_color=self.document._parse_color(style.get("border-color", "0,0,0,255")),
                    border_width=float(style.get("border-thickness", "1")),
                    font_bold="bold" in style.get("font-style", "plain"),
                    font_italic="italic" in style.get("font-style", "plain"),
                    font_underline="underline" in style.get("font-style", "plain"),
                    is_linker=False
                )
                self.temp_new_node = NodeItem(data, self.scene)
                self.temp_new_node.request_connection.connect(self.handle_connection_request)
                self.temp_new_node.node_moved.connect(self.node_moved)
                self.temp_new_node.node_selected.connect(self.node_selected)
                self.scene.addItem(self.temp_new_node)
                self.temp_new_node.setPos(pos)
                sa, da = closest_anchors(source_node, self.temp_new_node)
                self.temp_connection = ConnectionItem(source_node, sa, self.temp_new_node, da)
                self.scene.addItem(self.temp_connection)
            else:
                self.temp_new_node.setPos(pos)
                if self.temp_connection:
                    sa, da = closest_anchors(source_node, self.temp_new_node)
                    self.temp_connection.source_anchor = sa
                    self.temp_connection.dest_anchor = da
                    self.temp_connection.update_position()
            dest_item = self.scene.itemAt(pos, self.view.transform())
            for n in self.node_items.values():
                if n in (dest_item, source_node):
                    n.show_anchor_points()
                else:
                    n.hide_anchor_points()
            return
        if etype == QEvent.GraphicsSceneMouseRelease:
            pos = event.scenePos()
            dest_item = self.scene.itemAt(pos, self.view.transform())
            if self.temp_new_node is not None:
                if isinstance(dest_item, NodeItem) and dest_item not in (source_node, self.temp_new_node):
                    # Connecting to existing node
                    self.push_undo_state()
                    if self.temp_connection:
                        if self.temp_connection.scene() is not None:
                            self.scene.removeItem(self.temp_connection)
                        self.temp_connection = None
                    if self.temp_new_node.scene() is not None:
                        self.scene.removeItem(self.temp_new_node)
                    self.temp_new_node = None
                    if dest_item.data.is_linker or source_node.data.is_linker:
                        use_linker = False
                    else:
                        modifiers = QApplication.keyboardModifiers()
                        use_linker = not (modifiers & Qt.ShiftModifier)
                    if use_linker:
                        lid = str(uuid.uuid4())
                        style = self.document.default_linker_style if self.document.default_linker_style else self.document.default_concept_style
                        data = ConceptData(
                            lid, "????",
                            (source_node.pos().x() + dest_item.pos().x()) / 2,
                            (source_node.pos().y() + dest_item.pos().y()) / 2,
                            90, 20,
                            font_family=style.get("font-name", "Verdana"),
                            font_size=float(style.get("font-size", "12")),
                            font_color=self.document._parse_color(style.get("font-color", "0,0,0,255")),
                            fill_color=self.document._parse_color(style.get("background-color", "237,244,246,255")),
                            border_color=self.document._parse_color(style.get("border-color", "0,0,0,255")),
                            border_width=float(style.get("border-thickness", "1")),
                            font_bold="bold" in style.get("font-style", "plain"),
                            font_italic="italic" in style.get("font-style", "plain"),
                            font_underline="underline" in style.get("font-style", "plain"),
                            is_linker=True
                        )
                        self.document.concepts[lid] = data
                        linker_item = NodeItem(data, self.scene)
                        linker_item.request_connection.connect(self.handle_connection_request)
                        linker_item.node_moved.connect(self.node_moved)
                        linker_item.node_selected.connect(self.node_selected)
                        self.scene.addItem(linker_item)
                        self.node_items[lid] = linker_item
                        sa1, da1 = closest_anchors(source_node, linker_item)
                        conn1 = ConnectionItem(source_node, sa1, linker_item, da1)
                        conn1.arrow_end = False
                        sa2, da2 = closest_anchors(linker_item, dest_item)
                        conn2 = ConnectionItem(linker_item, sa2, dest_item, da2)
                        self.scene.addItem(conn1)
                        self.scene.addItem(conn2)
                        self.connection_items.extend([conn1, conn2])
                    else:
                        sa, da = closest_anchors(source_node, dest_item)
                        conn = ConnectionItem(source_node, sa, dest_item, da)
                        if dest_item.data.is_linker:
                            conn.arrow_end = False
                        self.scene.addItem(conn)
                        self.connection_items.append(conn)
                    for n in self.node_items.values():
                        n.hide_anchor_points()
                    self.update_model_from_scene()
                    return
                # Finalise new concept dropped in empty space
                cid = str(uuid.uuid4())
                self.temp_new_node.data.id = cid
                self.temp_new_node.data.x = pos.x()
                self.temp_new_node.data.y = pos.y()
                self.document.concepts[cid] = self.temp_new_node.data
                self.node_items[cid] = self.temp_new_node
                if source_node.data.is_linker:
                    use_linker = False
                else:
                    modifiers = QApplication.keyboardModifiers()
                    use_linker = not (modifiers & Qt.ShiftModifier)
                if use_linker:
                    lid = str(uuid.uuid4())
                    style_l = self.document.default_linker_style if self.document.default_linker_style else self.document.default_concept_style
                    ldata = ConceptData(
                        lid, "????",
                        (source_node.pos().x() + self.temp_new_node.pos().x()) / 2,
                        (source_node.pos().y() + self.temp_new_node.pos().y()) / 2,
                        90, 20,
                        font_family=style_l.get("font-name", "Verdana"),
                        font_size=float(style_l.get("font-size", "12")),
                        font_color=self.document._parse_color(style_l.get("font-color", "0,0,0,255")),
                        fill_color=self.document._parse_color(style_l.get("background-color", "237,244,246,255")),
                        border_color=self.document._parse_color(style_l.get("border-color", "0,0,0,255")),
                        border_width=float(style_l.get("border-thickness", "1")),
                        font_bold="bold" in style_l.get("font-style", "plain"),
                        font_italic="italic" in style_l.get("font-style", "plain"),
                        font_underline="underline" in style_l.get("font-style", "plain"),
                        is_linker=True
                    )
                    self.document.concepts[lid] = ldata
                    linker_item2 = NodeItem(ldata, self.scene)
                    linker_item2.request_connection.connect(self.handle_connection_request)
                    linker_item2.node_moved.connect(self.node_moved)
                    linker_item2.node_selected.connect(self.node_selected)
                    self.scene.addItem(linker_item2)
                    self.node_items[lid] = linker_item2
                    if self.temp_connection:
                        if self.temp_connection.scene() is not None:
                            self.scene.removeItem(self.temp_connection)
                    sa1, da1 = closest_anchors(source_node, linker_item2)
                    conn1 = ConnectionItem(source_node, sa1, linker_item2, da1)
                    conn1.arrow_end = False
                    sa2, da2 = closest_anchors(linker_item2, self.temp_new_node)
                    conn2 = ConnectionItem(linker_item2, sa2, self.temp_new_node, da2)
                    self.scene.addItem(conn1)
                    self.scene.addItem(conn2)
                    self.connection_items.extend([conn1, conn2])
                else:
                    if self.temp_connection:
                        sa, da = closest_anchors(source_node, self.temp_new_node)
                        self.temp_connection.source_anchor = sa
                        self.temp_connection.dest_anchor = da
                        self.temp_connection.update_position()
                        self.connection_items.append(self.temp_connection)
                    else:
                        sa, da = closest_anchors(source_node, self.temp_new_node)
                        conn = ConnectionItem(source_node, sa, self.temp_new_node, da)
                        self.scene.addItem(conn)
                        self.connection_items.append(conn)
                self.temp_new_node = None
                self.temp_connection = None
                for n in self.node_items.values():
                    n.hide_anchor_points()
                self.update_model_from_scene()
                return
        # No temp node: maybe connecting existing nodes directly
        if isinstance(dest_item, NodeItem) and dest_item is not source_node:
            self.push_undo_state()
            if dest_item.data.is_linker or source_node.data.is_linker:
                use_linker = False
            else:
                modifiers = QApplication.keyboardModifiers()
                use_linker = not (modifiers & Qt.ShiftModifier)
                if use_linker:
                    lid = str(uuid.uuid4())
                    style = self.document.default_linker_style if self.document.default_linker_style else self.document.default_concept_style
                    data = ConceptData(
                        lid, "????",
                        (source_node.pos().x() + dest_item.pos().x()) / 2,
                        (source_node.pos().y() + dest_item.pos().y()) / 2,
                        90, 20,
                        font_family=style.get("font-name", "Verdana"),
                        font_size=float(style.get("font-size", "12")),
                        font_color=self.document._parse_color(style.get("font-color", "0,0,0,255")),
                        fill_color=self.document._parse_color(style.get("background-color", "237,244,246,255")),
                        border_color=self.document._parse_color(style.get("border-color", "0,0,0,255")),
                        border_width=float(style.get("border-thickness", "1")),
                        font_bold="bold" in style.get("font-style", "plain"),
                        font_italic="italic" in style.get("font-style", "plain"),
                        font_underline="underline" in style.get("font-style", "plain"),
                        is_linker=True
                    )
                    self.document.concepts[lid] = data
                    linker_item = NodeItem(data, self.scene)
                    linker_item.request_connection.connect(self.handle_connection_request)
                    linker_item.node_moved.connect(self.node_moved)
                    linker_item.node_selected.connect(self.node_selected)
                    self.scene.addItem(linker_item)
                    self.node_items[lid] = linker_item
                    sa1, da1 = closest_anchors(source_node, linker_item)
                    conn1 = ConnectionItem(source_node, sa1, linker_item, da1)
                    conn1.arrow_end = False
                    sa2, da2 = closest_anchors(linker_item, dest_item)
                    conn2 = ConnectionItem(linker_item, sa2, dest_item, da2)
                    self.scene.addItem(conn1)
                    self.scene.addItem(conn2)
                    self.connection_items.extend([conn1, conn2])
                else:
                    sa, da = closest_anchors(source_node, dest_item)
                    conn = ConnectionItem(source_node, sa, dest_item, da)
                    if dest_item.data.is_linker:
                        conn.arrow_end = False
                    self.scene.addItem(conn)
                    self.connection_items.append(conn)
                for n in self.node_items.values():
                    n.hide_anchor_points()
                self.update_model_from_scene()
        else:
            for n in self.node_items.values():
                n.hide_anchor_points()
        return

    def node_moved(self, node: NodeItem) -> None:
        """Mark document as modified when a node moves."""
        self.set_modified(True)

    def node_selected(self, node: NodeItem) -> None:
        self.selection_changed()


class StartupDialog(QDialog):
    """Simple start-up dialog offering basic file operations."""

    def __init__(self, editor: "ConceptMapEditor") -> None:
        super().__init__(editor)
        self.editor = editor
        self.setWindowTitle("Benvenuto")
        layout = QVBoxLayout(self)
        img_label = QLabel()
        pixmap = QPixmap(os.path.join(os.path.dirname(__file__), "intro.png"))
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(img_label)
        new_btn = QPushButton("Nuova mappa")
        open_btn = QPushButton("Apri...")
        quit_btn = QPushButton("Esci")
        for btn in (new_btn, open_btn, quit_btn):
            layout.addWidget(btn)
        new_btn.clicked.connect(self.handle_new)
        open_btn.clicked.connect(self.handle_open)
        quit_btn.clicked.connect(self.reject)

    def handle_new(self) -> None:
        self.editor.new_file()
        self.accept()

    def handle_open(self) -> None:
        self.editor.open_file()
        self.accept()

def main() -> None:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "antmapicon.png")))
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    app.setStyleSheet(
        "QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }"
    )
    editor = ConceptMapEditor()
    start = StartupDialog(editor)
    if start.exec_() == QDialog.Accepted:
        editor.show()
        editor.autofit_view()
        sys.exit(app.exec_())
    sys.exit(0)


if __name__ == "__main__":
    main()
