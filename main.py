"""
PyQt5-based concept map editor supporting IHMC CXL files.

This application implements an interactive editor for concept maps
similar to IHMC CmapTools.  It can load and save CXL files,
create new maps, and allows the user to add, resize and connect
concepts (nodes) and linking phrases via an intuitive graphical
interface.

Features:

* **Double click** on empty canvas creates a new concept at that
  location with a text label.  A dialog prompts for the label.

* Each concept has a small **resize handle** in its bottom right
  corner.  Drag this handle to resize the concept; resizing is
  constrained so that the label text always fits within the
  bounding box.

* Each concept also has a **connection handle** at its top centre.
  Dragging this handle will start a connection.  When the drag ends:
  - If released over an existing concept, a connection is made
    between the two concepts.  If the **Shift** key is held during
    this operation, the connection is direct.  Otherwise, a
    linking phrase is created with a default label (prompted via
    dialog) and connected accordingly.
  - If released over empty space, a new concept is created at the
    drop location and connected from the original concept.  The
    Shift modifier similarly controls whether a linking phrase is
    inserted.

* **Dragging** a concept moves it around the canvas; all attached
  connections update dynamically.

* **Clicking** on a connection displays small handles at its
  endpoints.  Drag these handles to re-anchor the connection to
  different positions on the concept (top, bottom, left or right).

* **Selection and style editing**: clicking on a concept selects it
  and opens a docked style editor panel where properties such as
  label, font size, fill colour, border colour and border
  thickness can be modified.  Changes are reflected immediately.

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
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import (
    Qt,
    QRectF,
    QPointF,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QColor,
    QPen,
    QBrush,
    QFont,
    QFontMetrics,
    QPainter,
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
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QColorDialog,
    QFileDialog,
    QInputDialog,
    QAction,
    QMenu,
    QMessageBox,
)


##############################
# Data model and CXL parsing #
##############################

class ConceptData:
    """Represents conceptual data for a concept or linking phrase."""

    def __init__(self, cid: str, label: str, x: float, y: float, width: float, height: float,
                 font_family: str = "Verdana", font_size: float = 12.0,
                 font_color: str = "0,0,0,255", fill_color: str = "237,244,246,255",
                 border_color: str = "0,0,0,255", border_width: float = 1.0,
                 is_linker: bool = False) -> None:
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
        self.is_linker = is_linker


class ConnectionData:
    """Represents connection data between two node ids."""

    def __init__(self, cid: str, from_id: str, to_id: str) -> None:
        self.id = cid
        self.from_id = from_id
        self.to_id = to_id


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
        map_elem = self.root.find("c:map", self.ns) or self.root.find("map")
        if map_elem is None:
            raise ValueError("Il file non contiene un elemento <map>.")
        # Determine format
        self.appearance_mode = bool(map_elem.find("c:concept-appearance-list", self.ns))
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
            # Parse connections
            for conn in map_elem.findall("c:connection-list/c:connection", self.ns):
                cid = conn.get("id") or str(uuid.uuid4())
                from_id = conn.get("from-id") or ""
                to_id = conn.get("to-id") or ""
                self.connections.append(ConnectionData(cid, from_id, to_id))
            # Extract default styles
            ss = map_elem.find("c:style-sheet-list/c:style-sheet", self.ns)
            if ss is not None:
                c_style = ss.find("c:concept-style", self.ns)
                if c_style is not None:
                    self.default_concept_style = {
                        "font-name": c_style.get("font-name", "Verdana"),
                        "font-size": c_style.get("font-size", "12"),
                        "font-color": c_style.get("font-color", "0,0,0,255"),
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
                self.connections.append(ConnectionData(cid, from_id, to_id))
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
                c_style.set("background-color", self._color_to_rgba(first_c.fill_color))
                c_style.set("border-color", self._color_to_rgba(first_c.border_color))
                c_style.set("border-thickness", str(int(first_c.border_width)))
            first_l = next((n for n in self.concepts.values() if n.is_linker), None)
            if first_l is not None:
                l_style.set("font-name", first_l.font_family)
                l_style.set("font-size", str(int(first_l.font_size)))
                l_style.set("font-color", self._color_to_rgba(first_l.font_color))
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
            "background-color": c_style.get("background-color"),
            "border-color": c_style.get("border-color"),
            "border-thickness": c_style.get("border-thickness"),
        }
        self.default_linker_style = {
            "font-name": l_style.get("font-name"),
            "font-size": l_style.get("font-size"),
            "font-color": l_style.get("font-color"),
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
        # Find nearest anchor index
        min_dist = None
        best_index = 0
        for i, anchor in enumerate(node.anchor_positions()):
            d = (anchor - pos).manhattanLength()
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
        self.label_item = QGraphicsTextItem(self.data.label, self)
        self.label_item.setDefaultTextColor(QColor(self.data.font_color))
        self.label_item.setFont(QFont(self.data.font_family, int(self.data.font_size)))
        # Resizing handle (bottom right)
        self.handle_size = 8
        self.resizing = False
        # Connection handle (top centre)
        self.connection_handle_size = 10
        self.dragging_connection = False
        self.connection_preview: Optional[QGraphicsLineItem] = None
        # Connections list
        self.connections: List['ConnectionItem'] = []
        # Set initial position
        self.setPos(self.data.x, self.data.y)
        # Prepare bounding rect
        self._update_bounds()

    def _update_bounds(self) -> None:
        """Update the bounding rect based on label and data width/height."""
        # Adjust label width based on text
        metrics = QFontMetrics(QFont(self.data.font_family, int(self.data.font_size)))
        text_rect = metrics.boundingRect(self.data.label)
        # Ensure width >= text width + padding
        pad = 10
        min_w = text_rect.width() + pad * 2
        min_h = text_rect.height() + pad * 2
        if self.data.width < min_w:
            self.data.width = min_w
        if self.data.height < min_h:
            self.data.height = min_h
        # Update label position
        self.label_item.setPlainText(self.data.label)
        # Center label
        label_rect = self.label_item.boundingRect()
        # We will adjust in paint
        self.prepareGeometryChange()

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
        # Draw label centred
        metrics = QFontMetrics(QFont(self.data.font_family, int(self.data.font_size)))
        text_rect = metrics.boundingRect(self.data.label)
        text_x = (rect.width() - text_rect.width()) / 2
        text_y = (rect.height() - text_rect.height()) / 2 + metrics.ascent()
        painter.setFont(QFont(self.data.font_family, int(self.data.font_size)))
        painter.setPen(QPen(QColor(self.data.font_color)))
        painter.drawText(QPointF(text_x, text_y), self.data.label)
        # Resize handle
        painter.setBrush(QBrush(QColor("#888888")))
        painter.setPen(Qt.NoPen)
        resize_rect = QRectF(rect.right() - self.handle_size, rect.bottom() - self.handle_size, self.handle_size, self.handle_size)
        painter.drawRect(resize_rect)
        # Connection handle (top centre)
        painter.setBrush(QBrush(QColor("#6666FF")))
        ch_size = self.connection_handle_size
        ch_x = (rect.width() - ch_size) / 2
        ch_y = -ch_size / 2  # Half above the node
        painter.drawEllipse(QPointF(ch_x + ch_size / 2, ch_y + ch_size / 2), ch_size / 2, ch_size / 2)

    def anchor_positions(self) -> List[QPointF]:
        """Return available anchor points on the node for connections."""
        w = self.data.width
        h = self.data.height
        return [
            QPointF(w / 2, 0),            # top
            QPointF(w / 2, h),            # bottom
            QPointF(0, h / 2),            # left
            QPointF(w, h / 2),            # right
        ]

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
                # Create a preview line
                self.connection_preview = QGraphicsLineItem()
                pen = QPen(QColor("#000000"))
                pen.setStyle(Qt.DashLine)
                self.connection_preview.setPen(pen)
                self.scene_ref.addItem(self.connection_preview)
                self.connection_preview.setZValue(-1000)
                self.request_connection.emit(self, event)
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
            metrics = QFontMetrics(QFont(self.data.font_family, int(self.data.font_size)))
            text_rect = metrics.boundingRect(self.data.label)
            min_w = text_rect.width() + 20
            min_h = text_rect.height() + 20
            if new_w < min_w:
                new_w = min_w
            if new_h < min_h:
                new_h = min_h
            self.data.width = new_w
            self.data.height = new_h
            self.prepareGeometryChange()
            self.update()
            # Update connections
            for conn in self.connections:
                conn.update_position()
            return
        elif self.dragging_connection and self.connection_preview is not None:
            # Update preview line
            line = self.connection_preview.line()
            p0 = self.mapToScene(self.anchor_positions()[0])  # top anchor as start
            p1 = event.scenePos()
            self.connection_preview.setLine(p0.x(), p0.y(), p1.x(), p1.y())
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
            if self.connection_preview is not None:
                self.scene_ref.removeItem(self.connection_preview)
                self.connection_preview = None
            self.dragging_connection = False
            # Inform scene of drop
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
        return super().itemChange(change, value)


class ConnectionItem(QGraphicsLineItem):
    """Represents a connection line between two nodes."""

    def __init__(self, source: NodeItem, source_anchor: int, dest: NodeItem, dest_anchor: int) -> None:
        super().__init__()
        self.source = source
        self.dest = dest
        self.source_anchor = source_anchor
        self.dest_anchor = dest_anchor
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
        self.setZValue(-1)

    def update_position(self) -> None:
        """Recompute the endpoints of the line based on anchor positions."""
        s_anchor = self.source.anchor_positions()[self.source_anchor]
        d_anchor = self.dest.anchor_positions()[self.dest_anchor]
        # Map to scene coordinates
        s_point = self.source.mapToScene(s_anchor)
        d_point = self.dest.mapToScene(d_anchor)
        self.setLine(s_point.x(), s_point.y(), d_point.x(), d_point.y())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Toggle anchor handles
            if self.source_handle or self.dest_handle:
                self.hide_anchor_handles()
            else:
                self.show_anchor_handles()
            event.accept()
            return
        super().mousePressEvent(event)

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

    def hide_anchor_handles(self):
        """Remove anchor handles from scene."""
        if self.source_handle:
            scene = self.scene()
            if scene:
                scene.removeItem(self.source_handle)
                scene.removeItem(self.dest_handle)
            self.source_handle = None
            self.dest_handle = None


#####################################################
# Main editor window with scene, view and style panel
#####################################################

class StyleEditor(QWidget):
    """Panel for editing the style of a selected node."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.current_node: Optional[NodeItem] = None
        layout = QVBoxLayout(self)
        # Label
        self.label_edit = QLineEdit()
        layout.addWidget(QLabel("Etichetta:"))
        layout.addWidget(self.label_edit)
        # Font size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 72)
        layout.addWidget(QLabel("Dimensione font:"))
        layout.addWidget(self.font_size_spin)
        # Fill colour
        self.fill_btn = QPushButton("Scegli colore riempimento")
        layout.addWidget(self.fill_btn)
        # Border colour
        self.border_btn = QPushButton("Scegli colore bordo")
        layout.addWidget(self.border_btn)
        # Border thickness
        self.border_thick_spin = QSpinBox()
        self.border_thick_spin.setRange(0, 10)
        layout.addWidget(QLabel("Spessore bordo:"))
        layout.addWidget(self.border_thick_spin)
        # Apply button
        self.apply_btn = QPushButton("Applica")
        layout.addWidget(self.apply_btn)
        layout.addStretch(1)
        # Connect signals
        self.apply_btn.clicked.connect(self.apply_changes)
        self.fill_btn.clicked.connect(self.choose_fill_color)
        self.border_btn.clicked.connect(self.choose_border_color)

    def set_node(self, node: Optional[NodeItem]) -> None:
        """Set the node currently being edited."""
        self.current_node = node
        if node is None:
            self.label_edit.setText("")
            self.font_size_spin.setValue(12)
            self.border_thick_spin.setValue(1)
            self.fill_btn.setStyleSheet("")
            self.border_btn.setStyleSheet("")
            return
        # Populate fields
        self.label_edit.setText(node.data.label)
        self.font_size_spin.setValue(int(node.data.font_size))
        self.border_thick_spin.setValue(int(node.data.border_width))
        # Set button colours
        self.fill_btn.setStyleSheet(f"background-color: {node.data.fill_color}")
        self.border_btn.setStyleSheet(f"background-color: {node.data.border_color}")

    def choose_fill_color(self):
        if self.current_node is None:
            return
        colour = QColorDialog.getColor(QColor(self.current_node.data.fill_color), self, "Colore riempimento")
        if colour.isValid():
            self.current_node.data.fill_color = colour.name()
            self.fill_btn.setStyleSheet(f"background-color: {colour.name()}")
            self.current_node.update()

    def choose_border_color(self):
        if self.current_node is None:
            return
        colour = QColorDialog.getColor(QColor(self.current_node.data.border_color), self, "Colore bordo")
        if colour.isValid():
            self.current_node.data.border_color = colour.name()
            self.border_btn.setStyleSheet(f"background-color: {colour.name()}")
            self.current_node.update()

    def apply_changes(self):
        if self.current_node is None:
            return
        # Apply label
        text = self.label_edit.text()
        if text:
            self.current_node.data.label = text
        # Font size
        self.current_node.data.font_size = self.font_size_spin.value()
        # Border thickness
        self.current_node.data.border_width = self.border_thick_spin.value()
        # Update visuals
        self.current_node.update()
        self.current_node._update_bounds()
        # Update connections (in case size changed)
        for conn in self.current_node.connections:
            conn.update_position()


class ConceptMapEditor(QMainWindow):
    """Main window for concept map editing."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CXL Concept Map Editor (PyQt)")
        self.resize(1024, 768)
        # Data model
        self.document = CXLDocument()
        # Scene and view
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.setCentralWidget(self.view)
        # Style editor dock
        self.style_editor = StyleEditor(self)
        self.dock = QDockWidget("Proprietà", self)
        self.dock.setWidget(self.style_editor)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.hide()
        # Actions and menu
        self._create_actions()
        self._create_menu()
        # Map state
        self.node_items: Dict[str, NodeItem] = {}
        self.connection_items: List[ConnectionItem] = []
        # Scene events
        self.scene.setBackgroundBrush(QBrush(QColor("#F5F5F5")))
        self.scene.mouseDoubleClickEvent = self.scene_double_click

    def _create_actions(self) -> None:
        self.new_act = QAction("Nuovo", self)
        self.open_act = QAction("Apri…", self)
        self.save_act = QAction("Salva", self)
        self.save_as_act = QAction("Salva come…", self)
        self.exit_act = QAction("Esci", self)
        # Connect actions
        self.new_act.triggered.connect(self.new_file)
        self.open_act.triggered.connect(self.open_file)
        self.save_act.triggered.connect(self.save_file)
        self.save_as_act.triggered.connect(self.save_file_as)
        self.exit_act.triggered.connect(self.close)

    def _create_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(self.new_act)
        file_menu.addAction(self.open_act)
        file_menu.addAction(self.save_act)
        file_menu.addAction(self.save_as_act)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_act)

    def clear_scene(self) -> None:
        """Remove all items from scene and reset state."""
        self.scene.clear()
        self.node_items.clear()
        self.connection_items.clear()
        self.style_editor.set_node(None)
        self.dock.hide()

    def new_file(self) -> None:
        """Create a new blank map."""
        self.document.new_map()
        self.clear_scene()
        self.setWindowTitle("Nuova mappa - Editor concetti (PyQt)")

    def open_file(self) -> None:
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
            # Determine anchor indices (default top to top)
            src_anchor = 0
            dst_anchor = 1
            connection_item = ConnectionItem(src_item, src_anchor, dst_item, dst_anchor)
            self.scene.addItem(connection_item)
            self.connection_items.append(connection_item)
        self.setWindowTitle(f"{filepath} - Editor concetti (PyQt)")
        # Fit view
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

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
        QMessageBox.information(self, "Salvataggio", "File salvato con successo.")

    def save_file_as(self) -> None:
        """Save current map under a new name."""
        filepath, _ = QFileDialog.getSaveFileName(self, "Salva file CXL", "", "CXL (*.cxl)")
        if not filepath:
            return
        self.update_model_from_scene()
        try:
            self.document.save(filepath)
            self.setWindowTitle(f"{filepath} - Editor concetti (PyQt)")
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
        # Update connections
        self.document.connections.clear()
        for conn_item in self.connection_items:
            conn = ConnectionData(str(uuid.uuid4()), conn_item.source.data.id, conn_item.dest.data.id)
            self.document.connections.append(conn)

    ###################
    # Scene interaction
    ###################
    def scene_double_click(self, event) -> None:
        """Handle double click on scene to create new concept."""
        # Determine if click on empty area
        pos = event.scenePos()
        item = self.scene.itemAt(pos, self.view.transform())
        if item is None:
            # Prompt for label
            text, ok = QInputDialog.getText(self, "Nuovo concetto", "Etichetta del concetto:")
            if not ok or not text:
                return
            # Create new concept data
            cid = str(uuid.uuid4())
            style = self.document.default_linker_style if False else self.document.default_concept_style
            data = ConceptData(
                cid, text,
                pos.x(), pos.y(),
                120, 60,
                font_family=style.get("font-name", "Verdana"),
                font_size=float(style.get("font-size", "12")),
                font_color=self.document._parse_color(style.get("font-color", "0,0,0,255")),
                fill_color=self.document._parse_color(style.get("background-color", "237,244,246,255")),
                border_color=self.document._parse_color(style.get("border-color", "0,0,0,255")),
                border_width=float(style.get("border-thickness", "1")),
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
            event.accept()
            return
        # Otherwise default
        QGraphicsScene.mouseDoubleClickEvent(self.scene, event)

    def handle_connection_request(self, source_node: NodeItem, event) -> None:
        """Handle the start and end of a connection drag."""
        if event.type() == 2:  # MouseMove
            # Nothing to do here; preview handled inside node
            return
        # On release: determine destination
        pos = event.scenePos()
        dest_item = self.scene.itemAt(pos, self.view.transform())
        # If dest_item is NodeItem (and not the same) => connect
        if isinstance(dest_item, NodeItem) and dest_item is not source_node:
            # Determine if shift pressed
            modifiers = QApplication.keyboardModifiers()
            use_linker = not (modifiers & Qt.ShiftModifier)
            if use_linker:
                # Prompt for linking phrase label
                text, ok = QInputDialog.getText(self, "Nuova frase-legame", "Etichetta della frase-legame:")
                if not ok or not text:
                    text = ""
                # Create linking phrase node
                lid = str(uuid.uuid4())
                style = self.document.default_linker_style if self.document.default_linker_style else self.document.default_concept_style
                data = ConceptData(
                    lid, text,
                    (source_node.pos().x() + dest_item.pos().x()) / 2,
                    (source_node.pos().y() + dest_item.pos().y()) / 2,
                    90, 20,
                    font_family=style.get("font-name", "Verdana"),
                    font_size=float(style.get("font-size", "12")),
                    font_color=self.document._parse_color(style.get("font-color", "0,0,0,255")),
                    fill_color=self.document._parse_color(style.get("background-color", "237,244,246,255")),
                    border_color=self.document._parse_color(style.get("border-color", "0,0,0,255")),
                    border_width=float(style.get("border-thickness", "1")),
                    is_linker=True
                )
                self.document.concepts[lid] = data
                linker_item = NodeItem(data, self.scene)
                linker_item.request_connection.connect(self.handle_connection_request)
                linker_item.node_moved.connect(self.node_moved)
                linker_item.node_selected.connect(self.node_selected)
                self.scene.addItem(linker_item)
                self.node_items[lid] = linker_item
                # Create connections
                conn1 = ConnectionItem(source_node, 0, linker_item, 1)
                conn2 = ConnectionItem(linker_item, 0, dest_item, 1)
                self.scene.addItem(conn1)
                self.scene.addItem(conn2)
                self.connection_items.extend([conn1, conn2])
            else:
                # Direct connection
                conn = ConnectionItem(source_node, 0, dest_item, 1)
                self.scene.addItem(conn)
                self.connection_items.append(conn)
            # Update model connection list
            # Note: Document connections will be updated on save
            return
        # If no destination: create new concept
        # Prompt for new concept label
        text, ok = QInputDialog.getText(self, "Nuovo concetto", "Etichetta del concetto:")
        if not ok or not text:
            return
        # Create new node
        cid = str(uuid.uuid4())
        style = self.document.default_concept_style
        data = ConceptData(
            cid, text,
            pos.x(), pos.y(),
            120, 60,
            font_family=style.get("font-name", "Verdana"),
            font_size=float(style.get("font-size", "12")),
            font_color=self.document._parse_color(style.get("font-color", "0,0,0,255")),
            fill_color=self.document._parse_color(style.get("background-color", "237,244,246,255")),
            border_color=self.document._parse_color(style.get("border-color", "0,0,0,255")),
            border_width=float(style.get("border-thickness", "1")),
            is_linker=False
        )
        self.document.concepts[cid] = data
        new_item = NodeItem(data, self.scene)
        new_item.request_connection.connect(self.handle_connection_request)
        new_item.node_moved.connect(self.node_moved)
        new_item.node_selected.connect(self.node_selected)
        self.scene.addItem(new_item)
        new_item.setPos(pos)
        self.node_items[cid] = new_item
        # Connect from source to new concept
        modifiers = QApplication.keyboardModifiers()
        use_linker = not (modifiers & Qt.ShiftModifier)
        if use_linker:
            # Prompt for linking phrase label
            text2, ok2 = QInputDialog.getText(self, "Nuova frase-legame", "Etichetta della frase-legame:")
            if not ok2 or not text2:
                text2 = ""
            lid = str(uuid.uuid4())
            style_l = self.document.default_linker_style if self.document.default_linker_style else self.document.default_concept_style
            ldata = ConceptData(
                lid, text2,
                (source_node.pos().x() + new_item.pos().x()) / 2,
                (source_node.pos().y() + new_item.pos().y()) / 2,
                90, 20,
                font_family=style_l.get("font-name", "Verdana"),
                font_size=float(style_l.get("font-size", "12")),
                font_color=self.document._parse_color(style_l.get("font-color", "0,0,0,255")),
                fill_color=self.document._parse_color(style_l.get("background-color", "237,244,246,255")),
                border_color=self.document._parse_color(style_l.get("border-color", "0,0,0,255")),
                border_width=float(style_l.get("border-thickness", "1")),
                is_linker=True
            )
            self.document.concepts[lid] = ldata
            linker_item2 = NodeItem(ldata, self.scene)
            linker_item2.request_connection.connect(self.handle_connection_request)
            linker_item2.node_moved.connect(self.node_moved)
            linker_item2.node_selected.connect(self.node_selected)
            self.scene.addItem(linker_item2)
            self.node_items[lid] = linker_item2
            # Connections
            conn1 = ConnectionItem(source_node, 0, linker_item2, 1)
            conn2 = ConnectionItem(linker_item2, 0, new_item, 1)
            self.scene.addItem(conn1)
            self.scene.addItem(conn2)
            self.connection_items.extend([conn1, conn2])
        else:
            # Direct connection
            conn = ConnectionItem(source_node, 0, new_item, 1)
            self.scene.addItem(conn)
            self.connection_items.append(conn)

    def node_moved(self, node: NodeItem) -> None:
        """Update style editor when node moves; reserved for future use."""
        pass

    def node_selected(self, node: NodeItem) -> None:
        """Handle node selection: open style editor for concept nodes."""
        # Only show style editor for concepts (not linking phrases)
        if node.data.is_linker:
            self.style_editor.set_node(None)
            self.dock.hide()
            return
        self.style_editor.set_node(node)
        self.dock.show()


def main() -> None:
    app = QApplication(sys.argv)
    editor = ConceptMapEditor()
    editor.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()