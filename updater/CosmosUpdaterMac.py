"""
macOS progress window for the standalone updater: an AppKit window with a
status line and a progress bar, driven through the Objective-C runtime with
ctypes so the updater needs no GUI toolkit. Runs in the "--progress-ui"
process (see CosmosUpdaterProgress.py), so even a crash here can't
interrupt the update itself.

AppKit draws it, so it follows Light/Dark mode and the system accent color
on its own. Nothing here subclasses an Objective-C class: the main thread
pumps events in a short loop and applies commands that the stdin reader
thread queues.
"""

import ctypes
import queue
from ctypes import c_bool, c_char_p, c_double, c_long, c_ulong, c_void_p

from CosmosUpdaterProgress import start_command_reader

_objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
_foundation = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AppKit.framework/AppKit")

_objc.objc_getClass.argtypes = [c_char_p]
_objc.objc_getClass.restype = c_void_p
_objc.sel_registerName.argtypes = [c_char_p]
_objc.sel_registerName.restype = c_void_p
_MSG_SEND = ctypes.cast(_objc.objc_msgSend, c_void_p).value

# AppKit constants
NSApplicationActivationPolicyRegular = 0
NSWindowStyleMaskTitled = 1 << 0
NSBackingStoreBuffered = 2
NSProgressIndicatorStyleBar = 0
NSEventMaskAny = 0xFFFFFFFFFFFFFFFF

# Layout in points (origin at the bottom left)
WIDTH, HEIGHT = 440.0, 112.0
MARGIN = 20.0
BAR_HEIGHT = 20.0
LABEL_HEIGHT = 44.0


class NSRect(ctypes.Structure):
    _fields_ = [("x", c_double), ("y", c_double), ("width", c_double), ("height", c_double)]


_functions = {}
_selectors = {}


def _send(obj, selector, *args, restype=c_void_p, argtypes=()):
    """objc_msgSend cast to this exact signature (required on Apple
    silicon, where variadic and fixed arguments are passed differently)."""
    key = (restype, tuple(argtypes))
    function = _functions.get(key)
    if function is None:
        function = ctypes.CFUNCTYPE(restype, c_void_p, c_void_p, *argtypes)(_MSG_SEND)
        _functions[key] = function
    sel = _selectors.get(selector)
    if sel is None:
        sel = _selectors[selector] = _objc.sel_registerName(selector.encode("ascii"))
    return function(obj, sel, *args)


def _cls(name):
    return _objc.objc_getClass(name.encode("ascii"))


def _nsstring(text):
    return _send(_cls("NSString"), "stringWithUTF8String:", text.encode("utf-8"),
                 argtypes=[c_char_p])


def _new(class_name, init, *args, argtypes=()):
    return _send(_send(_cls(class_name), "alloc"), init, *args, argtypes=argtypes)


class _Window:
    def __init__(self, title, icon_path):
        self._commands = queue.Queue()
        self._closed = False
        self._indeterminate = None

        self._pool = _send(_cls("NSAutoreleasePool"), "new")
        self.app = _send(_cls("NSApplication"), "sharedApplication")
        _send(self.app, "setActivationPolicy:", NSApplicationActivationPolicyRegular,
              restype=c_bool, argtypes=[c_long])
        if icon_path:
            image = _new("NSImage", "initWithContentsOfFile:", _nsstring(icon_path),
                         argtypes=[c_void_p])
            if image:
                _send(self.app, "setApplicationIconImage:", image, restype=None,
                      argtypes=[c_void_p])
        _send(self.app, "finishLaunching", restype=None)

        # Titled only: no close/minimize buttons, it can't be cancelled
        # halfway through swapping files
        self.window = _new(
            "NSWindow", "initWithContentRect:styleMask:backing:defer:",
            NSRect(0, 0, WIDTH, HEIGHT), NSWindowStyleMaskTitled, NSBackingStoreBuffered, False,
            argtypes=[NSRect, c_ulong, c_ulong, c_bool])
        _send(self.window, "setReleasedWhenClosed:", False, restype=None, argtypes=[c_bool])
        _send(self.window, "setTitle:", _nsstring(title), restype=None, argtypes=[c_void_p])
        content = _send(self.window, "contentView")

        self.label = _send(_cls("NSTextField"), "wrappingLabelWithString:", _nsstring(""),
                           argtypes=[c_void_p])
        _send(self.label, "setSelectable:", False, restype=None, argtypes=[c_bool])
        _send(self.label, "setFrame:",
              NSRect(MARGIN, MARGIN + BAR_HEIGHT + 8, WIDTH - 2 * MARGIN, LABEL_HEIGHT),
              restype=None, argtypes=[NSRect])
        _send(content, "addSubview:", self.label, restype=None, argtypes=[c_void_p])

        self.bar = _new("NSProgressIndicator", "initWithFrame:",
                        NSRect(MARGIN, MARGIN, WIDTH - 2 * MARGIN, BAR_HEIGHT),
                        argtypes=[NSRect])
        _send(self.bar, "setStyle:", NSProgressIndicatorStyleBar, restype=None,
              argtypes=[c_ulong])
        _send(self.bar, "setMinValue:", 0.0, restype=None, argtypes=[c_double])
        _send(self.bar, "setMaxValue:", 100.0, restype=None, argtypes=[c_double])
        _send(content, "addSubview:", self.bar, restype=None, argtypes=[c_void_p])
        self._set_busy()

        _send(self.window, "center", restype=None)
        _send(self.window, "makeKeyAndOrderFront:", None, restype=None, argtypes=[c_void_p])
        _send(self.app, "activateIgnoringOtherApps:", True, restype=None, argtypes=[c_bool])

    def _set_busy(self):
        if self._indeterminate is not True:
            _send(self.bar, "setIndeterminate:", True, restype=None, argtypes=[c_bool])
            _send(self.bar, "startAnimation:", None, restype=None, argtypes=[c_void_p])
            self._indeterminate = True

    def _set_percent(self, percent):
        if self._indeterminate is not False:
            _send(self.bar, "stopAnimation:", None, restype=None, argtypes=[c_void_p])
            _send(self.bar, "setIndeterminate:", False, restype=None, argtypes=[c_bool])
            self._indeterminate = False
        _send(self.bar, "setDoubleValue:", float(percent), restype=None, argtypes=[c_double])

    def post(self, kind, value):
        """Called from the stdin reader thread; applied on the main thread."""
        self._commands.put((kind, value))

    def _apply_commands(self):
        while True:
            try:
                kind, value = self._commands.get_nowait()
            except queue.Empty:
                return
            if kind == "text":
                _send(self.label, "setStringValue:", _nsstring(value), restype=None,
                      argtypes=[c_void_p])
            elif kind == "progress":
                try:
                    self._set_percent(max(0, min(100, int(value))))
                except ValueError:
                    pass
            elif kind == "busy":
                self._set_busy()
            elif kind == "close":
                self._closed = True
                return

    def run(self):
        mode = c_void_p.in_dll(_foundation, "NSDefaultRunLoopMode").value
        ns_date = _cls("NSDate")
        while not self._closed:
            pool = _send(_cls("NSAutoreleasePool"), "new")
            until = _send(ns_date, "dateWithTimeIntervalSinceNow:", 0.05, argtypes=[c_double])
            event = _send(self.app, "nextEventMatchingMask:untilDate:inMode:dequeue:",
                          NSEventMaskAny, until, mode, True,
                          argtypes=[ctypes.c_uint64, c_void_p, c_void_p, c_bool])
            if event:
                _send(self.app, "sendEvent:", event, restype=None, argtypes=[c_void_p])
            self._apply_commands()
            _send(self.app, "updateWindows", restype=None)
            _send(pool, "drain", restype=None)
        _send(self.window, "orderOut:", None, restype=None, argtypes=[c_void_p])


def run(title, icon_path=None):
    window = _Window(title, icon_path)
    start_command_reader(window.post)
    window.run()
