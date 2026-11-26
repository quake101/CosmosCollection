"""
Windows progress window for the standalone updater: a plain Win32 window
with a status line and a progress bar, drawn through ctypes so the updater
needs no GUI toolkit. Runs in the "--progress-ui" process (see
CosmosUpdaterProgress.py).

The one-file build's manifest (PyInstaller's default) turns on Common
Controls 6, so the bar and text get the current Windows look. The window is
per-monitor DPI aware and re-lays itself out when dragged to a monitor with
a different scale.

Win32 windows don't follow dark mode on their own. When apps are set to
Dark (and High Contrast is off), this one asks DWM for a dark title bar,
paints a dark background and light text, and draws the bar unthemed in
dark colors: a dark track and the light shade of the user's accent color.
(Windows' own dark bar style, DarkMode_CopyEngine, has no busy animation:
its marquee is drawn light. It also reverts whenever the bar switches
between busy and percent.)
"""

import ctypes
import queue
import sys
from ctypes import wintypes

from CosmosUpdaterProgress import start_command_reader

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
shell32 = ctypes.WinDLL("shell32", use_last_error=True)
comctl32 = ctypes.WinDLL("comctl32", use_last_error=True)
uxtheme = ctypes.WinDLL("uxtheme", use_last_error=True)
dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)

LRESULT = ctypes.c_ssize_t
LONG_PTR = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_SETFONT = 0x0030
WM_CTLCOLORSTATIC = 0x0138
WM_DPICHANGED = 0x02E0
WM_USER = 0x0400
WM_APP = 0x8000

WS_CAPTION = 0x00C00000
WS_SYSMENU = 0x00080000
WS_CHILD = 0x40000000
WS_VISIBLE = 0x10000000
CS_NOCLOSE = 0x0200
SS_NOPREFIX = 0x0080
PBS_SMOOTH = 0x01
PBS_MARQUEE = 0x08
PBM_SETPOS = WM_USER + 2
PBM_SETRANGE32 = WM_USER + 6
PBM_SETBARCOLOR = WM_USER + 9
PBM_SETMARQUEE = WM_USER + 10
PBM_SETBKCOLOR = 0x2001  # CCM_SETBKCOLOR
GWL_STYLE = -16
GWL_EXSTYLE = -20
WS_EX_STATICEDGE = 0x00020000
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_FRAMECHANGED = 0x0020
COLOR_BTNFACE = 15
SW_SHOWNORMAL = 1
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
MONITOR_DEFAULTTONEAREST = 2
ICC_PROGRESS_CLASS = 0x20
IDI_APPLICATION = 32512
IDC_ARROW = 32512
SPI_GETHIGHCONTRAST = 0x0042
HCF_HIGHCONTRASTON = 0x00000001
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19  # Windows 10 before 20H1

# Windows 11's dark dialog colors (a COLORREF is 0x00BBGGRR)
DARK_BACKGROUND = 0x00202020
DARK_TEXT = 0x00FFFFFF
DARK_TRACK = 0x00383838  # the track of Windows' own dark progress bar
DARK_FILL_DEFAULT = 0x00FFCD60  # its fill, #60CDFF, if the accent can't be read

# Layout in 96-DPI pixels
CLIENT_WIDTH = 440
MARGIN = 18
LABEL_HEIGHT = 46  # room for up to three lines, e.g. a failure message
GAP = 10
BAR_HEIGHT = 16
CLIENT_HEIGHT = MARGIN + LABEL_HEIGHT + GAP + BAR_HEIGHT + MARGIN
WINDOW_STYLE = WS_CAPTION | WS_SYSMENU
WINDOW_EX_STYLE = 0


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT), ("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    ]


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


class INITCOMMONCONTROLSEX(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("dwICC", wintypes.DWORD)]


class HIGHCONTRASTW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwFlags", wintypes.DWORD),
                ("lpszDefaultScheme", wintypes.LPWSTR)]


def _declare():
    f = user32.DefWindowProcW
    f.argtypes, f.restype = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM], LRESULT
    f = user32.RegisterClassExW
    f.argtypes, f.restype = [ctypes.POINTER(WNDCLASSEXW)], wintypes.ATOM
    f = user32.CreateWindowExW
    f.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                  wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
    f.restype = wintypes.HWND
    for name in ("SendMessageW", "PostMessageW"):
        f = getattr(user32, name)
        f.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        f.restype = LRESULT if name == "SendMessageW" else wintypes.BOOL
    user32.SetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.restype = LRESULT
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.UpdateWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user32.AdjustWindowRectEx.argtypes = [ctypes.POINTER(wintypes.RECT), wintypes.DWORD,
                                          wintypes.BOOL, wintypes.DWORD]
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
    user32.MonitorFromPoint.restype = wintypes.HMONITOR
    user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MONITORINFO)]
    user32.GetSysColorBrush.argtypes = [ctypes.c_int]
    user32.GetSysColorBrush.restype = wintypes.HBRUSH
    user32.LoadIconW.argtypes = [wintypes.HINSTANCE, wintypes.LPVOID]
    user32.LoadIconW.restype = wintypes.HICON
    user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, wintypes.LPVOID]
    user32.LoadCursorW.restype = wintypes.HANDLE
    user32.PostQuitMessage.argtypes = [ctypes.c_int]
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    gdi32.CreateFontW.argtypes = [ctypes.c_int] * 5 + [wintypes.DWORD] * 8 + [wintypes.LPCWSTR]
    gdi32.CreateFontW.restype = wintypes.HFONT
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.CreateSolidBrush.argtypes = [wintypes.COLORREF]
    gdi32.CreateSolidBrush.restype = wintypes.HBRUSH
    gdi32.SetTextColor.argtypes = [wintypes.HDC, wintypes.COLORREF]
    gdi32.SetBkColor.argtypes = [wintypes.HDC, wintypes.COLORREF]
    user32.SystemParametersInfoW.argtypes = [wintypes.UINT, wintypes.UINT, wintypes.LPVOID, wintypes.UINT]
    uxtheme.SetWindowTheme.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR]
    dwmapi.DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD]
    shell32.ExtractIconExW.argtypes = [wintypes.LPCWSTR, ctypes.c_int, ctypes.POINTER(wintypes.HICON),
                                       ctypes.POINTER(wintypes.HICON), wintypes.UINT]
    comctl32.InitCommonControlsEx.argtypes = [ctypes.POINTER(INITCOMMONCONTROLSEX)]
    # Windows 10 1607+; callers fall back when these are missing
    if hasattr(user32, "GetDpiForWindow"):
        user32.GetDpiForWindow.argtypes = [wintypes.HWND]
        user32.GetDpiForWindow.restype = wintypes.UINT
    if hasattr(user32, "AdjustWindowRectExForDpi"):
        user32.AdjustWindowRectExForDpi.argtypes = [ctypes.POINTER(wintypes.RECT), wintypes.DWORD,
                                                    wintypes.BOOL, wintypes.DWORD, wintypes.UINT]
    if hasattr(user32, "SetProcessDpiAwarenessContext"):
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    if hasattr(user32, "GetWindowLongPtrW"):
        get_long, set_long = user32.GetWindowLongPtrW, user32.SetWindowLongPtrW
    else:  # 32-bit Python
        get_long, set_long = user32.GetWindowLongW, user32.SetWindowLongW
    get_long.argtypes, get_long.restype = [wintypes.HWND, ctypes.c_int], LONG_PTR
    set_long.argtypes, set_long.restype = [wintypes.HWND, ctypes.c_int, LONG_PTR], LONG_PTR
    return get_long, set_long


def _set_dpi_awareness():
    try:
        # Per-monitor v2: sharp text on every monitor, title bar scales too
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except AttributeError:
        pass
    try:
        user32.SetProcessDPIAware()
    except AttributeError:
        pass


def _use_dark_mode():
    """Whether apps are set to Dark in Settings > Personalization > Colors,
    with High Contrast off (its own colors come from the system)."""
    contrast = HIGHCONTRASTW()
    contrast.cbSize = ctypes.sizeof(HIGHCONTRASTW)
    if (user32.SystemParametersInfoW(SPI_GETHIGHCONTRAST, contrast.cbSize, ctypes.byref(contrast), 0)
            and contrast.dwFlags & HCF_HIGHCONTRASTON):
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as key:
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False  # before Windows 10 1809, or never set: light


def _dark_fill_color():
    """The light shade of the user's accent color, which dark-mode controls
    fill with: the second of the eight RGBA colors (lightest first) in
    Explorer's AccentPalette."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Accent") as key:
            palette = winreg.QueryValueEx(key, "AccentPalette")[0]
        red, green, blue = palette[4], palette[5], palette[6]
        return red | green << 8 | blue << 16
    except (OSError, IndexError, TypeError):
        return DARK_FILL_DEFAULT


def _dark_title_bar(hwnd):
    value = wintypes.BOOL(True)
    for attribute in (DWMWA_USE_IMMERSIVE_DARK_MODE, DWMWA_USE_IMMERSIVE_DARK_MODE_OLD):
        if dwmapi.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)) == 0:
            return


def _window_dpi(hwnd):
    try:
        dpi = user32.GetDpiForWindow(hwnd)
        if dpi:
            return dpi
    except AttributeError:
        pass
    return 96


def _app_icons():
    """The exe's own icon when frozen, else the stock application icon."""
    if getattr(sys, "frozen", False):
        large, small = wintypes.HICON(), wintypes.HICON()
        if shell32.ExtractIconExW(sys.executable, 0, ctypes.byref(large), ctypes.byref(small), 1):
            return large, small
    icon = user32.LoadIconW(None, ctypes.c_void_p(IDI_APPLICATION))
    return icon, icon


def _scale(value, dpi):
    return (value * dpi + 48) // 96


class _Window:
    def __init__(self, title):
        self._get_long, self._set_long = _declare()
        self._commands = queue.Queue()
        self._font = None
        self._closing = False
        self._marquee = False
        self._dark = _use_dark_mode()
        self._dark_brush = gdi32.CreateSolidBrush(DARK_BACKGROUND) if self._dark else None
        self._dark_fill = _dark_fill_color() if self._dark else None

        _set_dpi_awareness()
        icc = INITCOMMONCONTROLSEX(ctypes.sizeof(INITCOMMONCONTROLSEX), ICC_PROGRESS_CLASS)
        comctl32.InitCommonControlsEx(ctypes.byref(icc))

        hinstance = kernel32.GetModuleHandleW(None)
        large_icon, small_icon = _app_icons()
        self._wndproc = WNDPROC(self._on_message)  # keep a reference
        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.style = CS_NOCLOSE  # can't cancel halfway through swapping files
        wc.lpfnWndProc = self._wndproc
        wc.hInstance = hinstance
        wc.hIcon = large_icon
        wc.hIconSm = small_icon
        wc.hCursor = user32.LoadCursorW(None, ctypes.c_void_p(IDC_ARROW))
        wc.hbrBackground = self._dark_brush or user32.GetSysColorBrush(COLOR_BTNFACE)
        wc.lpszClassName = "CosmosCollectionUpdaterProgress"
        if not user32.RegisterClassExW(ctypes.byref(wc)):
            raise ctypes.WinError(ctypes.get_last_error())

        # Create on the monitor under the mouse (where the Update button was
        # just clicked), hidden, so its DPI is known before sizing it.
        work = self._work_area()
        self.hwnd = user32.CreateWindowExW(
            WINDOW_EX_STYLE, wc.lpszClassName, title, WINDOW_STYLE,
            work.left, work.top, 100, 100, None, None, hinstance, None)
        if not self.hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        if self._dark:
            _dark_title_bar(self.hwnd)

        self.label = user32.CreateWindowExW(
            0, "STATIC", "", WS_CHILD | WS_VISIBLE | SS_NOPREFIX,
            0, 0, 10, 10, self.hwnd, None, hinstance, None)
        self.bar = user32.CreateWindowExW(
            0, "msctls_progress32", "", WS_CHILD | WS_VISIBLE | PBS_SMOOTH,
            0, 0, 10, 10, self.hwnd, None, hinstance, None)
        user32.SendMessageW(self.bar, PBM_SETRANGE32, 0, 100)
        self._apply_bar_theme()

        dpi = _window_dpi(self.hwnd)
        width, height = self._outer_size(dpi)
        x = work.left + max(0, (work.right - work.left - width) // 2)
        y = work.top + max(0, (work.bottom - work.top - height) // 3)
        user32.SetWindowPos(self.hwnd, None, x, y, width, height, SWP_NOZORDER | SWP_NOACTIVATE)
        self._layout(dpi)
        self._set_busy()

        user32.ShowWindow(self.hwnd, SW_SHOWNORMAL)
        user32.UpdateWindow(self.hwnd)
        user32.SetForegroundWindow(self.hwnd)

    @staticmethod
    def _work_area():
        point = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(point))
        monitor = user32.MonitorFromPoint(point, MONITOR_DEFAULTTONEAREST)
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        user32.GetMonitorInfoW(monitor, ctypes.byref(info))
        return info.rcWork

    @staticmethod
    def _outer_size(dpi):
        rect = wintypes.RECT(0, 0, _scale(CLIENT_WIDTH, dpi), _scale(CLIENT_HEIGHT, dpi))
        try:
            user32.AdjustWindowRectExForDpi(ctypes.byref(rect), WINDOW_STYLE, False,
                                            WINDOW_EX_STYLE, dpi)
        except AttributeError:
            user32.AdjustWindowRectEx(ctypes.byref(rect), WINDOW_STYLE, False, WINDOW_EX_STYLE)
        return rect.right - rect.left, rect.bottom - rect.top

    def _layout(self, dpi):
        """Size the font and controls for dpi."""
        old_font = self._font
        # Segoe UI 9pt, the Windows dialog font
        self._font = gdi32.CreateFontW(-_scale(12, dpi), 0, 0, 0, 400, 0, 0, 0,
                                       1, 0, 0, 5, 0, "Segoe UI")  # 5 = ClearType
        user32.SendMessageW(self.label, WM_SETFONT, self._font, 1)
        margin, width = _scale(MARGIN, dpi), _scale(CLIENT_WIDTH - 2 * MARGIN, dpi)
        label_height = _scale(LABEL_HEIGHT, dpi)
        flags = SWP_NOZORDER | SWP_NOACTIVATE
        user32.SetWindowPos(self.label, None, margin, margin, width, label_height, flags)
        user32.SetWindowPos(self.bar, None, margin, margin + label_height + _scale(GAP, dpi),
                            width, _scale(BAR_HEIGHT, dpi), flags)
        if old_font:
            gdi32.DeleteObject(old_font)

    def _apply_bar_theme(self):
        """Dark mode: draw the bar unthemed, in dark colors. Re-applied after
        each switch between busy and percent, which changes its style."""
        if self._dark:
            uxtheme.SetWindowTheme(self.bar, "", "")
            user32.SendMessageW(self.bar, PBM_SETBKCOLOR, 0, DARK_TRACK)
            user32.SendMessageW(self.bar, PBM_SETBARCOLOR, 0, self._dark_fill)
            # Unthemed, the bar gives itself a sunken 3D edge (light gray and
            # white), which looks out of place on a dark window
            ex_style = self._get_long(self.bar, GWL_EXSTYLE)
            if ex_style & WS_EX_STATICEDGE:
                self._set_long(self.bar, GWL_EXSTYLE, ex_style & ~WS_EX_STATICEDGE)
                user32.SetWindowPos(self.bar, None, 0, 0, 0, 0,
                                    SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER)

    def _set_busy(self):
        if not self._marquee:
            style = self._get_long(self.bar, GWL_STYLE)
            self._set_long(self.bar, GWL_STYLE, style | PBS_MARQUEE)
            self._apply_bar_theme()
            user32.SendMessageW(self.bar, PBM_SETMARQUEE, 1, 30)
            self._marquee = True

    def _set_percent(self, percent):
        if self._marquee:
            user32.SendMessageW(self.bar, PBM_SETMARQUEE, 0, 0)
            style = self._get_long(self.bar, GWL_STYLE)
            self._set_long(self.bar, GWL_STYLE, style & ~PBS_MARQUEE)
            self._apply_bar_theme()
            self._marquee = False
        user32.SendMessageW(self.bar, PBM_SETPOS, percent, 0)

    def post(self, kind, value):
        """Called from the stdin reader thread."""
        self._commands.put((kind, value))
        user32.PostMessageW(self.hwnd, WM_APP, 0, 0)

    def _apply_commands(self):
        while True:
            try:
                kind, value = self._commands.get_nowait()
            except queue.Empty:
                return
            if kind == "text":
                user32.SetWindowTextW(self.label, value)
            elif kind == "progress":
                try:
                    self._set_percent(max(0, min(100, int(value))))
                except ValueError:
                    pass
            elif kind == "busy":
                self._set_busy()
            elif kind == "close" and not self._closing:
                self._closing = True
                user32.DestroyWindow(self.hwnd)
                return

    def _on_message(self, hwnd, msg, wparam, lparam):
        try:
            if msg == WM_APP:
                self._apply_commands()
                return 0
            if msg == WM_CLOSE:
                return 0  # no closing it mid-update (Alt+F4)
            if msg == WM_CTLCOLORSTATIC and self._dark:
                gdi32.SetTextColor(wparam, DARK_TEXT)
                gdi32.SetBkColor(wparam, DARK_BACKGROUND)
                return self._dark_brush
            if msg == WM_DPICHANGED:
                suggested = ctypes.cast(lparam, ctypes.POINTER(wintypes.RECT)).contents
                user32.SetWindowPos(hwnd, None, suggested.left, suggested.top,
                                    suggested.right - suggested.left,
                                    suggested.bottom - suggested.top,
                                    SWP_NOZORDER | SWP_NOACTIVATE)
                self._layout(wparam & 0xFFFF)
                return 0
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
        except Exception:
            pass
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def run(self):
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        for handle in (self._font, self._dark_brush):
            if handle:
                gdi32.DeleteObject(handle)


def run(title, _icon_path=None):
    window = _Window(title)
    start_command_reader(window.post)
    window.run()
