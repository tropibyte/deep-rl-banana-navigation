"""Record a GIF of a trained agent playing.

The vector-observation Banana build returns no visual observations at all, so
there are no frames to pull out of the environment. The only way to film it is
to run the Unity window with graphics enabled and screen-capture its client
area, which is what this module does (Windows/Win32; falls back to a full-screen
grab elsewhere).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_IS_WINDOWS = sys.platform.startswith("win")

# Width of the screen-right strip reserved for notification toasts, which
# are always-on-top and would otherwise be captured. Generous on purpose:
# a Windows 11 toast is ~360px, and losing a slice of the frame is a far
# better outcome than publishing someone's notifications.
TOAST_ZONE_PX = 480


def _find_unity_window(timeout: float = 25.0):
    """Return (hwnd, (l, t, r, b)) for the Unity client area, or None."""
    if not _IS_WINDOWS:
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    EnumWindows = user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    deadline = time.time() + timeout
    while time.time() < deadline:
        found = []

        def cb(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True
            cls = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cls, 256)
            title = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, title, 256)
            if "UnityWndClass" in cls.value or "Banana" in title.value:
                found.append(hwnd)
            return True

        EnumWindows(EnumWindowsProc(cb), 0)
        if found:
            hwnd = found[0]
            # Bring it to the front, but never resize or maximise it: the Unity
            # player owns its own resolution, and both resizing it and
            # maximising it provoke a D3D device reset that can block.
            HWND_TOPMOST, SWP_NOSIZE, SWP_NOMOVE = -1, 0x0001, 0x0002
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                SWP_NOSIZE | SWP_NOMOVE)
            user32.SetForegroundWindow(hwnd)

            # Wait for the client size to STOP changing before trusting it.
            # The player resizes itself asynchronously after launch, and a
            # rectangle measured mid-transition can be far larger than the
            # window really occupies -- which is how desktop content behind the
            # window ends up inside the recording.
            prev, stable = None, 0
            for _ in range(30):
                time.sleep(0.25)
                r = wintypes.RECT()
                user32.GetClientRect(hwnd, ctypes.byref(r))
                cur = (r.right, r.bottom)
                stable = stable + 1 if (cur == prev and cur[0] > 0) else 0
                prev = cur
                if stable >= 3:
                    break

            rect = wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            wr = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(wr))
            pt = wintypes.POINT(0, 0)
            user32.ClientToScreen(hwnd, ctypes.byref(pt))

            # Belt and braces: the client area can never be larger than the
            # window that contains it. If the two disagree, trust the smaller.
            w = min(rect.right, wr.right - wr.left)
            h = min(rect.bottom, wr.bottom - wr.top)
            right, bottom = pt.x + w, pt.y + h

            # Windows anchors notification toasts to the RIGHT EDGE OF THE
            # SCREEN and draws them above every other window, so anything in
            # that strip lands in the capture -- including whatever personal
            # content a toast happens to be showing.
            right = min(right, user32.GetSystemMetrics(0) - TOAST_ZONE_PX)

            box = (pt.x, pt.y, right, bottom)
            if box[2] > box[0] and box[3] > box[1]:
                return hwnd, box
        time.sleep(0.5)
    return None


def record_gif(checkpoint: str, out_path: str, episodes: int = 1, fps: int = 30,
               env_path: str | None = None, max_frames: int = 600,
               scale: float = 0.5, every: int = 2, crop_top: float = 0.0) -> Path:
    """Play greedily with graphics on, capturing the Unity window to a GIF."""
    import imageio.v2 as imageio
    from PIL import Image, ImageGrab

    from .agent import DQNAgent
    from .env import BananaEnv

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    agent = DQNAgent.load(checkpoint)
    frames: list[np.ndarray] = []
    scores: list[float] = []

    # graphics ON: there is nothing to film otherwise.
    #
    # train_mode=True even though nothing is being trained. The Banana build's
    # *inference* configuration steps at ~0.5 steps/s on this machine versus
    # ~21 steps/s for the training configuration (both measured with graphics
    # enabled), so filming a single 300-step episode in inference mode takes
    # over ten minutes. The rendered frames are identical either way -- only
    # the wall-clock pacing differs, and playback fps is set here anyway.
    with BananaEnv(exe_path=env_path, no_graphics=False, seed=0, train_mode=True) as env:
        win = _find_unity_window()
        if win is None:
            print("WARNING: could not locate the Unity window; grabbing the full screen.")
        box = win[1] if win else None
        print(f"capturing region {box}" if box else "capturing full screen")

        for ep in range(episodes):
            state = env.reset(train_mode=True)
            score, done, i = 0.0, False, 0
            while not done and len(frames) < max_frames:
                action = agent.act(state, greedy=True)
                state, reward, done, _ = env.step(action)
                score += reward
                if i % every == 0:
                    img = ImageGrab.grab(bbox=box, all_screens=True)
                    if crop_top > 0:
                        # The upper part of the frame is empty ceiling; cropping
                        # it puts the floor (where the bananas are) in frame.
                        img = img.crop((0, int(img.height * crop_top),
                                        img.width, img.height))
                    if scale != 1.0:
                        img = img.resize((int(img.width * scale), int(img.height * scale)),
                                         Image.LANCZOS)
                    frames.append(np.asarray(img.convert("RGB")))
                i += 1
            scores.append(score)
            print(f"  episode {ep + 1}: score {score:.0f}  ({len(frames)} frames)")

    if not frames:
        raise RuntimeError("No frames captured.")

    # A locked or screen-blanked Windows session makes ImageGrab return solid
    # black, which would silently produce a black GIF. Detect that and refuse
    # rather than shipping a broken asset.
    stack = np.stack(frames[:: max(1, len(frames) // 12)])
    if float(stack.std()) < 1.5:
        raise RuntimeError(
            "Captured frames are essentially uniform (std={:.2f}) - the Unity "
            "window was probably not visible. Screen capture needs an unlocked, "
            "active desktop session; re-run this command while logged in."
            .format(float(stack.std())))

    imageio.mimsave(out, frames, fps=fps, loop=0)
    print(f"scores: {scores}  mean {np.mean(scores):.1f}  frames {len(frames)}")
    return out
