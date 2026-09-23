"""
CYBERPUNK WEBCAM FILTER
=======================

WHAT THIS DOES
--------------
Turns your webcam feed into a "cyberpunk skin" style effect:
  - YOU get a glowing neon edge outline (like a holographic/robotic overlay)
  - The BACKGROUND behind you gets darkened and covered in a scrolling neon
    grid + moving scanlines, so it always looks like it's "alive" and moving,
    even if you sit still.
  - The whole frame gets a bit of chromatic aberration (RGB channel split)
    and the occasional glitch flicker for extra sci-fi flavor.

HOW IT WORKS (high level)
--------------------------
1. We use OpenCV's built-in `BackgroundSubtractorMOG2`. This watches the
   video over time and learns what the "static background" looks like.
   Anything that MOVES relative to that learned background (i.e. you)
   is flagged as "foreground". This is what lets us treat you and the
   background differently, with no extra AI models to download.
2. We build the neon background layer and the neon-edged "you" layer
   separately, each frame.
3. We use the foreground mask to blend them together: foreground pixels
   show the neon-edge you, background pixels show the moving grid.
4. A few small full-frame effects (scanlines, chromatic aberration,
   glitch) are applied on top for polish.

FILE STRUCTURE
--------------
This is a single, self-contained script on purpose (easy to read start to
finish, nothing hidden across multiple files). It's organized as:
    - CONFIG constants           -> tweak the look here
    - small effect functions     -> each does ONE visual trick
    - process_frame()            -> combines all the effects for one frame
    - main()                     -> opens the webcam and runs the loop

HOW TO RUN
----------
    pip install opencv-python numpy
    python cyberpunk_filter.py

Press 'q' to quit the window.
"""

import cv2
import numpy as np
import time

# --------------------------------------------------------------------------
# CONFIG - tweak these to change the vibe
# --------------------------------------------------------------------------
NEON_COLOR_A = (255, 0, 200)   # BGR - magenta, used for the "you" edge glow
NEON_COLOR_B = (255, 255, 0)   # BGR - cyan, the edge glow cycles towards this
GRID_COLOR = (255, 60, 0)      # BGR - blue-ish neon for the background grid
GRID_SPACING = 40              # pixels between grid lines
SCAN_LINE_SPACING = 6          # pixels between scanlines
SCAN_LINE_SPEED = 2            # how many pixels the scanlines shift per frame
GLITCH_CHANCE = 0.04           # probability (per frame) of a glitch pop
CHROMATIC_SHIFT = 3            # pixels of red/blue channel offset


def make_neon_background(frame, frame_count):
    """
    Takes the ORIGINAL frame and turns it into a dark, animated
    'cyberpunk backdrop': darker/desaturated base + a scrolling neon grid
    drawn on top. The grid position depends on frame_count, so it never
    stops moving even if nothing in the real scene changes.
    """
    # Darken and desaturate the real background so the neon grid stands out
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    dark = (dark.astype(np.float32) * 0.25).astype(np.uint8)  # dim it down

    h, w = frame.shape[:2]
    grid = np.zeros_like(frame)

    # Scroll offset: makes the grid crawl diagonally over time
    offset = (frame_count * 1) % GRID_SPACING

    # Vertical lines
    for x in range(-GRID_SPACING, w + GRID_SPACING, GRID_SPACING):
        cv2.line(grid, (x + offset, 0), (x + offset, h), GRID_COLOR, 1)
    # Horizontal lines
    for y in range(-GRID_SPACING, h + GRID_SPACING, GRID_SPACING):
        cv2.line(grid, (0, y + offset), (w, y + offset), GRID_COLOR, 1)

    # Blend grid glow onto the dark background
    background = cv2.addWeighted(dark, 1.0, grid, 0.5, 0)
    return background


def make_neon_person(frame, frame_count):
    """
    Takes the ORIGINAL frame and produces a glowing-edge 'hologram' version
    of whatever is in it (we only end up USING this where the foreground
    mask says "this is you"). Uses Canny edge detection, then colorizes
    and thickens the edges into a soft glow.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    glow = cv2.GaussianBlur(edges, (0, 0), sigmaX=3)

    # Cycle the glow color smoothly between NEON_COLOR_A and NEON_COLOR_B
    t = (np.sin(frame_count * 0.05) + 1) / 2  # 0..1 oscillation
    color = tuple(
        int(a * (1 - t) + b * t) for a, b in zip(NEON_COLOR_A, NEON_COLOR_B)
    )

    colored_glow = np.zeros_like(frame)
    for c in range(3):
        colored_glow[:, :, c] = (glow.astype(np.float32) / 255.0 * color[c]).astype(np.uint8)

    # Keep a dim, slightly desaturated version of you underneath the glow
    # so you're still recognisable, not just a wireframe.
    dim_you = (frame.astype(np.float32) * 0.35).astype(np.uint8)
    person = cv2.add(dim_you, colored_glow)
    return person


def add_scanlines(img, frame_count):
    """Draws faint horizontal lines that slide downward every frame."""
    h, w = img.shape[:2]
    out = img.copy()
    offset = (frame_count * SCAN_LINE_SPEED) % SCAN_LINE_SPACING
    overlay = out.copy()
    for y in range(offset, h, SCAN_LINE_SPACING):
        cv2.line(overlay, (0, y), (w, y), (0, 0, 0), 1)
    return cv2.addWeighted(overlay, 0.35, out, 0.65, 0)


def add_chromatic_aberration(img, shift=CHROMATIC_SHIFT):
    """Nudges the red and blue channels apart horizontally for a
    'glitchy lens' look."""
    b, g, r = cv2.split(img)
    b = np.roll(b, shift, axis=1)
    r = np.roll(r, -shift, axis=1)
    return cv2.merge((b, g, r))


def maybe_glitch(img, frame_count):
    """Occasionally slices a random horizontal band of the image and
    shifts it sideways - a classic digital-glitch pop."""
    if np.random.rand() > GLITCH_CHANCE:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    band_h = np.random.randint(6, 20)
    y = np.random.randint(0, max(1, h - band_h))
    shift = np.random.randint(-25, 25)
    out[y:y + band_h] = np.roll(out[y:y + band_h], shift, axis=1)
    return out


def process_frame(frame, mask, frame_count):
    """
    Combines every effect into the final cyberpunk frame:
      1. Build the neon background and neon 'you' separately.
      2. Use the foreground mask to pick, per pixel, which one to show.
      3. Apply full-frame polish effects on top.
    """
    background = make_neon_background(frame, frame_count)
    person = make_neon_person(frame, frame_count)

    # mask is 0/255 single channel -> make it 0..1 float, 3 channels,
    # so we can blend background/person smoothly instead of a hard cutout.
    mask_f = cv2.GaussianBlur(mask, (9, 9), 0).astype(np.float32) / 255.0
    mask_3 = cv2.merge([mask_f, mask_f, mask_f])

    combined = (person.astype(np.float32) * mask_3 +
                background.astype(np.float32) * (1 - mask_3)).astype(np.uint8)

    combined = add_scanlines(combined, frame_count)
    combined = add_chromatic_aberration(combined)
    combined = maybe_glitch(combined, frame_count)
    return combined


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Is another app using it?")
        return

    # MOG2 learns the background over the first couple of seconds -
    # stand out of frame briefly when you start it for the cleanest mask.
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32,
                                               detectShadows=False)

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror, feels more natural

        mask = fgbg.apply(frame)
        # Clean up small noise specks in the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                 np.ones((3, 3), np.uint8))

        output = process_frame(frame, mask, frame_count)

        cv2.imshow("Cyberpunk Filter (press q to quit)", output)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
