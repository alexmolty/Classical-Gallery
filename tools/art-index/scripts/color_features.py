"""Portable color descriptors. No model, NumPy, or image inference is needed."""
import math


VERSION = 1


def _round(value, digits=0):
    scale = 10 ** digits
    return math.floor(value * scale + .5) / scale if digits else math.floor(value + .5)


def _hsv(red, green, blue):
    high, low = max(red, green, blue), min(red, green, blue)
    delta = high - low
    hue = 0.0
    if delta:
        if high == red:
            hue = 60 * ((green - blue) / delta)
        elif high == green:
            hue = 60 * ((blue - red) / delta + 2)
        else:
            hue = 60 * ((red - green) / delta + 4)
        if hue < 0:
            hue += 360
    return hue, delta / high if high else 0.0, high / 255


def analyze_rgba(width, height, pixels):
    """Match color-features.js on an already downsampled RGBA image <=64 px."""
    if (type(width) is not int or type(height) is not int or not 1 <= width <= 64 or
            not 1 <= height <= 64 or len(pixels) != width * height * 4):
        raise ValueError("Expected a 1..64 by 1..64 RGBA image")
    bins, luminance = {}, [0] * 256
    total, saturation = 0, 0.0
    for offset in range(0, len(pixels), 4):
        red, green, blue, alpha = pixels[offset:offset + 4]
        if any(type(value) is not int or not 0 <= value <= 255 for value in (red, green, blue, alpha)):
            raise ValueError("RGBA channels must be bytes")
        if alpha < 16:
            continue
        hue, sat, value = _hsv(red, green, blue)
        if sat < .15 or value < .06:
            key = min(7, math.floor(value * 8))
        else:
            sat_bin = 0 if sat < .45 else 1 if sat < .75 else 2
            value_bin = 0 if value < .33 else 1 if value < .66 else 2
            key = 8 + math.floor(hue / 30) * 9 + sat_bin * 3 + value_bin
        row = bins.setdefault(key, [0, 0, 0, 0])
        row[0] += red
        row[1] += green
        row[2] += blue
        row[3] += 1
        luminance[_round(.2126 * red + .7152 * green + .0722 * blue)] += 1
        total += 1
        saturation += sat
    if not total:
        return None
    palette = []
    for _, (red, green, blue, count) in sorted(bins.items(), key=lambda item: (-item[1][3], item[0]))[:8]:
        palette.append([_round(red / count), _round(green / count), _round(blue / count), _round(count / total, 4)])
    hue, sat, value = _hsv(*palette[0][:3])

    def quantile(probability):
        rank, cumulative = max(1, math.ceil(total * probability)), 0
        for level, count in enumerate(luminance):
            cumulative += count
            if cumulative >= rank:
                return level

    return {"version": VERSION, "palette": palette,
            "dominantHue": _round(hue) % 360 if sat >= .15 and value >= .06 else None,
            "saturation": _round(saturation / total * 100, 1),
            "contrast": _round((quantile(.95) - quantile(.05)) / 255 * 100, 1)}


def analyze_image(image):
    """Decode/orient once at build time; preserve alpha and image proportions."""
    from PIL import Image, ImageOps
    sample = ImageOps.exif_transpose(image).convert("RGBA")
    sample.thumbnail((64, 64), Image.Resampling.LANCZOS)
    return analyze_rgba(sample.width, sample.height, sample.tobytes())


def analyze_legacy_preview(image, padding):
    """Exclude the old inference canvas's symmetric padding before analysis."""
    from PIL import Image, ImageChops
    image = image.convert("RGB")
    bounds = ImageChops.difference(image, Image.new("RGB", image.size, padding)).getbbox()
    if bounds:
        left, top, right, bottom = bounds
        width, height = image.size
        horizontal, vertical = min(left, width - right), min(top, height - bottom)
        if horizontal > vertical:
            image = image.crop((horizontal, 0, width - horizontal, height))
        elif vertical:
            image = image.crop((0, vertical, width, height - vertical))
    return analyze_image(image)


def validate(value):
    """Optional v1 fields must never make an older catalogue unreadable."""
    if not isinstance(value, dict) or value.get("version") != VERSION:
        return False
    palette = value.get("palette")
    if not isinstance(palette, list) or not 1 <= len(palette) <= 8:
        return False
    previous, total = 1, 0
    for row in palette:
        if not isinstance(row, list) or len(row) != 4:
            return False
        if any(type(channel) is not int or not 0 <= channel <= 255 for channel in row[:3]):
            return False
        share = row[3]
        if type(share) not in (int, float) or not .0001 <= share <= 1 or share > previous + 1e-10 or abs(share * 10000 - round(share * 10000)) >= 1e-7:
            return False
        previous, total = share, total + share
    for name in ("saturation", "contrast"):
        metric = value.get(name)
        if type(metric) not in (int, float) or not 0 <= metric <= 100 or abs(metric * 10 - round(metric * 10)) >= 1e-7:
            return False
    hue, saturation, brightness = _hsv(*palette[0][:3])
    expected = _round(hue) % 360 if saturation >= .15 and brightness >= .06 else None
    actual = value.get("dominantHue")
    return total <= 1.0004 + 1e-10 and actual == expected and (actual is None or type(actual) is int)
