"""Minimal brace-aware RTF -> text extractor (no deps). For thesis handoff prep."""
import re, sys

SKIP_DESTS = {'fonttbl', 'colortbl', 'stylesheet', 'pict', 'themedata',
              'colorschememapping', 'latentstyles', 'datastore', 'rsidtbl',
              'generator', 'info', 'xmlnstbl', 'listtable', 'listoverridetable',
              'mmathPr', 'wgrffmtfilter', 'pgptbl'}


def rtf_to_text(data):
    out = []
    i, n, depth = 0, len(data), 0
    skip_stack = []
    while i < n:
        c = data[i]
        if c == '{':
            depth += 1
            i += 1
            m = re.match(r"\\\*?\\?([a-zA-Z]+)", data[i:i + 40])
            cw = m.group(1) if m else None
            is_skip = (cw in SKIP_DESTS) or data[i:i + 2] == '\\*'
            skip_stack.append(depth if is_skip else None)
            continue
        if c == '}':
            if skip_stack and skip_stack[-1] == depth:
                skip_stack.pop()
            elif skip_stack:
                skip_stack.pop()
            depth -= 1
            i += 1
            continue
        skipping = any(x is not None for x in skip_stack)
        if c == '\\':
            m = re.match(r"\\([a-zA-Z]+)(-?\d+)? ?", data[i:])
            if m:
                w, arg = m.group(1), m.group(2)
                if not skipping:
                    if w in ('par', 'pard', 'line', 'sect'):
                        out.append('\n')
                    elif w == 'tab':
                        out.append('\t')
                    elif w == 'u' and arg:
                        try:
                            out.append(chr(int(arg) % 65536))
                        except Exception:
                            pass
                i += m.end()
                continue
            m = re.match(r"\\'([0-9a-fA-F]{2})", data[i:])
            if m:
                if not skipping:
                    try:
                        out.append(bytes([int(m.group(1), 16)]).decode('cp1252', 'ignore'))
                    except Exception:
                        pass
                i += m.end()
                continue
            m = re.match(r"\\([{}\\])", data[i:])
            if m:
                if not skipping:
                    out.append(m.group(1))
                i += m.end()
                continue
            i += 1
            continue
        if not skipping and c not in '\r\n':
            out.append(c)
        i += 1
    t = ''.join(out)
    t = re.sub(r'\n[ \t]+', '\n', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


if __name__ == '__main__':
    raw = open(sys.argv[1], 'rb').read().decode('latin-1', 'ignore')
    # Strip embedded-image hex payloads (huge runs of hex digits) so the
    # char-by-char parser isn't crushed by 20MB of \pict binary.
    raw = re.sub(r'(?:[0-9a-fA-F]\s*){300,}', ' ', raw)
    t = rtf_to_text(raw)
    t = t.encode('utf-8', 'replace').decode('utf-8')  # drop stray surrogates
    open(sys.argv[2], 'w', encoding='utf-8').write(t)
    print("CHARS:", len(t))
