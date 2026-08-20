from jku_mirror import JKU_BASE, _allowed_url, canonical_scheme_lines, extract_links


def test_allowed_url_is_host_and_prefix_scoped():
    assert _allowed_url(JKU_BASE + "foo.php?id=3")
    assert not _allowed_url("https://evil.example/research/matrix-multiplication/foo")
    assert not _allowed_url("https://www.algebra.uni-linz.ac.at/Students/foo")
    assert not _allowed_url(JKU_BASE + "logo.png")


def test_extract_links_includes_iframe_option_and_inline_js():
    text = '''
    <a href="scheme.php?id=1">one</a>
    <iframe src="frame/list.php"></iframe>
    <select><option value="show.php?scheme=7">seven</option><option value="23">bad</option></select>
    <script>const endpoint = "api/scheme.php?id=9";</script>
    '''
    got = set(extract_links(text, JKU_BASE))
    assert JKU_BASE + "scheme.php?id=1" in got
    assert JKU_BASE + "frame/list.php" in got
    assert JKU_BASE + "show.php?scheme=7" in got
    assert JKU_BASE + "api/scheme.php?id=9" in got
    assert all(not x.endswith("/23") for x in got)


def test_scheme_content_detection_plain_and_pre():
    lines = ["(a11+a12)*(b22-b23)*(c31+c32)" for _ in range(23)]
    assert canonical_scheme_lines(("\n".join(lines) + "\n").encode()) == lines
    wrapped = ("<html><pre>" + "\n".join(lines) + "</pre></html>").encode()
    assert canonical_scheme_lines(wrapped) == lines


def test_scheme_content_rejects_wrong_rank_and_shape():
    assert canonical_scheme_lines(("\n".join(["a11*b11*c11"] * 22)).encode()) is None
    assert canonical_scheme_lines(("\n".join(["a44*b11*c11"] * 23)).encode()) is None
