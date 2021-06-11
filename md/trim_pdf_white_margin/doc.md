
当一个 `pdf`文档的白空格很大的时候，希望可以将它们裁剪少一点。从网络中找了下，发现没有现成的工具可以实现这个功能。

所以用`python`实现了一个简单的脚本。

- `pdf`的每一个文字都有一个坐标（文字的左上角，文字的右下角），根据这个信息，可以从一页的文字中计算出整个`pdf`最合适的大小。

- 根据每一页的大小，可以将最适合整个`pdf`文档的大小计算出来。

- 然后进行裁剪，保存。

```python
import fitz

def get_tight_margin_rect(page):
    r = None
    page_rawdict = page.getText("rawdict")
    for block in page_rawdict["blocks"]:
        if block["type"] != 0:
            continue

        x0, y0, x1, y1 = block["bbox"]
        if r is None:
            r = fitz.Rect(x0, y0, x1, y1)
            continue
        x0 = min(x0, r.x0)
        y0 = min(y0, r.y0)
        x1 = max(x1, r.x1)
        y1 = max(y1, r.y1)
        r = fitz.Rect(x0, y0, x1, y1)
    return r


def get_doc_tight_margin_rect(document):
    dr = None
    for index, page in enumerate(document):
        r = get_tight_margin_rect(page)
        if r is None:
            continue
        if dr is None:
            dr = r
            continue
        x0 = min(r.x0, dr.x0)
        y0 = min(r.y0, dr.y0)
        x1 = max(r.x1, dr.x1)
        y1 = max(r.y1, dr.y1)
        dr = fitz.Rect(x0, y0, x1, y1)
    return dr

def create_clip_doc(origin, empty, clip, name):
    for index, page in enumerate(origin):
        page.setCropBox(clip)
        new_page = empty.newPage(index, page.rect.width, page.rect.height)
        new_page.showPDFpage(fitz.Rect(0, 0, page.rect.width, page.rect.height), origin, index)
    empty.save(name, garbage=3, deflate=True)

if __name__ == "__main__":
    origin = fitz.open("bv_cvxbook.pdf")
    doc = fitz.open()

    clip = get_doc_tight_margin_rect(origin)
    create_clip_doc(origin, doc, clip, "trim_bv_cvxbook.pdf")

```
