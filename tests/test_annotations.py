import pytest

from doc_redact.annotations import BBox, Document, RelativeBBox, load_documents, parse_labelstudio, save_documents


def test_intersection_uses_top_left_origin():
    region = BBox(left=0, top=100, width=50, height=20)
    lower_word = BBox(left=10, top=110, width=10, height=10)  # bottom half of region
    assert region.intersection_area(lower_word) == 100
    assert region.intersection_area(BBox(left=0, top=200, width=5, height=5)) == 0


def test_parse_labelstudio_converts_percentages_and_strips_upload_prefix():
    export = [
        {
            "data": {"image": "/data/upload/3/954fb992-89773a9d-d230.jpg"},
            "annotations": [
                {
                    "result": [
                        {
                            "original_width": 200,
                            "original_height": 1000,
                            "value": {"x": 10, "y": 50, "width": 25, "height": 1.5, "rectanglelabels": ["ssn"]},
                        }
                    ]
                }
            ],
        }
    ]
    [doc] = parse_labelstudio(export, "https://host/FUNSD")
    assert doc.url == "https://host/FUNSD/89773a9d-d230.jpg"
    assert doc.stem == "89773a9d-d230"
    assert doc.annotations[0].category == "ssn"
    assert doc.annotations[0].bbox == BBox(left=20, top=500, width=50, height=15)


def test_percent_boxes_load_and_convert_with_image_size(tmp_path):
    path = tmp_path / "mixed.json"
    path.write_text(
        '[{"url": "https://host/a.jpg", "annotations": ['
        '{"category": "ssn", "bbox": {"left": 10, "top": 20, "width": 30, "height": 40}},'
        '{"category": "name", "bbox": {"x": 50, "y": 25, "width": 10, "height": 2.5}}]}]'
    )
    [doc] = load_documents(path)
    assert isinstance(doc.annotations[1].bbox, RelativeBBox)
    pixels = doc.in_pixels((800, 1000))
    assert pixels.annotations[0].bbox == BBox(10, 20, 30, 40)
    assert pixels.annotations[1].bbox == BBox(left=400, top=250, width=80, height=25)
    save_documents([doc], path)
    assert load_documents(path) == [doc]  # percent boxes are written back as percent


def test_unknown_bbox_format_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('[{"url": "u", "annotations": [{"category": "ssn", "bbox": {"x1": 1, "y1": 2}}]}]')
    with pytest.raises(ValueError, match="unrecognized bbox format"):
        load_documents(path)


def test_documents_round_trip(tmp_path):
    docs = [Document(url="https://host/a.jpg", image="/up/x-a.jpg", annotations=[])]
    path = tmp_path / "docs.json"
    save_documents(docs, path)
    assert load_documents(path) == docs
