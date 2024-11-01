#
# Type definitions
#

from abc import abstractmethod
from typing import Dict, List, Optional, cast, Any
from marshmallow import Schema, fields, post_load
from loguru import logger

class Tag():
    # NOTE: Most tag types should have a corresponding TypedSchema which can be used to validate and unmarshal dictionary-like data into the tag object.

    # Tag types should implement a way to convert themselves back into a dictionary-like format
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


class TypedSchema(Schema):
    # TypedSchema implements unmarshal which should take arbitrary data, validate it with the schema, and return the correct object type
    @abstractmethod
    def unmarshal(self, data: Any) -> Tag:
        pass

class VideoTag(Tag):
    # VideoTag represents a single tag in a video, possibly containing a text label
    #
    # Has attributes
    # - start_time: int (required) (in milliseconds)
    # - end_time: int (required) (in milliseconds)
    # - text: str (optional) (the text of the tag, sometimes this is not relevant (i.e shot detection))
    # - confidence: float (optional) (the confidence of the tag)
    # - coalesce: bool (optional) (whether this tag should be coalesced with other tags when aggregated (i.e when aggregated)) TODO: remove this and place logic in config
    def __init__(self,  start_time: int, end_time: int, text: Optional[str]=None, data: Optional[dict]=None, confidence: Optional[float]=None, coalesce: Optional[bool]=None):
        self.start_time = start_time
        self.end_time = end_time
        self.text = text
        self.data = data
        self.confidence = confidence
        self.coalesce = coalesce

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        if self.data is not None:
            res["data"] = self.data
        if self.text is not None:
            res["text"] = self.text
        if self.confidence is not None:
            res["confidence"] = self.confidence
        if self.coalesce is not None:
            res["coalesce"] = self.coalesce
        return res

class VideoTagSchema(TypedSchema):
    start_time = fields.Integer()
    end_time = fields.Integer()
    data = fields.Dict(required=False)
    text = fields.String(required=False) 
    confidence = fields.Float(required=False)
    coalesce = fields.Boolean(required=False)

    def unmarshal(self, data: Any) -> VideoTag:
        data = self.load(data)
        return VideoTag(**data)

class AggTag(Tag):
    # AggTag represents a collection of tags in a video, aggregated over a certain time period
    #
    # Has attributes
    # - start_time: int (required) (in milliseconds)
    # - end_time: int (required) (in milliseconds)
    # - tags: dict (a dictionary mapping a track name to a list of video tags)
    def __init__(self, start_time: int, end_time: int, tags: Dict[str, List[VideoTag]]):
        self.start_time = start_time
        self.end_time = end_time
        self.tags = tags

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "tags": {k: [tag.to_dict() for tag in v] for k, v in self.tags.items()}
        }

class AggTagSchema(TypedSchema):
    start_time = fields.Integer()
    end_time = fields.Integer()
    tags = fields.Dict(fields.String(), fields.List(fields.Nested(VideoTagSchema)))
    
    def unmarshal(self, data: Any) -> AggTag:
        data = self.load(data)
        return AggTag(**data)
    
"""
The following Tag classes and their associated schemaas are private members of the FrameTag class and should not be used outside of the FrameTag class
"""

class _Box(Tag):
    # _Box represents a bounding box in an image
    def __init__(self, x1: float, x2: float, y1: float, y2: float, x3: Optional[float]=None, x4: Optional[float]=None, y3: Optional[float]=None, y4: Optional[float]=None):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        assert all(p is None for p in [x3, x4, y3, y4]) or all(p is not None for p in [x3, x4, y3, y4]), f"Either all or none of x3, x4, y3, y4 must be specified. Got x3={x3}, x4={x4}, y3={y3}, y4={y4}"
        if x3:
            self.x3 = x3
            self.x4 = x4
            self.y3 = y3
            self.y4 = y4

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "x1": self.x1,
            "x2": self.x2,
            "y1": self.y1,
            "y2": self.y2
        }
        if hasattr(self, "x3"):
            res["x3"] = self.x3
            res["x4"] = self.x4
            res["y3"] = self.y3
            res["y4"] = self.y4
        return res

class _BoxSchema(Schema):
    x1 = fields.Float()
    x2 = fields.Float()
    x3 = fields.Float(required=False)
    x4 = fields.Float(required=False)
    y1 = fields.Float()
    y2 = fields.Float()
    y3 = fields.Float(required=False)
    y4 = fields.Float(required=False)
    @post_load
    def make_box(self, data, **kwargs) -> _Box:
        return _Box(**data)

# _FrameTag represents a single tag in a frame
# private class for use in FrameTag
class _FrameTag(Tag):
    def __init__(self, text: str, confidence: float, box: Optional[_Box]=None, true_box: Optional[_Box]=None):
        self.text = text
        self.confidence = confidence
        self.box = box
        self.true_box = true_box

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "text": self.text,
            "confidence": self.confidence,
        }
        if self.box:
            res["box"] = self.box.to_dict()
        if self.true_box:
            res["true_box"] = self.true_box.to_dict()
        return res

class _FrameTagSchema(Schema):
    text = fields.String()
    confidence = fields.Float()
    box = fields.Nested(_BoxSchema, required=False) 
    true_box = fields.Nested(_BoxSchema, required=False)
    @post_load
    def make_tag(self, data, **kwargs) -> _FrameTag:
        return _FrameTag(**data)
    
class FrameTag(Tag):
    # FrameTag represents a collection of tags in a single frame (from a single model), see _FrameTag for the structure of the tags 
    #
    # Has attributes
    # - tags: List[_FrameTag] (required) (a list of individual frametags in the frame)
    # - timestamp_sec: int (optional) (the timestamp of the frame in milliseconds [NOTE: variable name is a misnomer!])
    # NOTE: timestamp_sec should typically always be included, some functionality may not work otherwise since we rely on this for knowing where in the video the frame is
    def __init__(self, tags: List[_FrameTag], timestamp_sec: Optional[int]=None):
        self.tags = tags
        self.timestamp_sec = timestamp_sec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tags": [tag.to_dict() for tag in self.tags],
            "timestamp_sec": self.timestamp_sec
        }

class FrameTagSchema(TypedSchema):    
    tags = fields.List(fields.Nested(_FrameTagSchema), missing=[], default=[])
    timestamp_sec = fields.Integer(required=False)

    @post_load
    def make_tag(self, data, **kwargs) -> FrameTag:
        return FrameTag(**data)
    
    def unmarshal(self, data) -> FrameTag:
        try:
            res = cast(FrameTag, self.load(data))
        except Exception as e:
            logger.error(f"Error unmarshalling {data}")
            raise e
        return res
    
# FrameTags represents a collection of FrameTags in a video
# maps frame_idx -> Tag
FrameTags = Dict[int, FrameTag]