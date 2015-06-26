"""autogenerated by genpy from hrl_multimodal_anomaly_detection/Rectangle.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct


class Rectangle(genpy.Message):
  _md5sum = "6f56da242edf56d7bb971ac8a43fd805"
  _type = "hrl_multimodal_anomaly_detection/Rectangle"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """int32 lowX
int32 lowY
int32 highX
int32 highY
int32 r
int32 g
int32 b
int32 thickness
"""
  __slots__ = ['lowX','lowY','highX','highY','r','g','b','thickness']
  _slot_types = ['int32','int32','int32','int32','int32','int32','int32','int32']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       lowX,lowY,highX,highY,r,g,b,thickness

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(Rectangle, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.lowX is None:
        self.lowX = 0
      if self.lowY is None:
        self.lowY = 0
      if self.highX is None:
        self.highX = 0
      if self.highY is None:
        self.highY = 0
      if self.r is None:
        self.r = 0
      if self.g is None:
        self.g = 0
      if self.b is None:
        self.b = 0
      if self.thickness is None:
        self.thickness = 0
    else:
      self.lowX = 0
      self.lowY = 0
      self.highX = 0
      self.highY = 0
      self.r = 0
      self.g = 0
      self.b = 0
      self.thickness = 0

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_struct_8i.pack(_x.lowX, _x.lowY, _x.highX, _x.highY, _x.r, _x.g, _x.b, _x.thickness))
    except struct.error as se: self._check_types(se)
    except TypeError as te: self._check_types(te)

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      end = 0
      _x = self
      start = end
      end += 32
      (_x.lowX, _x.lowY, _x.highX, _x.highY, _x.r, _x.g, _x.b, _x.thickness,) = _struct_8i.unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_struct_8i.pack(_x.lowX, _x.lowY, _x.highX, _x.highY, _x.r, _x.g, _x.b, _x.thickness))
    except struct.error as se: self._check_types(se)
    except TypeError as te: self._check_types(te)

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      end = 0
      _x = self
      start = end
      end += 32
      (_x.lowX, _x.lowY, _x.highX, _x.highY, _x.r, _x.g, _x.b, _x.thickness,) = _struct_8i.unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
_struct_8i = struct.Struct("<8i")
