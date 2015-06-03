"""autogenerated by genpy from hrl_feeding_task/AnglesTimeoutMsg.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct


class AnglesTimeoutMsg(genpy.Message):
  _md5sum = "6d1665d882173067cb8c7f03d5898f41"
  _type = "hrl_feeding_task/AnglesTimeoutMsg"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """# A representation of posture in free space, composed of an array of joint angles, along with a timeout value used in the mpcBaseAction() method setPostureGoal()

float64[7] angles
float64 timeout

"""
  __slots__ = ['angles','timeout']
  _slot_types = ['float64[7]','float64']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       angles,timeout

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(AnglesTimeoutMsg, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.angles is None:
        self.angles = [0.,0.,0.,0.,0.,0.,0.]
      if self.timeout is None:
        self.timeout = 0.
    else:
      self.angles = [0.,0.,0.,0.,0.,0.,0.]
      self.timeout = 0.

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
      buff.write(_struct_7d.pack(*self.angles))
      buff.write(_struct_d.pack(self.timeout))
    except struct.error as se: self._check_types(se)
    except TypeError as te: self._check_types(te)

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      end = 0
      start = end
      end += 56
      self.angles = _struct_7d.unpack(str[start:end])
      start = end
      end += 8
      (self.timeout,) = _struct_d.unpack(str[start:end])
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
      buff.write(self.angles.tostring())
      buff.write(_struct_d.pack(self.timeout))
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
      start = end
      end += 56
      self.angles = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=7)
      start = end
      end += 8
      (self.timeout,) = _struct_d.unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
_struct_7d = struct.Struct("<7d")
_struct_d = struct.Struct("<d")
