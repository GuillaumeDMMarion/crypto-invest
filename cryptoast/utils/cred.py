'''
Objects for credentials management.
'''
# other
import os

class Cred(object):
  '''
  Simple placeholder for credential management.

  Args:
    apis (list): All apis for which to keep key- and secret-values of.
    key_secret_names (list): The names of the key- and secret-values.
  '''
  def __init__(self, apis=None, key_secret_names=('KEY', 'SECRET')):
    self._apis = apis
    self._key_secret_names = key_secret_names

  @property
  def key_secret_names(self):
    return self._key_secret_names

  @property
  def apis(self):
    return self._apis

  def __repr__(self):
    if self.apis is not None:
      return str(self.apis)
    return super().__repr__()

  def __getitem__(self, key):
    try:
      assert(self.apis is not None)
    except AssertionError:
      raise KeyError('Empty apis arg at Cred intitialization.')
    return self.get(self.apis[key])

  def get(self, api, key_secret_names=None):
    """
    Args:
      api (str): The api name.
      key_secret_names (list): The names of the key- and secret-values.

    Returns:
      (generator) The key- and secret-values of the specified api.
    """
    if key_secret_names is None:
      key_secret_names = self.key_secret_names
    return (os.environ[api + '_' + name] for name in key_secret_names)

  def set(self, api, values, key_secret_names=None):
    """
    Args:
      api (str): The api name.
      key_secret_names (list): The names of the key- and secret-values.

    Returns:
      None; Sets the key- and secret-values of the specified api as environment variables.
    """
    if key_secret_names is None:
      key_secret_names = self.key_secret_names
    for value, name in zip(values, key_secret_names):
      assert (isinstance(value, str))
      os.environ[api+ '_' + name] = value
