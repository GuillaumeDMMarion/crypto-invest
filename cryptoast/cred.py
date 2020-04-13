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
  def __init__(self, apis=None, key_secret_names=None):
    self.apis = apis
    self.key_secret_names = key_secret_names

  def get(self, api, key_secret_names=None):
    """
    Args:
      api (str): The api name.
      key_secret_names (list): The names of the key- and secret-values.

    Returns:
      (generator) The key- and secret-values of the specified api.
    """
    if key_secret_names is not None:
      pass
    else:
      assert (self.key_secret_names is not None), 'No key-secret names provided.'
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
    if key_secret_names is not None:
      pass
    else:
      assert (self.key_secret_names is not None), 'No key-secret names provided.'
      key_secret_names = self.key_secret_names
    for value, name in zip(values, key_secret_names):
      assert (isinstance(value, str))
      os.environ[api+ '_' + name] = value
